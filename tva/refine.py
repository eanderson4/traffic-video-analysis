"""Landmark refinement: static vehicles as fixed points.

Vehicles that provably don't move — parked lots, and cars stopped for long
runs in the queue — are fixed world points observed in many frames. Each
frame's homography is polished with a RANSAC affine correction that maps
its observations of those landmarks onto their median world positions,
then corrections are temporally smoothed. This kills frame-local jitter
(the thing that makes parked cars flicker between speed colors) on top of
the mosaic-level drift fix in plate.py.

Run order: stabilize -> detect -> world -> plate -> world -> anchors ->
world (each world pass re-derives positions from the current homographies).
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d

from .stabilize import apply_h

MIN_RUN_S = 2.0        # stopped runs shorter than this aren't landmarks
MIN_LANDMARKS = 10     # per frame, else correction is interpolated
RANSAC_TOL = 5.0       # world px
PLANE_LAT = 3.0        # x car_len: only landmarks this close to the
                       # centerline (elevated carriageways are a different
                       # plane than ground-level lots; mixing them leaves
                       # parallax on the road we're analyzing)


def collect_landmarks(ws, plane=True):
    """[(world xy, {frame: pixel xy})] for static tracks and long stopped
    runs. With plane=True, restricted to the centerline's roadway."""
    wt = ws.load("world_tracks.json")
    fps = ws.meta["fps"]
    cl = None
    if plane:
        try:
            cl = ws.load("centerline.json")
        except FileNotFoundError:
            print("no centerline.json - anchoring on all landmarks")
    px_of = {}
    for tr in ws.load("tracks.json")["tracks"]:
        px_of[tr["id"]] = {o[0]: (o[1], o[2]) for o in tr["obs"]}
    marks = []
    for tr in wt["tracks"]:
        if cl is not None:
            from .kinematics import station_of
            mx = np.array([float(np.median(tr["x"]))])
            my = np.array([float(np.median(tr["y"]))])
            _, lat = station_of(cl["x"], cl["y"], mx, my)
            if lat[0] > PLANE_LAT * wt["car_len_px"]:
                continue
        obs = px_of.get(tr["id"], {})
        spans = []
        if tr["static"]:
            spans.append((0, len(tr["frames"]) - 1))
        else:
            for s, fa, fb in tr["runs"]:
                if s and (fb - fa) / fps >= MIN_RUN_S:
                    ia = tr["frames"].index(fa)
                    ib = tr["frames"].index(fb)
                    spans.append((ia, ib))
        for ia, ib in spans:
            fr = tr["frames"][ia:ib + 1]
            wx = float(np.median(tr["x"][ia:ib + 1]))
            wy = float(np.median(tr["y"][ia:ib + 1]))
            pix = {f: px_of_f for f in fr
                   if (px_of_f := obs.get(f)) is not None}
            if len(pix) >= int(MIN_RUN_S * fps * 0.5):
                marks.append(((wx, wy), pix))
    return marks


def anchors(ws, smooth=None):
    data = ws.load("homographies.json")
    Hs = [np.asarray(m) for m in data["H"]]
    n = len(Hs)
    marks = collect_landmarks(ws)
    if len(marks) < 3 * MIN_LANDMARKS:
        print(f"only {len(marks)} on-plane landmarks - falling back to all")
        marks = collect_landmarks(ws, plane=False)
    print(f"{len(marks)} static landmarks")

    # per-frame affine corrections (world coords), NaN where underconstrained
    corr = np.full((n, 6), np.nan)
    n_used = []
    for i in range(n):
        src, dst = [], []
        for (wx, wy), pix in marks:
            p = pix.get(i)
            if p is None:
                continue
            src.append(apply_h(Hs[i], [p])[0])
            dst.append((wx, wy))
        if len(src) < MIN_LANDMARKS:
            n_used.append(0)
            continue
        import cv2
        A, inl = cv2.estimateAffine2D(
            np.asarray(src, np.float32), np.asarray(dst, np.float32),
            method=cv2.RANSAC, ransacReprojThreshold=RANSAC_TOL)
        if A is None:
            n_used.append(0)
            continue
        corr[i] = A.ravel()
        n_used.append(int(inl.sum()))
    ok = ~np.isnan(corr[:, 0])
    print(f"corrections solved on {ok.sum()}/{n} frames "
          f"(median {np.median([u for u in n_used if u]):.0f} landmarks)")
    if ok.sum() < 5:
        print("too few frames constrained - skipping anchor refinement")
        return

    # Interpolate gaps + light temporal smoothing. smooth=False (fit each
    # frame's correction independently) was tried for ref-frame
    # homographies during R1+K1 integration to cancel their per-frame
    # registration jitter at this layer — it improved the landmark
    # residual but INJECTED comparable per-frame fit noise of its own
    # (~24 bbox landmarks/frame can't beat the jitter it removes; static
    # z_v slip went 16.8 -> 19.8 px/s). Jitter is owned upstream by
    # register.smooth_homographies; corrections stay smooth here.
    idx = np.arange(n)
    if smooth is None:
        smooth = True
    for c in range(6):
        corr[:, c] = np.interp(idx, idx[ok], corr[ok, c])
        if smooth:
            corr[:, c] = gaussian_filter1d(corr[:, c], 2.0)

    out = []
    resid_before = _residual(Hs, marks, None)
    for i in range(n):
        A = np.vstack([corr[i].reshape(2, 3), [0, 0, 1]])
        Hc = A @ Hs[i]
        out.append(Hc / Hc[2, 2])
    resid_after = _residual(out, marks, None)
    print(f"landmark residual: {resid_before:.2f}px -> {resid_after:.2f}px "
          "(world, median)")
    data["H"] = [H.tolist() for H in out]
    data["anchored"] = True
    ws.save("homographies.json", data)


def _residual(Hs, marks, _):
    r = []
    for (wx, wy), pix in marks:
        for f, p in pix.items():
            if f < len(Hs):
                q = apply_h(np.asarray(Hs[f]), [p])[0]
                r.append(float(np.hypot(q[0] - wx, q[1] - wy)))
    return float(np.median(r)) if r else float("nan")
