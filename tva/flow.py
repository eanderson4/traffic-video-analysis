"""Per-track LK optical-flow velocity measurements (z_v).

For every track observation at frame f with a next frame available:
Shi-Tomasi corners inside the bbox eroded ~25% (background pixels live at
the edges), pyramidal LK f -> f+1 with a forward-backward check, both
endpoint sets mapped through their OWN frames' homographies, median world
displacement / dt = a direct velocity measurement in reference-plane px/s.

Mapping each endpoint through its own frame's homography subtracts drone
ego-motion by construction, and — the key property — a stopped car reads
identically ~0 regardless of how badly the homography chain has drifted:
chain error is *shared* by consecutive frames, so it cancels in the
difference. That is why z_v stays sane at clip ends where z_p blows up.

Degenerate flow (motion blur, textureless roofs, glare, corners latched
onto neighbours) is rejected here (< MIN_CORNERS survivors, or high spread
of the per-corner world displacements); the Kalman applies a further
Mahalanobis gate at fuse time. Everything is cached to flow.json so the
filter can rerun without re-reading the 4K video.
"""
import cv2
import numpy as np

from .stabilize import apply_h

ERODE_FRAC = 0.25       # shrink bbox w,h by this before corner detection
MAX_CORNERS = 30
CORNER_QUALITY = 0.01
CORNER_MIN_DIST = 3
MIN_ROI_PX = 8          # skip boxes whose eroded side is smaller than this
MIN_CORNERS = 4         # reject measurement below this many survivors
FB_ERR_PX = 0.6         # forward-backward consistency gate (image px)
SPREAD_MAX_FRAC = 0.06  # reject if MAD of world displacements > frac*car_len
BG_MAX_CORNERS = 400    # background corners for per-frame slip estimation
BG_MIN_DIST = 30
BG_DILATE_FRAC = 0.15   # bbox dilation when masking vehicles out
BG_MIN_CORNERS = 20     # below this the frame's bias is left missing
LK_PARAMS = dict(
    winSize=(21, 21), maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def _median_car_len(tracks):
    sizes = [max(o[3], o[4]) for tr in tracks for o in tr["obs"]]
    return float(np.median(sizes)) if sizes else 100.0


def _roi_corners(gray, cx, cy, w, h):
    """Shi-Tomasi corners inside the eroded bbox, full-frame coords."""
    iw = w * (1.0 - ERODE_FRAC)
    ih = h * (1.0 - ERODE_FRAC)
    if min(iw, ih) < MIN_ROI_PX:
        return None
    x0 = int(round(cx - iw / 2))
    y0 = int(round(cy - ih / 2))
    x1, y1 = int(round(x0 + iw)), int(round(y0 + ih))
    H, W = gray.shape
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, W), min(y1, H)
    if x1 - x0 < MIN_ROI_PX or y1 - y0 < MIN_ROI_PX:
        return None
    p = cv2.goodFeaturesToTrack(gray[y0:y1, x0:x1], MAX_CORNERS,
                                CORNER_QUALITY, CORNER_MIN_DIST)
    if p is None:
        return None
    return p.reshape(-1, 2) + np.array([x0, y0], dtype=np.float32)


def run(ws):
    from .kinematics import jacobian_scale

    meta = ws.meta
    fps = meta["fps"]
    dt = 1.0 / fps
    Hs = [np.asarray(h) for h in ws.load("homographies.json")["H"]]
    data = ws.load("tracks.json")
    car_len = _median_car_len(data["tracks"])
    spread_max = SPREAD_MAX_FRAC * car_len

    # frame -> [(track_id, cx, cy, w, h)] for obs with a usable next frame
    per_frame = {}
    all_boxes = {}   # every obs regardless of next frame, for the bg mask
    for tr in data["tracks"]:
        for f, cx, cy, w, h, conf in tr["obs"]:
            all_boxes.setdefault(f, []).append((cx, cy, w, h))
            if f + 1 < len(Hs):
                per_frame.setdefault(f, []).append((tr["id"], cx, cy, w, h))

    out = {}   # track_id -> lists
    bias = [None] * max(len(Hs) - 1, 0)   # per-frame registration slip
    n_meas = n_rej_corners = n_rej_spread = 0
    cap = cv2.VideoCapture(meta["src"])
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"cannot read {meta['src']}")
    prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    f = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cur = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        obs_here = per_frame.get(f, [])
        # batch all tracks' corners into one LK call per frame pair
        batches = []
        for tid, cx, cy, w, h in obs_here:
            pts = _roi_corners(prev, cx, cy, w, h)
            if pts is None or len(pts) < MIN_CORNERS:
                n_rej_corners += 1
                continue
            batches.append((tid, cx, cy, pts))
        # background corners (all vehicle boxes masked out, dilated 15%):
        # their world displacement SHOULD be zero, so its median measures
        # the frame pair's registration slip — the common-mode error that
        # every z_v at this frame inherits. Cached so the filter can
        # subtract it. tid=None marks the pseudo-batch.
        if f < len(bias):
            mask = np.full(prev.shape, 255, np.uint8)
            Hh, Ww = prev.shape
            for cx, cy, w, h in all_boxes.get(f, []):
                dw, dh = w * (1 + BG_DILATE_FRAC), h * (1 + BG_DILATE_FRAC)
                mask[max(int(cy - dh / 2), 0):min(int(cy + dh / 2), Hh),
                     max(int(cx - dw / 2), 0):min(int(cx + dw / 2), Ww)] = 0
            bg = cv2.goodFeaturesToTrack(prev, BG_MAX_CORNERS,
                                         CORNER_QUALITY, BG_MIN_DIST,
                                         mask=mask)
            if bg is not None and len(bg) >= BG_MIN_CORNERS:
                batches.append((None, 0, 0, bg.reshape(-1, 2)))
        if batches:
            p0 = np.concatenate([b[3] for b in batches]).astype(
                np.float32).reshape(-1, 1, 2)
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None,
                                                 **LK_PARAMS)
            p0b, stb, _ = cv2.calcOpticalFlowPyrLK(cur, prev, p1, None,
                                                   **LK_PARAMS)
            fb = np.linalg.norm((p0 - p0b).reshape(-1, 2), axis=1)
            good = (st.ravel() == 1) & (stb.ravel() == 1) & (fb < FB_ERR_PX)
            p0f, p1f = p0.reshape(-1, 2), p1.reshape(-1, 2)
            i = 0
            for tid, cx, cy, pts in batches:
                n = len(pts)
                g = good[i:i + n]
                a, b = p0f[i:i + n][g], p1f[i:i + n][g]
                i += n
                if tid is None:
                    if g.sum() >= BG_MIN_CORNERS:
                        d = apply_h(Hs[f + 1], b) - apply_h(Hs[f], a)
                        m = np.median(d, axis=0)
                        bias[f] = [round(float(m[0] / dt), 3),
                                   round(float(m[1] / dt), 3), int(g.sum())]
                    continue
                if g.sum() < MIN_CORNERS:
                    n_rej_corners += 1
                    continue
                d = apply_h(Hs[f + 1], b) - apply_h(Hs[f], a)
                med = np.median(d, axis=0)
                spread = float(np.median(np.linalg.norm(d - med, axis=1)))
                jac = jacobian_scale(Hs[f], (cx, cy))
                # gate in image scale: where the chain's Jacobian has blown
                # up, world-px spread inflates with it and says nothing
                # about flow quality
                if spread / max(jac, 1e-6) > spread_max:
                    n_rej_spread += 1
                    continue
                rec = out.setdefault(tid, {"frames": [], "vx": [], "vy": [],
                                           "n": [], "spread": [], "jac": []})
                rec["frames"].append(int(f))
                rec["vx"].append(round(float(med[0] / dt), 2))
                rec["vy"].append(round(float(med[1] / dt), 2))
                rec["n"].append(int(g.sum()))
                rec["spread"].append(round(spread, 3))
                rec["jac"].append(round(jac, 3))
                n_meas += 1
        prev = cur
        f += 1
        if f % 50 == 0:
            print(f"frame {f}: {n_meas} measurements", flush=True)
    cap.release()

    ws.save("flow.json", {
        "car_len_px": car_len, "dt": dt,
        "erode_frac": ERODE_FRAC, "fb_err_px": FB_ERR_PX,
        "spread_max": spread_max,
        "bias": bias,
        "tracks": [{"id": tid, **rec} for tid, rec in sorted(out.items())],
    })
    n_bias = sum(1 for b in bias if b is not None)
    print(f"{n_meas} z_v measurements on {len(out)} tracks "
          f"(rejected: {n_rej_corners} few-corners, {n_rej_spread} spread); "
          f"bg slip measured on {n_bias}/{len(bias)} frame pairs")
