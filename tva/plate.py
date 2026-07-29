"""Median background plate in the reference plane, with drift refinement.

Warp every Nth frame through its homography onto a common canvas and take
the per-pixel median: moving vehicles ghost out, leaving road surface,
markings, and scenery. Long-stopped vehicles (deep gridlock) survive the
median — expect residue where the queue never moves.

Chained adjacent-frame homographies drift over long clips, which smears the
median. So each sampled frame is re-registered against a middle-out mosaic
(features matched in the overlap, RANSAC): corrections can't accumulate
because every sample aligns to the same global reference. The corrections
are then interpolated across all frames and folded back into
homographies.json (original preserved as homographies-raw.json), so world
tracks and renders de-drift too.

Canvas transform T maps world (reference-plane) coords -> plate pixels;
stored in plate.json.
"""
import os
import shutil

import cv2
import numpy as np

MAX_DIM = 4200
STEP = 8


def refine_correction(mosaic_gray, mosaic_mask, warp_gray, warp_mask):
    """Canvas->canvas forward homography F aligning a warped sample to the
    mosaic (identity on failure)."""
    overlap = cv2.erode((mosaic_mask & warp_mask) * 255,
                        np.ones((15, 15), np.uint8))
    if (overlap > 0).sum() < 20000:
        return np.eye(3), 0
    p0 = cv2.goodFeaturesToTrack(mosaic_gray, 900, 0.01, 14, mask=overlap)
    if p0 is None or len(p0) < 40:
        return np.eye(3), 0
    p1, st, _ = cv2.calcOpticalFlowPyrLK(
        mosaic_gray, warp_gray, p0, None, winSize=(41, 41), maxLevel=5,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01))
    good = st.ravel() == 1
    if good.sum() < 40:
        return np.eye(3), 0
    F, inl = cv2.findHomography(p1[good], p0[good], cv2.RANSAC, 3.0)
    if F is None:
        return np.eye(3), 0
    return F, int(inl.sum())


def build(ws, step=STEP, max_dim=MAX_DIM, refine=None):
    meta = ws.meta
    w, h = meta["width"], meta["height"]
    hom = ws.load("homographies.json")
    if refine is None:
        # Default ON for chained v1 homographies only. On ref-frame (R1)
        # homographies the mosaic drift refinement re-anchors on off-deck
        # content and "corrects" the deck registration by ~400 px — it
        # must not run there (R1 report, pipeline notes).
        refine = hom.get("method") != "ref-frame"
        if not refine:
            print("ref-frame homographies: mosaic drift refinement OFF")
    raw_path = ws.path("homographies-raw.json")
    if refine and not os.path.exists(raw_path):
        shutil.copy(ws.path("homographies.json"), raw_path)
    Hs = [np.asarray(m) for m in hom["H"]]

    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    ones = np.ones((4, 1))
    pts = []
    for H in Hs:
        p = np.hstack([corners, ones]) @ H.T
        pts.append(p[:, :2] / p[:, 2:3])
    pts = np.vstack(pts)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    sc = min(1.0, max_dim / max(x1 - x0, y1 - y0))
    T = np.array([[sc, 0, -x0 * sc], [0, sc, -y0 * sc], [0, 0, 1]])
    Tinv = np.linalg.inv(T)
    cw, ch = int((x1 - x0) * sc) + 2, int((y1 - y0) * sc) + 2
    print(f"canvas {cw}x{ch}, scale {sc:.3f}")

    cap = cv2.VideoCapture(meta["src"])
    frames, idxs = [], []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0 and i < len(Hs):
            frames.append(frame)
            idxs.append(i)
        i += 1
    cap.release()
    n = len(frames)
    print(f"{n} samples")

    def warp(k, F=np.eye(3)):
        M = F @ T @ Hs[idxs[k]]
        img = cv2.warpPerspective(frames[k], M, (cw, ch))
        msk = cv2.warpPerspective(np.ones((h, w), np.uint8), M, (cw, ch))
        return img, msk

    Fs = [np.eye(3)] * n
    if refine:
        mid = n // 2
        order = [mid]
        for d in range(1, n):
            for k in (mid - d, mid + d):
                if 0 <= k < n:
                    order.append(k)
        img, msk = warp(mid)
        mosaic, mmask = img, msk
        mg = cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY)
        for k in order[1:]:
            img, msk = warp(k)
            F, inl = refine_correction(
                mg, mmask, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), msk)
            Fs[k] = F
            if not np.allclose(F, np.eye(3)):
                img, msk = warp(k, F)
            fill = (mmask == 0) & (msk == 1)
            mosaic[fill] = img[fill]
            mmask |= msk
            mg = cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY)
        shifts = [float(np.hypot(*(F[:2, 2]))) for F in Fs]
        print(f"drift corrections: median {np.median(shifts):.1f}px, "
              f"max {max(shifts):.1f}px (canvas)")

    stack, masks = [], []
    for k in range(n):
        img, msk = warp(k, Fs[k])
        stack.append(img)
        masks.append(msk)
    arr = np.stack(stack).astype(np.float32)
    cov = np.stack(masks)
    arr[cov == 0] = np.nan
    plate = np.nanmedian(arr, axis=0)
    coverage = cov.sum(axis=0)
    plate[coverage == 0] = 0
    cv2.imwrite(ws.path("plate.png"), plate.astype(np.uint8))
    cv2.imwrite(f"{ws.qa}/plate-coverage.png",
                (255 * coverage / max(1, coverage.max())).astype(np.uint8))
    ws.save("plate.json", {"T": T.tolist(), "w": cw, "h": ch,
                           "step": step, "samples": n})
    print(f"wrote {ws.path('plate.png')}")

    if refine:
        # world-coords corrections at sample times, lerped across frames
        Cs = [Tinv @ F @ T for F in Fs]
        Cs = [C / C[2, 2] for C in Cs]
        out = []
        for i in range(len(Hs)):
            k = min(i // step, n - 1)
            k2 = min(k + 1, n - 1)
            f = (i - idxs[k]) / max(1, idxs[k2] - idxs[k]) if k2 > k else 0.0
            C = (1 - f) * Cs[k] + f * Cs[k2]
            Hc = C @ Hs[i]
            out.append((Hc / Hc[2, 2]).tolist())
        data = ws.load("homographies.json")
        data["H"] = out
        data["refined"] = True
        ws.save("homographies.json", data)
