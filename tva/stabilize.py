"""Per-frame homographies to a reference ground plane.

Chains adjacent-frame homographies estimated from Shi-Tomasi corners +
pyramidal LK flow + RANSAC. For planar (aerial/elevated) road scenes this
captures camera pan/zoom/rotation exactly; moving vehicles fall out as
RANSAC outliers, and in congested footage most vehicles are static anchors
anyway. H[i] maps frame-i pixel coords -> reference plane (frame 0 coords).

Known limit: pure chaining accumulates drift over long clips. QA it with
`tva world` (stopped-car world jitter) before trusting long sequences.
"""
import cv2
import numpy as np

# estimation runs at reduced scale for speed; H is rescaled to full-res coords
EST_WIDTH = 1920
MAX_CORNERS = 1500
CORNER_QUALITY = 0.005
CORNER_MIN_DIST = 18
RANSAC_TOL = 2.5  # px at estimation scale


def iter_gray(src, width):
    cap = cv2.VideoCapture(src)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h = int(frame.shape[0] * width / frame.shape[1])
        yield cv2.cvtColor(cv2.resize(frame, (width, h)), cv2.COLOR_BGR2GRAY)
    cap.release()


def adjacent_homography(g0, g1):
    """H mapping g1 coords -> g0 coords, plus inlier count."""
    p0 = cv2.goodFeaturesToTrack(g0, MAX_CORNERS, CORNER_QUALITY,
                                 CORNER_MIN_DIST)
    if p0 is None or len(p0) < 20:
        return np.eye(3), 0
    p1, st, _ = cv2.calcOpticalFlowPyrLK(
        g0, g1, p0, None, winSize=(31, 31), maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    good = st.ravel() == 1
    if good.sum() < 20:
        return np.eye(3), 0
    H, inl = cv2.findHomography(p1[good], p0[good], cv2.RANSAC, RANSAC_TOL)
    if H is None:
        return np.eye(3), 0
    return H, int(inl.sum())


def run(ws):
    meta = ws.meta
    scale = EST_WIDTH / meta["width"]
    S = np.diag([scale, scale, 1.0])
    S_inv = np.diag([1 / scale, 1 / scale, 1.0])

    Hs = [np.eye(3)]
    inliers = [0]
    prev = None
    for i, g in enumerate(iter_gray(meta["src"], EST_WIDTH)):
        if prev is not None:
            H_adj, ninl = adjacent_homography(prev, g)
            Hs.append(Hs[-1] @ H_adj)
            inliers.append(ninl)
            if i % 50 == 0:
                print(f"frame {i}: {ninl} inliers", flush=True)
        prev = g

    # rescale to full-res pixel coords, normalize
    out = []
    for H in Hs:
        Hf = S_inv @ H @ S
        out.append((Hf / Hf[2, 2]).tolist())
    ws.save("homographies.json", {"H": out, "inliers": inliers})


def apply_h(H, xy):
    """Map (n,2) points through a 3x3 homography."""
    pts = np.asarray(xy, dtype=np.float64)
    ones = np.ones((len(pts), 1))
    p = np.hstack([pts, ones]) @ np.asarray(H).T
    return p[:, :2] / p[:, 2:3]


def qa(ws, times=(0, 4, 8, 12, 16)):
    """Warp sample frames into the reference plane on a common canvas."""
    meta = ws.meta
    Hs = [np.asarray(h) for h in ws.load("homographies.json")["H"]]
    w, h, fps = meta["width"], meta["height"], meta["fps"]
    idxs = [min(int(t * fps), len(Hs) - 1) for t in times]

    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    warped = np.vstack([apply_h(Hs[i], corners) for i in idxs])
    x0, y0 = warped.min(axis=0)
    x1, y1 = warped.max(axis=0)
    out_scale = min(1.0, 3000.0 / max(x1 - x0, y1 - y0))
    T = np.array([[out_scale, 0, -x0 * out_scale],
                  [0, out_scale, -y0 * out_scale],
                  [0, 0, 1]])
    cw, ch = int((x1 - x0) * out_scale) + 2, int((y1 - y0) * out_scale) + 2

    cap = cv2.VideoCapture(meta["src"])
    acc = np.zeros((ch, cw, 3), dtype=np.float32)
    cnt = np.zeros((ch, cw, 1), dtype=np.float32)
    for k, i in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            continue
        wf = cv2.warpPerspective(frame, T @ Hs[i], (cw, ch))
        mask = cv2.warpPerspective(np.ones((h, w), np.float32), T @ Hs[i],
                                   (cw, ch))[..., None]
        acc += wf.astype(np.float32) * mask
        cnt += mask
        cv2.imwrite(str(ws.qa) + f"/stab-warp-t{int(i / fps)}.jpg", wf,
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
    cap.release()
    blend = (acc / np.maximum(cnt, 1e-6)).astype(np.uint8)
    cv2.imwrite(str(ws.qa) + "/stab-blend.jpg", blend,
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"QA: {ws.qa}/stab-blend.jpg (sharp road = good registration)")
