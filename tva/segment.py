"""Image segmentation on the background plate: SAM2 road surfaces +
painted-line lane boundaries.

The plate is a single stabilized, car-ghosted image, so segmentation runs
once, not per frame. Each inferred road prompts SAM2 with points along its
centerline -> road-surface mask -> polygon (world coords). Lane boundaries
come from the paint itself: a white top-hat filter inside the road mask
lifts thin bright markings; their signed lateral offsets from the
centerline form sharp peaks at each painted line. Boundaries update
roads.json (lanes = midpoints between adjacent boundaries).
"""
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from .stabilize import apply_h
from .roads import project_signed

LAT_CORRIDOR = 3.5     # x car_len: cap road mask laterally
TOPHAT_KERNEL = 15
PAINT_PCT = 97.0       # tophat percentile threshold inside the mask
PEAK_MIN_SEP_W = 20.0  # world px between painted lines


def run(ws, model="sam2.1_b.pt"):
    from ultralytics import SAM

    plate = cv2.imread(ws.path("plate.png"))
    pj = ws.load("plate.json")
    T = np.asarray(pj["T"])
    Tinv = np.linalg.inv(T)
    rdata = ws.load("roads.json")
    roads = rdata["roads"]
    car_len = ws.load("world_tracks.json")["car_len_px"]
    sam = SAM(model)
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    tophat = cv2.morphologyEx(
        gray, cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (TOPHAT_KERNEL, TOPHAT_KERNEL)))
    qa = plate.copy()
    segments = []
    for road in roads:
        cl = np.column_stack([road["x"], road["y"]])
        cpts = apply_h(T, cl)
        prompts = cpts[2:-2:4]
        res = sam(plate, points=prompts.tolist(),
                  labels=[1] * len(prompts), verbose=False)
        m = res[0].masks
        if m is None:
            print(f"road {road['id']}: SAM returned no mask")
            continue
        mask = (m.data.cpu().numpy().sum(axis=0) > 0).astype(np.uint8)
        if mask.shape != gray.shape:
            mask = cv2.resize(mask, (gray.shape[1], gray.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        # clamp to the road's lateral corridor so SAM bleed doesn't leak
        # onto neighboring surfaces
        ys, xs = np.nonzero(mask)
        wpts = apply_h(Tinv, np.column_stack([xs, ys]).astype(float))
        _, lat = project_signed(road["x"], road["y"], wpts[:, 0], wpts[:, 1])
        keep = np.abs(lat) < LAT_CORRIDOR * car_len
        mask[:] = 0
        mask[ys[keep], xs[keep]] = 1
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                np.ones((9, 9), np.uint8))

        # polygonize largest component
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        poly = cv2.approxPolyDP(cnt, 4.0, True).reshape(-1, 2).astype(float)
        poly_w = apply_h(Tinv, poly)

        # painted lane lines: bright thin marks inside the mask
        th = np.percentile(tophat[mask > 0], PAINT_PCT)
        pys, pxs = np.nonzero((tophat > th) & (mask > 0))
        pw = apply_h(Tinv, np.column_stack([pxs, pys]).astype(float))
        _, plat = project_signed(road["x"], road["y"], pw[:, 0], pw[:, 1])
        bw = 2.0
        lo, hi = -LAT_CORRIDOR * car_len, LAT_CORRIDOR * car_len
        edges = np.arange(lo, hi + bw, bw)
        hist, _ = np.histogram(plat, bins=edges)
        histS = gaussian_filter1d(hist.astype(float), 2.5)
        pk, _ = find_peaks(histS, distance=int(PEAK_MIN_SEP_W / bw),
                           prominence=0.15 * histS.max())
        bounds = sorted(float(edges[p] + bw / 2) for p in pk)
        if len(bounds) >= 2:
            road["boundaries"] = [round(b, 1) for b in bounds]
            road["lanes"] = [round((a + b) / 2, 1)
                             for a, b in zip(bounds, bounds[1:])]
            gaps = np.diff(bounds)
            road["lane_halfwidth"] = round(float(np.median(gaps)) / 2, 1)
        segments.append({"road": road["id"],
                         "polygon": [[round(float(x), 1), round(float(y), 1)]
                                     for x, y in poly_w]})
        print(f"road {road['id']}: mask {int(mask.sum())}px, "
              f"{len(bounds)} painted lines "
              f"({[round(b) for b in bounds]})")

        # QA tint + boundaries
        tint = qa.copy()
        tint[mask > 0] = (0.6 * tint[mask > 0] +
                          0.4 * np.array([255, 120, 0])).astype(np.uint8)
        qa = tint
        qa[pys, pxs] = (0, 255, 255)

    ws.save("segments.json", {"segments": segments})
    ws.save("roads.json", rdata)
    cv2.imwrite(f"{ws.qa}/segment-plate.jpg", qa,
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"QA: {ws.qa}/segment-plate.jpg")
