"""Vehicle detection + tracking: YOLO + ByteTrack at native resolution.

Writes tracks.json: {"tracks": [{"id", "cls", "obs": [[frame, cx, cy, w, h,
conf], ...]}, ...]}. Untracked detections (no id assigned yet) are dropped —
downstream kinematics needs identity over time.

Optional tiled ("SAHI-style") mode (tiles="2x2"): slice each frame into an
overlapping grid, detect per tile in one batched predict, merge across
tiles with class-aware NMS, and track the merged detections with a
persistent BYTETracker. Off by default; the default path is unchanged.
"""
from collections import defaultdict

# vehicle classes are picked from the model's own class names, so COCO
# (car/motorcycle/bus/truck) and VisDrone (car/van/truck/bus/motor) both work.
# COCO models are near-blind to nadir/top-down aerial cars — for drone
# footage use VisDrone weights (see README).
VEHICLE_NAMES = {"car", "van", "truck", "bus", "motorcycle", "motor"}


def run(ws, model="yolo11m.pt", imgsz=3840, conf=0.25, max_det=900,
        names=None, out="tracks.json", tiles=None, tile_overlap=0.15):
    from ultralytics import YOLO

    meta = ws.meta
    yolo = YOLO(model)
    names = names or VEHICLE_NAMES
    class_ids = [i for i, n in yolo.names.items() if n in names]
    class_names = {i: yolo.names[i] for i in class_ids}
    print(f"classes: {class_names}")
    if tiles is None:
        tracks, n = _track_stream(yolo, meta, class_ids, class_names,
                                  imgsz, conf, max_det)
        header = {"model": model, "imgsz": imgsz, "conf": conf,
                  "max_det": max_det}
    else:
        tracks, n = _track_tiled(yolo, meta, class_ids, class_names,
                                 imgsz, conf, max_det, tiles, tile_overlap)
        header = {"model": model, "imgsz": imgsz, "conf": conf,
                  "max_det": max_det, "tiles": tiles,
                  "tile_overlap": tile_overlap}
    ws.save(out, {
        **header,
        "tracks": [{"id": tid, **tr} for tid, tr in sorted(tracks.items())],
    })
    print(f"{len(tracks)} tracks, {n} observations")


def _track_stream(yolo, meta, class_ids, class_names, imgsz, conf, max_det):
    tracks = defaultdict(lambda: {"cls": None, "obs": []})
    n = 0
    for i, r in enumerate(yolo.track(
            source=meta["src"], imgsz=imgsz, conf=conf, max_det=max_det,
            classes=class_ids, tracker="bytetrack.yaml",
            stream=True, verbose=False)):
        b = r.boxes
        if b is not None and b.id is not None:
            ids = b.id.int().tolist()
            xywh = b.xywh.tolist()
            clss = b.cls.int().tolist()
            confs = b.conf.tolist()
            for tid, (cx, cy, w, h), c, cf in zip(ids, xywh, clss, confs):
                tr = tracks[tid]
                tr["cls"] = class_names.get(c, str(c))
                tr["obs"].append([i, round(cx, 1), round(cy, 1),
                                  round(w, 1), round(h, 1), round(cf, 3)])
            n += len(ids)
        if i % 50 == 0:
            print(f"frame {i}: {len(tracks)} tracks, {n} obs total",
                  flush=True)
    return tracks, n


def _tile_slices(w, h, cols, rows, overlap):
    """Grid of (x0, y0, x1, y1) tiles covering the frame, each tile
    extended by `overlap` fraction of its base size (clamped to frame)."""
    sw, sh = w / cols, h / rows
    tw, th = sw * (1 + overlap), sh * (1 + overlap)
    return [(round(c * sw), round(r * sh),
             min(round(c * sw + tw), w), min(round(r * sh + th), h))
            for r in range(rows) for c in range(cols)]


def _track_tiled(yolo, meta, class_ids, class_names, imgsz, conf, max_det,
                 tiles, tile_overlap):
    """SAHI-style: detect per overlapping tile (one batched predict per
    frame), merge with class-aware NMS, track the merged set with a
    persistent BYTETracker."""
    import cv2
    import numpy as np
    import torch
    from torchvision.ops import batched_nms
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import YAML, IterableSimpleNamespace
    from ultralytics.utils.checks import check_yaml

    cols, rows = (int(s) for s in tiles.lower().split("x"))
    # same tracker config yolo.track(tracker="bytetrack.yaml") would build
    cfg = IterableSimpleNamespace(**YAML.load(check_yaml("bytetrack.yaml")))
    tracker = BYTETracker(args=cfg)
    tracks = defaultdict(lambda: {"cls": None, "obs": []})
    n = 0
    cap = cv2.VideoCapture(meta["src"])
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        slices = _tile_slices(w, h, cols, rows, tile_overlap)
        imgs = [frame[y0:y1, x0:x1] for x0, y0, x1, y1 in slices]
        try:
            rs = yolo.predict(imgs, imgsz=imgsz, conf=conf, max_det=max_det,
                              classes=class_ids, verbose=False)
        except torch.cuda.OutOfMemoryError:
            # large imgsz x tile count can exceed VRAM in one batch —
            # per-tile predicts give the same detections
            torch.cuda.empty_cache()
            rs = [yolo.predict([img], imgsz=imgsz, conf=conf,
                               max_det=max_det, classes=class_ids,
                               verbose=False)[0] for img in imgs]
        dets = []
        for (x0, y0, _, _), r in zip(slices, rs):
            b = r.boxes
            if b is None or len(b) == 0:
                continue
            xyxy = b.xyxy.cpu().numpy()
            xyxy[:, [0, 2]] += x0
            xyxy[:, [1, 3]] += y0
            dets.append(np.column_stack(
                [xyxy, b.conf.cpu().numpy(), b.cls.cpu().numpy()]))
        if dets:
            d = np.concatenate(dets).astype(np.float32)
            keep = batched_nms(torch.from_numpy(d[:, :4]),
                               torch.from_numpy(d[:, 4]),
                               torch.from_numpy(d[:, 5]), 0.5)
            d = d[keep.numpy()]
            if len(d) > max_det:
                d = d[np.argsort(-d[:, 4])[:max_det]]
        else:
            d = np.zeros((0, 6), np.float32)
        # Boxes rows are xyxy+conf+cls; update() returns one row per active
        # track: [x1, y1, x2, y2, track_id, score, cls, det_idx]
        out = tracker.update(Boxes(d, (h, w)))
        if out.size:
            for x1, y1, x2, y2, tid, cf, c, _ in out.reshape(-1, 8).tolist():
                tr = tracks[int(tid)]
                tr["cls"] = class_names.get(int(c), str(int(c)))
                tr["obs"].append([i, round((x1 + x2) / 2, 1),
                                  round((y1 + y2) / 2, 1),
                                  round(x2 - x1, 1), round(y2 - y1, 1),
                                  round(float(cf), 3)])
                n += 1
        if i % 200 == 0:
            print(f"frame {i}: {len(tracks)} tracks, {n} obs total",
                  flush=True)
        i += 1
    cap.release()
    return tracks, n


def qa(ws, times=(0, 4, 8, 12, 16), tracks="tracks.json"):
    """Draw tracked boxes on sample frames."""
    import cv2

    meta = ws.meta
    fps = meta["fps"]
    suffix = "" if tracks == "tracks.json" else "-" + tracks.split(".")[0]
    per_frame = defaultdict(list)
    for tr in ws.load(tracks)["tracks"]:
        for f, cx, cy, w, h, conf in tr["obs"]:
            per_frame[f].append((tr["id"], cx, cy, w, h))

    cap = cv2.VideoCapture(meta["src"])
    for t in times:
        i = min(int(t * fps), meta["n_frames"] - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            continue
        for tid, cx, cy, w, h in per_frame.get(i, []):
            p0 = (int(cx - w / 2), int(cy - h / 2))
            p1 = (int(cx + w / 2), int(cy + h / 2))
            cv2.rectangle(frame, p0, p1, (0, 255, 0), 3)
            cv2.putText(frame, str(tid), (p0[0], p0[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"t={t}s  {len(per_frame.get(i, []))} boxes",
                    (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 3)
        cv2.imwrite(f"{ws.qa}/det{suffix}-t{t}.jpg", frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
    cap.release()
    print(f"QA: {ws.qa}/det{suffix}-t*.jpg")
