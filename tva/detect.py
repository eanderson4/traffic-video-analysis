"""Vehicle detection + tracking: YOLO + ByteTrack at native resolution.

Writes tracks.json: {"tracks": [{"id", "cls", "obs": [[frame, cx, cy, w, h,
conf], ...]}, ...]}. Untracked detections (no id assigned yet) are dropped —
downstream kinematics needs identity over time.
"""
from collections import defaultdict

# COCO vehicle classes
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def run(ws, model="yolo11m.pt", imgsz=3840, conf=0.25):
    from ultralytics import YOLO

    meta = ws.meta
    yolo = YOLO(model)
    tracks = defaultdict(lambda: {"cls": None, "obs": []})
    n = 0
    for i, r in enumerate(yolo.track(
            source=meta["src"], imgsz=imgsz, conf=conf,
            classes=VEHICLE_CLASSES, tracker="bytetrack.yaml",
            stream=True, verbose=False)):
        b = r.boxes
        if b is not None and b.id is not None:
            ids = b.id.int().tolist()
            xywh = b.xywh.tolist()
            clss = b.cls.int().tolist()
            confs = b.conf.tolist()
            for tid, (cx, cy, w, h), c, cf in zip(ids, xywh, clss, confs):
                tr = tracks[tid]
                tr["cls"] = CLASS_NAMES.get(c, str(c))
                tr["obs"].append([i, round(cx, 1), round(cy, 1),
                                  round(w, 1), round(h, 1), round(cf, 3)])
            n += len(ids)
        if i % 50 == 0:
            print(f"frame {i}: {len(tracks)} tracks, {n} obs total",
                  flush=True)
    ws.save("tracks.json", {
        "model": model, "imgsz": imgsz, "conf": conf,
        "tracks": [{"id": tid, **tr} for tid, tr in sorted(tracks.items())],
    })
    print(f"{len(tracks)} tracks, {n} observations")


def qa(ws, times=(0, 4, 8, 12, 16)):
    """Draw tracked boxes on sample frames."""
    import cv2

    meta = ws.meta
    fps = meta["fps"]
    per_frame = defaultdict(list)
    for tr in ws.load("tracks.json")["tracks"]:
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
        cv2.imwrite(f"{ws.qa}/det-t{t}.jpg", frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
    cap.release()
    print(f"QA: {ws.qa}/det-t*.jpg")
