"""Render analysis back onto the source video.

Per tracked vehicle: a speed-colored dot (red=stopped, green=free flow) with
a short trail of its actual world-plane path mapped into the current camera
view (trails stay glued to the road under pan/rotation). Stop-onset events
flash an expanding white ring — watching the rings march upstream IS the
shockwave. Output h264 via an ffmpeg rawvideo pipe.
"""
import subprocess

import cv2
import numpy as np

from .stabilize import apply_h
from .viz import speed_color

TRAIL_S = 1.2          # seconds of path behind each car
RING_S = 0.7           # stop-onset ring duration
LAYER_ALPHA = 0.8      # overlay blend


def run(ws, out=None, out_w=1920):
    meta = ws.meta
    fps, n = meta["fps"], meta["n_frames"]
    w, h = meta["width"], meta["height"]
    out = out or ws.path("overlay-speed.mp4")
    Hs = [np.asarray(m) for m in ws.load("homographies.json")["H"]]
    Hinv = [np.linalg.inv(m) for m in Hs]
    wt = ws.load("world_tracks.json")
    v_stop, v_move = wt["v_stop"], wt["v_move"]

    # per-track median bbox size (frame px) for marker radius
    size_of = {}
    for tr in ws.load("tracks.json")["tracks"]:
        size_of[tr["id"]] = float(np.median([max(o[3], o[4])
                                             for o in tr["obs"]]))

    # frame index -> [(track, obs index)]
    per_frame = [[] for _ in range(n)]
    for tr in wt["tracks"]:
        for k, f in enumerate(tr["frames"]):
            if f < n:
                per_frame[f].append((tr, k))
    rings = [(ev, ev["frame"]) for ev in wt["stop_events"]]
    trail_n = int(TRAIL_S * fps)
    ring_n = max(1, int(RING_S * fps))

    out_h = int(h * out_w / w)
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{out_w}x{out_h}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-crf", "18", "-preset", "medium",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", out],
        stdin=subprocess.PIPE)

    cap = cv2.VideoCapture(meta["src"])
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        layer = frame.copy()
        for tr, k in per_frame[i]:
            v = tr["speed"][k]
            col = speed_color(v, v_stop, v_move)[::-1]  # RGB -> BGR
            r = max(6, int(0.30 * size_of.get(tr["id"], 80)))
            # world path over the last TRAIL_S, drawn in this frame's view
            j0 = max(0, k - trail_n)
            wpts = np.column_stack([tr["x"][j0:k + 1], tr["y"][j0:k + 1]])
            fpts = apply_h(Hinv[i], wpts).astype(np.int32)
            if len(fpts) > 1:
                cv2.polylines(layer, [fpts], False, col, max(2, r // 4),
                              cv2.LINE_AA)
            cv2.circle(layer, tuple(fpts[-1]), r, col, -1, cv2.LINE_AA)
        for ev, f0 in rings:
            if f0 <= i < f0 + ring_n:
                age = (i - f0) / ring_n
                p = apply_h(Hinv[i], [(ev["x"], ev["y"])])[0].astype(int)
                rr = int(30 + 90 * age)
                c = int(255 * (1 - 0.6 * age))
                cv2.circle(layer, tuple(p), rr, (c, c, c), 4, cv2.LINE_AA)
        frame = cv2.addWeighted(layer, LAYER_ALPHA, frame,
                                1 - LAYER_ALPHA, 0)
        small = cv2.resize(frame, (out_w, out_h))
        enc.stdin.write(small.tobytes())
        if i % int(3 * fps) == 0:
            cv2.imwrite(f"{ws.qa}/render-t{int(i / fps)}.jpg", small,
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
        if i % 100 == 0:
            print(f"frame {i}/{n}", flush=True)
    cap.release()
    enc.stdin.close()
    enc.wait()
    print(f"wrote {out}")
