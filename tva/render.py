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

# per-road hues (BGR), cycled
ROAD_COLORS = [(255, 91, 46), (255, 200, 0), (180, 0, 255), (0, 200, 255),
               (0, 255, 140), (255, 128, 128)]


def road_geometry(road):
    """World-plane polylines to draw for one road: (kind, pts) with kind in
    edge|divider, plus chevron segments along the centerline."""
    cx, cy = np.asarray(road["x"]), np.asarray(road["y"])
    tang = np.gradient(np.column_stack([cx, cy]), axis=0)
    tang /= np.maximum(np.hypot(tang[:, 0], tang[:, 1])[:, None], 1e-9)
    nrm = np.column_stack([-tang[:, 1], tang[:, 0]])
    lanes = road["lanes"]
    hw = road["lane_halfwidth"]
    lines = []
    if len(lanes) > 1:
        for kind, off in (
                [("edge", lanes[0] - hw), ("edge", lanes[-1] + hw)] +
                [("divider", (a + b) / 2) for a, b in zip(lanes, lanes[1:])]):
            lines.append((kind, np.column_stack([cx, cy]) + nrm * off))
    else:
        # lanes unresolved: draw just the centerline
        lines.append(("divider", np.column_stack([cx, cy]) + nrm * lanes[0]))
    # direction chevrons on the centerline every ~4 vertices
    chevrons = []
    for i in range(2, len(cx) - 2, 4):
        tip = np.array([cx[i], cy[i]]) + tang[i] * 14
        b1 = np.array([cx[i], cy[i]]) - tang[i] * 10 + nrm[i] * 10
        b2 = np.array([cx[i], cy[i]]) - tang[i] * 10 - nrm[i] * 10
        chevrons.append(np.array([b1, tip, b2]))
    return lines, chevrons


def draw_roads(layer, roads, Hinv_i):
    for road in roads:
        col = ROAD_COLORS[road["id"] % len(ROAD_COLORS)]
        lines, chevrons = road_geometry(road)
        for kind, wpts in lines:
            fpts = apply_h(Hinv_i, wpts).astype(np.int32)
            cv2.polylines(layer, [fpts], False, col,
                          4 if kind == "edge" else 2, cv2.LINE_AA)
        for ch in chevrons:
            fpts = apply_h(Hinv_i, ch).astype(np.int32)
            cv2.polylines(layer, [fpts], False, col, 3, cv2.LINE_AA)


def run(ws, out=None, out_w=1920, layers=("roads", "cars")):
    meta = ws.meta
    fps, n = meta["fps"], meta["n_frames"]
    w, h = meta["width"], meta["height"]
    out = out or ws.path("overlay-speed.mp4")
    Hs = [np.asarray(m) for m in ws.load("homographies.json")["H"]]
    Hinv = [np.linalg.inv(m) for m in Hs]
    wt = ws.load("world_tracks.json")
    v_stop, v_move = wt["v_stop"], wt["v_move"]
    roads = []
    if "roads" in layers:
        try:
            roads = ws.load("roads.json")["roads"]
        except FileNotFoundError:
            print("roads.json missing - run `tva roads` first; skipping layer")

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
        if roads:
            draw_roads(layer, roads, Hinv[i])
        for tr, k in per_frame[i] if "cars" in layers else []:
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
