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
GAP_FILL_S = 0.6       # bridge detection dropouts up to this long
R_MAX = 40             # marker radius cap (src-frame px)
TRAIL_MIN_LEN = 0.5    # car_len; shorter world paths draw no trail
TRAIL_STRAIGHT = 0.8   # net/path ratio below this = registration curl
                       # (a real 90-degree turn over TRAIL_S is ~0.90)

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
    bounds = road.get("boundaries")
    if bounds and len(bounds) >= 2:
        # measured painted lines: outermost are edges, the rest dividers
        for kind, off in ([("edge", bounds[0]), ("edge", bounds[-1])] +
                          [("divider", b) for b in bounds[1:-1]]):
            lines.append((kind, np.column_stack([cx, cy]) + nrm * off))
    elif len(lanes) > 1:
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


def run(ws, out=None, out_w=1920, layers=("cars",), highlight=()):
    meta = ws.meta
    fps, n = meta["fps"], meta["n_frames"]
    w, h = meta["width"], meta["height"]
    out = out or ws.path("overlay-speed.mp4")
    Hs = [np.asarray(m) for m in ws.load("homographies.json")["H"]]
    Hinv = [np.linalg.inv(m) for m in Hs]
    wt = ws.load("world_tracks.json")
    v_stop, v_move = wt["v_stop"], wt["v_move"]
    car_len = wt["car_len_px"]
    roads = []
    if "roads" in layers:
        try:
            roads = ws.load("roads.json")["roads"]
        except FileNotFoundError:
            print("roads.json missing - run `tva roads` first; skipping layer")

    # per-track median bbox size (frame px) for marker radius, and observed
    # centers: markers are drawn on the DETECTION, never on the smoothed
    # world estimate (which coasts away from the car wherever the projection
    # is too distorted to trust measurements)
    size_of, px_of = {}, {}
    for tr in ws.load("tracks.json")["tracks"]:
        size_of[tr["id"]] = float(np.median([max(o[3], o[4])
                                             for o in tr["obs"]]))
        px_of[tr["id"]] = {o[0]: (o[1], o[2]) for o in tr["obs"]}

    # parked = static AND off every inferred road: don't paint them at all
    # (their "speed" is pure registration noise, so they twitch with color)
    parked = set()
    if roads:
        from .kinematics import station_of
        for tr in wt["tracks"]:
            if not tr.get("static"):
                continue
            mx = np.array([np.median(tr["x"])])
            my = np.array([np.median(tr["y"])])
            dmin = min(station_of(r["x"], r["y"], mx, my)[1][0]
                       for r in roads)
            if dmin > 3.5 * wt["car_len_px"]:
                parked.add(tr["id"])
        print(f"{len(parked)} parked vehicles hidden")

    # frame index -> [(track, obs index)]; fractional index = interpolated
    # frame inside a short detection dropout (dots persist through flicker)
    gap_max = int(GAP_FILL_S * fps)
    per_frame = [[] for _ in range(n)]
    for tr in wt["tracks"]:
        if tr["id"] in parked:
            continue
        F = tr["frames"]
        for k, f in enumerate(F):
            if f < n:
                per_frame[f].append((tr, float(k)))
        for k in range(len(F) - 1):
            g = F[k + 1] - F[k]
            if 1 < g <= gap_max:
                for f in range(F[k] + 1, min(F[k + 1], n)):
                    per_frame[f].append((tr, k + (f - F[k]) / g))
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
            k0 = int(k)
            frac = k - k0
            if frac:
                v = tr["speed"][k0] * (1 - frac) + tr["speed"][k0 + 1] * frac
            else:
                v = tr["speed"][k0]
            col = speed_color(v, v_stop, v_move)[::-1]  # RGB -> BGR
            hot = tr["id"] in highlight
            r = min(max(6, int(0.30 * size_of.get(tr["id"], 80))), R_MAX)
            if hot:
                r = int(r * 1.8)
            # world path over the last TRAIL_S, drawn in this frame's view,
            # translated so it ends exactly on the detected bbox center
            j0 = max(0, k0 - trail_n)
            wpts = np.column_stack([tr["x"][j0:k0 + 1], tr["y"][j0:k0 + 1]])
            if frac:
                tip = wpts[-1] * (1 - frac) + frac * np.array(
                    [tr["x"][k0 + 1], tr["y"][k0 + 1]])
                wpts = np.vstack([wpts, tip])
            fpts = apply_h(Hinv[i], wpts)
            obs_px = px_of[tr["id"]].get(i)
            if obs_px is None and frac:
                # inside a bridged dropout: interpolate the detected centers
                F = tr["frames"]
                p1 = np.asarray(px_of[tr["id"]][F[k0]], float)
                p2 = np.asarray(px_of[tr["id"]][F[k0 + 1]], float)
                obs_px = p1 * (1 - frac) + p2 * frac
            if obs_px is not None:
                fpts += np.asarray(obs_px) - fpts[-1]
            fpts = fpts.astype(np.int32)
            # trail only when the world path is path-like: far-field
            # registration wobble shows up as curls (low net/path ratio)
            seg = np.diff(wpts, axis=0)
            plen = float(np.hypot(seg[:, 0], seg[:, 1]).sum())
            net = float(np.hypot(*(wpts[-1] - wpts[0])))
            if (len(fpts) > 1 and plen > TRAIL_MIN_LEN * car_len
                    and net > TRAIL_STRAIGHT * plen):
                cv2.polylines(layer, [fpts], False, col, max(2, r // 4),
                              cv2.LINE_AA)
            if hot:
                cv2.circle(layer, tuple(fpts[-1]), r + 4, (255, 255, 255),
                           3, cv2.LINE_AA)
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
