"""World-plane kinematics: map tracks through homographies, compute speeds,
classify moving/stopped, extract stop-onset events, derive a road centerline.

All positions are reference-plane pixels (frame 0's ground plane), so a
stopped car is a fixed point regardless of camera motion. Speeds are px/s in
that plane; attach a physical scale later (lane width / car length).
"""
import math

import numpy as np
from scipy.signal import savgol_filter

from .stabilize import apply_h

MIN_TRACK_FRAMES = 12          # drop blips
STOP_SPEED_FRAC = 0.10         # stopped if speed < frac * median car length /s
MOVE_SPEED_FRAC = 0.22         # hysteresis: moving again above this
MIN_STATE_RUN_S = 0.4          # ignore state flickers shorter than this


def smooth(a, fps):
    win = max(5, int(round(fps * 0.6)) | 1)
    if len(a) < win:
        return np.asarray(a, dtype=float)
    return savgol_filter(a, win, 2)


def state_runs(stopped, fps):
    """Collapse a boolean stopped[] into runs, dropping sub-threshold flips."""
    min_run = max(1, int(MIN_STATE_RUN_S * fps))
    runs = []
    for i, s in enumerate(stopped):
        if runs and runs[-1][0] == s:
            runs[-1][2] = i
        else:
            runs.append([s, i, i])
    # absorb short runs into neighbors, iteratively
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for k, r in enumerate(runs):
            if r[2] - r[1] + 1 < min_run and len(runs) > 1:
                del runs[k]
                if 0 < k <= len(runs) - 1 and runs[k - 1][0] == runs[k][0]:
                    runs[k - 1][2] = runs[k][2]
                    del runs[k]
                elif k > 0:
                    runs[k - 1][2] = r[2]
                elif k == 0:
                    runs[0][1] = r[1]
                changed = True
                break
    return [(bool(s), a, b) for s, a, b in runs]


def run(ws):
    meta = ws.meta
    fps = meta["fps"]
    Hs = [np.asarray(h) for h in ws.load("homographies.json")["H"]]
    data = ws.load("tracks.json")

    # median car bbox size in world px -> speed thresholds
    sizes = []
    for tr in data["tracks"]:
        for f, cx, cy, w, h, conf in tr["obs"]:
            sizes.append(max(w, h))
    car_len = float(np.median(sizes)) if sizes else 100.0
    v_stop = STOP_SPEED_FRAC * car_len
    v_move = MOVE_SPEED_FRAC * car_len
    print(f"median car length ~{car_len:.0f}px -> stop<{v_stop:.0f}px/s, "
          f"move>{v_move:.0f}px/s")

    world_tracks = []
    stop_events = []
    for tr in data["tracks"]:
        obs = [o for o in tr["obs"] if o[0] < len(Hs)]
        if len(obs) < MIN_TRACK_FRAMES:
            continue
        frames = [o[0] for o in obs]
        pts = np.array([apply_h(Hs[f], [(cx, cy)])[0]
                        for f, cx, cy, *_ in obs])
        xs, ys = smooth(pts[:, 0], fps), smooth(pts[:, 1], fps)
        # central-difference speed on the (possibly gappy) frame index
        t = np.array(frames) / fps
        vx = np.gradient(xs, t)
        vy = np.gradient(ys, t)
        speed = np.hypot(vx, vy)

        # hysteresis state machine
        stopped = np.zeros(len(speed), dtype=bool)
        st = speed[0] < v_stop
        for i, v in enumerate(speed):
            if st and v > v_move:
                st = False
            elif not st and v < v_stop:
                st = True
            stopped[i] = st
        runs = state_runs(stopped, fps)
        for s, a, b in runs:
            if s and a > 0:  # moving -> stopped transition inside the track
                stop_events.append({
                    "track": tr["id"], "frame": frames[a],
                    "t": round(frames[a] / fps, 2),
                    "x": round(float(xs[a]), 1), "y": round(float(ys[a]), 1),
                })
        world_tracks.append({
            "id": tr["id"], "cls": tr["cls"], "frames": frames,
            "x": [round(float(v), 1) for v in xs],
            "y": [round(float(v), 1) for v in ys],
            "speed": [round(float(v), 1) for v in speed],
            "runs": [[s, frames[a], frames[b]] for s, a, b in runs],
        })

    ws.save("world_tracks.json", {
        "car_len_px": car_len, "v_stop": v_stop, "v_move": v_move,
        "tracks": world_tracks, "stop_events": stop_events,
    })
    print(f"{len(world_tracks)} world tracks, {len(stop_events)} stop events")
    centerline(ws)


def centerline(ws):
    """Road spine from vehicle motion: the track that covers the most
    stop-onset events (i.e. runs the length of the jammed carriageway),
    longest path as tiebreak. Without stop events, plain longest path —
    which can easily pick a free-flowing road elsewhere in frame, so refine
    by hand in the annotator when it matters. Station s = arc length (px)
    along this polyline via nearest-point projection.
    """
    data = ws.load("world_tracks.json")
    events = data["stop_events"]
    lat_tol = 2.0 * data["car_len_px"]
    ex = np.array([e["x"] for e in events])
    ey = np.array([e["y"] for e in events])
    best, best_key = None, (-1, 0.0)
    for tr in data["tracks"]:
        xs, ys = np.array(tr["x"]), np.array(tr["y"])
        d = float(np.hypot(np.diff(xs), np.diff(ys)).sum())
        covered = 0
        if len(events) and d > 3 * data["car_len_px"]:
            _, lat = station_of(xs, ys, ex, ey)
            covered = int((lat < lat_tol).sum())
        if (covered, d) > best_key:
            best, best_key = tr, (covered, d)
    best_len = best_key[1]
    if best is not None and len(events):
        print(f"spine track {best['id']}: covers {best_key[0]}/{len(events)}"
              " stop events")
    if best is None:
        print("no tracks for centerline")
        return
    xs, ys = np.array(best["x"]), np.array(best["y"])
    # resample to ~40 points by arc length
    seg = np.hypot(np.diff(xs), np.diff(ys))
    arc = np.concatenate([[0], np.cumsum(seg)])
    s_new = np.linspace(0, arc[-1], 40)
    ws.save("centerline.json", {
        "source_track": best["id"], "length_px": round(best_len, 1),
        "x": [round(float(v), 1) for v in np.interp(s_new, arc, xs)],
        "y": [round(float(v), 1) for v in np.interp(s_new, arc, ys)],
    })
    print(f"centerline from track {best['id']} ({best_len:.0f}px path)")


def station_of(cl_x, cl_y, px, py):
    """Arc-length station + lateral distance of point(s) vs centerline."""
    cx, cy = np.asarray(cl_x), np.asarray(cl_y)
    seg = np.hypot(np.diff(cx), np.diff(cy))
    arc = np.concatenate([[0], np.cumsum(seg)])
    best_s = np.zeros(len(px))
    best_d = np.full(len(px), np.inf)
    for i in range(len(cx) - 1):
        ax, ay = cx[i], cy[i]
        bx, by = cx[i + 1] - ax, cy[i + 1] - ay
        L2 = bx * bx + by * by or 1e-9
        t = np.clip(((px - ax) * bx + (py - ay) * by) / L2, 0, 1)
        qx, qy = ax + t * bx, ay + t * by
        d = np.hypot(px - qx, py - qy)
        m = d < best_d
        best_d[m] = d[m]
        best_s[m] = arc[i] + t[m] * seg[i]
    return best_s, best_d
