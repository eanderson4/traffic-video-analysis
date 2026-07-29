"""World-plane kinematics: map tracks through homographies, compute speeds,
classify moving/stopped, extract stop-onset events, derive a road centerline.

All positions are reference-plane pixels (frame 0's ground plane), so a
stopped car is a fixed point regardless of camera motion. Speeds are px/s in
that plane; attach a physical scale later (lane width / car length).

Speeds come from a constant-acceleration Kalman filter + RTS smoother per
track, not from differentiating positions: bbox-center jitter and residual
projection noise turn finite differences into sinusoidal garbage, while the
acceleration prior only admits speed profiles a real car could drive. The
smoother is two-pass (uses future observations), so stop onsets are sharp
rather than lagged.
"""
import math

import numpy as np

from .stabilize import apply_h

MIN_TRACK_FRAMES = 12          # drop blips
STOP_SPEED_FRAC = 0.10         # stopped if speed < frac * median car length /s
MOVE_SPEED_FRAC = 0.22         # hysteresis: moving again above this
MIN_STATE_RUN_S = 0.4          # ignore state flickers shorter than this
SIGMA_MEAS_FRAC = 0.04         # position noise, frac of car_len (bbox jitter)
SIGMA_JERK_FRAC = 1.0          # white-jerk process intensity, frac of car_len


def _rts_1d(t, z, sig_meas, sig_jerk):
    """1-D constant-acceleration Kalman filter + RTS smoother.

    State [pos, vel, acc]; measurement = pos. sig_meas is per-sample (array).
    Returns smoothed (pos, vel). Handles irregular dt (gappy tracks).
    """
    n = len(z)
    q, R = sig_jerk ** 2, np.asarray(sig_meas) ** 2
    xf = np.zeros((n, 3))
    Pf = np.zeros((n, 3, 3))
    xp = np.zeros((n, 3))
    Pp = np.zeros((n, 3, 3))
    Fs = np.zeros((n, 3, 3))
    x = np.array([z[0], 0.0, 0.0])
    s0 = float(np.median(np.sqrt(R)))
    P = np.diag([R[0], (50 * s0) ** 2, (50 * s0) ** 2])
    for k in range(n):
        if k > 0:
            dt = t[k] - t[k - 1]
            F = np.array([[1, dt, dt * dt / 2], [0, 1, dt], [0, 0, 1]])
            Q = q * np.array(
                [[dt ** 5 / 20, dt ** 4 / 8, dt ** 3 / 6],
                 [dt ** 4 / 8, dt ** 3 / 3, dt ** 2 / 2],
                 [dt ** 3 / 6, dt ** 2 / 2, dt]])
            x = F @ x
            P = F @ P @ F.T + Q
        else:
            F = np.eye(3)
        xp[k], Pp[k], Fs[k] = x, P, F
        K = P[:, 0] / (P[0, 0] + R[k])
        x = x + K * (z[k] - x[0])
        P = P - np.outer(K, P[0, :])
        xf[k], Pf[k] = x, P
    xs = xf.copy()
    for k in range(n - 2, -1, -1):
        C = Pf[k] @ Fs[k + 1].T @ np.linalg.inv(Pp[k + 1])
        xs[k] = xf[k] + C @ (xs[k + 1] - xp[k + 1])
    return xs[:, 0], xs[:, 1]


def _local_sigma(t, res, floor, win_s=0.5):
    """Per-sample noise scale: median |residual| in a +/-win_s window."""
    out = np.empty(len(res))
    for k in range(len(res)):
        a = np.searchsorted(t, t[k] - win_s)
        b = np.searchsorted(t, t[k] + win_s, side="right")
        out[k] = max(floor, float(np.median(res[a:b])))
    return out


def smooth_track(t, pts, car_len, jac_scale=None):
    """Smoothed positions and velocities for an (n,2) world-point array.

    Measurement noise is adaptive twice over: jac_scale (local world-px per
    image-px of the homography) inflates it where the projection amplifies —
    near-horizon geometry during aggressive camera moves turns 1px of bbox
    jitter into 10+ world px — and a second pass re-weights by each
    segment's actual residuals, so motion-blurred moving cars are trusted
    less than pin-sharp stopped ones instead of everything sharing one
    global sigma.
    """
    sm = SIGMA_MEAS_FRAC * car_len
    sj = SIGMA_JERK_FRAC * car_len
    sig = np.full(len(t), sm)
    if jac_scale is not None:
        sig = sm * np.maximum(np.asarray(jac_scale), 1.0)

    def run_pass(sig_arr):
        xs, vx = _rts_1d(t, pts[:, 0], sig_arr, sj)
        ys, vy = _rts_1d(t, pts[:, 1], sig_arr, sj)
        return xs, ys, vx, vy

    xs, ys, vx, vy = run_pass(sig)
    res = np.hypot(pts[:, 0] - xs, pts[:, 1] - ys)
    sig = np.maximum(sig, _local_sigma(t, res, sm))
    return run_pass(sig)


def jacobian_scale(H, p, eps=2.0):
    """Local world-px per image-px of homography H at image point p."""
    q0 = apply_h(H, [p])[0]
    qx = apply_h(H, [(p[0] + eps, p[1])])[0]
    qy = apply_h(H, [(p[0], p[1] + eps)])[0]
    return float((np.hypot(*(qx - q0)) + np.hypot(*(qy - q0))) / (2 * eps))


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
        t = np.array(frames) / fps
        jac = np.array([jacobian_scale(Hs[f], (cx, cy))
                        for f, cx, cy, *_ in obs])
        xs, ys, vx, vy = smooth_track(t, pts, car_len, jac_scale=jac)
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
        for ri, (s, a, b) in enumerate(runs):
            if not (s and a > 0 and ri > 0):
                continue
            # a real stop needs real motion first: the preceding moving run
            # must cover >= 2 car lengths over >= 0.8 s, otherwise it's just
            # gridlock creep re-triggering the threshold.
            pa, pb = runs[ri - 1][1], runs[ri - 1][2]
            path = float(np.hypot(np.diff(xs[pa:pb + 1]),
                                  np.diff(ys[pa:pb + 1])).sum())
            dur = (frames[pb] - frames[pa]) / fps
            if path < 2.0 * car_len or dur < 0.8:
                continue
            # sub-frame onset: interpolate the v_stop crossing
            t_on = t[a]
            if a > 0 and speed[a - 1] > v_stop > speed[a]:
                frac = (speed[a - 1] - v_stop) / (speed[a - 1] - speed[a])
                t_on = t[a - 1] + frac * (t[a] - t[a - 1])
            stop_events.append({
                "track": tr["id"], "frame": frames[a],
                "t": round(float(t_on), 3),
                "x": round(float(xs[a]), 1), "y": round(float(ys[a]), 1),
            })
        # static = parked or gridlocked the whole time we saw it: barely any
        # net displacement or path over a multi-second life. These are the
        # strongest registration anchors in the scene (see refine.anchors).
        life = (frames[-1] - frames[0]) / fps
        net = float(np.hypot(xs[-1] - xs[0], ys[-1] - ys[0]))
        path_total = float(np.hypot(np.diff(xs), np.diff(ys)).sum())
        static = bool(life >= 3.0 and net < 1.5 * car_len
                      and path_total < 4.0 * car_len)
        world_tracks.append({
            "id": tr["id"], "cls": tr["cls"], "static": static,
            "frames": frames,
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
