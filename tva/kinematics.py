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

When flow.json is present (`tva flow`), each axis fuses TWO measurements:
z_p (homography-mapped bbox center, adaptive R_p as before) and z_v (median
LK-corner world displacement / dt, own R_v, Mahalanobis-gated). z_v is
immune to homography-chain drift — consecutive frames share the chain error
— so speeds stay bounded at clip ends even on corrupted registration, and a
stopped car reads ~0 by construction. Before filtering, a Montanino-Punzo
prefilter drops impossible position jumps (> 2 car_len/frame) and clips
accel spikes in the z_v stream.
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
ZV_GATE_SIGMA = 3.5            # Mahalanobis gate on velocity measurements
ZV_SIGMA_IMG_PX = 0.15         # LK sub-pixel matching noise floor (image px)
JUMP_CARLEN_PER_FRAME = 2.0    # prefilter: drop z_p jumps beyond this
ZV_AMAX_CARLEN = 2.5           # prefilter: accel bound for z_v spikes (/s^2)
ZV_SPIKE_WIN_S = 0.2           # rolling-median half-window for spike clip
WAVELET_SCALES_S = (0.16, 0.24, 0.36, 0.54, 0.8)  # Mexican-hat widths
WAVELET_SEARCH_BACK_S = 2.0    # look this far before the stop run start
WAVELET_SEARCH_FWD_S = 0.5    # ... and this far after


def _rts_1d(t, z, sig_meas, sig_jerk, zv=None, sig_v=None):
    """1-D constant-acceleration Kalman filter + RTS smoother.

    State [pos, vel, acc]; measurements are pos (always) and, where zv is
    finite, velocity (H = [0,1,0]) with per-sample sigma sig_v, gated at
    ZV_GATE_SIGMA Mahalanobis. sig_meas is per-sample (array). Returns
    smoothed (pos, vel, acc). Handles irregular dt (gappy tracks).
    """
    n = len(z)
    q, R = sig_jerk ** 2, np.asarray(sig_meas) ** 2
    have_v = np.zeros(n, dtype=bool) if zv is None else np.isfinite(zv)
    Rv = None if sig_v is None else np.asarray(sig_v) ** 2
    xf = np.zeros((n, 3))
    Pf = np.zeros((n, 3, 3))
    xp = np.zeros((n, 3))
    Pp = np.zeros((n, 3, 3))
    Fs = np.zeros((n, 3, 3))
    v0 = zv[0] if (zv is not None and have_v[0]) else 0.0
    x = np.array([z[0], v0, 0.0])
    s0 = float(np.median(np.sqrt(R)))
    sv0 = np.sqrt(Rv[0]) if (Rv is not None and have_v[0]) else 50 * s0
    P = np.diag([R[0], sv0 ** 2, (50 * s0) ** 2])
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
        if have_v[k]:
            S = P[1, 1] + Rv[k]
            innov = zv[k] - x[1]
            if innov * innov < (ZV_GATE_SIGMA ** 2) * S:
                K = P[:, 1] / S
                x = x + K * innov
                P = P - np.outer(K, P[1, :])
        xf[k], Pf[k] = x, P
    xs = xf.copy()
    for k in range(n - 2, -1, -1):
        C = Pf[k] @ Fs[k + 1].T @ np.linalg.inv(Pp[k + 1])
        xs[k] = xf[k] + C @ (xs[k + 1] - xp[k + 1])
    return xs[:, 0], xs[:, 1], xs[:, 2]


def _local_sigma(t, res, floor, win_s=0.5):
    """Per-sample noise scale: median |residual| in a +/-win_s window."""
    out = np.empty(len(res))
    for k in range(len(res)):
        a = np.searchsorted(t, t[k] - win_s)
        b = np.searchsorted(t, t[k] + win_s, side="right")
        out[k] = max(floor, float(np.median(res[a:b])))
    return out


def smooth_track(t, pts, car_len, jac_scale=None, zv=None, sig_v=None):
    """Smoothed positions and velocities for an (n,2) world-point array.

    Measurement noise is adaptive twice over: jac_scale (local world-px per
    image-px of the homography) inflates it where the projection amplifies —
    near-horizon geometry during aggressive camera moves turns 1px of bbox
    jitter into 10+ world px — and a second pass re-weights by each
    segment's actual residuals, so motion-blurred moving cars are trusted
    less than pin-sharp stopped ones instead of everything sharing one
    global sigma.

    zv: optional (n,2) LK velocity measurements (NaN rows where absent),
    sig_v: per-sample velocity sigma. Fused per axis with its own R_v.
    """
    sm = SIGMA_MEAS_FRAC * car_len
    sj = SIGMA_JERK_FRAC * car_len
    sig = np.full(len(t), sm)
    if jac_scale is not None:
        sig = sm * np.maximum(np.asarray(jac_scale), 1.0)
    zvx = zv[:, 0] if zv is not None else None
    zvy = zv[:, 1] if zv is not None else None

    def run_pass(sig_arr):
        xs, vx, _ = _rts_1d(t, pts[:, 0], sig_arr, sj, zvx, sig_v)
        ys, vy, _ = _rts_1d(t, pts[:, 1], sig_arr, sj, zvy, sig_v)
        return xs, ys, vx, vy

    xs, ys, vx, vy = run_pass(sig)
    res = np.hypot(pts[:, 0] - xs, pts[:, 1] - ys)
    sig = np.maximum(sig, _local_sigma(t, res, sm))
    return run_pass(sig)


def prefilter_jumps(frames, pts, car_len):
    """Montanino-Punzo impossible-point screen: greedily keep observations
    whose world displacement from the last kept one stays under
    JUMP_CARLEN_PER_FRAME car lengths per frame of gap (~220 km/h at 25fps —
    nothing on a jammed road moves that fast; ID-switch teleports and
    projective blowups do). Returns indices kept."""
    keep = [0]
    for k in range(1, len(pts)):
        gap = max(frames[k] - frames[keep[-1]], 1)
        lim = JUMP_CARLEN_PER_FRAME * car_len * gap
        if float(np.hypot(*(pts[k] - pts[keep[-1]]))) <= lim:
            keep.append(k)
    return np.asarray(keep)


def clip_zv_spikes(t, zv, car_len):
    """Clip z_v accel spikes to a rolling-median envelope: each component is
    limited to (local median ± a_max*win + floor). Montanino-Punzo step 2
    for the velocity stream — glare/blur frames occasionally survive the
    flow spread gate as a lone wild sample."""
    out = zv.copy()
    allow = ZV_AMAX_CARLEN * car_len * ZV_SPIKE_WIN_S + 0.3 * car_len
    fin = np.isfinite(zv[:, 0])
    idx = np.flatnonzero(fin)
    if len(idx) < 5:
        return out
    tf = t[idx]
    for ax in (0, 1):
        vals = zv[idx, ax]
        med = np.empty(len(idx))
        for j in range(len(idx)):
            a = np.searchsorted(tf, tf[j] - ZV_SPIKE_WIN_S)
            b = np.searchsorted(tf, tf[j] + ZV_SPIKE_WIN_S, side="right")
            med[j] = np.median(vals[a:b])
        out[idx, ax] = np.clip(vals, med - allow, med + allow)
    return out


def _ricker(width, a):
    """Mexican-hat wavelet, unit-energy (scipy.signal.ricker was removed in
    scipy 1.15; this is the same kernel)."""
    x = np.arange(width) - (width - 1) / 2.0
    A = 2.0 / (np.sqrt(3.0 * a) * np.pi ** 0.25)
    return A * (1 - (x / a) ** 2) * np.exp(-(x ** 2) / (2 * a ** 2))


def wavelet_onset(t, speed, a_idx, fps):
    """Deceleration-onset time via Mexican-hat wavelet energy (Zheng et al.
    2011): the energy peak of the CWT of the speed series marks the arrival
    of the deceleration wave at this vehicle, robust to residual noise.
    Searches around the hysteresis stop-run start a_idx; parabolic sub-frame
    interpolation on the energy peak. Returns time or None."""
    n = len(speed)
    if n < 8:
        return None
    E = np.zeros(n)
    for w_s in WAVELET_SCALES_S:
        a = max(w_s * fps, 1.5)
        width = min(int(10 * a) | 1, 2 * n - 1)
        c = np.convolve(speed, _ricker(width, a), mode="same")
        E += c * c
    lo = max(0, a_idx - int(WAVELET_SEARCH_BACK_S * fps))
    hi = min(n, a_idx + int(WAVELET_SEARCH_FWD_S * fps) + 1)
    if hi - lo < 3:
        return None
    p = lo + int(np.argmax(E[lo:hi]))
    frac = 0.0
    if 0 < p < n - 1:
        den = E[p - 1] - 2 * E[p] + E[p + 1]
        if den < 0:
            frac = float(np.clip(0.5 * (E[p - 1] - E[p + 1]) / den, -0.5, 0.5))
    return float(np.interp(p + frac, np.arange(n), t))


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


def load_flow(ws):
    """flow.json -> {track_id: {frame: (vx, vy, sigma_v)}}, or None.

    sigma_v combines the LK sub-pixel matching floor (ZV_SIGMA_IMG_PX image
    px, amplified by the local homography Jacobian) with the standard error
    of the median world displacement (1.858 * MAD / sqrt(n))."""
    import os
    if not os.path.exists(ws.path("flow.json")):
        return None
    fl = ws.load("flow.json")
    dt = fl["dt"]
    out = {}
    for tr in fl["tracks"]:
        rec = {}
        for f, vx, vy, n, sp, jac in zip(tr["frames"], tr["vx"], tr["vy"],
                                         tr["n"], tr["spread"], tr["jac"]):
            sig = max(ZV_SIGMA_IMG_PX * max(jac, 1.0),
                      1.858 * sp / math.sqrt(n)) / dt
            rec[f] = (vx, vy, sig)
        out[tr["id"]] = rec
    return out


def run(ws):
    meta = ws.meta
    fps = meta["fps"]
    Hs = [np.asarray(h) for h in ws.load("homographies.json")["H"]]
    data = ws.load("tracks.json")
    flow = load_flow(ws)
    print("velocity fusion: ON (flow.json)" if flow else
          "velocity fusion: OFF (no flow.json; run `tva flow` first)")

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

    n_jump_dropped = 0
    world_tracks = []
    stop_events = []
    for tr in data["tracks"]:
        obs = [o for o in tr["obs"] if o[0] < len(Hs)]
        if len(obs) < MIN_TRACK_FRAMES:
            continue
        frames = [o[0] for o in obs]
        pts = np.array([apply_h(Hs[f], [(cx, cy)])[0]
                        for f, cx, cy, *_ in obs])
        # Montanino-Punzo prefilter: impossible position jumps
        keep = prefilter_jumps(frames, pts, car_len)
        if len(keep) < MIN_TRACK_FRAMES:
            continue
        n_jump_dropped += len(obs) - len(keep)
        obs = [obs[k] for k in keep]
        frames = [o[0] for o in obs]
        pts = pts[keep]
        t = np.array(frames) / fps
        jac = np.array([jacobian_scale(Hs[f], (cx, cy))
                        for f, cx, cy, *_ in obs])
        zv = sig_v = None
        if flow is not None and tr["id"] in flow:
            rec = flow[tr["id"]]
            zv = np.full((len(frames), 2), np.nan)
            sig_v = np.ones(len(frames))
            for i, f in enumerate(frames):
                if f in rec:
                    vx_m, vy_m, sg = rec[f]
                    zv[i] = (vx_m, vy_m)
                    sig_v[i] = sg
            zv = clip_zv_spikes(t, zv, car_len)
        xs, ys, vx, vy = smooth_track(t, pts, car_len, jac_scale=jac,
                                      zv=zv, sig_v=sig_v)
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
            t_wav = wavelet_onset(t, speed, a, fps)
            stop_events.append({
                "track": tr["id"], "frame": frames[a],
                "t": round(float(t_on), 3),
                "t_wav": round(t_wav, 3) if t_wav is not None
                         else round(float(t_on), 3),
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
        "velocity_fusion": flow is not None,
        "tracks": world_tracks, "stop_events": stop_events,
    })
    if n_jump_dropped:
        print(f"prefilter dropped {n_jump_dropped} impossible-jump obs")
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
