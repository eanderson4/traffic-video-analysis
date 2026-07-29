"""Kinematic QA metrics (Punzo 2011 + K1 additions) and the wave fit.

`tva qa --work <dir>` prints the metrics and writes qa/report.json:
- M4 jerk share: fraction of |jerk| above 15 m/s^3 (car-length scale,
  1 car_len ~ 4.5 m), from the published speed series — computed the same
  way for baseline and fused runs so the comparison is fair.
- M4 stopped speeds: speed samples inside hysteresis 'stopped' runs, in
  car_len/s; a working estimator pins these ~0 (< 0.1 car_len/s).
- M4 integral consistency: | integral(v dt) - path(x,y) | / path per track —
  speed states must re-integrate to the positions they were filtered with.
- Parked-car flow validation: median |z_v| on static tracks must be
  < 0.05 car_len/s — direct proof the flow measurement is unbiased.
- M5 wave fit: Theil-Sen line through stop-onset (t, station) points near
  the centerline; slope = wave speed, bootstrap CI, sanity band 10-25 km/h.
"""
import numpy as np

CAR_LEN_M = 4.5            # fleet-mean passenger-car length prior
JERK_MAX_MS3 = 15.0        # Punzo 2011 physical jerk bound
STOPPED_MAX_CARLEN_S = 0.10
PARKED_ZV_MAX_CARLEN_S = 0.05
WAVE_BAND_KMH = (10.0, 25.0)
WAVE_MAX_LAT_CARLEN = 3.0  # stop events this far off the centerline excluded
N_BOOT = 2000


def _theil_sen(t, s):
    """Theil-Sen slope + intercept (median of pairwise slopes)."""
    t, s = np.asarray(t, float), np.asarray(s, float)
    slopes = []
    for i in range(len(t)):
        dt = t[i + 1:] - t[i]
        ok = np.abs(dt) > 1e-9
        slopes.extend(((s[i + 1:] - s[i])[ok] / dt[ok]).tolist())
    m = float(np.median(slopes))
    return m, float(np.median(s - m * t))


def jerk_share(tracks, fps, car_len):
    """Fraction of samples with |d2(speed)/dt2| > 15 m/s^3 equivalent."""
    lim = JERK_MAX_MS3 / CAR_LEN_M * car_len   # px/s^3
    bad = tot = 0
    for tr in tracks:
        t = np.asarray(tr["frames"], float) / fps
        v = np.asarray(tr["speed"], float)
        if len(v) < 5:
            continue
        jerk = np.gradient(np.gradient(v, t), t)
        bad += int((np.abs(jerk) > lim).sum())
        tot += len(jerk)
    return bad / max(tot, 1), tot


def stopped_speeds(tracks, car_len):
    """Per stopped-run median speed, in car_len/s.

    Per-run medians, not raw samples: the hysteresis run legitimately
    contains transition samples up to v_move (0.22 car_len/s) at its edges,
    so a raw p95 would test the hysteresis band, not the estimator. A
    'stopped car reads < 0.1 car_len/s' when the run's typical speed does.
    """
    vals = []
    for tr in tracks:
        fidx = {f: i for i, f in enumerate(tr["frames"])}
        for s, fa, fb in tr["runs"]:
            if not s:
                continue
            a, b = fidx[fa], fidx[fb]
            if b - a + 1 < 5:
                continue
            vals.append(float(np.median(tr["speed"][a:b + 1])) / car_len)
    return np.asarray(vals)


def integral_drift(tracks, fps, car_len):
    """Per moving track: |trapz(speed) - polyline path| / path."""
    drifts = []
    for tr in tracks:
        t = np.asarray(tr["frames"], float) / fps
        v = np.asarray(tr["speed"], float)
        x, y = np.asarray(tr["x"]), np.asarray(tr["y"])
        path = float(np.hypot(np.diff(x), np.diff(y)).sum())
        if path < 3.0 * car_len or len(t) < 10:
            continue
        drifts.append(abs(float(np.trapezoid(v, t)) - path) / path)
    return np.asarray(drifts)


def parked_flow(ws, wt):
    """Median |z_v| per static track in car_len/s, raw (jac-normalized, as
    measured) and corrected (minus the static-vehicle common-mode bias the
    filter subtracts — partly circular since those tracks defined the bias,
    but per-track medians still expose track-level residual noise).
    Returns (raw, corrected) arrays or None if no flow.json."""
    from .kinematics import load_flow
    flow = load_flow(ws)
    if flow is None:
        return None
    car_len = wt["car_len_px"]
    raw, cor, bias = [], [], []
    for tr in wt["tracks"]:
        if not tr["static"]:
            continue
        rec = flow.get(tr["id"])
        if rec is not None and len(rec) >= 10:
            raw.append(float(np.median([np.hypot(vx, vy)
                                        for vx, vy, _ in rec.values()]))
                       / car_len)
        mags = [m for m in (tr.get("zv_mag") or []) if m is not None]
        if len(mags) >= 10:
            # what the filter actually consumed (slip-field corrected)
            cor.append(float(np.median(mags)) / car_len)
        if tr.get("zv_med"):
            # signed per-track median = systematic offset = the actual
            # "flow is unbiased" test (|median| >> 0 would mean the
            # measurement pushes stopped cars in a consistent direction;
            # median-of-magnitudes only measures the noise floor)
            bias.append(float(np.hypot(*tr["zv_med"])) / car_len)
    return np.asarray(raw), np.asarray(cor), np.asarray(bias)


def wave_fit(ws, wt, rng=None, use_wavelet=True, fps=25.0):
    """Wave speed: connected slow region -> front boundary -> onset fit.

    A jammed road holds several slow pockets at once, and cars deep inside
    the jam keep creeping and re-stopping, so a single line through ALL
    stop onsets is meaningless (residual MAD was ~350-500 px on the hero
    clip). Instead, Ji-et-al-style:
      1. station = projection on a robust PCA axis of the event cloud
         (a spine-track centerline can clamp stations at its ends);
      2. Edie-bin the near-axis jam-direction tracks into a (t, station)
         speed field; slow = mean speed < v_move; take the LARGEST
         connected slow component = the main jam;
      3. its upstream boundary per time column is the stop-wave front;
         Theil-Sen through the boundary gives a field-based slope;
      4. M5 proper: Theil-Sen + bootstrap CI through the stop-onset events
         lying within WAVE_MAX_LAT_CARLEN car_len of that front line —
         onsets of cars joining the back of the queue. Internal re-stop
         events are reported but excluded from the fit.
    'backward' means the front moves against the traffic direction of the
    event tracks."""
    from scipy import ndimage

    car_len = wt["car_len_px"]
    events = wt["stop_events"]
    if len(events) < 3:
        return None
    P = np.array([[e["x"], e["y"]] for e in events], float)
    key = "t_wav" if use_wavelet else "t"
    et = np.array([e.get(key, e["t"]) for e in events])
    keep = np.ones(len(P), bool)
    axis = np.array([1.0, 0.0])
    for _ in range(3):
        c = P[keep].mean(axis=0)
        _, _, vt = np.linalg.svd(P[keep] - c, full_matrices=False)
        axis = vt[0]
        lat_all = np.abs((P - c) @ np.array([-axis[1], axis[0]]))
        keep = lat_all < WAVE_MAX_LAT_CARLEN * car_len
        if keep.sum() < 3:
            return None
    perp = np.array([-axis[1], axis[0]])
    s_all = (P - c) @ axis
    # traffic direction along the axis, from the event tracks' net motion
    by_id = {tr["id"]: tr for tr in wt["tracks"]}
    proj = []
    for e in events:
        tr = by_id.get(e["track"])
        if tr is None:
            continue
        d = np.array([tr["x"][-1] - tr["x"][0], tr["y"][-1] - tr["y"][0]])
        if np.hypot(*d) > 2 * car_len:
            proj.append(float(d @ axis))
    traffic_sign = float(np.sign(np.median(proj))) if proj else -1.0

    # Edie-binned slow field over near-axis jam-direction tracks
    S, T, V = [], [], []
    for tr in wt["tracks"]:
        Q = np.stack([tr["x"], tr["y"]], 1) - c
        if np.median(np.abs(Q @ perp)) > WAVE_MAX_LAT_CARLEN * car_len:
            continue
        d = Q[-1] - Q[0]
        if np.hypot(*d) > 2 * car_len and (d @ axis) * traffic_sign < 0:
            continue   # opposite carriageway
        S.append(Q @ axis)
        T.append(np.asarray(tr["frames"], float) / fps)
        V.append(np.asarray(tr["speed"], float))
    S, T, V = map(np.concatenate, (S, T, V))
    dtb, dsb = 0.4, car_len
    tb = np.arange(T.min(), T.max() + dtb, dtb)
    sb = np.arange(S.min(), S.max() + dsb, dsb)
    if len(tb) < 4 or len(sb) < 4:
        return None
    ti = np.clip(np.digitize(T, tb) - 1, 0, len(tb) - 2)
    si = np.clip(np.digitize(S, sb) - 1, 0, len(sb) - 2)
    num = np.zeros((len(sb) - 1, len(tb) - 1))
    den = np.zeros_like(num)
    np.add.at(num, (si, ti), V)
    np.add.at(den, (si, ti), 1)
    slow = (den > 0) & (num / np.maximum(den, 1) < wt["v_move"])
    lab, ncomp = ndimage.label(slow)
    if ncomp == 0:
        return None
    big = int(np.argmax(ndimage.sum(slow, lab, range(1, ncomp + 1)))) + 1
    comp = lab == big
    bt, bs = [], []
    for j in range(comp.shape[1]):
        rows = np.flatnonzero(comp[:, j])
        if len(rows) < 3:
            continue
        r = rows.max() if traffic_sign < 0 else rows.min()
        bt.append(tb[j] + dtb / 2)
        bs.append(sb[r] + (dsb if traffic_sign < 0 else 0.0))
    if len(bt) < 4:
        return None
    w_field, ic_field = _theil_sen(np.array(bt), np.array(bs))

    # onset events on the front line
    d_line = np.abs(s_all - (w_field * et + ic_field))
    on_front = keep & (d_line < WAVE_MAX_LAT_CARLEN * car_len)
    to_kmh = CAR_LEN_M / car_len * 3.6
    out = {
        "onset": key,
        "n_events_total": len(events),
        "n_events_near_axis": int(keep.sum()),
        "n_events_on_front": int(on_front.sum()),
        "field_slope_px_s": round(w_field, 2),
        "field_speed_kmh": round(abs(w_field) * to_kmh, 2),
        "backward": bool(w_field * traffic_sign < 0),
    }
    if on_front.sum() < 4:
        out["note"] = "too few onsets on front; field slope only"
        out["in_band"] = bool(WAVE_BAND_KMH[0] <= abs(w_field) * to_kmh
                              <= WAVE_BAND_KMH[1])
        return out
    t, s = et[on_front], s_all[on_front]
    slope, icpt = _theil_sen(t, s)
    rng = rng or np.random.default_rng(3)
    boots = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(t), len(t))
        if len(np.unique(t[idx])) < 2:
            continue
        boots.append(_theil_sen(t[idx], s[idx])[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    resid = s - (slope * t + icpt)
    out.update({
        "slope_px_s": round(slope, 2),
        "speed_kmh": round(abs(slope) * to_kmh, 2),
        "ci95_kmh": [round(abs(v) * to_kmh, 2)
                     for v in sorted((lo, hi), key=abs)] if lo * hi > 0
                    else [round(min(lo, hi) * to_kmh, 2),
                          round(max(lo, hi) * to_kmh, 2)],
        "ci95_px_s": [round(lo, 2), round(hi, 2)],
        "resid_mad_px": round(float(np.median(np.abs(resid))), 1),
        "in_band": bool(WAVE_BAND_KMH[0] <= abs(slope) * to_kmh
                        <= WAVE_BAND_KMH[1] and lo * hi > 0),
    })
    return out


def run(ws, tag=None):
    fps = ws.meta["fps"]
    wt = ws.load("world_tracks.json")
    car_len = wt["car_len_px"]
    tracks = wt["tracks"]

    js, n_samples = jerk_share(tracks, fps, car_len)
    sv = stopped_speeds(tracks, car_len)
    dr = integral_drift(tracks, fps, car_len)
    pf = parked_flow(ws, wt)
    wf = wave_fit(ws, wt)
    wf_cross = wave_fit(ws, wt, use_wavelet=False)

    rep = {
        "fusion": bool(wt.get("velocity_fusion")),
        "n_tracks": len(tracks), "car_len_px": car_len,
        "M4_jerk_share": round(js, 5), "M4_jerk_samples": n_samples,
        "M4_stopped_runs_carlen_s": {
            "median": round(float(np.median(sv)), 4) if len(sv) else None,
            "p95": round(float(np.percentile(sv, 95)), 4) if len(sv)
                   else None,
            "n_runs": len(sv),
            "share_below_0p1": round(float(
                (sv < STOPPED_MAX_CARLEN_S).mean()), 3) if len(sv)
                else None,
            "pass": bool(len(sv) and
                         np.percentile(sv, 95) < STOPPED_MAX_CARLEN_S),
        },
        "M4_integral_drift": {
            "median": round(float(np.median(dr)), 4) if len(dr) else None,
            "p90": round(float(np.percentile(dr, 90)), 4) if len(dr)
                   else None,
            "n_tracks": len(dr),
        },
        "parked_zv_carlen_s": None if pf is None else {
            "raw_mag_median": round(float(np.median(pf[0])), 4)
                              if len(pf[0]) else None,
            "raw_mag_p90": round(float(np.percentile(pf[0], 90)), 4)
                           if len(pf[0]) else None,
            "corrected_mag_median": round(float(np.median(pf[1])), 4)
                                    if len(pf[1]) else None,
            "signed_bias_median": round(float(np.median(pf[2])), 4)
                                  if len(pf[2]) else None,
            "signed_bias_p90": round(float(np.percentile(pf[2], 90)), 4)
                               if len(pf[2]) else None,
            "n_static_tracks": len(pf[0]),
            "pass_mag_raw": bool(len(pf[0]) and
                                 np.median(pf[0]) < PARKED_ZV_MAX_CARLEN_S),
            "pass_bias": bool(len(pf[2]) and
                              np.median(pf[2]) < PARKED_ZV_MAX_CARLEN_S),
        },
        "M5_wave": wf,
        "M5_wave_hysteresis_onsets": wf_cross,
    }
    name = f"report-{tag}.json" if tag else "report.json"
    import json
    import os
    os.makedirs(ws.qa, exist_ok=True)
    with open(f"{ws.qa}/{name}", "w") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps(rep, indent=1))
    print(f"wrote {ws.qa}/{name}")
    return rep
