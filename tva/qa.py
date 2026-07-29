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
    """All speed samples inside 'stopped' hysteresis runs, in car_len/s."""
    vals = []
    for tr in tracks:
        fidx = {f: i for i, f in enumerate(tr["frames"])}
        for s, fa, fb in tr["runs"]:
            if not s:
                continue
            a, b = fidx[fa], fidx[fb]
            vals.extend(v / car_len for v in tr["speed"][a:b + 1])
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
    """Median |z_v| per static track, in car_len/s. None if no flow.json."""
    import os
    if not os.path.exists(ws.path("flow.json")):
        return None
    fl = ws.load("flow.json")
    car_len = wt["car_len_px"]
    static = {tr["id"] for tr in wt["tracks"] if tr["static"]}
    meds = []
    for tr in fl["tracks"]:
        if tr["id"] not in static or len(tr["vx"]) < 10:
            continue
        mag = np.hypot(tr["vx"], tr["vy"])
        meds.append(float(np.median(mag)) / car_len)
    return np.asarray(meds)


def wave_fit(ws, wt, rng=None):
    """Theil-Sen wave speed through stop-onset (t, station) points."""
    from .kinematics import station_of
    cl = ws.load("centerline.json")
    car_len = wt["car_len_px"]
    events = wt["stop_events"]
    if len(events) < 3:
        return None
    ex = np.array([e["x"] for e in events])
    ey = np.array([e["y"] for e in events])
    et = np.array([e.get("t_wav", e["t"]) for e in events])
    s, lat = station_of(cl["x"], cl["y"], ex, ey)
    m = lat < WAVE_MAX_LAT_CARLEN * car_len
    t, s = et[m], s[m]
    if len(t) < 3:
        return None
    slope, icpt = _theil_sen(t, s)
    to_kmh = CAR_LEN_M / car_len * 3.6
    rng = rng or np.random.default_rng(3)
    boots = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(t), len(t))
        if len(np.unique(t[idx])) < 2:
            continue
        boots.append(_theil_sen(t[idx], s[idx])[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    resid = s - (slope * t + icpt)
    return {
        "n_events_total": len(events), "n_events_fit": int(m.sum()),
        "slope_px_s": round(slope, 2),
        "speed_kmh": round(abs(slope) * to_kmh, 2),
        "slope_kmh_signed": round(slope * to_kmh, 2),
        "ci95_kmh_signed": [round(lo * to_kmh, 2), round(hi * to_kmh, 2)],
        "ci95_px_s": [round(lo, 2), round(hi, 2)],
        "resid_mad_px": round(float(np.median(np.abs(resid))), 1),
        "in_band": bool(WAVE_BAND_KMH[0] <= abs(slope) * to_kmh
                        <= WAVE_BAND_KMH[1]),
    }


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

    rep = {
        "fusion": bool(wt.get("velocity_fusion")),
        "n_tracks": len(tracks), "car_len_px": car_len,
        "M4_jerk_share": round(js, 5), "M4_jerk_samples": n_samples,
        "M4_stopped_speed_carlen_s": {
            "median": round(float(np.median(sv)), 4) if len(sv) else None,
            "p95": round(float(np.percentile(sv, 95)), 4) if len(sv)
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
            "median": round(float(np.median(pf)), 4) if len(pf) else None,
            "max": round(float(pf.max()), 4) if len(pf) else None,
            "n_static_tracks": len(pf),
            "pass": bool(len(pf) and
                         np.median(pf) < PARKED_ZV_MAX_CARLEN_S),
        },
        "M5_wave": wf,
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
