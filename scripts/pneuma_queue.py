#!/usr/bin/env python3
"""Queueing-theory taunt, pNEUMA drone edition (Athens, signalized arterials).

Same analysis as scripts/sind_queue.py but for pNEUMA: no signal CSV, so
green windows are inferred from the data itself — a discharge onset (burst
of stop-bar crossings after a standing queue) marks green start; green ends
when flow across the stop bar stops (same trick as SinD Changchun).

pNEUMA CSV format (verified on 20181029_d2_0800_0830.csv):
  header row, then one row per track; ';' delimited;
  cols 0-3: track_id, type, traveled_d, avg_speed (track metadata)
  then repeating 6-col blocks: lat, lon, speed(km/h), lon_acc, lat_acc,
  time(s, global, 25 Hz, empty cells for missing timesteps).
Coordinates WGS84 -> local meters via equirectangular around session centroid.

Usage:
  python3 scripts/pneuma_queue.py <session.csv> recon      # QA trajectory map
  python3 scripts/pneuma_queue.py <session.csv> analyze    # queue taunt
Writes qa PNGs to testdata/datasets/pneuma/qa/.
"""
import os
import sys

import numpy as np

STOP_SPEED = 0.5          # m/s
MAX_SAT_HEADWAY_S = 6.0   # gaps longer than this are not saturated flow
PARKED_MAX_S = 150.0      # stopped-in-box longer than this = parked, not queued
MOTOR = ("Car", "Medium Vehicle", "Heavy Vehicle", "Bus", "Motorcycle")
R_EARTH = 6371000.0

def load_tracks(path, keep=MOTOR, cache=True):
    """Parse pNEUMA CSV -> list of dicts (tid, type, t, x, y, speed in m/s)."""
    cpath = path + ".npz"
    if cache and os.path.exists(cpath):
        z = np.load(cpath, allow_pickle=True)
        return z["tracks"].tolist()
    tracks = []
    with open(path) as f:
        f.readline()  # header
        for line in f:
            if not line.strip():
                continue
            head, _, rest = line.partition(";" * 3 + " ")  # not reliable; use split
            parts = line.split(";", 4)
            if len(parts) < 5:
                continue
            tid = int(parts[0])
            typ = parts[1].strip()
            rest = parts[4].strip()
            # empty cells (missing timesteps) -> nan
            while ";;" in rest:
                rest = rest.replace(";;", ";nan;")
            if rest.endswith(";"):
                rest += "nan"
            vals = np.fromstring(rest, sep=";")
            n = len(vals) // 6
            vals = vals[: n * 6].reshape(n, 6)
            lat, lon, spd, t = vals[:, 0], vals[:, 1], vals[:, 2], vals[:, 5]
            ok = ~np.isnan(t)
            if typ not in keep or ok.sum() < 5:
                continue
            tracks.append(dict(tid=tid, type=typ, t=t[ok], lat=lat[ok],
                               lon=lon[ok], speed=spd[ok] / 3.6))
    lat0 = np.mean([np.mean(c["lat"]) for c in tracks])
    lon0 = np.mean([np.mean(c["lon"]) for c in tracks])
    kx = np.cos(np.radians(lat0)) * R_EARTH
    for c in tracks:
        c["x"] = (c.pop("lon") - lon0) * kx * np.pi / 180.0
        c["y"] = (c.pop("lat") - lat0) * R_EARTH * np.pi / 180.0
    if cache:
        np.savez_compressed(cpath, tracks=np.array(tracks, dtype=object))
    return tracks


def recon(session, tracks, qa_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 14))
    colors = {"Car": "tab:blue", "Medium Vehicle": "tab:orange",
              "Heavy Vehicle": "tab:red", "Bus": "tab:purple",
              "Motorcycle": "tab:green", "Taxi": "tab:cyan"}
    for c in tracks:
        ax.plot(c["x"], c["y"], lw=0.4, alpha=0.5,
                color=colors.get(c["type"], "gray"))
    for typ, col in colors.items():
        ax.plot([], [], lw=2, color=col, label=typ)
    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m, local)")
    ax.set_ylabel("y (m, local)")
    ax.set_title(f"{session}: {len(tracks)} tracks")
    fig.tight_layout()
    out = os.path.join(qa_dir, f"trajectories-{session}.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


# ---- per-session geometry, in local meters (iterated from recon plots) ----
# Approach is defined in rotated coords: s = pos @ dir (along travel),
# w = pos @ perp (lateral). Lines/boxes are then axis-aligned in (s, w),
# so the analyze() logic mirrors sind_queue.py with dir=(1,0).
FOUR_WHEEL = ("Car", "Medium Vehicle", "Heavy Vehicle", "Bus")

SESSIONS = {
    # Junction at ~(12,80) local: one-way 2-lane street (dir 0.836,0.548)
    # heading NE into the junction; queue on the SW leg. Stop bar s=38
    # (= point ~(-3,74) map coords, SW corner of the junction box).
    "20181029_d2_0800_0830": dict(
        approach=dict(name="sw-leg-ne", dir=(0.83634174, 0.54820844),
                      in_line=(-10.0, (61.0, 74.0)),
                      out_line=(38.0, (61.0, 74.0)),
                      queue_box=(2.0, 38.0, 61.0, 74.0)),
        green_gap=15.0,      # min stop-bar flow gap (s) separating platoons
        green_offset=2.0,    # start-up lost time before first crossing (s)
    ),
}


def rotate(tracks, d, keep=FOUR_WHEEL):
    """Project tracks onto approach coords: x->s (along), y->w (lateral)."""
    d = np.asarray(d)
    p = np.array([-d[1], d[0]])
    out = []
    for c in tracks:
        if c["type"] not in keep:
            continue
        xy = np.column_stack([c["x"], c["y"]])
        out.append(dict(tid=c["tid"], t=c["t"], s=xy @ d, w=xy @ p,
                        speed=c["speed"]))
    return out


def crossings(car, line):
    """Times the car crosses the s=coord line going +s, w within (lo,hi)."""
    coord, (lo, hi) = line
    sd = car["s"] - coord
    out = []
    for i in range(len(sd) - 1):
        if sd[i] < 0 <= sd[i + 1] and lo <= (car["w"][i] + car["w"][i + 1]) / 2 <= hi:
            f = -sd[i] / (sd[i + 1] - sd[i] or 1e-9)
            out.append(car["t"][i] + f * (car["t"][i + 1] - car["t"][i]))
    return out


def infer_greens(dep_times, gap, offset):
    """Green windows from stop-bar discharge bursts (no signal CSV).

    A gap > `gap` seconds between consecutive stop-bar crossings separates
    discharge platoons; green starts `offset` s before a platoon's first
    crossing (start-up lost time) and ends `offset` s after its last.
    """
    dep = sorted(dep_times)
    if len(dep) < 10:
        return [], None, None
    cuts = [0] + [i + 1 for i in range(len(dep) - 1)
                  if dep[i + 1] - dep[i] > gap] + [len(dep)]
    greens = []
    for a, b in zip(cuts, cuts[1:]):
        if b - a >= 2:      # a real platoon, not a stray pair
            greens.append((dep[a] - offset, dep[b - 1] + offset))
    if len(greens) < 2:
        return greens, None, None
    starts = np.array([g0 for g0, _ in greens])
    C = float(np.median(np.diff(starts)))
    g = float(np.median([b - a for a, b in greens]))
    return greens, C, g


def analyze(ap, cars, greens, t_end, dt=0.1):
    s0, s1, w0, w1 = ap["queue_box"]
    arrivals, departures = {}, {}
    q = np.zeros(int(t_end / dt) + 1, int)
    waits = []
    parked = 0
    for car in cars:
        a = crossings(car, ap["in_line"])
        d = crossings(car, ap["out_line"])
        if a:
            arrivals[car["tid"]] = a[0]
        if d:
            departures[car["tid"]] = d[-1]
        in_q = ((car["s"] >= s0) & (car["s"] <= s1) &
                (car["w"] >= w0) & (car["w"] <= w1) &
                (car["speed"] < STOP_SPEED))
        if not in_q.any():
            continue
        # parked vehicles (Athens curb lane): stopped far longer than any
        # red phase — not signal queue, exclude from Q and waits
        t_in = car["t"][in_q]
        if t_in[-1] - t_in[0] > PARKED_MAX_S:
            parked += 1
            continue
        # one count per vehicle per dt bin (25 Hz samples, 10 Hz grid)
        idx = np.unique((car["t"][in_q] / dt).astype(int).clip(0, len(q) - 1))
        q[idx] += 1
        start = None
        for ti, qi in zip(car["t"], in_q):
            if qi and start is None:
                start = ti
            elif not qi and start is not None:
                waits.append((car["tid"], start, ti - start))
                start = None
        if start is not None:
            waits.append((car["tid"], start, car["t"][-1] - start))
    dist = ap["out_line"][0] - ap["in_line"][0]
    v_free = float(np.median(
        [np.percentile(c["speed"][c["speed"] > 2 * STOP_SPEED], 75)
         for c in cars if (c["speed"] > 2 * STOP_SPEED).any()]))
    delays = {tid: departures[tid] - arrivals[tid] - dist / v_free
              for tid in set(arrivals) & set(departures)
              if departures[tid] > arrivals[tid]}
    truncated = len(set(arrivals) - set(departures))
    dep_t = sorted(departures.values())
    sat_gaps = [b - a for a, b in zip(dep_t, dep_t[1:])
                if b - a < MAX_SAT_HEADWAY_S
                and any(g0 <= a <= g1 for g0, g1 in greens)
                and q[min(int(a / dt), len(q) - 1)] > 0]
    sat = 1.0 / np.mean(sat_gaps) if len(sat_gaps) >= 8 else None
    return dict(name=ap["name"], arrivals=arrivals, departures=departures,
                q=q, dt=dt, waits=waits, delays=delays, truncated=truncated,
                v_free=v_free, dist=dist, sat=sat, n_sat=len(sat_gaps),
                parked=parked)


def cycle_taunt(r, greens, t_end):
    """Per-cycle measured vs D/D/1, same theory as sind_queue.cycle_taunt."""
    if not r["sat"] or len(greens) < 2:
        return None, []
    lam = len(r["arrivals"]) / t_end
    s, dt, q = r["sat"], r["dt"], r["q"]
    rows = []
    starts = [g0 for g0, _ in greens] + [t_end]
    for i in range(len(greens)):
        a, b = starts[i], starts[i + 1]
        g_len = greens[i][1] - greens[i][0]
        red = (b - a) - g_len
        ia, ib = int(a / dt), min(int(b / dt), len(q) - 1)
        if ib <= ia:
            continue
        qm = int(q[ia:ib].max())
        area = float(q[ia:ib].sum() * dt)
        q_theory = lam * red
        clears = s > lam
        d_theory = (0.5 * lam * red * (red + lam * red / (s - lam))
                    if clears else None)
        rows.append(dict(cycle=i, t0=a, q_max=qm, area=area, red=red,
                         q_theory=q_theory, d_theory=d_theory))
    return rows, lam


def qa_geometry(session, tracks, ap, qa_dir):
    """Trajectory zoom with stop bar / in-line / queue box overlaid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = np.asarray(ap["dir"])
    p = np.array([-d[1], d[0]])
    fig, ax = plt.subplots(figsize=(12, 10))
    for c in tracks:
        m = (c["x"] > -60) & (c["x"] < 60) & (c["y"] > 30) & (c["y"] < 120)
        if m.any():
            ax.plot(c["x"][m], c["y"][m], lw=0.5, alpha=0.3, color="tab:blue")
    def line_pts(l):
        coord, (wlo, whi) = l
        return [d * coord + p * wlo, d * coord + p * whi]
    for l, col, lab in ((ap["in_line"], "lime", "in-line"),
                        (ap["out_line"], "red", "stop bar")):
        pts = np.array(line_pts(l))
        ax.plot(pts[:, 0], pts[:, 1], color=col, lw=2.5, label=lab)
    s0, s1, w0, w1 = ap["queue_box"]
    box = np.array([d * s + p * w for s, w in
                    ((s0, w0), (s1, w0), (s1, w1), (s0, w1), (s0, w0))])
    ax.plot(box[:, 0], box[:, 1], color="red", lw=1.2, ls="--",
            label="queue box")
    ax.legend()
    ax.set_xlim(-60, 60)
    ax.set_ylim(30, 120)
    ax.set_aspect("equal")
    ax.set_title(f"{session} approach {ap['name']}: geometry")
    out = os.path.join(qa_dir, f"geometry-{session}-{ap['name']}.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot(session, r, greens, cyc, qa_dir, qa_out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grid = np.arange(len(r["q"])) * r["dt"]
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 11), sharex=True,
                                        height_ratios=[1, 1, 0.7])
    at = sorted(r["arrivals"].values())
    dt_ = sorted(r["departures"].values())
    ax1.step(at, np.arange(1, len(at) + 1), where="post",
             label=f"arrivals at in-line ({len(at)})")
    ax1.step(dt_, np.arange(1, len(dt_) + 1), where="post",
             label=f"departures at stop bar ({len(dt_)})")
    for g0, g1 in greens:
        ax1.axvspan(g0, g1, color="green", alpha=0.10)
    ax1.legend(loc="upper left")
    ax1.set_ylabel("cumulative vehicles")
    ax1.set_title(f"{session} {r['name']}: Newell curves (green = inferred)")
    ax2.fill_between(grid, r["q"], step="post", color="tab:red", alpha=0.6)
    for g0, g1 in greens:
        ax2.axvspan(g0, g1, color="green", alpha=0.10)
    ax2.set_ylabel("queue Q(t) (veh)")
    if cyc:
        xm = [x["q_max"] for x in cyc]
        xt = [x["q_theory"] for x in cyc]
        x0 = [x["t0"] for x in cyc]
        w = 18
        ax3.bar([t - w / 2 - 1 for t in x0], xm, w, label="measured q_max",
                align="edge")
        ax3.bar([t + w / 2 + 1 for t in x0], xt, w, label="λr theory",
                align="edge")
        ax3.set_xlabel("t (s)  [one pair per cycle, at cycle start]")
        ax3.set_ylabel("max queue (veh)")
        ax3.legend()
    fig.tight_layout()
    fig.savefig(qa_out, dpi=110)
    plt.close(fig)
    return qa_out


def run_analyze(path, session, qa_dir, tracks, t_end):
    cfg = SESSIONS[session]
    ap = cfg["approach"]
    geo = qa_geometry(session, tracks, ap, qa_dir)
    print(f"QA geometry: {geo}")
    cars = rotate(tracks, ap["dir"])
    # first pass: departures only -> green inference -> full analysis
    pre_dep = []
    for car in cars:
        pre_dep += crossings(car, ap["out_line"])
    greens, C, g = infer_greens(pre_dep, cfg["green_gap"], cfg["green_offset"])
    print(f"inferred {len(greens)} greens: C={C and round(C,1)}s "
          f"g={g and round(g,1)}s")
    for g0, g1 in greens:
        print(f"   green {g0:7.1f}-{g1:7.1f} ({g1 - g0:5.1f}s)")
    r = analyze(ap, cars, greens, t_end)
    cyc, lam = cycle_taunt(r, greens, t_end)
    d = list(r["delays"].values())
    s_vph = r["sat"] * 3600 if r["sat"] else None
    cap = r["sat"] * g / C if (r["sat"] and C and g) else None
    X = lam / cap if cap else None
    print(f"approach {ap['name']}: arr {len(r['arrivals'])} "
          f"dep {len(r['departures'])} trunc {r['truncated']} "
          f"parked-excl {r['parked']}, "
          f"λ {lam * 3600:.0f} veh/h, s {s_vph and round(s_vph)} veh/h "
          f"(n={r['n_sat']}), v_free {r['v_free'] * 3.6:.0f} km/h")
    if X:
        d1 = C * (1 - g / C) ** 2 / (2 * (1 - min(1, X) * g / C))
        print(f"cap {cap * 3600:.0f} veh/h, X={X:.2f}; delay meas mean "
              f"{np.mean(d):.1f}s med {np.median(d):.1f}s vs D/D/1 {d1:.1f}s")
    if cyc:
        print(f"{'cyc':>3} {'t0':>7} {'red':>5} {'qmax':>4} {'λr':>5} "
              f"{'delay veh·s':>11} {'D/D/1':>8}")
        for x in cyc:
            print(f"{x['cycle']:>3} {x['t0']:>7.0f} {x['red']:>5.0f} "
                  f"{x['q_max']:>4} {x['q_theory']:>5.1f} {x['area']:>11.0f} "
                  f"{x['d_theory'] if x['d_theory'] is not None else float('nan'):>8.0f}")
        mq = np.mean([x["q_max"] for x in cyc])
        tq = np.mean([x["q_theory"] for x in cyc])
        ma = np.mean([x["area"] for x in cyc])
        ta = [x["d_theory"] for x in cyc if x["d_theory"] is not None]
        print(f"means: q_max {mq:.1f} vs λr {tq:.1f}; delay {ma:.0f} veh·s "
              + (f"vs D/D/1 {np.mean(ta):.0f} veh·s" if ta else "(never clears)"))
    out = os.path.join(qa_dir, f"qa-pneuma-{session}-{ap['name']}.png")
    print(f"QA: {plot(session, r, greens, cyc, qa_dir, out)}")


def main():
    path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "recon"
    session = os.path.basename(path).replace(".csv", "")
    qa_dir = os.path.join(os.path.dirname(path), "qa")
    os.makedirs(qa_dir, exist_ok=True)
    tracks = load_tracks(path)
    t_end = max(c["t"][-1] for c in tracks)
    n_t = sum(len(c["t"]) for c in tracks)
    print(f"{len(tracks)} motor tracks, {n_t} samples, t_end {t_end:.0f}s")
    if mode == "recon":
        print(recon(session, tracks, qa_dir))
    elif mode == "analyze":
        run_analyze(path, session, qa_dir, tracks, t_end)


if __name__ == "__main__":
    main()
