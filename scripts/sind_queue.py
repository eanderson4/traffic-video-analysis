#!/usr/bin/env python3
"""Queueing-theory taunt, SinD ground-truth edition.

Measures, per approach of a SinD recording (smoothed vehicle tracks: meters,
~30fps timestamps, persistent track ids; plus the signal CSV where present):

- arrivals      in_line crossings, matched by track id
- queue joins   cars starting a stopped run inside the queue box
- departures    stop-bar crossings
- queue Q(t)    stopped (<STOP_SPEED) inside the queue box
- delay/car     t_out - t_in - free-flow travel over the approach
                (persistent ids: a real per-car measurement)
- saturation    1/mean headway of stop-bar crossings during green with a
                queue present

Then the taunt — textbook predictions from the measured lambda / saturation /
signal timing against the measured mean delay:

- D/D/1 uniform delay:  d = C(1-g/C)^2 / (2(1 - min(1,X)*g/C)),  X = lambda/c
- Webster (1958):       d1 above + d2 = X^2 / (2*lam*(1-X))
                        (random-arrival correction; blows up as X -> 1)
- oversaturated (X>=1): steady-state theory just says "unbounded" —
                        we print the measured queue growth rate instead.

Usage:  python3 scripts/sind_queue.py <sind-recording-dir>
Writes qa PNGs next to the CSVs (qa-sind-<approach>.png).
"""
import os
import sys

import numpy as np
import pandas as pd

STOP_SPEED = 0.5          # m/s
MOTOR = ("car", "truck", "bus")
MAX_SAT_HEADWAY_S = 6.0   # gaps longer than this are not saturated flow

# Per-recording geometry in SinD local coords (meters), from recon maps.
# line = (coord, (lo, hi)): crossing line perpendicular to travel direction.
RECORDINGS = {
    "tianjin-8_2_1": dict(
        tracks="Veh_smoothed_tracks.csv", lights="TrafficLight_8_2_1.csv",
        light_map={},   # undersaturated; signal assignment not decoded
        approaches=[
            dict(name="west",  dir=(1, 0),
                 in_line=(-20.0, (10.5, 15.0)), out_line=(9.0, (10.5, 15.0)),
                 queue_box=(-20, 9, 10.5, 15.0)),
            dict(name="east",  dir=(-1, 0),
                 in_line=(33.0, (16.5, 21.0)), out_line=(20.0, (16.5, 21.0)),
                 queue_box=(20, 33, 16.5, 21.0)),
            dict(name="north", dir=(0, -1),
                 in_line=(33.0, (9.5, 13.5)), out_line=(16.0, (9.5, 13.5)),
                 queue_box=(9.5, 13.5, 16, 33)),
            dict(name="south", dir=(0, 1),
                 in_line=(-8.0, (15.5, 18.5)), out_line=(12.0, (15.5, 18.5)),
                 queue_box=(15.5, 18.5, -8, 12)),
        ]),
    "changchun-507_009": dict(
        tracks="Veh_smoothed_tracks.csv", lights="Traffic_Lights.csv",
        light_map={"north": "Vehicle Traffic light 1",
                   "south": "Vehicle Traffic light 1",
                   "west": "Vehicle Traffic light 2",
                   "east": "Vehicle Traffic light 2"},
        approaches=[
            dict(name="north", dir=(0, -1),    # southbound carriageway
                 in_line=(35.0, (-29, -21)), out_line=(4.0, (-29, -21)),
                 queue_box=(-29, -21, 4, 76)),
            dict(name="south", dir=(0, 1),     # northbound carriageway
                 in_line=(-40.0, (-19, -11)), out_line=(-10.0, (-19, -11)),
                 queue_box=(-19, -11, -83, -10)),
            dict(name="west",  dir=(1, 0),     # eastbound
                 in_line=(-45.0, (-10, -4)), out_line=(-29.0, (-10, -4)),
                 queue_box=(-64, -29, -10, -4)),
            dict(name="east",  dir=(-1, 0),    # westbound
                 in_line=(20.0, (-3, 4)), out_line=(-11.0, (-3, 4)),
                 queue_box=(-11, 24, -3, 4)),
        ]),
}


def load_cars(path):
    df = pd.read_csv(path)
    df = df[df.agent_type.isin(MOTOR)]
    cars = []
    for tid, g in df.groupby("track_id"):
        g = g.sort_values("frame_id")
        t = g.timestamp_ms.to_numpy() / 1000.0
        cars.append(dict(tid=int(tid), t=t, x=g.x.to_numpy(),
                         y=g.y.to_numpy(),
                         speed=np.hypot(g.vx, g.vy)))
    return cars


def load_green_windows(path, col):
    """(green windows, C, g) from a SinD traffic-light CSV, state 1 = green."""
    tl = pd.read_csv(path)
    greens, cur, t0 = [], None, 0.0
    for _, r in tl.iterrows():
        st, tt = r[col], r["timestamp(ms)"] / 1000.0
        if st == 1 and cur != 1:
            t0 = tt
        elif st != 1 and cur == 1:
            greens.append((t0, tt))
        cur = st
    if len(greens) < 2:
        return greens, None, None
    starts = np.array([a for a, _ in greens])
    C = float(np.median(np.diff(starts)))
    g = float(np.median([b - a for a, b in greens]))
    return greens, C, g


def crossings(car, line, dir_):
    """Times the car crosses the line in the travel direction."""
    coord, (lo, hi) = line
    along = car["x"] if dir_[0] else car["y"]
    lat = car["y"] if dir_[0] else car["x"]
    sd = (along - coord) * (dir_[0] or dir_[1])
    out = []
    for i in range(len(sd) - 1):
        if sd[i] < 0 <= sd[i + 1] and lo <= (lat[i] + lat[i + 1]) / 2 <= hi:
            f = -sd[i] / (sd[i + 1] - sd[i] or 1e-9)
            out.append(car["t"][i] + f * (car["t"][i + 1] - car["t"][i]))
    return out


def analyze(ap, cars, greens, t_end, dt=0.1):
    x0, x1, y0, y1 = ap["queue_box"]
    arrivals, departures, joins = {}, {}, []
    q = np.zeros(int(t_end / dt) + 1, int)
    waits = []                       # (tid, t_join, duration)
    for car in cars:
        a = crossings(car, ap["in_line"], ap["dir"])
        d = crossings(car, ap["out_line"], ap["dir"])
        if a:
            arrivals[car["tid"]] = a[0]
        if d:
            departures[car["tid"]] = d[-1]
        in_q = ((car["x"] >= x0) & (car["x"] <= x1) &
                (car["y"] >= y0) & (car["y"] <= y1) &
                (car["speed"] < STOP_SPEED))
        if not in_q.any():
            continue
        idx = (car["t"] / dt).astype(int).clip(0, len(q) - 1)
        np.add.at(q, idx[in_q], 1)
        start = None
        for ti, qi in zip(car["t"], in_q):
            if qi and start is None:
                start = ti
                joins.append((car["tid"], ti))
            elif not qi and start is not None:
                waits.append((car["tid"], start, ti - start))
                start = None
        if start is not None:
            waits.append((car["tid"], start, car["t"][-1] - start))
    dist = abs(ap["out_line"][0] - ap["in_line"][0])
    v_free = float(np.median(
        [np.percentile(c["speed"][c["speed"] > 2 * STOP_SPEED], 75)
         for c in cars if (c["speed"] > 2 * STOP_SPEED).any()]))
    delays = {tid: departures[tid] - arrivals[tid] - dist / v_free
              for tid in set(arrivals) & set(departures)
              if departures[tid] > arrivals[tid]}
    truncated = len(set(arrivals) - set(departures))
    # saturation: headways of stop-bar departures during green + queue>0
    dep_t = sorted(departures.values())
    sat_gaps = [b - a for a, b in zip(dep_t, dep_t[1:])
                if b - a < MAX_SAT_HEADWAY_S
                and any(g0 <= a <= g1 for g0, g1 in greens)
                and q[min(int(a / dt), len(q) - 1)] > 0]
    sat = 1.0 / np.mean(sat_gaps) if len(sat_gaps) >= 8 else None
    return dict(name=ap["name"], arrivals=arrivals, joins=joins,
                departures=departures, q=q, dt=dt, waits=waits,
                delays=delays, truncated=truncated, v_free=v_free,
                dist=dist, sat=sat)


def taunt(r, C, g, t_end):
    """Measured vs D/D/1-uniform vs Webster, given measured λ, s, C, g."""
    lam = len(r["arrivals"]) / t_end          # veh/s
    d_meas = float(np.mean(list(r["delays"].values()))) if r["delays"] else None
    out = dict(lam_vph=lam * 3600, sat_vph=(r["sat"] * 3600
                                            if r["sat"] else None),
               delay_meas=d_meas)
    if not (C and g and r["sat"] and lam > 0):
        return out
    s, cap = r["sat"], r["sat"] * g / C
    X = lam / cap
    out.update(C=C, g=g, X=X)
    d1 = C * (1 - g / C) ** 2 / (2 * (1 - min(1, X) * g / C))
    out["d_dd1"] = d1
    if X < 1:
        d2 = X ** 2 / (2 * lam * (1 - X))
        out["d_webster"] = d1 + min(d2, C)   # Webster's own d2 cap sanity
    else:
        out["d_webster"] = None
        # deterministic oversaturated growth: queue grows at lam - cap
        out["growth_vps"] = lam - cap
    return out


def cycle_taunt(r, greens, t_end):
    """Per-signal-cycle comparison, D/D/1 against measurement.

    Cycle k = [green_start_k, green_start_k+1). Theory (deterministic uniform
    arrivals λ, discharge s during green g, red r = C-g):
      q_max  = λ·r                    (red-phase accumulation)
      delay  = ½·λ·r · (r + λ·r/(s−λ)) per cycle, if the queue clears
    Measured: max Q over the cycle and ∫Q dt (total stopped veh·s).
    """
    if not r["sat"] or len(greens) < 2:
        return None
    lam = len(r["arrivals"]) / t_end
    s, dt = r["sat"], r["dt"]
    q = r["q"]
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
        rows.append(dict(cycle=i, q_max=qm, area=area, red=red,
                         q_theory=q_theory, d_theory=d_theory))
    if not rows:
        return None
    agg = dict(
        n=len(rows),
        q_max_meas=float(np.mean([x["q_max"] for x in rows])),
        q_max_theory=float(np.mean([x["q_theory"] for x in rows])),
        area_meas=float(np.mean([x["area"] for x in rows])),
        area_theory=float(np.mean([x["d_theory"] for x in rows
                                   if x["d_theory"] is not None]))
        if any(x["d_theory"] is not None for x in rows) else None,
        clears_all=all(x["d_theory"] is not None for x in rows))
    return agg


def plot(rec_dir, r, greens, t_end):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grid = np.arange(len(r["q"])) * r["dt"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    at = sorted(r["arrivals"].values())
    dt_ = sorted(r["departures"].values())
    ax1.step(at, np.arange(1, len(at) + 1), where="post",
             label=f"arrivals at in-line ({len(at)})")
    ax1.step(dt_, np.arange(1, len(dt_) + 1), where="post",
             label=f"departures at stop bar ({len(dt_)})")
    for g0, g1 in greens:
        ax1.axvspan(g0, g1, color="green", alpha=0.08)
    ax1.legend(loc="upper left")
    ax1.set_ylabel("cumulative vehicles")
    ax2.fill_between(grid, r["q"], step="post", color="tab:red", alpha=0.6)
    for g0, g1 in greens:
        ax2.axvspan(g0, g1, color="green", alpha=0.08)
    ax2.set_ylabel("queue length (veh)")
    ax2.set_xlabel("t (s)  [green bands = approach green]")
    d = r["delays"]
    ax1.set_title(f"{r['name']}: q̄ {r['q'].mean():.1f} max {r['q'].max()}, "
                  f"delay med {np.median(list(d.values())) if d else 0:.0f}s, "
                  f"wait {sum(w[2] for w in r['waits']):.0f} veh·s")
    fig.tight_layout()
    out = os.path.join(rec_dir, f"qa-sind-{r['name']}.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main():
    rec_dir = sys.argv[1].rstrip("/")
    key = next(k for k in RECORDINGS if k in rec_dir)
    cfg = RECORDINGS[key]
    cars = load_cars(os.path.join(rec_dir, cfg["tracks"]))
    t_end = max(c["t"][-1] for c in cars)
    print(f"{len(cars)} motor vehicles, {t_end:.0f}s")
    print(f"{'appr':>6} {'arr':>4} {'join':>4} {'dep':>4} {'λ/h':>5} "
          f"{'q̄':>5} {'qmax':>4} {'d_med':>5} {'d_mean':>6} {'s/h':>5} "
          f"{'X':>5} {'d_dd1':>6} {'d_web':>6}")
    for ap in cfg["approaches"]:
        greens, C, g = [], None, None
        if ap["name"] in cfg["light_map"]:
            greens, C, g = load_green_windows(
                os.path.join(rec_dir, cfg["lights"]),
                cfg["light_map"][ap["name"]])
        r = analyze(ap, cars, greens, t_end)
        t = taunt(r, C, g, t_end)
        d = list(r["delays"].values())
        print(f"{r['name']:>6} {len(r['arrivals']):>4} {len(r['joins']):>4} "
              f"{len(r['departures']):>4} {t.get('lam_vph', 0):>5.0f} "
              f"{r['q'].mean():>5.1f} {r['q'].max():>4} "
              f"{np.median(d) if d else 0:>5.0f} "
              f"{np.mean(d) if d else 0:>6.1f} "
              f"{t['sat_vph'] if t.get('sat_vph') else 0:>5.0f} "
              f"{t['X'] if t.get('X') else 0:>5.2f} "
              f"{t['d_dd1'] if t.get('d_dd1') else 0:>6.1f} "
              f"{t['d_webster'] if t.get('d_webster') else 0:>6.1f}")
        if t.get("growth_vps"):
            print(f"        X>=1: oversaturated, queue grows "
                  f"{t['growth_vps'] * 3600:.0f} veh/h beyond capacity "
                  f"(truncated exits: {r['truncated']})")
        ct = cycle_taunt(r, greens, t_end)
        if ct:
            print(f"        per cycle (n={ct['n']}): q_max meas "
                  f"{ct['q_max_meas']:.1f} vs λr {ct['q_max_theory']:.1f}; "
                  f"cycle delay meas {ct['area_meas']:.0f} veh·s"
                  + (f" vs D/D/1 {ct['area_theory']:.0f} veh·s"
                     if ct["area_theory"] is not None
                     else " (D/D/1: never clears)"))
        out = plot(rec_dir, r, greens, t_end)
        print(f"        QA: {out}")


if __name__ == "__main__":
    main()
