"""Intersection queue analysis: per-approach arrival rates, queue length,
stopped wait, and a queueing-theory scoreboard (measured vs M/D/1, M/M/1).

Inputs (workdir):
- world_tracks.json (`tva world`): smoothed reference-plane paths, speeds,
  moving/stopped runs.
- intersection.json (hand-authored, reference-plane coordinates — with a
  static camera these are frame-0 pixels):

    {"approaches": [
      {"name": "north",
       "dir": [0, 1],                  travel direction toward the junction
       "in_line": [[x1,y1],[x2,y2]],   upstream counting line (arrivals)
       "out_line": [[x1,y1],[x2,y2]],  stop-bar line (departures/discharge)
       "queue_poly": [[x,y], ...]},    stopped cars inside here are queued
      ...]}

Definitions:
- arrival: smoothed path crosses in_line in the travel direction.
- departure: crosses out_line in the travel direction (discharge).
- queued at frame f: inside queue_poly and in a stopped run (world_tracks
  hysteresis runs, same source the wave analysis uses).
- wait: contiguous queued samples per car, summed -> vehicle-seconds.

Theory scoreboard per approach: lambda (arrival rate, veh/s), mu (observed
discharge rate while a queue exists, veh/s), rho=lambda/mu, then
M/D/1 mean queue Lq=rho^2/(2(1-rho)) and M/M/1 Lq=rho^2/(1-rho) against the
measured mean queue. Undersaturated only — rho>=1 means the deterministic
models diverge and we say so instead of printing a silly number.

Outputs: queue.json + qa/xing-annotation.png (lines/polys over frame 0 with
all tracks) + qa/xing-<name>.png (Newell cumulative curves + queue timeline).
"""
import json
import math
import os

import cv2
import numpy as np

REFRACTORY_S = 2.0      # one crossing event per track per line within this
MIN_DEPARTURES_FOR_MU = 3


def _signed_dist(pts, p0, n):
    return (pts - np.asarray(p0)) @ n


def _line_normal(p0, p1, direction):
    """Unit normal to the line, oriented so travel along `direction` moves
    from the negative side to the positive side."""
    d = np.asarray(p1, float) - np.asarray(p0, float)
    n = np.array([-d[1], d[0]])
    n /= np.hypot(*n) or 1e-9
    if n @ np.asarray(direction, float) < 0:
        n = -n
    return n


def line_events(tr, p0, p1, direction, fps):
    """(time, track_id) crossings of the line in the travel direction."""
    n = _line_normal(p0, p1, direction)
    t = np.asarray(tr["frames"]) / fps
    pts = np.column_stack([tr["x"], tr["y"]])
    sd = _signed_dist(pts, p0, n)
    events = []
    last_t = -1e9
    for i in range(len(sd) - 1):
        if sd[i] < 0 <= sd[i + 1] and t[i + 1] - last_t > REFRACTORY_S:
            frac = -sd[i] / (sd[i + 1] - sd[i] or 1e-9)
            events.append((float(t[i] + frac * (t[i + 1] - t[i])), tr["id"]))
            last_t = float(t[i + 1])
    return events


def in_poly(poly, x, y):
    """Ray-casting point-in-polygon (poly is [[x,y], ...])."""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def queued_samples(tr, poly, fps):
    """Per-sample queue membership -> (frames[], queued bool[]) using the
    track's stopped runs + the queue polygon."""
    frames = tr["frames"]
    runs = [(a, b) for stopped, a, b in tr["runs"] if stopped]
    out = np.zeros(len(frames), dtype=bool)
    for i, f in enumerate(frames):
        if any(a <= f <= b for a, b in runs):
            if in_poly(poly, tr["x"][i], tr["y"][i]):
                out[i] = True
    return np.asarray(frames), out


def wait_runs(frames, queued, fps):
    """Contiguous queued stretches -> [(t_start, duration_s)]."""
    out = []
    start = None
    prev = None
    for f, q in zip(frames, queued):
        if q and start is None:
            start = f
        elif not q and start is not None:
            out.append((start / fps, (prev - start + 1) / fps))
            start = None
        prev = f
    if start is not None:
        out.append((start / fps, (prev - start + 1) / fps))
    return out


def analyze_approach(ap, tracks, n_frames, fps):
    name = ap["name"]
    arrivals, departures = [], []
    q = np.zeros(n_frames, dtype=int)
    waits = []                      # (track_id, t_start, duration_s)
    queued_ids = set()
    for tr in tracks:
        arrivals += line_events(tr, *ap["in_line"], ap["dir"], fps)
        departures += line_events(tr, *ap["out_line"], ap["dir"], fps)
        frames, queued = queued_samples(tr, ap["queue_poly"], fps)
        for f, qf in zip(frames, queued):
            if qf and f < n_frames:
                q[f] += 1
        for t0, dur in wait_runs(frames, queued, fps):
            waits.append((tr["id"], t0, dur))
            queued_ids.add(tr["id"])
    arrivals.sort()
    departures.sort()
    duration = n_frames / fps
    lam = len(arrivals) / duration
    queued_mask = q > 0
    disch_t = float(queued_mask.sum()) / fps
    dep_during = [d for d in departures
                  if queued_mask[min(int(d[0] * fps), n_frames - 1)]]
    mu = (len(dep_during) / disch_t
          if len(dep_during) >= MIN_DEPARTURES_FOR_MU and disch_t > 0 else None)
    total_wait = sum(d for _, _, d in waits)
    res = {
        "name": name,
        "n_arrivals": len(arrivals),
        "n_departures": len(departures),
        "arrivals": [[round(t, 2), tid] for t, tid in arrivals],
        "departures": [[round(t, 2), tid] for t, tid in departures],
        "lambda_vps": round(lam, 4),
        "lambda_vph": round(lam * 3600, 1),
        "mu_vps": None if mu is None else round(mu, 4),
        "queue_len": q.tolist(),
        "mean_queue": round(float(q.mean()), 2),
        "max_queue": int(q.max()),
        "n_cars_queued": len(queued_ids),
        "total_wait_veh_s": round(total_wait, 1),
        "mean_wait_s": (round(total_wait / len(queued_ids), 1)
                        if queued_ids else 0.0),
        "waits": [[tid, round(t0, 2), round(d, 1)] for tid, t0, d in waits],
    }
    # queueing-theory scoreboard
    if mu and lam > 0:
        rho = lam / mu
        res["rho"] = round(rho, 3)
        if rho < 1:
            res["md1_mean_queue"] = round(rho ** 2 / (2 * (1 - rho)), 2)
            res["mm1_mean_queue"] = round(rho ** 2 / (1 - rho), 2)
            res["md1_mean_wait_s"] = round(
                res["md1_mean_queue"] / lam, 1)
        else:
            res["md1_mean_queue"] = None
            res["mm1_mean_queue"] = None
            res["md1_mean_wait_s"] = None
            res["note"] = ("rho >= 1: oversaturated, steady-state models "
                           "diverge (queue grows without bound)")
    return res


def plot_annotation(ws, approaches, tracks):
    """Frame 0 + every track path + the hand-drawn lines/polys: the check
    that the annotation actually sits on the roads."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    meta = ws.meta
    cap = cv2.VideoCapture(meta["src"])
    ok, frame = cap.read()
    cap.release()
    fig, ax = plt.subplots(figsize=(16, 9))
    if ok:
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    for tr in tracks:
        ax.plot(tr["x"], tr["y"], "-", lw=0.4, color="cyan", alpha=0.35)
    for ap in approaches:
        for key, color in (("in_line", "lime"), ("out_line", "red")):
            (x1, y1), (x2, y2) = ap[key]
            ax.plot([x1, x2], [y1, y2], color=color, lw=2)
        px, py = zip(*(ap["queue_poly"] + [ap["queue_poly"][0]]))
        ax.plot(px, py, color="yellow", lw=1.5)
        cx = float(np.mean([p[0] for p in ap["queue_poly"]]))
        cy = float(np.mean([p[1] for p in ap["queue_poly"]]))
        ax.text(cx, cy, ap["name"], color="yellow", fontsize=14,
                ha="center", va="center")
    ax.set_xlim(0, meta["width"])
    ax.set_ylim(meta["height"], 0)
    ax.set_title("intersection annotation over frame 0 "
                 "(green=in, red=out, yellow=queue poly)")
    fig.tight_layout()
    out = os.path.join(ws.qa, "xing-annotation.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"QA: {out}")


def plot_approach(ws, res, n_frames, fps):
    """Newell cumulative curves + queue-length timeline for one approach."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t_grid = np.arange(n_frames) / fps
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    at = [t for t, _ in res["arrivals"]]
    dt = [t for t, _ in res["departures"]]
    ax1.step(at, np.arange(1, len(at) + 1), where="post", color="tab:blue",
             label=f"arrivals A(t)  λ={res['lambda_vph']:.0f} veh/h")
    ax1.step(dt, np.arange(1, len(dt) + 1), where="post", color="tab:orange",
             label="departures D(t)")
    ax1.set_ylabel("cumulative vehicles")
    ax1.legend(loc="upper left")
    ax1.set_title(f"{res['name']}: mean queue {res['mean_queue']} "
                  f"(max {res['max_queue']}), "
                  f"total wait {res['total_wait_veh_s']:.0f} veh·s "
                  f"across {res['n_cars_queued']} cars")
    ax2.fill_between(t_grid, res["queue_len"], step="post",
                     color="tab:red", alpha=0.6)
    ax2.set_xlabel("t (s)")
    ax2.set_ylabel("queue length (veh)")
    ax2.set_ylim(bottom=0)
    fig.tight_layout()
    out = os.path.join(ws.qa, f"xing-{res['name']}.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def run(ws):
    if not os.path.exists(ws.path("intersection.json")):
        raise SystemExit(
            f"need {ws.path('intersection.json')} — see tva/queue.py "
            "docstring for the schema (reference-plane coords)")
    xing = ws.load("intersection.json")
    meta = ws.meta
    fps, n_frames = meta["fps"], meta["n_frames"]
    data = ws.load("world_tracks.json")
    # NOTE: no static-track filter here — a car queued at a red for most of
    # the clip IS "static" by the kinematics definition (no net motion), and
    # those are exactly the cars we want. Statics never cross counting lines,
    # and off-road parked cars are excluded by drawing queue polys on the
    # road only.
    tracks = data["tracks"]
    plot_annotation(ws, xing["approaches"], tracks)

    results = [analyze_approach(ap, tracks, n_frames, fps)
               for ap in xing["approaches"]]
    ws.save("queue.json", {"fps": fps, "n_frames": n_frames,
                           "duration_s": round(n_frames / fps, 1),
                           "approaches": results})

    hdr = (f"{'approach':<10} {'arr':>4} {'dep':>4} {'λ veh/h':>8} "
           f"{'μ veh/s':>8} {'ρ':>6} {'q̄':>5} {'qmax':>5} "
           f"{'wait veh·s':>10} {'M/D/1 q̄':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        mu = f"{r['mu_vps']:.3f}" if r["mu_vps"] is not None else "—"
        rho = f"{r['rho']:.2f}" if "rho" in r else "—"
        md1 = (f"{r['md1_mean_queue']:.2f}"
               if r.get("md1_mean_queue") is not None else "—")
        print(f"{r['name']:<10} {r['n_arrivals']:>4} {r['n_departures']:>4} "
              f"{r['lambda_vph']:>8.0f} {mu:>8} {rho:>6} "
              f"{r['mean_queue']:>5} {r['max_queue']:>5} "
              f"{r['total_wait_veh_s']:>10.0f} {md1:>8}")
        out = plot_approach(ws, r, n_frames, fps)
        print(f"  QA: {out}")
