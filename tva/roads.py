"""Infer roads and lanes from world-plane vehicle tracks.

Roads: rasterize all track motion into a coarse flow grid (per-cell presence
count + net displacement direction), then flood-fill cells into directed
corridors — adjacent cells join a road when their flow directions agree, so
opposing carriageways split automatically. Lanes: within a road, signed
lateral offsets of every observation from the road centerline form a
multimodal distribution; smoothed-histogram peaks are lane centers.

Output roads.json: per road a centerline polyline (world px, ordered with
traffic), lane center offsets, and per-lane half-width.
"""
import math
from collections import defaultdict

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

CELL = 24              # world px per flow-grid cell
MIN_CELL_HITS = 3      # cells with fewer observations are noise
ANG_TOL = math.radians(55)  # max direction difference between joined cells
MIN_ROAD_CELLS = 40    # drop tiny fragments
N_CL_BINS = 36         # centerline refinement bins along the spine


def flow_grid(wt):
    grid = defaultdict(lambda: [0, 0.0, 0.0])  # cell -> [hits, sum dx, sum dy]
    cell_pts = defaultdict(list)               # cell -> [(track idx, obs idx)]
    for ti, tr in enumerate(wt["tracks"]):
        xs, ys = np.array(tr["x"]), np.array(tr["y"])
        for k in range(len(xs)):
            c = (int(xs[k] // CELL), int(ys[k] // CELL))
            g = grid[c]
            g[0] += 1
            if k + 1 < len(xs):
                g[1] += xs[k + 1] - xs[k]
                g[2] += ys[k + 1] - ys[k]
            cell_pts[c].append((ti, k))
    return grid, cell_pts


def cluster_cells(grid):
    """Flood-fill same-direction neighboring cells into road components."""
    ang = {}
    for c, (n, dx, dy) in grid.items():
        if n >= MIN_CELL_HITS and math.hypot(dx, dy) > 1.0:
            ang[c] = math.atan2(dy, dx)
    comp = {}
    comps = []
    for seed in ang:
        if seed in comp:
            continue
        cid = len(comps)
        stack, members = [seed], []
        comp[seed] = cid
        while stack:
            c = stack.pop()
            members.append(c)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nb = (c[0] + dx, c[1] + dy)
                    if nb in ang and nb not in comp:
                        d = abs(ang[nb] - ang[c])
                        if min(d, 2 * math.pi - d) < ANG_TOL:
                            comp[nb] = cid
                            stack.append(nb)
        comps.append(members)
    keep = [m for m in comps if len(m) >= MIN_ROAD_CELLS]
    keep.sort(key=len, reverse=True)
    return keep


def project_signed(cl_x, cl_y, px, py):
    """Station + SIGNED lateral offset (left of travel = negative) of points
    vs a centerline polyline."""
    cx, cy = np.asarray(cl_x, float), np.asarray(cl_y, float)
    px, py = np.asarray(px, float), np.asarray(py, float)
    seg = np.hypot(np.diff(cx), np.diff(cy))
    arc = np.concatenate([[0], np.cumsum(seg)])
    best_s = np.zeros(len(px))
    best_d = np.full(len(px), np.inf)
    best_sign = np.ones(len(px))
    for i in range(len(cx) - 1):
        ax, ay = cx[i], cy[i]
        bx, by = cx[i + 1] - ax, cy[i + 1] - ay
        L2 = bx * bx + by * by or 1e-9
        t = np.clip(((px - ax) * bx + (py - ay) * by) / L2, 0, 1)
        qx, qy = ax + t * bx, ay + t * by
        d = np.hypot(px - qx, py - qy)
        m = d < best_d
        cross = bx * (py - ay) - by * (px - ax)
        best_d[m] = d[m]
        best_s[m] = arc[i] + t[m] * seg[i]
        sgn = np.sign(cross[m])
        sgn[sgn == 0] = 1
        best_sign[m] = sgn
    return best_s, best_d * best_sign


def infer(ws):
    wt = ws.load("world_tracks.json")
    car_len = wt["car_len_px"]
    grid, cell_pts = flow_grid(wt)
    comps = cluster_cells(grid)
    print(f"{len(comps)} road components "
          f"({[len(c) for c in comps[:8]]}... cells)")

    roads = []
    for cid, members in enumerate(comps):
        mset = set(members)
        # observations belonging to this road
        obs = [(ti, k) for c in members for (ti, k) in cell_pts[c]]
        per_track = defaultdict(list)
        for ti, k in obs:
            per_track[ti].append(k)
        # spine: track covering the most distance ON this road (obs count
        # would pick a parked car in gridlock)
        def on_road_path(ti):
            ks = sorted(per_track[ti])
            xs = np.array([wt["tracks"][ti]["x"][k] for k in ks])
            ys = np.array([wt["tracks"][ti]["y"][k] for k in ks])
            return float(np.hypot(np.diff(xs), np.diff(ys)).sum())

        spine_ti = max(per_track, key=on_road_path)
        path_len = on_road_path(spine_ti)
        if path_len < 4 * car_len:
            continue
        sp = wt["tracks"][spine_ti]
        sp_x, sp_y = np.array(sp["x"]), np.array(sp["y"])

        px = np.array([wt["tracks"][ti]["x"][k] for ti, k in obs])
        py = np.array([wt["tracks"][ti]["y"][k] for ti, k in obs])
        s, lat = project_signed(sp_x, sp_y, px, py)
        # keep points laterally near the spine (same carriageway)
        near = np.abs(lat) < 3.5 * car_len
        if near.sum() < 200:
            continue
        s, lat, px, py = s[near], lat[near], px[near], py[near]

        # refine centerline: spine shifted to the mean lateral per station bin
        bins = np.linspace(0, s.max(), N_CL_BINS + 1)
        cl_pts = []
        for b0, b1 in zip(bins, bins[1:]):
            m = (s >= b0) & (s < b1)
            if m.sum() < 10:
                continue
            cl_pts.append((float(np.median(px[m])), float(np.median(py[m])),
                           (b0 + b1) / 2))
        if len(cl_pts) < 4:
            continue
        cl_x = gaussian_filter1d([p[0] for p in cl_pts], 3.0)
        cl_y = gaussian_filter1d([p[1] for p in cl_pts], 3.0)

        # sanity gate: a road centerline shouldn't wind more than a full
        # turn — that's a parking lot / mixed-direction blob, not a road
        dx, dy = np.diff(cl_x), np.diff(cl_y)
        heading = np.unwrap(np.arctan2(dy, dx))
        if np.abs(np.diff(heading)).sum() > 2 * math.pi:
            print(f"  (skipping wandering component, {len(members)} cells)")
            continue

        # lanes: peaks of the signed-lateral histogram around the centerline
        s2, lat2 = project_signed(cl_x, cl_y, px, py)
        bw = 4.0
        lo, hi = np.percentile(lat2, [2, 98])
        edges = np.arange(lo, hi + bw, bw)
        hist, _ = np.histogram(lat2, bins=edges)
        histS = gaussian_filter1d(hist.astype(float), 1.6)
        min_sep = max(2, int(0.9 * car_len * 0.45 / bw))  # ~lane width apart
        pk, _ = find_peaks(histS, distance=min_sep,
                           prominence=0.12 * histS.max())
        lane_offsets = [float(edges[p] + bw / 2) for p in pk]
        if not lane_offsets:
            lane_offsets = [0.0]
        gaps = np.diff(lane_offsets)
        lane_hw = float(np.median(gaps) / 2) if len(gaps) else car_len * 0.35

        roads.append({
            "id": len(roads), "cells": len(members),
            "n_obs": int(near.sum()),
            "x": [round(float(v), 1) for v in cl_x],
            "y": [round(float(v), 1) for v in cl_y],
            "lanes": [round(v, 1) for v in sorted(lane_offsets)],
            "lane_halfwidth": round(lane_hw, 1),
        })
        print(f"road {roads[-1]['id']}: {len(members)} cells, "
              f"{near.sum()} obs, {len(lane_offsets)} lanes "
              f"(offsets {[round(v) for v in sorted(lane_offsets)]})")
    ws.save("roads.json", {"cell_px": CELL, "roads": roads})
