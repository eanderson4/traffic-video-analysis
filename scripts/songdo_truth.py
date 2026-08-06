"""Ground-truth stats from a Songdo Traffic session CSV, for validating
tva pipeline output against EPFL/KAIST trajectories.

Usage: python3 scripts/songdo_truth.py <session.csv> [--drone N]
"""
import argparse

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--drone", type=int, default=None,
                    help="restrict to one Drone_ID (matches one video's "
                         "footprint)")
    ap.add_argument("--stop-kmh", type=float, default=2.0,
                    help="speed below which a vehicle counts as stopped")
    ap.add_argument("--max-t", type=float, default=None,
                    help="clip to the first N seconds (sample videos are "
                         "the first 60s of a session)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    t = pd.to_timedelta(df["Local_Time"]).dt.total_seconds()
    df["t"] = t - t.min()
    if args.max_t is not None:
        df = df[df["t"] <= args.max_t]
    if args.drone is not None:
        df = df[df["Drone_ID"] == args.drone]
    print(f"rows {len(df)}, span {df['t'].max():.1f}s, "
          f"drones {sorted(df['Drone_ID'].unique())}")

    vis = df[df["Visibility"] == 1]
    u = vis.groupby("Vehicle_ID")
    print(f"unique vehicles (visible): {u.ngroups}")
    cls = u["Vehicle_Class"].first().value_counts().sort_index()
    names = {0: "car/van", 1: "bus", 2: "truck", 3: "motorcycle"}
    for c, n in cls.items():
        print(f"  class {c} ({names.get(c, '?')}): {n}")

    # per road-section throughput (vehicles seen per section)
    sec = u["Road_Section"].first().value_counts()
    print("vehicles per road section:", dict(sec))

    # stopped-vehicle time series, 1s bins
    stopped = vis[vis["Vehicle_Speed"] < args.stop_kmh]
    bins = stopped.groupby(stopped["t"].astype(int))["Vehicle_ID"] \
                  .nunique()
    full = pd.Series(0, index=range(int(df["t"].max()) + 1))
    full.update(bins)
    print("stopped vehicles per second (t:count):")
    print(" ".join(f"{i}:{v}" for i, v in full.items()))

    sp = vis["Vehicle_Speed"].dropna()
    print(f"speed km/h: median {sp.median():.1f} "
          f"p10 {sp.quantile(0.1):.1f} p90 {sp.quantile(0.9):.1f}")
    # per-vehicle wait: total stopped time while visible
    dt = df.groupby("Vehicle_ID")["t"].diff().median()
    waits = (vis[vis["Vehicle_Speed"] < args.stop_kmh]
             .groupby("Vehicle_ID").size() * dt)
    print(f"wait veh-s (stopped time, visible): {waits.sum():.0f} "
          f"across {(waits > 1).sum()} vehicles, "
          f"mean {waits[waits > 1].mean():.1f}s "
          f"max {waits.max():.1f}s")


if __name__ == "__main__":
    main()
