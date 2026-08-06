"""Compare tva pipeline output against Songdo ground truth for one clip.

Reads a tva work dir (world_tracks.json) + the matching Songdo session CSV
(--drone = the video's Drone_ID, --max-t = clip length) and diffs:
- unique vehicle counts
- stopped-vehicle time series (1s bins): ours vs truth, correlation
- total stopped vehicle-seconds

Usage: python3 scripts/songdo_compare.py --work testdata/songdo-a \
    --truth testdata/datasets/songdo/truth-a/2022-10-07_A_PM5.csv \
    --drone 1 --max-t 60
"""
import argparse
import json

import numpy as np
import pandas as pd


def ours_stopped_series(wt, fps, n_frames):
    """Stopped vehicles per second from tva world_tracks stopped runs."""
    dur = int(n_frames / fps) + 1
    out = np.zeros(dur, int)
    for tr in wt["tracks"]:
        for stopped, a, b in tr["runs"]:
            if not stopped:
                continue
            for s in range(int(a / fps), min(int(b / fps) + 1, dur)):
                out[s] += 1
    return out


def truth_stopped_series(csv, drone, max_t, stop_kmh=2.0):
    df = pd.read_csv(csv)
    t = pd.to_timedelta(df["Local_Time"]).dt.total_seconds()
    df["t"] = t - t.min()
    df = df[(df["t"] <= max_t) & (df["Drone_ID"] == drone)]
    vis = df[df["Visibility"] == 1]
    n_unique = vis["Vehicle_ID"].nunique()
    stopped = vis[vis["Vehicle_Speed"] < stop_kmh]
    bins = stopped.groupby(stopped["t"].astype(int))["Vehicle_ID"].nunique()
    out = np.zeros(int(max_t) + 1, int)
    for k, v in bins.items():
        out[int(k)] = v
    return out, n_unique, len(stopped) / 29.97


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--drone", type=int, required=True)
    ap.add_argument("--max-t", type=float, required=True)
    args = ap.parse_args()

    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from tva.workspace import Workspace
    ws = Workspace(args.work)
    meta = ws.meta
    wt = json.loads(open(ws.path("world_tracks.json")).read())
    fps, n = meta["fps"], meta["n_frames"]

    ours = ours_stopped_series(wt, fps, n)
    truth, n_truth, wait_s_truth = truth_stopped_series(
        args.truth, args.drone, args.max_t)
    k = min(len(ours), len(truth))
    ours, truth = ours[:k], truth[:k]

    n_ours = len(wt["tracks"])
    print(f"unique tracks:   ours {n_ours}   truth {n_truth}")
    print(f"stopped series ({k}s):")
    print("  t: " + " ".join(f"{i:>3d}" for i in range(k)))
    print("  o: " + " ".join(f"{v:>3d}" for v in ours))
    print("  T: " + " ".join(f"{v:>3d}" for v in truth))
    r = np.corrcoef(ours, truth)[0, 1]
    mae = np.abs(ours.astype(float) - truth).mean()
    print(f"correlation {r:.3f}   MAE {mae:.1f} vehicles   "
          f"mean ours {ours.mean():.1f} vs truth {truth.mean():.1f}")


if __name__ == "__main__":
    main()
