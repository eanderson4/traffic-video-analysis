"""R1 metrics M1-M3 for the CURRENT homographies.json in a work dir.

Run `python3 -m tva world --work <dir>` first so world_tracks.json /
centerline.json are derived from the same homographies being scored.

M1  static-landmark residual (world px, refine.py machinery), median + p90,
    mid-clip (frames 100-350) vs last 2 s (frames 376-425).
M2  local Jacobian scale sqrt|det J| of H at the jam-road centroid per
    frame; max/median ratio (v1 blows up 10-14x at clip end; target < 2).
M3  median frame-px distance between inv(H) @ smoothed-world position and
    the detected bbox center, at probe frames 300/380/415/424
    (v1: 1.5/14/69/65 px; target <= 5 px at all four).

Usage: python3 scripts/r1_metrics.py --work testdata/hero --label v1
Appends a JSON record to <work>/qa/r1-metrics.jsonl and prints a table.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tva.workspace import Workspace                     # noqa: E402
from tva.stabilize import apply_h                       # noqa: E402
from tva.refine import collect_landmarks                # noqa: E402

MID = (100, 350)
LAST = (376, 425)
PROBES = (300, 380, 415, 424)


def m1_static_residual(ws, Hs):
    marks = collect_landmarks(ws)
    by_frame = {}
    for (wx, wy), pix in marks:
        for f, p in pix.items():
            if f < len(Hs):
                q = apply_h(Hs[f], [p])[0]
                by_frame.setdefault(f, []).append(
                    float(np.hypot(q[0] - wx, q[1] - wy)))
    out = {}
    for name, (a, b) in (("mid", MID), ("last2s", LAST)):
        r = [v for f, vs in by_frame.items() if a <= f <= b for v in vs]
        out[name] = {"median": float(np.median(r)) if r else None,
                     "p90": float(np.percentile(r, 90)) if r else None,
                     "n": len(r)}
    return out, len(marks)


def jam_centroid(ws):
    wt = ws.load("world_tracks.json")
    ev = wt["stop_events"]
    if ev:
        return float(np.median([e["x"] for e in ev])), \
               float(np.median([e["y"] for e in ev]))
    cl = ws.load("centerline.json")
    return float(np.median(cl["x"])), float(np.median(cl["y"]))


def m2_jacobian(ws, Hs, cw):
    """sqrt|det J| of frame->world H at the image point that maps to the
    jam centroid, per frame."""
    eps = 2.0
    scales = []
    for H in Hs:
        p = apply_h(np.linalg.inv(H), [cw])[0]
        q0 = apply_h(H, [p])[0]
        qx = apply_h(H, [(p[0] + eps, p[1])])[0]
        qy = apply_h(H, [(p[0], p[1] + eps)])[0]
        J = np.column_stack([(qx - q0) / eps, (qy - q0) / eps])
        scales.append(float(np.sqrt(abs(np.linalg.det(J)))))
    s = np.asarray(scales)
    return {"median": float(np.median(s)), "max": float(s.max()),
            "max_over_median": float(s.max() / np.median(s)),
            "argmax": int(s.argmax())}


def m3_dot_offset(ws, Hs):
    wt = ws.load("world_tracks.json")
    det = {}
    for tr in ws.load("tracks.json")["tracks"]:
        for f, cx, cy, *_ in tr["obs"]:
            det[(tr["id"], f)] = (cx, cy)
    out = {}
    for pf in PROBES:
        Hinv = np.linalg.inv(Hs[pf])
        d = []
        for tr in wt["tracks"]:
            if pf not in tr["frames"]:
                continue
            i = tr["frames"].index(pf)
            c = det.get((tr["id"], pf))
            if c is None:
                continue
            p = apply_h(Hinv, [(tr["x"][i], tr["y"][i])])[0]
            d.append(float(np.hypot(p[0] - c[0], p[1] - c[1])))
        out[str(pf)] = {"median": float(np.median(d)) if d else None,
                        "n": len(d)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()
    ws = Workspace(args.work)
    data = ws.load("homographies.json")
    Hs = [np.asarray(h) for h in data["H"]]

    m1, n_marks = m1_static_residual(ws, Hs)
    m2 = m2_jacobian(ws, Hs, jam_centroid(ws))
    m3 = m3_dot_offset(ws, Hs)

    rec = {"label": args.label, "method": data.get("method", "chained-v1"),
           "M1": m1, "M1_landmarks": n_marks, "M2": m2, "M3": m3}
    os.makedirs(ws.qa, exist_ok=True)
    with open(os.path.join(ws.qa, "r1-metrics.jsonl"), "a") as fh:
        fh.write(json.dumps(rec) + "\n")

    print(f"\n== {args.label} ({rec['method']}) ==")
    print(f"M1 mid    frames {MID}:  median {m1['mid']['median']:.2f}  "
          f"p90 {m1['mid']['p90']:.2f}  (n={m1['mid']['n']})")
    print(f"M1 last2s frames {LAST}: median {m1['last2s']['median']:.2f}  "
          f"p90 {m1['last2s']['p90']:.2f}  (n={m1['last2s']['n']})")
    print(f"M1 ratio last2s/mid (median): "
          f"{m1['last2s']['median'] / m1['mid']['median']:.2f}")
    print(f"M2 jac scale: median {m2['median']:.3f}  max {m2['max']:.3f}  "
          f"max/median {m2['max_over_median']:.2f}  (argmax f{m2['argmax']})")
    row = "  ".join(f"f{p}: {m3[str(p)]['median']:.1f}px(n={m3[str(p)]['n']})"
                    for p in PROBES)
    print(f"M3 dot offset: {row}")


if __name__ == "__main__":
    main()
