"""Queue-story video: raw footage -> schematic world view -> schematic with
live counters and per-car wait timers.

Three acts:
  0 .. RAW_END s        source frames, queue lines/polys glued to the road
                        (world->image via per-frame homographies, so camera
                        motion can't shake them off), cars as state dots
  RAW_END .. +FADE s    crossfade to the schematic
  .. end                schematic world plane (dark map, lanes as polys,
                        cars as rotated bboxes with heading triangles,
                        signal heads at the stop bars) + per-approach
                        counters (arrived / queued / total wait) + a
                        ticking timer over every stopped car

The schematic lives entirely in the reference plane: camera zoom/drift in
the source footage vanishes, and everything is driven by the same
intersection.json + queue.analyze_approach numbers as `tva queue`.
"""
import os

import cv2
import numpy as np

from .queue import analyze_approach
from .stabilize import apply_h

RAW_END_S = 6.0
FADE_S = 2.0
SIGNAL_GAP_S = 6.0    # stop-bar crossings closer than this = one green window
SIGNAL_TAIL_S = 2.0   # window stays green this long past the last crossing
BG = (18, 18, 22)
ROAD = (46, 46, 54)
IN_LINE = (80, 220, 80)      # BGR, greenish
OUT_LINE = (60, 60, 230)     # reddish
QUEUE_POLY = (200, 160, 40)
STATE_BGR = {"g": (90, 220, 90), "a": (60, 200, 240), "r": (80, 80, 240),
             "p": (70, 70, 70)}
HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX


class Scene:
    """All per-frame lookups the renderer needs, precomputed once."""

    def __init__(self, ws):
        meta = ws.meta
        self.fps, self.n = meta["fps"], meta["n_frames"]
        self.w, self.h = meta["width"], meta["height"]
        self.src = meta["src"]
        hom = ws.load("homographies.json")
        self.Hinv = [np.linalg.inv(np.asarray(H)) for H in hom["H"]]
        wt = ws.load("world_tracks.json")
        self.v_stop, self.v_move = wt["v_stop"], wt["v_move"]
        self.car_len = wt["car_len_px"]
        self.xing = ws.load("intersection.json")
        tracks = wt["tracks"]
        self.approaches = [analyze_approach(ap, tracks, self.n, self.fps)
                           for ap in self.xing["approaches"]]
        # per-track detection boxes (image space, for the raw phase) and
        # world-space sizes + headings (for bbox-style schematic)
        raw = ws.load("tracks.json")
        self.bbox = {tr["id"]: {o[0]: (o[1], o[2], o[3], o[4])
                                for o in tr["obs"]}
                     for tr in raw["tracks"]}
        self.size = {}
        for tr in raw["tracks"]:
            L = float(np.median([max(o[3], o[4]) for o in tr["obs"]]))
            W = float(np.median([min(o[3], o[4]) for o in tr["obs"]]))
            self.size[tr["id"]] = (L, W)
        self.heading = {}
        for tr in tracks:
            ang = np.arctan2(np.gradient(tr["y"]), np.gradient(tr["x"]))
            filled, last = [], 0.0
            for a, sp in zip(ang, tr["speed"]):
                if sp > self.v_stop:
                    last = float(a)
                filled.append(last)
            self.heading[tr["id"]] = dict(zip(tr["frames"], filled))
        # frame -> list of (x, y, state) across all tracks
        from .queue import in_poly
        polys = [ap["queue_poly"] for ap in self.xing["approaches"]]
        self.dots = {}
        for tr in tracks:
            stopped_runs = [(a, b) for s, a, b in tr["runs"] if s]
            for i, f in enumerate(tr["frames"]):
                if f >= self.n:
                    continue
                x, y = tr["x"][i], tr["y"][i]
                stopped = any(a <= f <= b for a, b in stopped_runs)
                queued = stopped and any(in_poly(p, x, y) for p in polys)
                if queued or (stopped and not tr.get("static")):
                    st = "r"
                elif tr.get("static"):
                    st = "p"          # parked / never moved: dim gray
                else:
                    st = "a" if tr["speed"][i] < self.v_move else "g"
                self.dots.setdefault(f, []).append((x, y, st, tr["id"]))
        # wait runs: tid -> (t_start, t_end) list, and tid -> frame -> (x,y)
        self.waits = []
        for ap in self.approaches:
            for tid, t0, dur in ap["waits"]:
                self.waits.append((tid, t0, t0 + dur))
        self.pos = {}
        for tr in tracks:
            self.pos[tr["id"]] = dict(zip(tr["frames"],
                                          zip(tr["x"], tr["y"])))
        # per-approach counter series: arrivals times, queue per frame,
        # cumulative wait veh·s per frame
        self.counters = []
        for ap in self.approaches:
            arr_t = np.array(sorted(t for t, _ in ap["arrivals"]))
            qlen = np.array(ap["queue_len"] + [0] *
                            (self.n - len(ap["queue_len"])))[:self.n]
            self.counters.append(dict(
                name=ap["name"], arr_t=arr_t, qlen=qlen,
                wait=np.cumsum(qlen) / self.fps))
        # signal state per frame: explicit phases if the annotation has
        # them ({"signal": {"phases": [[t0, t1, "green"|"red"], ...]}}),
        # else inferred — stop-bar flow within the trailing window = green
        self.signal = {}
        for ap, res in zip(self.xing["approaches"], self.approaches):
            phases = (ap.get("signal") or {}).get("phases")
            green = np.zeros(self.n, bool)
            if phases:
                for t0, t1, st in phases:
                    if st == "green":
                        green[int(t0 * self.fps):int(t1 * self.fps)] = True
            else:
                # infer green windows: maximal runs of stop-bar crossings
                # with gaps < GAP_S, extended SIGNAL_TAIL_S past the last
                # crossing (a lone crossing still flashes green briefly —
                # someone did enter the junction)
                dep = np.sort([t for t, _ in res["departures"]])
                if len(dep):
                    starts = [0] + [i for i in range(1, len(dep))
                                    if dep[i] - dep[i - 1] > SIGNAL_GAP_S]
                    for si, s in enumerate(starts):
                        e = (starts[si + 1] - 1
                             if si + 1 < len(starts) else len(dep) - 1)
                        f0 = int(dep[s] * self.fps)
                        f1 = min(int((dep[e] + SIGNAL_TAIL_S) * self.fps) + 1,
                                 self.n)
                        green[f0:f1] = True
            self.signal[ap["name"]] = green

    def active_waits(self, t):
        for tid, t0, t1 in self.waits:
            if t0 <= t < t1:
                yield tid, t - t0

    def state_counts(self, f):
        return [(ap["name"], int(np.searchsorted(ap["arr_t"], f / self.fps)),
                 int(ap["qlen"][f]), ap["wait"][f])
                for ap in self.counters]


def schematic_canvas(scn):
    """Fixed world->canvas affine fitted to the queue polys (+ margin)."""
    pts = np.array([p for ap in scn.xing["approaches"]
                    for p in ap["queue_poly"]], float)
    lo, hi = pts.min(0), pts.max(0)
    span = np.maximum(hi - lo, 1)
    margin = 0.35 * span
    lo, hi = lo - margin, hi + margin
    span = hi - lo
    scale = min(scn.w / span[0], scn.h / span[1])
    off = (np.array([scn.w, scn.h]) - span * scale) / 2

    def to_px(p):
        q = (np.asarray(p, float) - lo) * scale + off
        return q.astype(int)

    return to_px, scale


def draw_schematic(scn, f, to_px, scale):
    img = np.full((scn.h, scn.w, 3), BG, np.uint8)
    t = f / scn.fps
    diag = float(np.hypot(scn.w, scn.h)) / scale      # world units on canvas
    for ap in scn.xing["approaches"]:
        poly = np.array(ap["queue_poly"], float)
        c = poly.mean(0)
        d = np.asarray(ap["dir"], float)
        d /= np.hypot(*d)
        n = np.array([-d[1], d[0]])
        offs = (poly - c) @ n
        lo, hi = offs.min(), offs.max()
        band = np.array([c - d * diag + n * lo, c + d * diag + n * lo,
                         c + d * diag + n * hi, c - d * diag + n * hi])
        cv2.fillPoly(img, [np.array([to_px(p) for p in band], np.int32)],
                     ROAD)
    for ap in scn.xing["approaches"]:
        poly = np.array([to_px(p) for p in ap["queue_poly"]], np.int32)
        cv2.polylines(img, [poly], True, QUEUE_POLY, 2)
        for key, col in (("in_line", IN_LINE), ("out_line", OUT_LINE)):
            a, b = to_px(ap[key][0]), to_px(ap[key][1])
            cv2.line(img, tuple(a), tuple(b), col, 3)
    for x, y, st, tid in scn.dots.get(f, []):
        L, W = scn.size.get(tid, (scn.car_len, 0.55 * scn.car_len))
        rad = scn.heading.get(tid, {}).get(f, 0.0)
        Lpx, Wpx = max(L * scale, 6), max(W * scale, 4)
        rect = (tuple(to_px((x, y)).astype(float)), (Lpx, Wpx),
                np.degrees(rad))
        box = cv2.boxPoints(rect).astype(np.int32)
        col = STATE_BGR[st]
        cv2.fillConvexPoly(img, box, col)
        # heading triangle on the front edge (SinD style): the box's L axis
        # points along (cos, sin)(rad), so the front edge sits at +L/2
        u = np.array([np.cos(rad), np.sin(rad)])
        v = np.array([-u[1], u[0]])
        c = np.array(rect[0])
        base = c + u * (Lpx / 2)
        tri = np.array([base + v * Wpx * 0.42, base - v * Wpx * 0.42,
                        base + u * max(5.0, Lpx * 0.35)], np.int32)
        cv2.fillConvexPoly(img, tri, tuple(int(x * 0.55) for x in col))
    # timers over stopped cars: greedily stack labels upward so packed
    # queues don't pile every timer on the same spot
    r = max(4, int(0.35 * scn.car_len * scale))
    placed = []
    for tid, waited in sorted(scn.active_waits(t), key=lambda w: -w[1]):
        p = next((q for df in range(-2, 3)
                  if (q := self_pos(scn, tid, f + df)) is not None), None)
        if p is None:
            continue
        px = to_px(p)
        txt = f"{waited:4.1f}s"
        (tw, th), _ = cv2.getTextSize(txt, HUD_FONT, 0.55, 1)
        for k in range(8):
            org = (px[0] - 18, px[1] - r - 6 - k * (th + 6))
            rect = (org[0] - 2, org[1] - th - 2, org[0] + tw + 2,
                    org[1] + 4)
            if not any(rect[0] < q[2] and rect[2] > q[0] and
                       rect[1] < q[3] and rect[3] > q[1] for q in placed):
                placed.append(rect)
                cv2.putText(img, txt, org, HUD_FONT, 0.55,
                            (240, 240, 240), 1, cv2.LINE_AA)
                break
    draw_signals(img, scn, f, to_px)
    draw_hud(img, scn, f)
    return img


def draw_signals(img, scn, f, to_px):
    """3-lamp signal head beside each stop bar (red top / amber / green
    bottom); the lit lamp follows scn.signal for this frame."""
    for ap in scn.xing["approaches"]:
        green = scn.signal[ap["name"]][f]
        (x1, y1), (x2, y2) = ap["out_line"]
        mid_w = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        mid = to_px(mid_w).astype(float)
        d = np.array([x2 - x1, y2 - y1], float)
        d /= np.hypot(*d) or 1e-9
        n = np.array([-d[1], d[0]])
        # sit on the side away from the queue polygon
        c = np.array(ap["queue_poly"], float).mean(0)
        if (mid_w - c) @ n < 0:
            n = -n
        base = mid + n * 34
        wpx, hpx = 16, 40
        tl = (int(base[0] - wpx / 2), int(base[1] - hpx / 2))
        br = (int(base[0] + wpx / 2), int(base[1] + hpx / 2))
        cv2.rectangle(img, tl, br, (28, 28, 32), -1)
        cv2.rectangle(img, tl, br, (90, 90, 90), 1)
        for frac, col, lit in ((0.22, (60, 60, 230), not green),
                               (0.50, (60, 200, 240), False),
                               (0.78, (80, 220, 80), green)):
            p = (int(base[0]), int(tl[1] + frac * hpx))
            cv2.circle(img, p, 5, col if lit else (48, 48, 52), -1)


def self_pos(scn, tid, f):
    return scn.pos.get(tid, {}).get(f)


def draw_hud(img, scn, f):
    counts = {name: (arrived, queued, wait)
              for name, arrived, queued, wait in scn.state_counts(f)}
    order = [n for n in ("west", "east", "south", "north") if n in counts]
    for i, name in enumerate(order):
        arrived, queued, wait = counts[name]
        txt = (f"{name}  arrived {arrived}  queue {queued}  "
               f"wait {wait:.0f} veh-s")
        right = i % 2 == 1
        bottom = i >= 2
        (tw, th), _ = cv2.getTextSize(txt, HUD_FONT, 0.7, 1)
        x = scn.w - tw - 18 if right else 18
        y = scn.h - 18 if bottom else 12 + th
        cv2.rectangle(img, (x - 8, y - th - 8), (x + tw + 8, y + 8),
                      (0, 0, 0), -1)
        cv2.putText(img, txt, (x, y), HUD_FONT, 0.7, (230, 230, 230), 1,
                    cv2.LINE_AA)


def draw_overlay(frame, scn, f):
    """Raw-phase: queue lines/polys + state dots glued to the road."""
    layer = frame.astype(np.float32)
    ann = np.zeros_like(frame)
    Hinv_i = scn.Hinv[min(f, len(scn.Hinv) - 1)]
    for ap in scn.xing["approaches"]:
        poly = np.array(apply_h(Hinv_i, ap["queue_poly"]), np.int32)
        cv2.polylines(ann, [poly], True, QUEUE_POLY, 2)
        for key, col in (("in_line", IN_LINE), ("out_line", OUT_LINE)):
            pts = apply_h(Hinv_i, ap[key]).astype(np.int32)
            cv2.line(ann, tuple(pts[0]), tuple(pts[1]), col, 3)
    m = ann.any(axis=2)
    layer[m] = 0.45 * layer[m] + 0.55 * ann[m].astype(np.float32)
    out = layer.astype(np.uint8)
    for x, y, st, tid in scn.dots.get(f, []):
        bb = scn.bbox.get(tid, {}).get(f)
        if bb is not None:      # draw the actual detection box
            cx, cy, bw, bh = bb
            p1 = (int(cx - bw / 2), int(cy - bh / 2))
            p2 = (int(cx + bw / 2), int(cy + bh / 2))
            cv2.rectangle(out, p1, p2, STATE_BGR[st], 2)
        else:
            p = apply_h(Hinv_i, [(x, y)])[0].astype(int)
            if 0 <= p[0] < scn.w and 0 <= p[1] < scn.h:
                cv2.circle(out, tuple(p), 7, STATE_BGR[st], -1)
    return out


def run(ws, out=None):
    scn = Scene(ws)
    out = out or ws.path("xing-queue.mp4")
    to_px, scale = schematic_canvas(scn)
    cap = cv2.VideoCapture(scn.src)
    wr = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), scn.fps,
                         (scn.w, scn.h))
    raw_end = RAW_END_S * scn.fps
    fade_end = raw_end + FADE_S * scn.fps
    f = 0
    while f < scn.n:
        ok, frame = cap.read()
        if not ok:
            break
        if f < raw_end:
            img = draw_overlay(frame, scn, f)
        else:
            sch = draw_schematic(scn, f, to_px, scale)
            if f < fade_end:
                a = (f - raw_end) / (fade_end - raw_end)
                img = cv2.addWeighted(draw_overlay(frame, scn, f), 1 - a,
                                      sch, a, 0)
            else:
                img = sch
        wr.write(img)
        f += 1
        if f % 200 == 0:
            print(f"frame {f}/{scn.n}")
    cap.release()
    wr.release()
    print(f"wrote {out}")
