"""Queue-story video: annotated footage throughout — after a short crisp
intro the background blurs/darkens so the annotation pops, and everything
(rotated bboxes with heading triangles, movement-lane ribbons extracted
from the tracks themselves, crossing-fitted queue lines/polys, signal
heads, wait timers whose final seconds fly into the HUD bucket when the
wait ends, per-approach dashboard) stays glued to the road via per-frame
homographies, so camera zoom/drift can't shake it off.

Track gaps (detection dropouts) are backfilled by interpolation up to
GAP_FILL_S so cars don't blink in and out; state comes from the track's
stopped runs, which already span the gaps. Headings come from the car's
movement-lane tangent when it's on one — lanes run in the travel
direction, so backwards boxes and tilt die at the source.

Driven by the same intersection.json + queue.analyze_approach numbers as
`tva queue`. --frames renders picked stills to qa/ for fast overlay
iteration instead of a full video.
"""

import cv2
import numpy as np

from .queue import analyze_approach, classify_movements, in_poly
from .stabilize import apply_h

RAW_END_S = 4.0       # intro: crisp footage, annotation already on
FADE_S = 1.5          # background blur/darken ramps in over this
BLUR_K = 31           # background blur kernel (odd)
BG_DARKEN = 0.55      # blurred background brightness
GAP_FILL_S = 1.5      # backfill detection dropouts up to this long
SMOOTH_K = 5          # temporal smoothing window for box centers (frames)
HEAD_SMOOTH_K = 9     # circular smoothing window for headings (frames)
FILL_ALPHA = 0.55     # car fill opacity; borders/triangles stay crisp
FLY_S = 1.2           # wait-end "+Ns" label flight time to the HUD corner
MIN_FLY_DUR_S = 2.0   # shorter waits end quietly (no flight spam)
MAX_ASPECT = 3.8      # L/W beyond this = zebra stripe, not a vehicle
MIN_MEAN_CONF = 0.5   # below this a track is suspect...
STATIC_MAX_DISP = 60  # ...and if it also never moves this far (ref px,
                      # ~1.2 car lengths) it is a static hallucination
AMBER_FACTOR = 2.5    # amber band top = v_move * this (~15 km/h slow-roll);
                      # the default (v_move ~ 6 km/h) reads as stopped
SIGNAL_GAP_S = 6.0    # stop-bar crossings closer than this = one green window
SIGNAL_TAIL_S = 2.0   # window stays green this long past the last crossing
PATH_N = 64           # arc-length samples per movement centerline
LANE_MIN_TRACKS = 3   # fewer member tracks than this = not a lane, skip
LANE_ALPHA = 0.30     # movement-ribbon blend strength (subtler than lines)
FIT_MIN_PTS = 3       # crossing cluster smaller than this: draw hand line
BG = (18, 18, 22)
IN_LINE = (80, 220, 80)      # BGR, greenish
OUT_LINE = (60, 60, 230)     # reddish
QUEUE_POLY = (200, 160, 40)
LANE_COL = (190, 200, 205)   # movement ribbons: neutral light gray-blue
STATE_BGR = {"g": (90, 220, 90), "a": (60, 200, 240), "r": (80, 80, 240),
             "p": (70, 70, 70)}
HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX

RAIN_START_S = 6.0    # first equation drops once the blur has settled
RAIN_EVERY_S = 2.6    # spawn cadence
RAIN_FALL_S = 1.7     # fall duration (ease-out cubic)
RAIN_SETTLE_A = 0.55  # resting opacity — annotation stays the star
RAIN_MAX_W = 0.42     # no equation wider than this fraction of the frame
# (tex, target height px); Little's law is the hero: it falls LAST and
# lands dead center as the finale
EQUATIONS = [
    (r"$P(N{=}k)=\frac{(\lambda t)^k e^{-\lambda t}}{k!}$", 30),
    (r"$f(t)=\mu e^{-\mu t}$", 32),
    (r"$\rho=\lambda/\mu$", 34),
    (r"$L=\frac{\rho}{1-\rho}$", 32),
    (r"$W=\frac{1}{\mu-\lambda}$", 32),
    (r"$W_q=\frac{\lambda\,E[S^2]}{2(1-\rho)}$", 30),
    (r"$W_q\approx\frac{\rho}{1-\rho}\,\frac{c_a^2+c_s^2}{2}\,E[S]$", 28),
    (r"$L=\lambda W$", 84),
]
# landing slots as (x, y) frame fractions, in EQUATIONS order; spread to
# the four quadrants, clear of the corner HUD blocks, Kingman (wide)
# low-center, and Little's law last, dead center
RAIN_SLOTS = [(0.26, 0.13), (0.74, 0.13), (0.14, 0.32), (0.86, 0.32),
              (0.20, 0.80), (0.80, 0.80), (0.50, 0.90), (0.50, 0.50)]


class Scene:
    """All per-frame lookups the renderer needs, precomputed once."""

    def __init__(self, ws):
        meta = ws.meta
        self.fps, self.n = meta["fps"], meta["n_frames"]
        self.w, self.h = meta["width"], meta["height"]
        self.src = meta["src"]
        hom = ws.load("homographies.json")
        # world->image maps, with broken frames (sudden scale collapse/spike
        # — the stabilizer occasionally loses the plot for a few frames)
        # replaced by the previous good one
        self.Hinv, last, last_s = [], None, None
        for H in hom["H"]:
            Hinv = np.linalg.inv(np.asarray(H))
            s = _proj_scale(Hinv, self.w / 2, self.h / 2)
            if last_s is not None and not 0.67 < s / last_s < 1.5:
                Hinv = last
            else:
                last, last_s = Hinv, s
            self.Hinv.append(Hinv)
        wt = ws.load("world_tracks.json")
        self.v_stop, self.v_move = wt["v_stop"], wt["v_move"]
        self.car_len = wt["car_len_px"]
        self.xing = ws.load("intersection.json")
        # per-track world-space sizes from detection boxes. Boxes are image
        # space, so each obs is scaled back to the reference plane first —
        # with a zooming camera, raw image px conflate car size with zoom
        # (a track seen mostly late would read 2-3x too big and then get
        # scaled up AGAIN at projection time)
        raw = ws.load("tracks.json")
        self.size = {}
        skip = set()
        for tr in raw["tracks"]:
            Ls, Ws = [], []
            for o in tr["obs"]:
                if o[0] >= len(self.Hinv):
                    continue
                s = _proj_scale(self.Hinv[o[0]], self.w / 2,
                                self.h / 2) / 100.0
                Ls.append(max(o[3], o[4]) / s)
                Ws.append(min(o[3], o[4]) / s)
            if Ls:
                L, W = float(np.median(Ls)), float(np.median(Ws))
                if L / (W or 1) > MAX_ASPECT:
                    # at high zoom the detector fires on individual zebra
                    # stripes (long thin "bus"/"truck" boxes) — drop them
                    skip.add(tr["id"])
                else:
                    self.size[tr["id"]] = (L, W)
        if skip:
            print(f"dropped {len(skip)} stripe-like tracks "
                  f"(aspect > {MAX_ASPECT})")
        # static low-confidence hallucinations: the detector occasionally
        # invents a box on a bench / roof corner / sidewalk clutter and
        # holds it for a second or two. Real vehicles either move or are
        # seen with high confidence (a queued car is static but conf~0.9).
        conf = {tr["id"]: float(np.mean([o[5] for o in tr["obs"]]))
                for tr in raw["tracks"] if tr["obs"]}
        static = set()
        for tr in wt["tracks"]:
            if tr["id"] in skip or not tr["x"]:
                continue
            x0, y0 = tr["x"][0], tr["y"][0]
            disp = max(float(np.hypot(x - x0, y - y0))
                       for x, y in zip(tr["x"], tr["y"]))
            if conf.get(tr["id"], 1.0) < MIN_MEAN_CONF and \
                    disp < STATIC_MAX_DISP:
                static.add(tr["id"])
                self.size.pop(tr["id"], None)
        if static:
            print(f"dropped {len(static)} static low-conf tracks "
                  f"(mean conf < {MIN_MEAN_CONF}, disp < {STATIC_MAX_DISP})")
        skip |= static
        tracks = [tr for tr in wt["tracks"] if tr["id"] not in skip]
        self.approaches = [analyze_approach(ap, tracks, self.n, self.fps)
                           for ap in self.xing["approaches"]]
        # per-track raw headings from path gradients, carried while stopped
        raw_h = {}
        for tr in tracks:
            ang = np.arctan2(np.gradient(tr["y"]), np.gradient(tr["x"]))
            filled, last = [], 0.0
            for a, sp in zip(ang, tr["speed"]):
                if sp > self.v_stop:
                    last = float(a)
                filled.append(last)
            raw_h[tr["id"]] = dict(zip(tr["frames"], filled))
        # pass 1: smoothed, gap-filled per-track position series (the
        # render-time persistence trick from the hero pipeline) — cached
        # for movement-lane extraction and reused by the dots pass
        polys = [ap["queue_poly"] for ap in self.xing["approaches"]]
        max_gap = int(GAP_FILL_S * self.fps)
        series = {}
        for tr in tracks:
            fr = np.asarray(tr["frames"])
            full = np.arange(fr[0], fr[-1] + 1)
            xi = _smooth(np.interp(full, fr, tr["x"]), SMOOTH_K)
            yi = _smooth(np.interp(full, fr, tr["y"]), SMOOTH_K)
            spi = np.interp(full, fr, tr["speed"])
            keep = np.zeros(len(full), bool)
            for a, b in zip(fr[:-1], fr[1:]):
                if b - a - 1 <= max_gap:
                    keep[a - fr[0]: b - fr[0] + 1] = True
            for f in fr:
                keep[f - fr[0]] = True
            series[tr["id"]] = (full, xi, yi, spi, keep)
        # movement lanes: median centerline per (entry, exit) movement
        self.move_paths, self.tid_path = _movement_paths(
            self.xing["approaches"], tracks, series, self.fps, self.car_len)
        # counting lines as DRAWN: refit each to its actual crossing-point
        # cluster (the hand-authored lines still drive the counting)
        self.fit_lines = _fitted_lines(
            self.xing["approaches"], self.approaches, series, self.fps)
        # pass 2: backfilled per-frame dots + headings
        self.dots = {}
        self.pos = {}
        self.heading = {}
        for tr in tracks:
            full, xi, yi, spi, keep = series[tr["id"]]
            stopped_runs = [(a, b) for s, a, b in tr["runs"] if s]
            h_obs = raw_h[tr["id"]]
            h_arr, last_h = [], 0.0
            for f in full:
                if f in h_obs:
                    last_h = h_obs[f]
                h_arr.append(last_h)
            h_arr = np.arctan2(_smooth(np.sin(h_arr), HEAD_SMOOTH_K),
                               _smooth(np.cos(h_arr), HEAD_SMOOTH_K))
            # cars don't reverse in these scenes: flip headings pointing
            # >90° against the track's overall travel direction — noisy
            # low-speed path gradients otherwise read as backwards boxes
            dx, dy = xi[-1] - xi[0], yi[-1] - yi[0]
            if dx * dx + dy * dy > 120 ** 2:
                flip = np.cos(h_arr - np.arctan2(dy, dx)) < 0
                h_arr = h_arr.copy()
                h_arr[flip] += np.pi
            path = self.move_paths.get(self.tid_path.get(tr["id"]))
            pos_d, h_d = {}, {}
            lock_pos, lock_h = None, 0.0
            for i, f in enumerate(full):
                if f >= self.n:
                    break
                if not keep[i]:
                    continue
                x, y = xi[i], yi[i]
                stopped = any(a <= f <= b for a, b in stopped_runs)
                queued_ap = None
                if stopped:
                    for ap, p in zip(self.xing["approaches"], polys):
                        if in_poly(p, x, y):
                            queued_ap = ap
                            break
                if queued_ap is not None or (stopped and
                                             not tr.get("static")):
                    st = "r"
                elif tr.get("static"):
                    st = "p"          # parked / never moved: not drawn
                else:
                    st = "a" if spi[i] < self.v_move * AMBER_FACTOR else "g"
                # heading: the car's movement-lane tangent when it is on
                # one (kills backwards triangles AND tilt in one stroke);
                # else the queued snap / smoothed path gradient
                h = None
                if path is not None:
                    h = _path_heading(path, x, y, 1.5 * self.car_len)
                if h is None:
                    if queued_ap is not None:
                        h = float(np.arctan2(queued_ap["dir"][1],
                                             queued_ap["dir"][0]))
                    else:
                        h = float(h_arr[i])
                # stopped->moving handoff: hold the heading the car had
                # when it stopped until it has actually displaced ~a car
                # length — crawl-speed gradients are noise, and switching
                # sources the instant the wheels turn reads as a flip.
                # Past that, lane-less tracks head along the displacement
                # from the stop point until the gradient catches up.
                if stopped:
                    lock_pos, lock_h = (x, y), h
                elif lock_pos is not None:
                    d2 = ((x - lock_pos[0]) ** 2 + (y - lock_pos[1]) ** 2)
                    if d2 < (0.9 * self.car_len) ** 2:
                        h = lock_h
                    else:
                        if path is None:
                            h = float(np.arctan2(y - lock_pos[1],
                                                 x - lock_pos[0]))
                        if d2 > (2.5 * self.car_len) ** 2:
                            lock_pos = None
                h_d[f] = float(h)
                self.dots.setdefault(f, []).append((x, y, st, tr["id"]))
                pos_d[f] = (x, y)
            # feather any remaining heading steps (lock release, lane
            # handoffs) with a short circular smoothing pass
            if h_d:
                fs = sorted(h_d)
                ha = np.array([h_d[f] for f in fs])
                ha = np.arctan2(_smooth(np.sin(ha), HEAD_SMOOTH_K),
                                _smooth(np.cos(ha), HEAD_SMOOTH_K))
                h_d = dict(zip(fs, ha))
            self.pos[tr["id"]] = pos_d
            self.heading[tr["id"]] = h_d
        # wait runs with their approach; flight-worthy ends are gated below
        # on the inferred light actually being green
        self.waits = []        # (tid, t0, t1, approach name)
        raw_ends = []          # candidate wait ends, pre signal-gating
        for ap in self.approaches:
            for tid, t0, dur, trunc in ap["waits"]:
                self.waits.append((tid, t0, t0 + dur, ap["name"]))
                # truncated waits ended because the car left the view (or
                # the clip ended), not because it departed — no flight for
                # a wait we never saw end
                if dur >= MIN_FLY_DUR_S and not trunc:
                    raw_ends.append((tid, t0 + dur, ap["name"], dur))
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
        # else inferred from stop-bar crossing runs (a lone crossing still
        # flashes green briefly — someone did enter the junction)
        self.signal = {}
        for ap, res in zip(self.xing["approaches"], self.approaches):
            phases = (ap.get("signal") or {}).get("phases")
            green = np.zeros(self.n, bool)
            if phases:
                for t0, t1, st in phases:
                    if st == "green":
                        green[int(t0 * self.fps):int(t1 * self.fps)] = True
            else:
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
        # flight-worthy wait ends: the wait is not truncated AND the same
        # car actually crosses its approach's stop bar shortly after the
        # wait ends. Leaving the poly while stationary (drift), creeping
        # forward within the queue, or exiting the frame are none of them
        # a departure; a stop-bar crossing is the physical discharge event
        # (and it is what the inferred green windows are built from, so
        # this also pins flights to the light actually changing).
        self.wait_ends = []    # (tid, t1, approach name, final wait s)
        dep_t = {}
        for ap in self.approaches:
            for t, tid in ap["departures"]:
                dep_t.setdefault((ap["name"], tid), []).append(t)
        for tid, t1, name, dur in raw_ends:
            if any(t1 - 2.0 <= td <= t1 + 8.0
                   for td in dep_t.get((name, tid), [])):
                self.wait_ends.append((tid, t1, name, dur))
        self.rain = False

    def active_waits(self, t):
        for tid, t0, t1, name in self.waits:
            if t0 <= t < t1:
                yield tid, t - t0, name

    def cycle_wait(self, t):
        """Live accumulated wait per approach: sum over cars waiting right
        now of how long each has been waiting (resets as queues clear)."""
        cyc = {}
        for tid, t0, t1, name in self.waits:
            if t0 <= t < t1:
                cyc[name] = cyc.get(name, 0.0) + (t - t0)
        return cyc

    def state_counts(self, f):
        return [(ap["name"], int(np.searchsorted(ap["arr_t"], f / self.fps)),
                 int(ap["qlen"][f]), ap["wait"][f])
                for ap in self.counters]


def self_pos(scn, tid, f):
    return scn.pos.get(tid, {}).get(f)


def _proj_scale(M, cx, cy):
    """How many image pixels 100 reference-plane px map to under M."""
    p0 = M @ [cx, cy, 1.0]
    p0 /= p0[2]
    p1 = M @ [cx + 100.0, cy, 1.0]
    p1 /= p1[2]
    return float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))


def _smooth(a, k):
    """Centered moving average with edge padding (anti-twitch)."""
    if len(a) < k:
        return a
    ap = np.pad(a, k // 2, mode="edge")
    return np.convolve(ap, np.ones(k) / k, mode="valid")


def _movement_paths(approaches, tracks, series, fps, car_len):
    """Median centerline per (entry, exit) movement, from member tracks'
    smoothed paths (entry crossing -> track end) resampled to PATH_N
    arc-length samples. Returns ({movement key: (pts, unit tangents)},
    {tid: movement key}). Entry = in_line crossing, or the approach whose
    queue polygon holds the track's first point (cars already inside the
    view); exit leg = the approach whose OUTbound direction best matches
    the track's net displacement. Paths run in the travel direction, so
    their tangents double as ground-truth headings."""
    moves = classify_movements(approaches, tracks, fps)
    groups, tid_key = {}, {}
    for tr in tracks:
        m = moves.get(tr["id"]) or {}
        full, xi, yi = series[tr["id"]][:3]
        entry, t0 = m.get("entry"), m.get("t")
        if entry is None:
            # started inside the view: adopt the approach whose queue
            # polygon holds the track's first point, if any
            for ai, ap in enumerate(approaches):
                if in_poly(ap["queue_poly"], xi[0], yi[0]):
                    entry, t0 = ai, full[0] / fps
                    break
        if entry is None:
            continue
        i0 = 0 if t0 is None else int(np.searchsorted(full, t0 * fps))
        xs, ys = xi[i0:], yi[i0:]
        if len(xs) < 8:
            continue
        s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(xs),
                                                    np.diff(ys)))])
        if s[-1] < car_len:
            continue
        q = np.linspace(0, s[-1], PATH_N)
        path = np.column_stack([np.interp(q, s, xs), np.interp(q, s, ys)])
        # exit leg: best match between net displacement and an outbound
        # direction (None = the car never left — still queued at clip end)
        dx, dy = xi[-1] - xi[i0], yi[-1] - yi[i0]
        ai_out = None
        if dx * dx + dy * dy > 150 ** 2:
            net = np.array([dx, dy])
            net /= np.hypot(*net)
            best, ai_out = 0.6, None
            for ai, ap in enumerate(approaches):
                c = float(-(net @ np.array(ap["dir"], float)))
                if c > best:
                    best, ai_out = c, ai
        key = (entry, ai_out)
        groups.setdefault(key, []).append(path)
        tid_key[tr["id"]] = key
    paths = {}
    for key, members in groups.items():
        if len(members) < LANE_MIN_TRACKS:
            continue
        pts = np.median(np.stack(members), axis=0)
        tx, ty = np.gradient(pts[:, 0]), np.gradient(pts[:, 1])
        norm = np.hypot(tx, ty)
        norm[norm == 0] = 1e-9
        paths[key] = (pts, np.column_stack([tx / norm, ty / norm]))
    tid_key = {tid: k for tid, k in tid_key.items() if k in paths}
    return paths, tid_key


def _path_heading(path, x, y, max_d):
    """Heading from the movement centerline's tangent at the nearest
    sample, or None when the car sits too far off its lane to trust it."""
    pts, tan = path
    d2 = (pts[:, 0] - x) ** 2 + (pts[:, 1] - y) ** 2
    i = int(np.argmin(d2))
    if d2[i] > max_d * max_d:
        return None
    return float(np.arctan2(tan[i, 1], tan[i, 0]))


def _fitted_lines(approaches, results, series, fps):
    """Per approach, refit each counting line to the cluster of its actual
    crossing points (PCA major axis over the cluster extent, small margin).
    Counting is untouched — this only changes how the lines are drawn."""
    out = {}
    for ap, res in zip(approaches, results):
        fitted = {}
        for key, events in (("in_line", res["arrivals"]),
                            ("out_line", res["departures"])):
            pts = []
            for t, tid in events:
                ser = series.get(tid)
                if ser is None:
                    continue
                full, xi, yi = ser[0], ser[1], ser[2]
                fr = float(np.clip(t * fps, full[0], full[-1]))
                pts.append((float(np.interp(fr, full, xi)),
                            float(np.interp(fr, full, yi))))
            if len(pts) >= FIT_MIN_PTS:
                P = np.asarray(pts)
                c = P.mean(0)
                _, _, vt = np.linalg.svd(P - c, full_matrices=False)
                proj = (P - c) @ vt[0]
                fitted[key] = [c + vt[0] * (proj.min() - 40),
                               c + vt[0] * (proj.max() + 40)]
        out[ap["name"]] = fitted
    return out


def draw_frame(frame, scn, f, mix):
    """Annotation over footage; mix=0 crisp background, mix=1 fully
    blurred/darkened."""
    Hinv_i = scn.Hinv[min(f, len(scn.Hinv) - 1)]
    if mix > 0:
        bg = cv2.GaussianBlur(frame, (BLUR_K, BLUR_K), 0)
        dark = bg.astype(np.float32) * BG_DARKEN
        base = cv2.addWeighted(frame.astype(np.float32), 1 - mix, dark, mix,
                               0).astype(np.uint8)
    else:
        base = frame
    # queue lines/polys, blended into the background
    layer = base.astype(np.float32)
    # movement lanes first (deepest annotation layer): soft ribbons with
    # arrowheads showing where each (entry, exit) movement actually flows
    if scn.move_paths:
        lane = np.zeros_like(base)
        scale = _proj_scale(Hinv_i, scn.w / 2, scn.h / 2) / 100.0
        thick = max(2, int(scn.car_len * 0.32 * scale))
        for pts, _tan in scn.move_paths.values():
            p = apply_h(Hinv_i, pts).astype(np.int32)
            cv2.polylines(lane, [p], False, LANE_COL, thick, cv2.LINE_AA)
            for j in {len(p) // 2, len(p) - 3}:   # mid + exit arrowheads
                u = (p[min(j + 2, len(p) - 1)] - p[max(j - 2, 0)]
                     ).astype(float)
                n = np.hypot(*u) or 1e-9
                u /= n
                v = np.array([-u[1], u[0]])
                tip = p[j].astype(float) + u * thick * 1.2
                tri = np.array([tip,
                                p[j] - u * thick * 1.4 + v * thick * 1.0,
                                p[j] - u * thick * 1.4 - v * thick * 1.0],
                               np.int32)
                cv2.fillConvexPoly(lane, tri, LANE_COL)
        m = lane.any(axis=2)
        layer[m] = ((1 - LANE_ALPHA) * layer[m] +
                    LANE_ALPHA * lane[m].astype(np.float32))
    ann = np.zeros_like(base)
    for ap in scn.xing["approaches"]:
        poly = np.array(apply_h(Hinv_i, ap["queue_poly"]), np.int32)
        cv2.polylines(ann, [poly], True, QUEUE_POLY, 2)
        fit = scn.fit_lines.get(ap["name"], {})
        for key, col in (("in_line", IN_LINE), ("out_line", OUT_LINE)):
            pts = apply_h(Hinv_i, fit.get(key, ap[key])).astype(np.int32)
            cv2.line(ann, tuple(pts[0]), tuple(pts[1]), col, 3)
    m = ann.any(axis=2)
    layer[m] = 0.45 * layer[m] + 0.55 * ann[m].astype(np.float32)
    out = layer.astype(np.uint8)
    # cars: translucent rotated bbox + crisp border + heading triangle.
    # Fills go on an overlay so the dulled background shows through;
    # borders/triangles are drawn solid on top.
    shapes = []
    for x, y, st, tid in scn.dots.get(f, []):
        if st == "p":
            continue            # parked/off-road statics: not part of the story
        L, W = scn.size.get(tid, (scn.car_len, 0.55 * scn.car_len))
        rad = scn.heading.get(tid, {}).get(f, 0.0)
        u = np.array([np.cos(rad), np.sin(rad)])
        v = np.array([-u[1], u[0]])
        c = np.array([x, y])
        corners = [c + u * (L / 2) * sx + v * (W / 2) * sy
                   for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
        base_c = c + u * (L / 2)
        tri = [base_c + v * W * 0.42, base_c - v * W * 0.42,
               base_c + u * L * 0.35]
        shapes.append((apply_h(Hinv_i, corners).astype(np.int32),
                       apply_h(Hinv_i, tri).astype(np.int32), STATE_BGR[st],
                       st))
    overlay = out.copy()
    for box, _, col, _ in shapes:
        cv2.fillConvexPoly(overlay, box, col)
    out = cv2.addWeighted(overlay, FILL_ALPHA, out, 1 - FILL_ALPHA, 0)
    for box, tri, col, st in shapes:
        border = tuple(min(255, int(c * 1.35 + 40)) for c in col)
        cv2.polylines(out, [box], True, border, 2, cv2.LINE_AA)
        cv2.fillConvexPoly(out, tri, tuple(int(c * 0.55) for c in col))
    draw_signals(out, scn, f, Hinv_i)
    draw_timers(out, scn, f, Hinv_i)
    draw_wait_flights(out, scn, f, Hinv_i)
    if scn.rain:
        draw_rain(out, scn, f)
    draw_hud(out, scn, f)
    return out


def _hud_order(names):
    """Compass approaches in fixed corner order; custom names keep their
    config order after them."""
    compass = [n for n in ("west", "east", "south", "north") if n in names]
    return compass + [n for n in names if n not in compass]


def hud_anchor(scn, name):
    """Rough center of an approach's HUD corner block (fly-to target)."""
    order = _hud_order([ap["name"] for ap in scn.counters])
    i = order.index(name)
    x = scn.w - 170 if i % 2 == 1 else 170
    y = scn.h - 45 if i >= 2 else 45
    return np.array([x, y], float)


def draw_wait_flights(img, scn, f, Hinv_i):
    """When a wait ends, its final seconds fly from the car to the
    approach's HUD corner: ease-out path, growing, fading."""
    t = f / scn.fps
    for tid, t1, name, dur in scn.wait_ends:
        p = (t - t1) / FLY_S
        if not 0 <= p <= 1:
            continue
        f1 = int(t1 * scn.fps)
        wp = next((q for df in range(-2, 3)
                   if (q := self_pos(scn, tid, f1 + df)) is not None), None)
        if wp is None:
            continue
        start = apply_h(Hinv_i, [wp])[0]
        e = 1 - (1 - p) ** 3                    # ease-out cubic
        pos = start + (hud_anchor(scn, name) - start) * e
        _draw_fly_text(img, f"+{dur:.0f}s", pos, 0.8 + 0.5 * p, 1 - p * p)


def _draw_fly_text(img, txt, pos, scale, alpha):
    """Alpha-blended text via a small ROI overlay (putText has no alpha)."""
    (tw, th), _ = cv2.getTextSize(txt, HUD_FONT, scale, 2)
    x, y = int(pos[0]), int(pos[1])
    x0, y0 = min(max(x - tw // 2, 0), img.shape[1] - 1), \
        min(max(y - th - 4, 0), img.shape[0] - 1)
    x1 = min(x0 + tw + 8, img.shape[1])
    y1 = min(y0 + th + 10, img.shape[0])
    if x1 - x0 < 4 or y1 - y0 < 4:
        return
    roi = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
    cv2.putText(roi, txt, (x - tw // 2 - x0, y - y0), HUD_FONT, scale,
                (250, 250, 250), 2, cv2.LINE_AA)
    m = roi.any(axis=2)
    sub = img[y0:y1, x0:x1]
    sub[m] = (sub[m] * (1 - alpha) + roi[m] * alpha).astype(np.uint8)


def draw_signals(img, scn, f, Hinv_i):
    """3-lamp signal head at each stop bar (red top / amber / green
    bottom); the lit lamp follows scn.signal for this frame. Sits just
    past the stop bar on the driver's right — where the far-side head
    actually hangs in right-hand traffic."""
    for ap in scn.xing["approaches"]:
        green = scn.signal[ap["name"]][f]
        p0, p1 = (np.array(pt, float) for pt in ap["out_line"])
        mid = (p0 + p1) / 2
        d = np.array(ap["dir"], float)
        d /= np.hypot(*d) or 1e-9
        right = np.array([-d[1], d[0]])   # driver's right, y-down coords
        end = p0 if (p0 - mid) @ right > (p1 - mid) @ right else p1
        base = apply_h(Hinv_i, [end + d * 38.0 + right * 22.0])[0]
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


def draw_timers(img, scn, f, Hinv_i):
    """Ticking wait timers over stopped cars, greedily stacked upward so
    packed queues don't pile every timer on the same spot."""
    t = f / scn.fps
    placed = []
    for tid, waited, _ in sorted(scn.active_waits(t), key=lambda w: -w[1]):
        p = next((q for df in range(-2, 3)
                  if (q := self_pos(scn, tid, f + df)) is not None), None)
        if p is None:
            continue
        px = apply_h(Hinv_i, [p])[0].astype(int)
        txt = f"{waited:4.1f}s"
        (tw, th), _ = cv2.getTextSize(txt, HUD_FONT, 0.55, 1)
        for k in range(8):
            org = (px[0] - 18, px[1] - 14 - k * (th + 6))
            rect = (org[0] - 2, org[1] - th - 2, org[0] + tw + 2,
                    org[1] + 4)
            if not any(rect[0] < q[2] and rect[2] > q[0] and
                       rect[1] < q[3] and rect[3] > q[1] for q in placed):
                placed.append(rect)
                cv2.putText(img, txt, org, HUD_FONT, 0.55,
                            (240, 240, 240), 1, cv2.LINE_AA)
                break


def draw_hud(img, scn, f):
    t = f / scn.fps
    cyc = scn.cycle_wait(t)
    # a HUD line pulses for a moment when a flight lands in its bucket
    landed = {name for _, t1, name, _ in scn.wait_ends
              if t1 + FLY_S - 0.5 <= t <= t1 + FLY_S + 0.2}
    counts = {name: (arrived, queued, wait)
              for name, arrived, queued, wait in scn.state_counts(f)}
    order = _hud_order(list(counts))
    for i, name in enumerate(order):
        arrived, queued, wait = counts[name]
        txt = (f"{name}  arr {arrived}  q {queued}  "
               f"cyc {cyc.get(name, 0.0):.0f}s  tot {wait:.0f} veh-s")
        right = i % 2 == 1
        bottom = i >= 2
        big = name in landed
        fs = 0.85 if big else 0.7
        fg = (255, 255, 255) if big else (230, 230, 230)
        (tw, th), _ = cv2.getTextSize(txt, HUD_FONT, fs, 1)
        x = scn.w - tw - 18 if right else 18
        y = scn.h - 18 if bottom else 12 + th
        cv2.rectangle(img, (x - 8, y - th - 8), (x + tw + 8, y + 8),
                      (0, 0, 0), -1)
        cv2.putText(img, txt, (x, y), HUD_FONT, fs, fg, 1, cv2.LINE_AA)


def draw_rain(img, scn, f):
    """Queueing-theory equations fall from the sky and settle along the
    bottom: a visual table of contents for the models that claim to
    explain this scene."""
    t = f / scn.fps
    sprites = _rain_images(scn.w)
    for i, sprite in enumerate(sprites):
        p = (t - (RAIN_START_S + i * RAIN_EVERY_S)) / RAIN_FALL_S
        if p <= 0:
            continue
        lx, ly = RAIN_SLOTS[i]
        lx, ly = lx * scn.w, ly * scn.h
        angle = ((i * 37) % 17) - 8            # deterministic tilt, deg
        if p < 1:
            e = 1 - (1 - p) ** 3               # ease-out cubic
            x = lx + np.sin(p * 2 * np.pi) * 34 * (1 - p)
            y = -sprite.shape[0] + (ly + sprite.shape[0]) * e
            alpha = min(1.0, p * 3) * 0.95
        else:
            settle = min(1.0, (p - 1) * RAIN_FALL_S / 0.5)
            x, y = lx, ly
            alpha = 0.95 + (RAIN_SETTLE_A - 0.95) * settle
        _stamp(img, sprite, x, y, angle, alpha)


_RAIN_CACHE = None


def _rain_images(frame_w):
    """Render each equation once via matplotlib mathtext -> trimmed BGRA
    sprite scaled to its target height (and clamped to RAIN_MAX_W)."""
    global _RAIN_CACHE
    if _RAIN_CACHE is None:
        _RAIN_CACHE = []
        for tex, target_h in EQUATIONS:
            spr = _eq_bgra(tex, target_h)
            if spr.shape[1] > RAIN_MAX_W * frame_w:
                s = RAIN_MAX_W * frame_w / spr.shape[1]
                spr = cv2.resize(spr, None, fx=s, fy=s,
                                 interpolation=cv2.INTER_AREA)
            _RAIN_CACHE.append(spr)
    return _RAIN_CACHE


def _eq_bgra(tex, target_h):
    """Render mathtext to a white-text BGRA sprite. math_to_image's
    color/transparency handling varies by version, so render the default
    black-on-white and convert luminance -> alpha ourselves."""
    from io import BytesIO

    from matplotlib.mathtext import math_to_image
    buf = BytesIO()
    math_to_image(tex, buf, dpi=300, format="png")
    gray = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8),
                        cv2.IMREAD_GRAYSCALE)
    alpha = 255 - gray                          # text darkness -> opacity
    ys, xs = np.where(alpha > 8)
    alpha = alpha[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    bgra = np.full((*alpha.shape, 4), 255, np.uint8)
    bgra[:, :, 3] = alpha
    s = target_h / bgra.shape[0]
    return cv2.resize(bgra, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def _stamp(img, bgra, cx, cy, angle, alpha):
    """Alpha-blend a BGRA sprite (rotated by angle deg) centered (cx, cy)."""
    h, w = bgra.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    nw = int(h * abs(M[0, 1]) + w * abs(M[0, 0]))
    nh = int(h * abs(M[0, 0]) + w * abs(M[0, 1]))
    M[0, 2] += (nw - w) / 2
    M[1, 2] += (nh - h) / 2
    rot = cv2.warpAffine(bgra, M, (nw, nh))
    x0, y0 = int(round(cx - nw / 2)), int(round(cy - nh / 2))
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(img.shape[1], x0 + nw), min(img.shape[0], y0 + nh)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    spr = rot[sy0:sy0 + dy1 - dy0, sx0:sx0 + dx1 - dx0]
    a = spr[:, :, 3:4].astype(np.float32) / 255.0 * alpha
    roi = img[dy0:dy1, dx0:dx1].astype(np.float32)
    img[dy0:dy1, dx0:dx1] = (roi * (1 - a) +
                             spr[:, :, :3].astype(np.float32) * a
                             ).astype(np.uint8)


def run(ws, out=None, rain=False, bg_darken=None, blur_k=None,
        frames=None, lanes=True):
    if bg_darken is not None:
        global BG_DARKEN
        BG_DARKEN = bg_darken
    if blur_k is not None:
        global BLUR_K
        BLUR_K = blur_k | 1  # kernel must stay odd
    scn = Scene(ws)
    scn.rain = rain
    if not lanes:
        scn.move_paths = {}
    raw_end = RAW_END_S * scn.fps
    fade_end = raw_end + FADE_S * scn.fps

    def mix_at(f):
        if f < raw_end:
            return 0.0
        return min(1.0, (f - raw_end) / (fade_end - raw_end))

    if frames:
        # QA stills mode: render picked frames to qa/xing-fN.png and stop —
        # the fast iteration loop for overlay work
        cap = cv2.VideoCapture(scn.src)
        for f in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(f, scn.n - 1))
            ok, frame = cap.read()
            if not ok:
                continue
            path = ws.path(f"qa/xing-f{f}.png")
            cv2.imwrite(path, draw_frame(frame, scn, f, mix_at(f)))
            print(path)
        cap.release()
        return
    out = out or ws.path("xing-queue-rain.mp4" if rain else "xing-queue.mp4")
    cap = cv2.VideoCapture(scn.src)
    wr = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), scn.fps,
                         (scn.w, scn.h))
    f = 0
    while f < scn.n:
        ok, frame = cap.read()
        if not ok:
            break
        wr.write(draw_frame(frame, scn, f, mix_at(f)))
        f += 1
        if f % 200 == 0:
            print(f"frame {f}/{scn.n}")
    cap.release()
    wr.release()
    print(f"wrote {out}")
