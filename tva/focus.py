"""Focus-lane recovery + stop-wave state pipeline.

The focus lane (focus_lane.json) is the hand-picked lane whose cars get the
loud styling so the stop wave marching upstream is impossible to miss. This
module is the single home for deciding WHICH cars those are and WHAT STATE
they show, shared by the render (render.py), the wave views (viz.py) and
the lane editor (lane_edit.py):

- recover(): the render-time recovery pass - hand-annotated cars join as
  full tracks, early track fragments get chained onto focus tracks, and
  queued cars are held back to frame 0 - so every consumer works from the
  same tracks the overlay will draw.
- displayed_states(): engine quantization + hand overrides + commit/onset
  filtering - the exact state sequence and ring points the overlay draws.
- the state vocabulary (names, codes, colors) and onset-ring parameters;
  the editor ships them to its client via /data instead of re-deriving
  them in JS.
"""
from types import SimpleNamespace

import numpy as np

from . import overrides
from .stabilize import apply_h

RING_S = 0.7           # stop-onset ring duration
RING_R0 = 40           # onset ring start radius (src-frame px)
RING_R1 = 120          # radius growth over the ring's life
RING_FADE = 0.5        # brightness fraction lost by ring end
STITCH_GAP_S = 8       # max dropout when chaining early fragments
BACKFILL_V = 1.5       # x v_stop: "already stopped at first detection"
STOP_ENTER = 1.0       # x v_stop: candidate stopped below this
STOP_EXIT = 1.8        # x v_stop: leave stopped only above this, sustained
BRAKE_ENTER = 1.7      # x v_move: candidate braking below this
BRAKE_EXIT = 2.2       # x v_move: back to rolling only above this
STATE_DEB_S = 0.2      # a new state must hold this long to commit
# quantized focus-lane states; RGB matching the continuous ramp's anchors
STATE_RGB = {"rolling": (60, 200, 90), "braking": (235, 170, 50),
             "stopped": (235, 60, 50)}
STATE_CODE = {"rolling": "r", "braking": "b", "stopped": "s"}
CODE_STATE = {c: s for s, c in STATE_CODE.items()}


def focus_lane_ids(wt, roads, spec):
    """Track ids riding the hand-picked lane in focus_lane.json: median
    signed offset from the road centerline inside spec's offset_band."""
    road = roads[spec["road"]]
    rx, ry = np.asarray(road["x"]), np.asarray(road["y"])
    tang = np.gradient(np.column_stack([rx, ry]), axis=0)
    tang /= np.maximum(np.hypot(tang[:, 0], tang[:, 1])[:, None], 1e-9)
    nrm = np.column_stack([-tang[:, 1], tang[:, 0]])
    lo, hi = spec["offset_band"]
    ids = set(spec.get("include", []))
    for tr in wt["tracks"]:
        x, y = np.asarray(tr["x"]), np.asarray(tr["y"])
        d2 = (rx[None, :] - x[:, None]) ** 2 + (ry[None, :] - y[:, None]) ** 2
        j = d2.argmin(axis=1)
        if np.median(np.sqrt(d2[np.arange(len(x)), j])) > spec["max_dist"]:
            continue
        off = np.median((x - rx[j]) * nrm[j, 0] + (y - ry[j]) * nrm[j, 1])
        if lo <= off <= hi:
            ids.add(tr["id"])
    return ids - set(spec.get("exclude", []))


def hidden_ids(ws, wt, roads):
    """Parked/scenery tracks the render never paints.

    statics.json: hand-enforced known-fixed zones (parking lots, depots) in
    world px — registration extrapolation drifts in off-plane corners, so
    parked cars there read fake speeds no automatic static test catches.
    Plus static-and-off-every-road culling: that "speed" is pure
    registration noise, so they twitch with color.
    """
    parked = set()
    try:
        zones = ws.load("statics.json")["zones"]
    except FileNotFoundError:
        zones = []
    if zones:
        from matplotlib.path import Path
        paths = [Path(z["polygon"]) for z in zones]
        for tr in wt["tracks"]:
            p = (np.median(tr["x"]), np.median(tr["y"]))
            if any(pt.contains_point(p) for pt in paths):
                parked.add(tr["id"])
        print(f"{len(parked)} vehicles inside static zones hidden")
    if roads:
        from .kinematics import station_of
        for tr in wt["tracks"]:
            if not tr.get("static"):
                continue
            mx = np.array([np.median(tr["x"])])
            my = np.array([np.median(tr["y"])])
            dmin = min(station_of(r["x"], r["y"], mx, my)[1][0]
                       for r in roads)
            if dmin > 3.5 * wt["car_len_px"]:
                parked.add(tr["id"])
        print(f"{len(parked)} parked vehicles hidden")
    return parked


def manual_world(ws, Hs, fps):
    """Hand-annotated cars from the lane editor (manual_tracks.json):
    keyframe image-px points, linearly interpolated per frame, lifted to
    world via the per-frame homography; speed from the smoothed world path.
    """
    try:
        man = ws.load("manual_tracks.json")["tracks"]
    except FileNotFoundError:
        return []
    out = []
    for m in man:
        pts = sorted(map(tuple, m["points"]))
        if not pts:
            continue
        px = {pts[0][0]: (pts[0][1], pts[0][2])}
        for (fa, xa, ya), (fb, xb, yb) in zip(pts, pts[1:]):
            for f in range(fa, fb + 1):
                t = (f - fa) / (fb - fa) if fb > fa else 0.0
                px[f] = (xa + (xb - xa) * t, ya + (yb - ya) * t)
        F = sorted(px)
        w = np.array([apply_h(Hs[f], [px[f]])[0] for f in F])
        sm = w.copy()
        if len(w) > 7:
            k = np.ones(7) / 7
            sm = np.column_stack([np.convolve(w[:, i], k, "same")
                                  for i in (0, 1)])
            sm[:3], sm[-3:] = w[:3], w[-3:]
        d = np.gradient(sm, axis=0) if len(w) > 1 else np.zeros_like(w)
        sp = (np.hypot(d[:, 0], d[:, 1]) * fps).tolist()
        out.append({"track": {"id": m["id"], "frames": F,
                              "x": w[:, 0].tolist(), "y": w[:, 1].tolist(),
                              "speed": sp},
                    "id": m["id"], "px": px})
    return out


def lane_guide_path(wt, focus, roads, spec, car_len):
    """World-plane centerline of the focus lane, fitted from the selected
    cars' paths: bin their world points by station along the road, median
    per bin, light smoothing."""
    road = roads[spec["road"]]
    rx, ry = np.asarray(road["x"]), np.asarray(road["y"])
    s_cum = np.concatenate(
        [[0], np.cumsum(np.hypot(np.diff(rx), np.diff(ry)))])
    S, X, Y = [], [], []
    for tr in wt["tracks"]:
        if tr["id"] not in focus:
            continue
        x, y = np.asarray(tr["x"]), np.asarray(tr["y"])
        d2 = (rx[None, :] - x[:, None]) ** 2 + (ry[None, :] - y[:, None]) ** 2
        j = d2.argmin(axis=1)
        S.extend(s_cum[j])
        X.extend(x)
        Y.extend(y)
    if not S:
        return None
    S, X, Y = map(np.asarray, (S, X, Y))
    step = car_len / 2
    p = []
    for b in np.arange(S.min(), S.max() + step, step):
        sel = (S >= b) & (S < b + step)
        if sel.sum() >= 6:
            p.append((np.median(X[sel]), np.median(Y[sel])))
    if len(p) < 2:
        return None
    p = np.asarray(p)
    if len(p) > 8:
        # "valid" trims the sparse noisy end bins instead of keeping them
        k = np.ones(7) / 7
        p = np.column_stack([np.convolve(p[:, i], k, "valid")
                             for i in (0, 1)])
    return p


def guide_frame(guide, step=5.0):
    """Densified guide polyline + arc-length stations + unit normals, for
    station/offset queries against the fitted lane."""
    d = np.hypot(*np.diff(guide, axis=0).T)
    s = np.concatenate([[0], np.cumsum(d)])
    si = np.arange(0.0, s[-1] + step, step)
    g = np.column_stack([np.interp(si, s, guide[:, 0]),
                         np.interp(si, s, guide[:, 1])])
    t = np.gradient(g, axis=0)
    t /= np.maximum(np.hypot(t[:, 0], t[:, 1])[:, None], 1e-9)
    return g, si, np.column_stack([-t[:, 1], t[:, 0]])


def locate_on_guide(gf, x, y):
    """(station, signed lateral offset) of world points w.r.t. the guide."""
    g, si, n = gf
    x, y = np.asarray(x, float), np.asarray(y, float)
    d2 = (g[None, :, 0] - x[:, None]) ** 2 + (g[None, :, 1] - y[:, None]) ** 2
    j = d2.argmin(axis=1)
    off = (x - g[j, 0]) * n[j, 0] + (y - g[j, 1]) * n[j, 1]
    return si[j], off


def stitch_focus(wt, focus, gf, car_len, fps, px_of, exclude, parked):
    """Chain early track fragments onto focus-lane tracks.

    The oblique early view fragments tracks: the same physical car carries
    a throwaway id until detection stabilizes, then gets a fresh one. For
    each focus track, walk backwards through candidate fragments (real
    detections, on the lane, continuous in time and position) and merge
    them in; the focus gap bridge then interpolates across the joins."""
    by_id = {tr["id"]: tr for tr in wt["tracks"]}
    max_gap = int(STITCH_GAP_S * fps)
    used = set()
    for fid in sorted(focus & set(by_id),
                      key=lambda t: by_id[t]["frames"][0]):
        if fid < 0 or fid in used or fid not in focus:
            continue
        tr = by_id[fid]
        while tr["frames"][0] > 0:
            f0 = tr["frames"][0]
            p0 = np.array([tr["x"][0], tr["y"][0]])
            k = min(10, len(tr["frames"]) - 1)
            vel = (np.array([tr["x"][k], tr["y"][k]]) - p0) \
                / max(tr["frames"][k] - f0, 1)
            off0 = locate_on_guide(gf, [p0[0]], [p0[1]])[1][0]
            best = None
            for c in wt["tracks"]:
                cid = c["id"]
                if (cid == fid or cid in used or cid in exclude
                        or cid in parked):
                    continue
                # manual cars (negative ids) are hand-placed history for a
                # later real track: any length, any gap - the annotator
                # says the car was there
                if cid >= 0 and len(c["frames"]) < 3:
                    continue
                fc = c["frames"][-1]
                if not (0 < f0 - fc and
                        (cid < 0 or f0 - fc <= max_gap)):
                    continue
                pc = np.array([c["x"][-1], c["y"][-1]])
                if abs(locate_on_guide(gf, [pc[0]], [pc[1]])[1][0]
                       - off0) > 25:
                    continue
                # where was this car at fc? back-extrapolate its own entry
                # velocity; a jammed car predicts (essentially) in place
                d = float(np.hypot(*(pc - (p0 - vel * (f0 - fc)))))
                if d <= max(0.8 * car_len,
                            0.3 * np.hypot(*vel) * (f0 - fc)):
                    if best is None or d < best[0]:
                        best = (d, c)
            if best is None:
                break
            c = best[1]
            used.add(c["id"])
            sp = list(c["speed"])
            if c["id"] < 0 and len(sp) == 1:
                # a single-keyframe manual car carries no speed of its own;
                # inherit the entry speed of the track it becomes
                sp = [tr["speed"][0]]
            tr["frames"] = list(c["frames"]) + list(tr["frames"])
            tr["x"] = list(c["x"]) + list(tr["x"])
            tr["y"] = list(c["y"]) + list(tr["y"])
            tr["speed"] = sp + list(tr["speed"])
            px_of[fid].update(px_of[c["id"]])
            focus.discard(c["id"])
    if used:
        wt["tracks"] = [t for t in wt["tracks"] if t["id"] not in used]
        print(f"stitched {len(used)} early fragments into focus tracks")
    return used


def backfill_focus(wt, focus, car_len, v_stop, px_of, Hinv, wh, skip):
    """Hold near-stationary focus cars at their first observed position
    back to frame 0. The early oblique view misses cars deep in the queue;
    a jammed car barely moves, so where detection eventually locks on is
    where the car has been sitting all along. Cars still moving at first
    detection genuinely arrived (or merged in) and get no synthetic
    history - that is what fragment stitching and manual keyframes are
    for."""
    w, h = wh
    occ = {}    # frame -> world positions of focus cars, for dedup
    for tr in wt["tracks"]:
        if tr["id"] in focus:
            for f, x, y in zip(tr["frames"], tr["x"], tr["y"]):
                occ.setdefault(f, []).append((x, y))
    n_fill = 0
    for tr in wt["tracks"]:
        tid = tr["id"]
        f0 = tr["frames"][0]
        if tid not in focus or tid < 0 or tid in skip or f0 == 0:
            continue
        v0 = float(np.median(tr["speed"][:10]))
        if v0 > BACKFILL_V * v_stop:
            continue
        x0, y0 = tr["x"][0], tr["y"][0]
        # anchor: keep the drawn dot continuous with the detection at f0
        det = px_of[tid].get(f0)
        delta = (np.asarray(det, float)
                 - apply_h(Hinv[f0], [(x0, y0)])[0]
                 if det is not None else np.zeros(2))
        add = []
        for f in range(f0 - 1, -1, -1):
            if any((x - x0) ** 2 + (y - y0) ** 2 < (0.65 * car_len) ** 2
                   for x, y in occ.get(f, [])):
                break
            p = apply_h(Hinv[f], [(x0, y0)])[0] + delta
            if not (-40 <= p[0] <= w + 40 and -40 <= p[1] <= h + 40):
                break
            add.append((f, p))
        if not add:
            continue
        add.reverse()
        tr["frames"] = [f for f, _ in add] + list(tr["frames"])
        tr["x"] = [x0] * len(add) + list(tr["x"])
        tr["y"] = [y0] * len(add) + list(tr["y"])
        tr["speed"] = [v0] * len(add) + list(tr["speed"])
        for f, p in add:
            px_of[tid][f] = (float(p[0]), float(p[1]))
            occ.setdefault(f, []).append((x0, y0))
        n_fill += 1
    if n_fill:
        print(f"backfilled {n_fill} queued cars to their first frames")


def focus_states(speed, fps, v_stop, v_move):
    """Quantize a focus car's smoothed speed into rolling/braking/stopped.

    The continuous ramp leaves cars hovering in amber and flickering with
    measurement noise; the wave wants crisp hand-offs. Hysteresis bands
    (enter vs exit thresholds off v_stop/v_move) plus a debounce (the
    candidate state must hold STATE_DEB_S) make each car flip to solid red
    once, crisply, and not flicker back. Symmetric both ways: a car that
    rolls again walks back amber then green. Returns the state per sample;
    derive commits to stopped with overrides.stop_commits."""
    s_in, s_out = STOP_ENTER * v_stop, STOP_EXIT * v_stop
    b_in, b_out = BRAKE_ENTER * v_move, BRAKE_EXIT * v_move
    deb = max(1, int(STATE_DEB_S * fps))

    def classify(v, st):
        if st == "stopped":
            return st if v <= s_out else (
                "braking" if v <= b_out else "rolling")
        if st == "braking":
            return "stopped" if v < s_in else (
                st if v <= b_out else "rolling")
        return "stopped" if v < s_in else ("braking" if v < b_in else st)

    v0 = speed[0] if speed else 0.0
    st = "stopped" if v0 < s_in else ("braking" if v0 < b_in else "rolling")
    states = []
    pend, cnt = None, 0
    for v in speed:
        # classify against the PENDING state once a transition starts, so
        # the hysteresis band keeps it alive: entry begins past the enter
        # threshold and holds anywhere this side of the exit threshold
        tgt = classify(v, pend if pend is not None else st)
        if tgt == st:
            pend, cnt = None, 0
        elif tgt == pend:
            cnt += 1
            if cnt >= deb:
                st, pend, cnt = tgt, None, 0
        else:
            pend, cnt = tgt, 1
        states.append(st)
    return states


def onset_flips(states, flips):
    """Stop commits that earn a ring/pop. A commit rings only if the car
    has ROLLED (green) at some point since the previous commit — or ever,
    for the first one. A car that has never been green never rings: a
    start-red -> amber -> red wobble is the same stop, not an onset."""
    out, armed = [], False
    prev = 0
    for j in flips:
        if "rolling" in states[prev:j]:
            armed = True
        if armed:
            out.append(j)
            armed = False
        prev = j
    return out


def order_wave(wt, state_of, gf):
    """Make the stop front continuous IN SPACE: ordered by station along
    the lane, a car may not commit to stopped before its downstream
    neighbor. Cars measured flipping early (queue pockets that pre-date
    the clip, lumpy decel) hold braking until the front reaches them;
    their pop/ring moves with the corrected commit. Only first flips are
    ordered — a genuine restart + re-stop later is its own event.

    Flag-gated (--monotonic-wavefront) and currently known-rough: it can
    move a ring onto a sample that isn't stopped, it rewrites hand-pinned
    stopped samples, and a stale second ring index can survive the move.
    Fix before ever turning the flag on for a deliverable."""
    by_id = {tr["id"]: tr for tr in wt["tracks"]}
    rows = []
    for tid, (states, fl) in state_of.items():
        if fl:
            tr = by_id[tid]
            s, _ = locate_on_guide(gf, [tr["x"][fl[0]]],
                                   [tr["y"][fl[0]]])
            rows.append((s[0], tid))
    rows.sort()
    front = -1
    n_fix = 0
    for _, tid in rows:
        states, fl = state_of[tid]
        tr = by_id[tid]
        j = min(np.searchsorted(tr["frames"], max(front, tr["frames"][fl[0]])),
                len(states) - 1)
        front = tr["frames"][j]
        if j > fl[0]:
            for k in range(fl[0], j):
                if states[k] == "stopped":
                    states[k] = "braking"
            fl[0] = j
            n_fix += 1
    if n_fix:
        print(f"wave order: held {n_fix} early flips to the marching front")


def recover(ws):
    """The focus-lane recovery pass, shared by render/wave views/editor.

    Loads world tracks and the hand-picked lane spec, merges hand-annotated
    manual cars in as full tracks (always in the focus set), then recovers
    the cars detection loses early: chain real fragments first, refit the
    lane guide, hold already-stopped cars back to frame 0. Returns a
    namespace:

      meta, wt, roads, spec, Hs, Hinv   inputs, loaded once
      band        offset_band baseline ids (the editor stores deltas)
      focus       final focus-lane track ids (manual cars included)
      manual      the merged manual cars ({track, id, px})
      px_of       track id -> {frame: (x, y)} detected centers, extended
                  for manual/stitched/backfilled frames
      parked      hidden parked/scenery ids
      guide, gf   fitted lane polyline + densified guide frame (or None)
      stitched    fragment ids absorbed into focus tracks
      first_real  focus id -> first non-backfilled frame (editor styling)
    """
    meta = ws.meta
    fps = meta["fps"]
    wt = ws.load("world_tracks.json")
    try:
        roads = ws.load("roads.json")["roads"]
    except FileNotFoundError:
        roads = []
    try:
        spec = ws.load("focus_lane.json")
    except FileNotFoundError:
        spec = None
    px_of = {}
    for tr in ws.load("tracks.json")["tracks"]:
        px_of[tr["id"]] = {o[0]: (o[1], o[2]) for o in tr["obs"]}
    Hs = [np.asarray(m) for m in ws.load("homographies.json")["H"]]
    Hinv = [np.linalg.inv(m) for m in Hs]

    band, focus = set(), set()
    if roads and spec:
        band = focus_lane_ids(wt, roads,
                              {**spec, "include": [], "exclude": []})
        focus = focus_lane_ids(wt, roads, spec)

    # hand-annotated cars join as full tracks, always in the focus set
    manual = manual_world(ws, Hs, fps)
    for m in manual:
        wt["tracks"].append(m["track"])
        px_of[m["id"]] = m["px"]
        focus.add(m["id"])

    parked = hidden_ids(ws, wt, roads)

    guide = gf = None
    stitched, first_real = set(), {}
    if focus and roads and spec:
        guide = lane_guide_path(wt, focus, roads, spec, wt["car_len_px"])
    if guide is not None:
        stitched = stitch_focus(wt, focus, guide_frame(guide),
                                wt["car_len_px"], fps, px_of,
                                set(spec.get("exclude", [])), parked)
        guide = lane_guide_path(wt, focus, roads, spec, wt["car_len_px"])
        first_real = {tr["id"]: tr["frames"][0]
                      for tr in wt["tracks"] if tr["id"] in focus}
        backfill_focus(wt, focus, wt["car_len_px"], wt["v_stop"], px_of,
                       Hinv, (meta["width"], meta["height"]),
                       set(spec.get("no_backfill", [])))
        gf = guide_frame(guide)
    if focus:
        print(f"{len(focus)} vehicles in focus lane")
    return SimpleNamespace(
        meta=meta, wt=wt, roads=roads, spec=spec, Hs=Hs, Hinv=Hinv,
        band=band, focus=focus, manual=manual, px_of=px_of, parked=parked,
        guide=guide, gf=gf, stitched=stitched, first_real=first_real)


def displayed_states(ctx, ovs=None, monotonic_wave=False):
    """What the overlay draws, per focus car: {id: (state per sample,
    onset ring sample indices)}.

    Engine quantization first, then hand overrides (sticky from each pin's
    frame until the engine's next change), then rings fire on
    onset-filtered commits of the DISPLAYED sequence, so ring + pop always
    coincide with the visible flip. monotonic_wave optionally holds early
    flips to the marching front (see order_wave's caveats).
    """
    fps = ctx.meta["fps"]
    wt = ctx.wt
    ovs = ovs or {}
    state_of = {}
    for tr in wt["tracks"]:
        if tr["id"] in ctx.focus:
            states = focus_states(tr["speed"], fps,
                                  wt["v_stop"], wt["v_move"])
            overrides.apply(states, tr["frames"], ovs.get(tr["id"]))
            flips = onset_flips(states, overrides.stop_commits(states))
            state_of[tr["id"]] = (states, flips)
    if state_of and monotonic_wave and ctx.gf is not None:
        order_wave(wt, state_of, ctx.gf)
    return state_of
