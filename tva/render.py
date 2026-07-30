"""Render analysis back onto the source video.

Per tracked vehicle: a speed-colored dot (red=stopped, green=free flow) with
a short trail of its actual world-plane path mapped into the current camera
view (trails stay glued to the road under pan/rotation). Stop-onset events
flash an expanding white ring — watching the rings march upstream IS the
shockwave. Output h264 via an ffmpeg rawvideo pipe.
"""
import subprocess

import cv2
import numpy as np

from . import overrides
from .stabilize import apply_h
from .viz import speed_color

TRAIL_S = 1.2          # seconds of path behind each car
RING_S = 0.7           # stop-onset ring duration
LAYER_ALPHA = 0.8      # overlay blend
GAP_FILL_S = 0.6       # bridge detection dropouts up to this long
R_MAX = 40             # marker radius cap (src-frame px)
TRAIL_MIN_LEN = 0.5    # car_len; shorter world paths draw no trail
TRAIL_STRAIGHT = 0.8   # net/path ratio below this = registration curl
                       # (a real 90-degree turn over TRAIL_S is ~0.90)
FOCUS_DOT_R = 29       # focus-lane marker radius (src-frame px), uniform:
                       # state/color is the story, not apparent vehicle size
FOCUS_RING_W = 5       # focus-lane white outline thickness
FOCUS_ALPHA = 0.75     # loud-dot opacity: the vehicle stays visible under it
CORRIDOR_HW = 48       # focus-lane corridor minimum half-width (world px)
CORRIDOR_PAD = 15      # corridor clearance beyond the outermost dot centers:
                       # edges hug the loud-dot envelope, minimal dead space
CORRIDOR_EXT = 2500    # end extension so the edges always exit the frame
CORRIDOR_DIM = 0.55    # brightness outside the corridor (1.0 = no dim)
STITCH_GAP_S = 8       # max dropout when chaining early fragments
BACKFILL_V = 1.5       # x v_stop: "already stopped at first detection"
STOP_ENTER = 1.0       # x v_stop: candidate stopped below this
STOP_EXIT = 1.8        # x v_stop: leave stopped only above this, sustained
BRAKE_ENTER = 1.7      # x v_move: candidate braking below this
BRAKE_EXIT = 2.2       # x v_move: back to rolling only above this
STATE_DEB_S = 0.2      # a new state must hold this long to commit
POP_S = 0.4            # dot pop duration when a focus car commits to stopped
POP_SCALE = 1.4        # peak pop radius multiplier
# quantized focus-lane states; RGB matching the continuous ramp's anchors
STATE_RGB = {"rolling": (60, 200, 90), "braking": (235, 170, 50),
             "stopped": (235, 60, 50)}

# per-road hues (BGR), cycled
ROAD_COLORS = [(255, 91, 46), (255, 200, 0), (180, 0, 255), (0, 200, 255),
               (0, 255, 140), (255, 128, 128)]


def road_geometry(road):
    """World-plane polylines to draw for one road: (kind, pts) with kind in
    edge|divider, plus chevron segments along the centerline."""
    cx, cy = np.asarray(road["x"]), np.asarray(road["y"])
    tang = np.gradient(np.column_stack([cx, cy]), axis=0)
    tang /= np.maximum(np.hypot(tang[:, 0], tang[:, 1])[:, None], 1e-9)
    nrm = np.column_stack([-tang[:, 1], tang[:, 0]])
    lanes = road["lanes"]
    hw = road["lane_halfwidth"]
    lines = []
    bounds = road.get("boundaries")
    if bounds and len(bounds) >= 2:
        # measured painted lines: outermost are edges, the rest dividers
        for kind, off in ([("edge", bounds[0]), ("edge", bounds[-1])] +
                          [("divider", b) for b in bounds[1:-1]]):
            lines.append((kind, np.column_stack([cx, cy]) + nrm * off))
    elif len(lanes) > 1:
        for kind, off in (
                [("edge", lanes[0] - hw), ("edge", lanes[-1] + hw)] +
                [("divider", (a + b) / 2) for a, b in zip(lanes, lanes[1:])]):
            lines.append((kind, np.column_stack([cx, cy]) + nrm * off))
    else:
        # lanes unresolved: draw just the centerline
        lines.append(("divider", np.column_stack([cx, cy]) + nrm * lanes[0]))
    # direction chevrons on the centerline every ~4 vertices
    chevrons = []
    for i in range(2, len(cx) - 2, 4):
        tip = np.array([cx[i], cy[i]]) + tang[i] * 14
        b1 = np.array([cx[i], cy[i]]) - tang[i] * 10 + nrm[i] * 10
        b2 = np.array([cx[i], cy[i]]) - tang[i] * 10 - nrm[i] * 10
        chevrons.append(np.array([b1, tip, b2]))
    return lines, chevrons


def draw_roads(layer, roads, Hinv_i):
    for road in roads:
        col = ROAD_COLORS[road["id"] % len(ROAD_COLORS)]
        lines, chevrons = road_geometry(road)
        for kind, wpts in lines:
            fpts = apply_h(Hinv_i, wpts).astype(np.int32)
            cv2.polylines(layer, [fpts], False, col,
                          4 if kind == "edge" else 2, cv2.LINE_AA)
        for ch in chevrons:
            fpts = apply_h(Hinv_i, ch).astype(np.int32)
            cv2.polylines(layer, [fpts], False, col, 3, cv2.LINE_AA)


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


def lane_corridor(guide, lo=-CORRIDOR_HW, hi=CORRIDOR_HW):
    """Two world-plane edge polylines bracketing the focus lane: the
    fitted guide centerline offset to lo/hi, ends extended along their
    tangents far enough to always exit the frame."""
    if guide is None or len(guide) < 2:
        return None
    a = min(5, len(guide) - 1)
    t0 = guide[0] - guide[a]
    t1 = guide[-1] - guide[-1 - a]
    t0 /= max(np.hypot(*t0), 1e-9)
    t1 /= max(np.hypot(*t1), 1e-9)
    g = np.vstack([guide[0] + t0 * CORRIDOR_EXT, guide,
                   guide[-1] + t1 * CORRIDOR_EXT])
    tang = np.gradient(g, axis=0)
    tang /= np.maximum(np.hypot(tang[:, 0], tang[:, 1])[:, None], 1e-9)
    nrm = np.column_stack([-tang[:, 1], tang[:, 0]])
    return g + nrm * lo, g + nrm * hi


def neon(col):
    """Push a BGR color to full saturation/value: the focus-lane variant of
    the speed ramp (same red=stopped/green=moving semantics, louder)."""
    px = np.uint8([[col]])
    hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)
    hsv[0, 0, 1:] = 255
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def focus_states(speed, fps, v_stop, v_move):
    """Quantize a focus car's smoothed speed into rolling/braking/stopped.

    The continuous ramp leaves cars hovering in amber and flickering with
    measurement noise; the wave wants crisp hand-offs. Hysteresis bands
    (enter vs exit thresholds off v_stop/v_move) plus a debounce (the
    candidate state must hold STATE_DEB_S) make each car flip to solid red
    once, crisply, and not flicker back. Symmetric both ways: a car that
    rolls again walks back amber then green. Returns (state per sample,
    sample indices where the state commits to stopped)."""
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
    states, flips = [], []
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
                if tgt == "stopped":
                    flips.append(len(states))
                st, pend, cnt = tgt, None, 0
        else:
            pend, cnt = tgt, 1
        states.append(st)
    return states, flips


def onset_flips(states, flips):
    """Stop commits that earn a ring/pop: the first, and any later one only
    if the car has rolled (green) again since the previous commit. A
    red->amber->red wobble is the same stop, not a new onset."""
    out, armed = [], True
    for k, j in enumerate(flips):
        if armed:
            out.append(j)
        end = flips[k + 1] if k + 1 < len(flips) else len(states)
        armed = "rolling" in states[j:end]
    return out


def order_wave(wt, state_of, gf):
    """Make the stop front continuous IN SPACE: ordered by station along
    the lane, a car may not commit to stopped before its downstream
    neighbor. Cars measured flipping early (queue pockets that pre-date
    the clip, lumpy decel) hold braking until the front reaches them;
    their pop/ring moves with the corrected commit. Only first flips are
    ordered — a genuine restart + re-stop later is its own event."""
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


def run(ws, out=None, out_w=1920, layers=("cars",), highlight=(),
        monotonic_wave=False):
    meta = ws.meta
    fps, n = meta["fps"], meta["n_frames"]
    w, h = meta["width"], meta["height"]
    out = out or ws.path("overlay-speed.mp4")
    Hs = [np.asarray(m) for m in ws.load("homographies.json")["H"]]
    Hinv = [np.linalg.inv(m) for m in Hs]
    wt = ws.load("world_tracks.json")
    v_stop, v_move = wt["v_stop"], wt["v_move"]
    car_len = wt["car_len_px"]
    # roads.json is loaded even when the layer is off: parked culling needs it
    roads_draw = "roads" in layers
    try:
        roads = ws.load("roads.json")["roads"]
    except FileNotFoundError:
        roads = []
        if roads_draw:
            print("roads.json missing - run `tva roads` first; skipping layer")

    # per-track median bbox size (frame px) for marker radius, and observed
    # centers: markers are drawn on the DETECTION, never on the smoothed
    # world estimate (which coasts away from the car wherever the projection
    # is too distorted to trust measurements)
    size_of, px_of = {}, {}
    for tr in ws.load("tracks.json")["tracks"]:
        size_of[tr["id"]] = float(np.median([max(o[3], o[4])
                                             for o in tr["obs"]]))
        px_of[tr["id"]] = {o[0]: (o[1], o[2]) for o in tr["obs"]}

    # focus_lane.json: hand-picked lane whose cars get the loud styling so
    # the stop wave marching up one lane is impossible to miss
    spec = None
    try:
        spec = ws.load("focus_lane.json")
        focus = focus_lane_ids(wt, roads, spec) if roads else set()
    except FileNotFoundError:
        focus = set()

    # hand-annotated cars join as full tracks, always in the focus set
    for m in manual_world(ws, Hs, fps):
        wt["tracks"].append(m["track"])
        px_of[m["id"]] = m["px"]
        size_of[m["id"]] = 80
        focus.add(m["id"])

    parked = hidden_ids(ws, wt, roads)

    corridor = None
    stitched = set()
    guide = (lane_guide_path(wt, focus, roads, spec, car_len)
             if focus and roads and spec else None)
    if guide is not None:
        # recover the cars detection loses early: chain real fragments
        # first, then hold already-stopped cars back to frame 0, then refit
        # the guide and size the corridor off the actual dot offsets
        stitched = stitch_focus(wt, focus, guide_frame(guide), car_len, fps,
                                px_of, set(spec.get("exclude", [])), parked)
        guide = lane_guide_path(wt, focus, roads, spec, car_len)
        backfill_focus(wt, focus, car_len, v_stop, px_of, Hinv, (w, h),
                       set(spec.get("no_backfill", [])))
        gf = guide_frame(guide)
        offs = np.concatenate(
            [locate_on_guide(gf, tr["x"], tr["y"])[1]
             for tr in wt["tracks"] if tr["id"] in focus])
        # 1st/99th percentile, not true extremes: one stray loud dot (an
        # adjacent-lane car passing the offset_band test) would otherwise
        # stretch the corridor for the whole clip; a 1% trim is ~100
        # samples here, far more than any single fragment carries
        corridor = lane_corridor(
            guide,
            min(-CORRIDOR_HW, np.percentile(offs, 1) - CORRIDOR_PAD),
            max(CORRIDOR_HW, np.percentile(offs, 99) + CORRIDOR_PAD))
    if focus:
        print(f"{len(focus)} vehicles in focus lane")
        # quantize each focus car to rolling/braking/stopped AFTER the
        # recovery passes, so states line up with the final speed arrays;
        # hand overrides stick until the engine's next change, and
        # rings/pops fire on onset flips of the DISPLAYED sequence
        ovs = overrides.load(ws)
        if ovs:
            print(f"{sum(map(len, ovs.values()))} hand state overrides")
        state_of = {}
        for tr in wt["tracks"]:
            if tr["id"] in focus:
                states, _ = focus_states(tr["speed"], fps, v_stop, v_move)
                overrides.apply(states, tr["frames"], ovs.get(tr["id"]))
                flips = onset_flips(states, overrides.stop_commits(states))
                state_of[tr["id"]] = (states, flips)
        if guide is not None and monotonic_wave:
            order_wave(wt, state_of, gf)
    pop_n = max(1, int(POP_S * fps))
    # focus onset rings ride the DISPLAYED commits to stopped (state machine
    # plus hand overrides), so ring + pop always coincide with the visible
    # flip. kinematics stop_events predate stitching/backfill: manual cars
    # have none, stitched cars kept theirs under the discarded fragment id —
    # those events are dropped from the quiet rings below since the focus
    # ring now speaks for them
    focus_rings = []
    for tr in wt["tracks"]:
        if focus and tr["id"] in state_of:
            for j in state_of[tr["id"]][1]:
                focus_rings.append(
                    (tr["frames"][j], tr["x"][j], tr["y"][j]))

    # frame index -> [(track, obs index)]; fractional index = interpolated
    # frame inside a short detection dropout (dots persist through flicker)
    gap_max = int(GAP_FILL_S * fps)
    per_frame = [[] for _ in range(n)]
    for tr in wt["tracks"]:
        if tr["id"] in parked:
            continue
        F = tr["frames"]
        for k, f in enumerate(F):
            if f < n:
                per_frame[f].append((tr, float(k)))
        # focus cars bridge any dropout (a jammed car barely moves, so the
        # interpolation is safe); others only short flicker
        gm = 10 ** 9 if tr["id"] in focus else gap_max
        for k in range(len(F) - 1):
            g = F[k + 1] - F[k]
            if 1 < g <= gm:
                for f in range(F[k] + 1, min(F[k + 1], n)):
                    per_frame[f].append((tr, k + (f - F[k]) / g))
    rings = [(ev, ev["frame"]) for ev in wt["stop_events"]
             if ev["track"] not in stitched]
    trail_n = int(TRAIL_S * fps)
    ring_n = max(1, int(RING_S * fps))

    out_h = int(h * out_w / w)
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{out_w}x{out_h}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-crf", "18", "-preset", "medium",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", out],
        stdin=subprocess.PIPE)

    cap = cv2.VideoCapture(meta["src"])
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        layer = frame.copy()
        focus_ops = []          # focus-lane draws go on top of the blend
        if roads_draw and roads:
            draw_roads(layer, roads, Hinv[i])
        for tr, k in per_frame[i] if "cars" in layers else []:
            k0 = int(k)
            frac = k - k0
            if frac:
                v = tr["speed"][k0] * (1 - frac) + tr["speed"][k0 + 1] * frac
            else:
                v = tr["speed"][k0]
            col = speed_color(v, v_stop, v_move)[::-1]  # RGB -> BGR
            hot = tr["id"] in highlight
            r = min(max(6, int(0.30 * size_of.get(tr["id"], 80))), R_MAX)
            if hot:
                r = int(r * 1.8)
            # world path over the last TRAIL_S, drawn in this frame's view,
            # translated so it ends exactly on the detected bbox center
            j0 = max(0, k0 - trail_n)
            wpts = np.column_stack([tr["x"][j0:k0 + 1], tr["y"][j0:k0 + 1]])
            if frac:
                tip = wpts[-1] * (1 - frac) + frac * np.array(
                    [tr["x"][k0 + 1], tr["y"][k0 + 1]])
                wpts = np.vstack([wpts, tip])
            fpts = apply_h(Hinv[i], wpts)
            obs_px = px_of[tr["id"]].get(i)
            if obs_px is None and frac:
                # inside a bridged dropout: interpolate the detected centers
                F = tr["frames"]
                p1 = np.asarray(px_of[tr["id"]][F[k0]], float)
                p2 = np.asarray(px_of[tr["id"]][F[k0 + 1]], float)
                obs_px = p1 * (1 - frac) + p2 * frac
            if obs_px is not None:
                fpts += np.asarray(obs_px) - fpts[-1]
            fpts = fpts.astype(np.int32)
            # trail only when the world path is path-like: far-field
            # registration wobble shows up as curls (low net/path ratio)
            seg = np.diff(wpts, axis=0)
            plen = float(np.hypot(seg[:, 0], seg[:, 1]).sum())
            net = float(np.hypot(*(wpts[-1] - wpts[0])))
            trail_ok = (len(fpts) > 1 and plen > TRAIL_MIN_LEN * car_len
                        and net > TRAIL_STRAIGHT * plen)
            if tr["id"] in focus:
                states, flips = state_of[tr["id"]]
                fcol = neon(STATE_RGB[states[k0]][::-1])
                fr = FOCUS_DOT_R
                for j in flips:     # pop on commit to stopped (sin pulse)
                    age = (i - tr["frames"][j]) / pop_n
                    if 0 <= age < 1:
                        fr = int(fr * (1 + (POP_SCALE - 1)
                                       * np.sin(np.pi * age)))
                        break
                focus_ops.append((fpts, fcol, fr, trail_ok))
                continue
            if trail_ok:
                cv2.polylines(layer, [fpts], False, col, max(2, r // 4),
                              cv2.LINE_AA)
            if hot:
                cv2.circle(layer, tuple(fpts[-1]), r + 4, (255, 255, 255),
                           3, cv2.LINE_AA)
            cv2.circle(layer, tuple(fpts[-1]), r, col, -1, cv2.LINE_AA)
        for ev, f0 in rings:
            if f0 <= i < f0 + ring_n and ev["track"] not in focus:
                age = (i - f0) / ring_n
                p = apply_h(Hinv[i], [(ev["x"], ev["y"])])[0].astype(int)
                rr = int(30 + 90 * age)
                c = int(255 * (1 - 0.6 * age))
                cv2.circle(layer, tuple(p), rr, (c, c, c), 4, cv2.LINE_AA)
        frame = cv2.addWeighted(layer, LAYER_ALPHA, frame,
                                1 - LAYER_ALPHA, 0)
        # spotlight corridor: everything outside the lane (video AND the
        # quiet overlays) dulls, complexity stays legible underneath;
        # edge lines bracket the loud dots so the lane reads as a shape
        if corridor is not None:
            e1 = apply_h(Hinv[i], corridor[0])
            e2 = apply_h(Hinv[i], corridor[1])
            poly = np.vstack([e1, e2[::-1]]).astype(np.int32)
            mask = np.zeros(frame.shape[:2], np.uint8)
            cv2.fillPoly(mask, [poly], 255)
            mask = cv2.blur(mask, (81, 81))
            a = CORRIDOR_DIM + (1 - CORRIDOR_DIM) * (
                mask.astype(np.float32) / 255)
            frame = (frame.astype(np.float32) * a[..., None]) \
                .astype(np.uint8)
            for e in (e1, e2):
                cv2.polylines(frame, [e.astype(np.int32)], False,
                              (235, 235, 235), 3, cv2.LINE_AA)
        # focus lane on top, undimmed: neon speed color, bigger, thick
        # white outline, heavier trail - blended at FOCUS_ALPHA so the
        # vehicle underneath stays visible
        if focus_ops:
            ov = frame.copy()
            for fpts, col, r, trail_ok in focus_ops:
                if trail_ok:
                    cv2.polylines(ov, [fpts], False, col, max(3, r // 3),
                                  cv2.LINE_AA)
                cv2.circle(ov, tuple(fpts[-1]), r + FOCUS_RING_W // 2 + 1,
                           (255, 255, 255), FOCUS_RING_W, cv2.LINE_AA)
                cv2.circle(ov, tuple(fpts[-1]), r, col, -1, cv2.LINE_AA)
            frame = cv2.addWeighted(ov, FOCUS_ALPHA, frame,
                                    1 - FOCUS_ALPHA, 0)
        for f0, ex, ey in focus_rings:
            if f0 <= i < f0 + ring_n:
                age = (i - f0) / ring_n
                p = apply_h(Hinv[i], [(ex, ey)])[0].astype(int)
                rr = int(40 + 120 * age)
                c = int(255 * (1 - 0.5 * age))
                cv2.circle(frame, tuple(p), rr, (c, c, 255), 7, cv2.LINE_AA)
        small = cv2.resize(frame, (out_w, out_h))
        enc.stdin.write(small.tobytes())
        if i % int(3 * fps) == 0:
            cv2.imwrite(f"{ws.qa}/render-t{int(i / fps)}.jpg", small,
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
        if i % 100 == 0:
            print(f"frame {i}/{n}", flush=True)
    cap.release()
    enc.stdin.close()
    enc.wait()
    print(f"wrote {out}")
