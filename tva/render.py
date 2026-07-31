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
# the focus-lane pipeline (recovery, state quantization, ring params) lives
# in focus.py; re-exported here so existing `from .render import X` users
# keep working
from .focus import (RING_FADE, RING_R0, RING_R1, RING_S, STATE_RGB,
                    backfill_focus, displayed_states, focus_lane_ids,
                    focus_states, guide_frame, hidden_ids, lane_guide_path,
                    locate_on_guide, manual_world, onset_flips, order_wave,
                    recover, stitch_focus)
from .stabilize import apply_h
from .viz import speed_color

TRAIL_S = 1.2          # seconds of path behind each car
LAYER_ALPHA = 0.8      # overlay blend
GAP_FILL_S = 0.6       # bridge detection dropouts up to this long
R_MAX = 40             # marker radius cap (src-frame px)
TRAIL_MIN_LEN = 0.5    # car_len; shorter world paths draw no trail
TRAIL_STRAIGHT = 0.8   # net/path ratio below this = registration curl
                       # (a real 90-degree turn over TRAIL_S is ~0.90)
FOCUS_DOT_R = 29       # focus-lane marker radius (src-frame px), uniform:
                       # state/color is the story, not apparent vehicle size
FOCUS_RING_W = 5       # focus-lane white outline thickness
FOCUS_ALPHA = 0.75     # loud-dot outline/trail opacity
FOCUS_FILL = 0.68      # inner fill opacity: the vehicle shows through a
                       # touch more than the rim does
CORRIDOR_HW = 48       # focus-lane corridor minimum half-width (world px)
CORRIDOR_PAD = 15      # corridor clearance beyond the outermost dot centers:
                       # edges hug the loud-dot envelope, minimal dead space
CORRIDOR_EXT = 2500    # end extension so the edges always exit the frame
CORRIDOR_DIM = 0.55    # brightness outside the corridor (1.0 = no dim)
POP_S = 0.4            # dot pop duration when a focus car commits to stopped
POP_SCALE = 1.4        # peak pop radius multiplier

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


def run(ws, out=None, out_w=1920, layers=("cars",), highlight=(),
        monotonic_wave=False):
    meta = ws.meta
    fps, n = meta["fps"], meta["n_frames"]
    w, h = meta["width"], meta["height"]
    out = out or ws.path("overlay-speed.mp4")
    # focus-lane recovery (manual cars, fragment stitching, queued-car
    # backfill) is shared with the wave views and the lane editor via
    # focus.recover, so every consumer draws the same dots
    ctx = recover(ws)
    wt = ctx.wt
    Hinv = ctx.Hinv
    v_stop, v_move = wt["v_stop"], wt["v_move"]
    car_len = wt["car_len_px"]
    roads = ctx.roads
    roads_draw = "roads" in layers
    if roads_draw and not roads:
        print("roads.json missing - run `tva roads` first; skipping layer")
    focus = ctx.focus
    parked = ctx.parked
    stitched = ctx.stitched
    px_of = ctx.px_of

    # per-track median bbox size (frame px) for marker radius; markers are
    # drawn on the DETECTION, never on the smoothed world estimate (which
    # coasts away from the car wherever the projection is too distorted to
    # trust measurements)
    size_of = {tr["id"]: float(np.median([max(o[3], o[4])
                                          for o in tr["obs"]]))
               for tr in ws.load("tracks.json")["tracks"]}
    for m in ctx.manual:
        size_of[m["id"]] = 80

    # spotlight corridor bracketing the loud dots, sized off their actual
    # offsets: 1st/99th percentile, not true extremes, so one stray loud
    # dot (an adjacent-lane car passing the offset_band test) can't stretch
    # the corridor for the whole clip; a 1% trim is ~100 samples here, far
    # more than any single fragment carries
    corridor = None
    if ctx.guide is not None:
        offs = np.concatenate(
            [locate_on_guide(ctx.gf, tr["x"], tr["y"])[1]
             for tr in wt["tracks"] if tr["id"] in focus])
        corridor = lane_corridor(
            ctx.guide,
            min(-CORRIDOR_HW, np.percentile(offs, 1) - CORRIDOR_PAD),
            max(CORRIDOR_HW, np.percentile(offs, 99) + CORRIDOR_PAD))

    # quantized rolling/braking/stopped per focus car: engine states, hand
    # overrides sticky until the engine's next change, rings on
    # onset-filtered commits of the DISPLAYED sequence - the shared
    # focus.py pipeline, so the wave views and editor agree with the render
    state_of = {}
    if focus:
        ovs = overrides.load(ws)
        if ovs:
            print(f"{sum(map(len, ovs.values()))} hand state overrides")
        state_of = displayed_states(ctx, ovs, monotonic_wave)
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
        # white outline, heavier trail - outline/trail at FOCUS_ALPHA,
        # inner fill one notch more transparent (FOCUS_FILL)
        if focus_ops:
            ov = frame.copy()
            for fpts, col, r, trail_ok in focus_ops:
                if trail_ok:
                    cv2.polylines(ov, [fpts], False, col, max(3, r // 3),
                                  cv2.LINE_AA)
                cv2.circle(ov, tuple(fpts[-1]), r + FOCUS_RING_W // 2 + 1,
                           (255, 255, 255), FOCUS_RING_W, cv2.LINE_AA)
            frame = cv2.addWeighted(ov, FOCUS_ALPHA, frame,
                                    1 - FOCUS_ALPHA, 0)
            ov = frame.copy()
            for fpts, col, r, trail_ok in focus_ops:
                cv2.circle(ov, tuple(fpts[-1]), r, col, -1, cv2.LINE_AA)
            frame = cv2.addWeighted(ov, FOCUS_FILL, frame,
                                    1 - FOCUS_FILL, 0)
        for f0, ex, ey in focus_rings:
            if f0 <= i < f0 + ring_n:
                age = (i - f0) / ring_n
                p = apply_h(Hinv[i], [(ex, ey)])[0].astype(int)
                rr = int(RING_R0 + RING_R1 * age)
                c = int(255 * (1 - RING_FADE * age))
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
