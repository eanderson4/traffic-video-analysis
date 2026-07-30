"""Diagnostic + presentation renders from world-plane artifacts."""
import numpy as np
from PIL import Image, ImageDraw

from .kinematics import station_of


def speed_color(v, v_stop, v_move):
    """Red (stopped) -> amber -> green (free flow)."""
    if v <= v_stop:
        return (235, 60, 50)
    if v >= 2.5 * v_move:
        return (60, 200, 90)
    f = (v - v_stop) / (2.5 * v_move - v_stop)
    if f < 0.5:
        g = f * 2
        return (235, int(60 + g * (170 - 60)), 50)
    g = (f - 0.5) * 2
    return (int(235 + g * (60 - 235)), int(170 + g * (200 - 170)),
            int(50 + g * (90 - 50)))


def spacetime(ws, out_w=1400, out_h=1000):
    """Space-time diagram: station (px along road) vs time, one polyline per
    track, colored by speed. Shockwaves appear as diagonal red/green
    boundaries."""
    meta = ws.meta
    fps = meta["fps"]
    wt = ws.load("world_tracks.json")
    cl = ws.load("centerline.json")
    v_stop, v_move = wt["v_stop"], wt["v_move"]
    max_lat = 3.0 * wt["car_len_px"]  # keep cars near the spine only

    img = Image.new("RGB", (out_w, out_h), (16, 18, 24))
    d = ImageDraw.Draw(img)
    t_max = meta["n_frames"] / fps
    s_max = cl["length_px"]

    def X(t):
        return 70 + (out_w - 100) * t / t_max

    def Y(s):
        return out_h - 60 - (out_h - 110) * s / s_max

    kept = 0
    for tr in wt["tracks"]:
        px, py = np.array(tr["x"]), np.array(tr["y"])
        s, lat = station_of(cl["x"], cl["y"], px, py)
        if np.median(lat) > max_lat:
            continue
        # drop opposite-direction traffic (net motion against the spine)
        if s[-1] - s[0] < -wt["car_len_px"]:
            continue
        kept += 1
        ts = np.array(tr["frames"]) / fps
        for i in range(len(ts) - 1):
            d.line([(X(ts[i]), Y(s[i])), (X(ts[i + 1]), Y(s[i + 1]))],
                   fill=speed_color(tr["speed"][i], v_stop, v_move), width=3)

    for ev in wt["stop_events"]:
        s, lat = station_of(cl["x"], cl["y"],
                            np.array([ev["x"]]), np.array([ev["y"]]))
        if lat[0] > max_lat:
            continue
        x, y = X(ev["t"]), Y(s[0])
        d.ellipse([x - 6, y - 6, x + 6, y + 6], outline=(255, 255, 255),
                  width=2)

    d.text((70, 12), f"space-time  ({kept} tracks near centerline; "
           "circles = stop onset; red=stopped green=moving)",
           fill=(220, 220, 220))
    d.text((10, out_h // 2), "station ->", fill=(160, 160, 160))
    d.text((out_w // 2, out_h - 40), "time ->", fill=(160, 160, 160))
    out = f"{ws.qa}/spacetime.png"
    img.save(out)
    print(f"wrote {out}")


def _focus_rows(ws, monotonic_wave=False):
    """Focus-lane recovery + state pipeline shared by the wave QA outputs.

    Mirrors the render exactly (stitch, backfill, quantized states, and the
    optional monotonic-wavefront pass), so these views show the states the
    overlay draws. Returns (meta, rows, s_lo, s_hi) with rows =
    (track, station array, states, flip sample indices) per focus car.
    """
    from .render import (backfill_focus, focus_lane_ids, focus_states,
                         guide_frame, hidden_ids, lane_guide_path,
                         locate_on_guide, manual_world, onset_flips,
                         order_wave, stitch_focus)

    meta = ws.meta
    fps = meta["fps"]
    wt = ws.load("world_tracks.json")
    v_stop, v_move = wt["v_stop"], wt["v_move"]
    car_len = wt["car_len_px"]
    roads = ws.load("roads.json")["roads"]
    spec = ws.load("focus_lane.json")
    focus = focus_lane_ids(wt, roads, spec)
    px_of = {}
    for tr in ws.load("tracks.json")["tracks"]:
        px_of[tr["id"]] = {o[0]: (o[1], o[2]) for o in tr["obs"]}
    Hs = [np.asarray(m) for m in ws.load("homographies.json")["H"]]
    for m in manual_world(ws, Hs, fps):
        wt["tracks"].append(m["track"])
        px_of[m["id"]] = m["px"]
        focus.add(m["id"])
    parked = hidden_ids(ws, wt, roads)
    guide = lane_guide_path(wt, focus, roads, spec, car_len)
    stitch_focus(wt, focus, guide_frame(guide), car_len, fps, px_of,
                 set(spec.get("exclude", [])), parked)
    guide = lane_guide_path(wt, focus, roads, spec, car_len)
    Hinv = [np.linalg.inv(m) for m in Hs]
    backfill_focus(wt, focus, car_len, v_stop, px_of, Hinv,
                   (meta["width"], meta["height"]),
                   set(spec.get("no_backfill", [])))
    gf = guide_frame(guide)
    state_of = {}
    for tr in wt["tracks"]:
        if tr["id"] in focus:
            states, flips = focus_states(tr["speed"], fps, v_stop, v_move)
            state_of[tr["id"]] = (states, onset_flips(states, flips))
    if monotonic_wave:
        order_wave(wt, state_of, gf)

    rows = []
    s_lo, s_hi = np.inf, -np.inf
    for tr in wt["tracks"]:
        if tr["id"] not in focus:
            continue
        s, _ = locate_on_guide(gf, tr["x"], tr["y"])
        states, flips = state_of[tr["id"]]
        rows.append((tr, s, states, flips))
        s_lo, s_hi = min(s_lo, s.min()), max(s_hi, s.max())
    return meta, rows, s_lo, s_hi


def focus_wave(ws, out_w=1800, out_h=1000, monotonic_wave=False):
    """Focus-lane wave diagram: the loud dots laid out in 1D.

    x = time, y = station along the fitted lane guide; every sample of every
    focus track is one dot colored by its quantized rolling/braking/stopped
    state, white circles mark commits to stopped. A clean shockwave reads as
    a single diagonal red boundary marching upstream; noise reads as
    speckle off the boundary.
    """
    from .render import STATE_RGB

    meta, rows, s_lo, s_hi = _focus_rows(ws, monotonic_wave)
    fps, n = meta["fps"], meta["n_frames"]

    img = Image.new("RGB", (out_w, out_h), (16, 18, 24))
    d = ImageDraw.Draw(img)
    t_max = n / fps

    def X(f):
        return 70 + (out_w - 100) * (f / fps) / t_max

    def Y(s):
        return out_h - 60 - (out_h - 110) * (s - s_lo) / (s_hi - s_lo)

    for tr, s, states, flips in rows:
        for k, f in enumerate(tr["frames"]):
            if f >= n:
                continue
            x, y = X(f), Y(s[k])
            d.ellipse([x - 2, y - 2, x + 2, y + 2],
                      fill=STATE_RGB[states[k]])
        for j in flips:
            x, y = X(tr["frames"][j]), Y(s[j])
            d.ellipse([x - 6, y - 6, x + 6, y + 6], outline=(255, 255, 255),
                      width=2)
    d.text((70, 12), f"focus-lane wave ({len(rows)} tracks; "
           "y = station along lane, circles = commit to stopped)",
           fill=(220, 220, 220))
    d.text((10, out_h // 2), "station ->", fill=(160, 160, 160))
    d.text((out_w // 2, out_h - 40), "time (s) ->", fill=(160, 160, 160))
    out = f"{ws.qa}/focus-wave.png"
    img.save(out)
    print(f"wrote {out}")


def wave_video(ws, out_w=1920, out_h=260, monotonic_wave=False):
    """1D strip video: the loud dots on the straightened lane, playing in
    real time. x = station (downstream/front of queue at left, upstream at
    right): cars travel right-to-left, the stop wave marches left-to-right.
    Dot color = quantized state, white rings = commit to stopped."""
    import subprocess

    from .render import STATE_RGB

    meta, rows, s_lo, s_hi = _focus_rows(ws, monotonic_wave)
    fps, n = meta["fps"], meta["n_frames"]
    mid = out_h // 2 + 10
    rad = 10
    ring_n = max(1, int(0.5 * fps))

    def X(s):
        return 60 + (out_w - 120) * (s - s_lo) / (s_hi - s_lo)

    out = ws.path("focus-wave.mp4")
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{out_w}x{out_h}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-crf", "18", "-preset", "medium",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", out],
        stdin=subprocess.PIPE)
    for i in range(n):
        img = Image.new("RGB", (out_w, out_h), (16, 18, 24))
        d = ImageDraw.Draw(img)
        d.line([(X(s_lo), mid), (X(s_hi), mid)], fill=(60, 64, 72), width=2)
        for tr, s, states, flips in rows:
            F = tr["frames"]
            if i < F[0] or i > F[-1]:
                continue
            k = min(np.searchsorted(F, i, "right") - 1, len(states) - 1)
            x = X(s[k])
            d.ellipse([x - rad, mid - rad, x + rad, mid + rad],
                      fill=STATE_RGB[states[k]])
            for j in flips:
                age = i - F[j]
                if 0 <= age < ring_n:
                    rr = rad + 4 + 26 * age / ring_n
                    d.ellipse([x - rr, mid - rr, x + rr, mid + rr],
                              outline=(255, 255, 255), width=3)
        d.text((60, 16), f"t = {i / fps:5.2f}s", fill=(220, 220, 220))
        d.text((X(s_lo) - 20, out_h - 32), "downstream (front of queue)",
               fill=(160, 160, 160))
        d.text((X(s_hi) - 150, out_h - 32), "upstream ->",
               fill=(160, 160, 160))
        enc.stdin.write(np.asarray(img).tobytes())
    enc.stdin.close()
    enc.wait()
    print(f"wrote {out}")


def speed_qa(ws, n_tracks=10, track_ids=None, suffix=""):
    """Raw vs smoothed speed profiles for tracks with stop events.

    Grey = central-difference speed of the raw projected bbox centers (what
    the smoother has to work with); color = the smoothed speed we publish.
    If the color curve still wiggles like the grey one, the smoother is too
    loose; if it lags the stop, too stiff. Green dots = |z_v| LK-flow
    velocity measurements when flow.json exists.

    track_ids: explicit list of ids to plot (default: longest tracks that
    have stop events). suffix goes into the output filename.
    """
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .stabilize import apply_h

    fps = ws.meta["fps"]
    Hs = [np.asarray(h) for h in ws.load("homographies.json")["H"]]
    wt = ws.load("world_tracks.json")
    v_stop, v_move = wt["v_stop"], wt["v_move"]
    ev_of = {}
    for ev in wt["stop_events"]:
        ev_of.setdefault(ev["track"], []).append(ev["t"])
    raw_obs = {tr["id"]: tr["obs"] for tr in ws.load("tracks.json")["tracks"]}
    flow = {}
    if os.path.exists(ws.path("flow.json")):
        flow = {tr["id"]: tr for tr in ws.load("flow.json")["tracks"]}

    if track_ids:
        by_id = {tr["id"]: tr for tr in wt["tracks"]}
        picks = [by_id[i] for i in track_ids if i in by_id]
    else:
        picks = sorted((tr for tr in wt["tracks"] if tr["id"] in ev_of),
                       key=lambda tr: -len(tr["frames"]))[:n_tracks]
    fig, axes = plt.subplots(len(picks), 1, figsize=(12, 2.1 * len(picks)),
                             sharex=True)
    for ax, tr in zip(np.atleast_1d(axes), picks):
        obs = [o for o in raw_obs[tr["id"]] if o[0] < len(Hs)]
        t_raw = np.array([o[0] for o in obs]) / fps
        pts = np.array([apply_h(Hs[o[0]], [(o[1], o[2])])[0] for o in obs])
        v_raw = np.hypot(np.gradient(pts[:, 0], t_raw),
                         np.gradient(pts[:, 1], t_raw))
        t_sm = np.array(tr["frames"]) / fps
        ax.plot(t_raw, v_raw, color="0.65", lw=1, label="raw diff")
        if tr.get("zv_mag"):
            zt = [tf for tf, m in zip(t_sm, tr["zv_mag"]) if m is not None]
            zm = [m for m in tr["zv_mag"] if m is not None]
            ax.plot(zt, zm, ".", color="#1a9c50", ms=2.5,
                    label="|z_v| (corrected)")
        elif tr["id"] in flow:
            fl = flow[tr["id"]]
            ax.plot(np.array(fl["frames"]) / fps,
                    np.hypot(fl["vx"], fl["vy"]), ".", color="#1a9c50",
                    ms=2.5, label="|z_v| flow")
        ax.plot(t_sm, tr["speed"], color="#2e5bff", lw=2, label="smoothed")
        ax.axhline(v_stop, color="#eb3c32", lw=0.8, ls="--")
        ax.axhline(v_move, color="#3cc85a", lw=0.8, ls="--")
        for te in ev_of.get(tr["id"], []):
            ax.axvline(te, color="#eb3c32", lw=1.2)
        ax.set_ylabel(f"tr {tr['id']}", fontsize=8)
        ax.set_ylim(0, max(np.percentile(v_raw, 98), 3 * v_move))
    np.atleast_1d(axes)[0].legend(loc="upper right", fontsize=8)
    np.atleast_1d(axes)[-1].set_xlabel("t (s)")
    fig.suptitle("speed profiles: raw central-diff vs published (px/s); "
                 "red vline = stop onset", fontsize=10)
    fig.tight_layout()
    out = f"{ws.qa}/speed-profiles{suffix}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out}")


def world_map(ws):
    """All world tracks + centerline + stop events on one canvas."""
    wt = ws.load("world_tracks.json")
    cl = ws.load("centerline.json")
    v_stop, v_move = wt["v_stop"], wt["v_move"]

    xs = np.concatenate([tr["x"] for tr in wt["tracks"]])
    ys = np.concatenate([tr["y"] for tr in wt["tracks"]])
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    sc = min(1.0, 2200.0 / max(x1 - x0, y1 - y0))
    W, H = int((x1 - x0) * sc) + 80, int((y1 - y0) * sc) + 80

    def P(x, y):
        return (40 + (x - x0) * sc, 40 + (y - y0) * sc)

    img = Image.new("RGB", (W, H), (16, 18, 24))
    d = ImageDraw.Draw(img)
    for tr in wt["tracks"]:
        for i in range(len(tr["x"]) - 1):
            d.line([P(tr["x"][i], tr["y"][i]),
                    P(tr["x"][i + 1], tr["y"][i + 1])],
                   fill=speed_color(tr["speed"][i], v_stop, v_move), width=2)
    d.line([P(x, y) for x, y in zip(cl["x"], cl["y"])],
           fill=(46, 91, 255), width=4)
    for ev in wt["stop_events"]:
        x, y = P(ev["x"], ev["y"])
        d.ellipse([x - 5, y - 5, x + 5, y + 5], outline=(255, 255, 255),
                  width=2)
    out = f"{ws.qa}/world-map.png"
    img.save(out)
    print(f"wrote {out}")
