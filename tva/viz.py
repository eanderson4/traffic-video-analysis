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
