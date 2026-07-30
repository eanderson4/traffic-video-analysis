"""Interactive focus-lane editor: local web page to hand-pick loud cars.

Serves the clip plus every rendered track's detected centers; click a dot
to toggle it in/out of the focus set, Save writes include/exclude back to
focus_lane.json. Clicks are stored as deltas against the offset_band
baseline, so retuning the band later keeps manual picks intact.
"""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from .render import (backfill_focus, focus_lane_ids, focus_states,
                     guide_frame, hidden_ids, lane_guide_path, manual_world,
                     stitch_focus)

VIEW_W = 1920  # frames served to the browser at this width

STATE_CODE = {"rolling": "r", "braking": "b", "stopped": "s"}


def build_data(ws):
    meta = ws.meta
    wt = ws.load("world_tracks.json")
    try:
        roads = ws.load("roads.json")["roads"]
    except FileNotFoundError:
        roads = []
    hidden = hidden_ids(ws, wt, roads)
    try:
        spec = ws.load("focus_lane.json")
    except FileNotFoundError:
        spec = {"road": 0, "offset_band": [0, 0], "max_dist": 0,
                "include": [], "exclude": []}
    band = set()
    if roads:
        band = focus_lane_ids(wt, roads,
                              {**spec, "include": [], "exclude": []})
    px_of = {}
    for tr in ws.load("tracks.json")["tracks"]:
        px_of[tr["id"]] = {o[0]: (o[1], o[2]) for o in tr["obs"]}

    # mirror the render's recovery pass (fragment stitching + queued-car
    # backfill) so the editor shows exactly the dots the render will draw;
    # backfilled frames are tagged so the client styles them apart
    first_real = {}
    eng_states = {}
    if roads:
        focus = focus_lane_ids(wt, roads, spec)
        Hs = [np.asarray(m) for m in ws.load("homographies.json")["H"]]
        for m in manual_world(ws, Hs, meta["fps"]):
            wt["tracks"].append(m["track"])
            px_of[m["id"]] = m["px"]
            focus.add(m["id"])
        guide = (lane_guide_path(wt, focus, roads, spec, wt["car_len_px"])
                 if focus else None)
        if guide is not None:
            stitch_focus(wt, focus, guide_frame(guide), wt["car_len_px"],
                         meta["fps"], px_of,
                         set(spec.get("exclude", [])), hidden)
            for tr in wt["tracks"]:
                if tr["id"] in focus:
                    first_real[tr["id"]] = tr["frames"][0]
            Hinv = [np.linalg.inv(m) for m in Hs]
            backfill_focus(wt, focus, wt["car_len_px"], wt["v_stop"],
                           px_of, Hinv, (meta["width"], meta["height"]),
                           set(spec.get("no_backfill", [])))
            # engine states (pre-override) per focus car; the client
            # re-derives displayed states/rings as pins are edited
            for tr in wt["tracks"]:
                if tr["id"] not in focus:
                    continue
                eng, _ = focus_states(tr["speed"], meta["fps"],
                                      wt["v_stop"], wt["v_move"])
                eng_states[tr["id"]] = eng

    tracks = []
    for tr in wt["tracks"]:
        if tr["id"] in hidden or tr["id"] < 0:
            continue
        eng = eng_states.get(tr["id"])
        F, X, Y, V, S = [], [], [], [], []
        for k, f in enumerate(tr["frames"]):
            p = px_of.get(tr["id"], {}).get(f)
            if p is None:
                continue
            F.append(f)
            X.append(round(p[0]))
            Y.append(round(p[1]))
            V.append(round(tr["speed"][k], 1))
            if eng is not None:
                S.append(STATE_CODE[eng[k]])
        if F:
            t = {"id": tr["id"], "f": F, "x": X, "y": Y, "v": V}
            bf = first_real.get(tr["id"])
            if bf is not None and F[0] < bf:
                t["bf"] = bf
            if eng is not None:
                t["es"] = "".join(S)          # engine states, pre-override
            tracks.append(t)
    try:
        man = ws.load("manual_tracks.json")["tracks"]
    except FileNotFoundError:
        man = []
    try:
        ovs_data = ws.load("state_overrides.json")
    except FileNotFoundError:
        ovs_data = {"overrides": []}
    return {
        "work": ws.dir,
        "manual": man,
        "n_frames": meta["n_frames"], "fps": meta["fps"],
        "width": meta["width"], "height": meta["height"],
        "v_stop": wt["v_stop"], "v_move": wt["v_move"],
        "band": sorted(band),
        "include": sorted(spec.get("include", [])),
        "exclude": sorted(spec.get("exclude", [])),
        "overrides": ovs_data.get("overrides", []),
        "tracks": tracks,
    }


def run(ws, port=8123):
    state = {"data": json.dumps(build_data(ws)).encode()}
    with open(os.path.join(os.path.dirname(__file__),
                           "lane_edit.html"), "rb") as fh:
        html = fh.read()
    meta = ws.meta
    n = meta["n_frames"]
    view_h = round(meta["height"] * VIEW_W / meta["width"] / 2) * 2
    cap = cv2.VideoCapture(meta["src"])
    lock = threading.Lock()
    last = {"pos": -2}  # skip the expensive random seek on sequential reads

    # sparse-keyframe sources make random seeks cost seconds; decode the
    # whole clip once in the background into an in-memory JPEG cache
    # (~0.2 MB/frame at 1920w) so scrubbing is instant once warm
    frames = {}

    def warm():
        c = cv2.VideoCapture(meta["src"])
        for i in range(n):
            ok, fr = c.read()
            if not ok:
                break
            fr = cv2.resize(fr, (VIEW_W, view_h))
            _, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frames[i] = buf.tobytes()
        c.release()
        print(f"frame cache warm ({len(frames)} frames)")

    threading.Thread(target=warm, daemon=True).start()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, ctype, body):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, "text/html", html)
            elif self.path == "/data":
                self._send(200, "application/json", state["data"])
            elif self.path.startswith("/frame/"):
                i = int(self.path.rsplit("/", 1)[1].split(".")[0])
                i = max(0, min(n - 1, i))
                if i in frames:
                    self._send(200, "image/jpeg", frames[i])
                    return
                with lock:
                    if i != last["pos"] + 1:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ok, frame = cap.read()
                    last["pos"] = i
                if not ok:
                    self._send(404, "text/plain", b"no frame")
                    return
                frame = cv2.resize(frame, (VIEW_W, view_h))
                _, buf = cv2.imencode(".jpg", frame,
                                      [cv2.IMWRITE_JPEG_QUALITY, 85])
                self._send(200, "image/jpeg", buf.tobytes())
            else:
                self._send(404, "text/plain", b"not found")

        def do_POST(self):
            if self.path != "/save":
                self._send(404, "text/plain", b"not found")
                return
            body = json.loads(
                self.rfile.read(int(self.headers["Content-Length"])))
            try:
                spec = ws.load("focus_lane.json")
            except FileNotFoundError:
                spec = {"road": 0, "offset_band": [0, 0], "max_dist": 0}
            spec["include"] = sorted(body["include"])
            spec["exclude"] = sorted(body["exclude"])
            with open(ws.path("focus_lane.json"), "w") as fh:
                json.dump(spec, fh, indent=1)
            if "manual" in body:
                with open(ws.path("manual_tracks.json"), "w") as fh:
                    json.dump({"tracks": body["manual"]}, fh, indent=1)
            if "overrides" in body:
                with open(ws.path("state_overrides.json"), "w") as fh:
                    json.dump({"overrides": body["overrides"]}, fh,
                              indent=1)
            print(f"saved focus_lane.json  include={spec['include']}  "
                  f"exclude={spec['exclude']}  "
                  f"overrides={len(body.get('overrides', []))}")
            state["data"] = json.dumps(build_data(ws)).encode()
            self._send(200, "application/json", b'{"ok": true}')

    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"lane editor: http://127.0.0.1:{port}  (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
