"""Interactive focus-lane editor: local web page to hand-pick loud cars.

Serves the clip plus every rendered track's detected centers; click a dot
to toggle it in/out of the focus set, Save writes include/exclude back to
focus_lane.json. Clicks are stored as deltas against the offset_band
baseline, so retuning the band later keeps manual picks intact.

Displayed states and onset rings are computed server-side by the same
focus.py pipeline the render runs: /data ships them per track, and POST
/preview recomputes them with the client's unsaved pins. The client keeps
no state-machine logic of its own, so the editor always shows exactly what
the next render will draw.
"""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

from . import focus, overrides

VIEW_W = 1920  # frames served to the browser at this width


def _kept(ctx, tr):
    """Sample indices with a detected center - the ones the client gets."""
    px = ctx.px_of.get(tr["id"], {})
    return [k for k, f in enumerate(tr["frames"]) if f in px]


def _state_payload(ctx, state_of):
    """track id -> {"ss": displayed states as a code string aligned with
    the samples _kept ships, "rings": onset ring frames}."""
    by_id = {tr["id"]: tr for tr in ctx.wt["tracks"]}
    out = {}
    for tid, (states, flips) in state_of.items():
        tr = by_id[tid]
        ks = _kept(ctx, tr)
        kset = set(ks)
        out[tid] = {
            "ss": "".join(focus.STATE_CODE[states[k]] for k in ks),
            "rings": [tr["frames"][j] for j in flips if j in kset],
        }
    return out


def build_data(ws, ctx):
    meta = ctx.meta
    wt = ctx.wt
    spec = ctx.spec or {}
    state_of = focus.displayed_states(ctx, overrides.load(ws))
    states_payload = _state_payload(ctx, state_of)

    tracks = []
    for tr in wt["tracks"]:
        if tr["id"] in ctx.parked or tr["id"] < 0:
            continue
        F, X, Y, V = [], [], [], []
        for k in _kept(ctx, tr):
            f = tr["frames"][k]
            p = ctx.px_of[tr["id"]][f]
            F.append(f)
            X.append(round(p[0]))
            Y.append(round(p[1]))
            V.append(round(tr["speed"][k], 1))
        if F:
            t = {"id": tr["id"], "f": F, "x": X, "y": Y, "v": V}
            bf = ctx.first_real.get(tr["id"])
            if bf is not None and F[0] < bf:
                t["bf"] = bf
            sp = states_payload.get(tr["id"])
            if sp is not None:
                t["ss"] = sp["ss"]        # displayed states, pins applied
                t["rings"] = sp["rings"]  # onset ring frames
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
        "band": sorted(ctx.band),
        "include": sorted(spec.get("include", [])),
        "exclude": sorted(spec.get("exclude", [])),
        "overrides": ovs_data.get("overrides", []),
        # the state vocabulary + ring params, single-sourced from focus.py,
        # so the client renders them instead of keeping its own copies
        "states": {s: {"code": focus.STATE_CODE[s],
                       "rgb": list(focus.STATE_RGB[s])}
                   for s in overrides.STATES},
        "ring": {"secs": focus.RING_S, "r0": focus.RING_R0,
                 "r1": focus.RING_R1, "fade": focus.RING_FADE},
        "tracks": tracks,
    }


def run(ws, port=8123):
    box = {"ctx": focus.recover(ws)}
    box["data"] = json.dumps(build_data(ws, box["ctx"])).encode()
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

        def _body(self):
            return json.loads(
                self.rfile.read(int(self.headers["Content-Length"])))

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, "text/html", html)
            elif self.path == "/data":
                self._send(200, "application/json", box["data"])
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
            if self.path == "/preview":
                # the client's unsaved pins through the real Python
                # pipeline (focus.displayed_states) - the client keeps no
                # state-machine logic of its own
                body = self._body()
                ovs = {}
                for tid, f, st in body.get("pins", []):
                    if st in overrides.STATES:
                        ovs.setdefault(tid, []).append((f, st))
                state_of = focus.displayed_states(box["ctx"], ovs)
                self._send(200, "application/json", json.dumps(
                    {"tracks": _state_payload(box["ctx"], state_of)}
                ).encode())
                return
            if self.path != "/save":
                self._send(404, "text/plain", b"not found")
                return
            body = self._body()
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
                overrides.save_all(ws, body["overrides"])
            print(f"saved focus_lane.json  include={spec['include']}  "
                  f"exclude={spec['exclude']}  "
                  f"overrides={len(body.get('overrides', []))}")
            # rebuild the recovered context so the next /data reflects the
            # new focus set (a car just toggled loud gets states and can be
            # pinned without a manual reload)
            box["ctx"] = focus.recover(ws)
            box["data"] = json.dumps(build_data(ws, box["ctx"])).encode()
            self._send(200, "application/json", b'{"ok": true}')

    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"lane editor: http://127.0.0.1:{port}  (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
