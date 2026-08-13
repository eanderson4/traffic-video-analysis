import argparse

from . import workspace


def main():
    ap = argparse.ArgumentParser(prog="tva")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="probe video, create work dir")
    p.add_argument("src")
    p.add_argument("--work", required=True)

    p = sub.add_parser("stabilize")
    p.add_argument("--work", required=True)
    p.add_argument("--selftest", action="store_true",
                   help="benchmark estimators on known synthetic warps "
                        "(no video processing, no homographies.json)")

    for name in ("register", "world", "flow", "spacetime",
                 "worldmap", "roads", "plate", "segment", "anchors",
                 "queue"):
        p = sub.add_parser(name)
        p.add_argument("--work", required=True)

    p = sub.add_parser("xingvid")
    p.add_argument("--work", required=True)
    p.add_argument("--rain", action="store_true",
                   help="rain queueing-theory equations from the sky "
                        "(Little's law, Poisson, M/M/1, Kingman...)")
    p.add_argument("--bg-darken", type=float, default=None,
                   help="blurred background brightness 0-1 (default 0.55; "
                        "raise for dark/dusk footage)")
    p.add_argument("--blur-k", type=int, default=None,
                   help="background blur kernel, odd (default 31)")
    p.add_argument("--frames", default=None,
                   help="comma-separated frame numbers: render QA stills "
                        "to qa/xing-fN.png instead of a full video")
    p.add_argument("--no-lanes", action="store_true",
                   help="hide the extracted movement-lane ribbons")

    p = sub.add_parser("speedqa")
    p.add_argument("--work", required=True)
    p.add_argument("--tracks", default="",
                   help="comma-separated track ids (default: auto-pick)")
    p.add_argument("--suffix", default="", help="output filename suffix")

    p = sub.add_parser("waveqa", help="focus-lane 1D space-time wave diagram")
    p.add_argument("--work", required=True)
    p.add_argument("--monotonic-wavefront", action="store_true",
                   help="enforce spatially ordered stop flips down the "
                        "focus lane (storytelling; off = honest measurement)")

    p = sub.add_parser("wavevid", help="focus-lane 1D strip video "
                                       "(dots on a straightened lane)")
    p.add_argument("--work", required=True)
    p.add_argument("--monotonic-wavefront", action="store_true")

    p = sub.add_parser("override", help="pin a focus car's state from a "
                                        "frame on, sticky until the "
                                        "engine's next state change")
    p.add_argument("--work", required=True)
    p.add_argument("--track", type=int)
    p.add_argument("--frame", type=int)
    p.add_argument("--state", choices=("rolling", "braking", "stopped"))
    p.add_argument("--remove", action="store_true",
                   help="drop overrides for --track (all, or only --frame)")
    p.add_argument("--list", action="store_true")

    p = sub.add_parser("qa", help="kinematic QA metrics + wave fit")
    p.add_argument("--work", required=True)
    p.add_argument("--tag", default="", help="write qa/report-<tag>.json")

    p = sub.add_parser("render")
    p.add_argument("--work", required=True)
    p.add_argument("--roads", action="store_true",
                   help="draw inferred road/lane overlay (off by default)")
    p.add_argument("--highlight", default="",
                   help="comma-separated track ids to enlarge + outline")
    p.add_argument("--monotonic-wavefront", action="store_true")

    p = sub.add_parser("edit", help="interactive focus-lane editor "
                                    "(click cars loud/quiet in a browser)")
    p.add_argument("--work", required=True)
    p.add_argument("--port", type=int, default=8123)

    p = sub.add_parser("detect")
    p.add_argument("--work", required=True)
    p.add_argument("--model", default="yolo11m.pt")
    p.add_argument("--imgsz", type=int, default=3840)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--max-det", type=int, default=900,
                   help="per-frame detection cap (ultralytics default 300 "
                        "saturates on dense scenes at full imgsz)")
    p.add_argument("--tiles", default=None,
                   help="tiled (SAHI-style) inference grid, COLSxROWS e.g. "
                        "2x2 (default off = whole-frame)")
    p.add_argument("--tile-overlap", type=float, default=0.15,
                   help="fractional overlap between tiles (default 0.15)")
    p.add_argument("--out", default="tracks.json",
                   help="output filename inside the work dir")

    args = ap.parse_args()
    if args.cmd == "init":
        workspace.init(args.src, args.work)
        return
    ws = workspace.Workspace(args.work)
    if args.cmd == "stabilize":
        from . import stabilize
        if args.selftest:
            stabilize.selftest(ws)
        else:
            stabilize.run(ws)
            stabilize.qa(ws)
    elif args.cmd == "register":
        from . import register, stabilize
        register.run(ws)
        stabilize.qa(ws)
    elif args.cmd == "detect":
        from . import detect
        detect.run(ws, model=args.model, imgsz=args.imgsz, conf=args.conf,
                   max_det=args.max_det, tiles=args.tiles,
                   tile_overlap=args.tile_overlap, out=args.out)
        detect.qa(ws, tracks=args.out)
    elif args.cmd == "world":
        from . import kinematics
        kinematics.run(ws)
    elif args.cmd == "flow":
        from . import flow
        flow.run(ws)
    elif args.cmd == "qa":
        from . import qa
        qa.run(ws, tag=args.tag or None)
    elif args.cmd == "spacetime":
        from . import viz
        viz.spacetime(ws)
    elif args.cmd == "worldmap":
        from . import viz
        viz.world_map(ws)
    elif args.cmd == "roads":
        from . import roads
        roads.infer(ws)
    elif args.cmd == "anchors":
        from . import refine
        refine.anchors(ws)
    elif args.cmd == "queue":
        from . import queue as queue_mod
        queue_mod.run(ws)
    elif args.cmd == "xingvid":
        from . import xingvid
        frames = ([int(s) for s in args.frames.split(",") if s.strip()]
                  if args.frames else None)
        xingvid.run(ws, rain=args.rain, bg_darken=args.bg_darken,
                    blur_k=args.blur_k, frames=frames,
                    lanes=not args.no_lanes)
    elif args.cmd == "plate":
        from . import plate
        plate.build(ws)
    elif args.cmd == "segment":
        from . import segment
        segment.run(ws)
    elif args.cmd == "speedqa":
        from . import viz
        ids = [int(s) for s in args.tracks.split(",") if s.strip()]
        viz.speed_qa(ws, track_ids=ids or None, suffix=args.suffix)
    elif args.cmd == "waveqa":
        from . import viz
        viz.focus_wave(ws, monotonic_wave=args.monotonic_wavefront)
    elif args.cmd == "wavevid":
        from . import viz
        viz.wave_video(ws, monotonic_wave=args.monotonic_wavefront)
    elif args.cmd == "override":
        from . import overrides
        if args.list:
            for tid, ovs in sorted(overrides.load(ws).items()):
                for f, st in ovs:
                    print(f"track {tid}: {st} from frame {f}")
        elif args.remove:
            if args.track is None:
                ap.error("--remove needs --track")
            overrides.remove(ws, args.track, args.frame)
        else:
            if None in (args.track, args.frame, args.state):
                ap.error("need --track, --frame and --state "
                         "(or --list / --remove)")
            overrides.add(ws, args.track, args.frame, args.state)
    elif args.cmd == "edit":
        from . import lane_edit
        lane_edit.run(ws, port=args.port)
    elif args.cmd == "render":
        from . import render
        layers = ("roads", "cars") if args.roads else ("cars",)
        hot = {int(s) for s in args.highlight.split(",") if s.strip()}
        render.run(ws, layers=layers, highlight=hot,
                   monotonic_wave=args.monotonic_wavefront)


if __name__ == "__main__":
    main()
