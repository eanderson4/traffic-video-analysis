import argparse

from . import workspace


def main():
    ap = argparse.ArgumentParser(prog="tva")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="probe video, create work dir")
    p.add_argument("src")
    p.add_argument("--work", required=True)

    for name in ("stabilize", "register", "world", "spacetime", "worldmap",
                 "roads", "plate", "segment", "anchors", "speedqa"):
        p = sub.add_parser(name)
        p.add_argument("--work", required=True)

    p = sub.add_parser("render")
    p.add_argument("--work", required=True)
    p.add_argument("--roads", action="store_true",
                   help="draw inferred road/lane overlay (off by default)")
    p.add_argument("--highlight", default="",
                   help="comma-separated track ids to enlarge + outline")

    p = sub.add_parser("detect")
    p.add_argument("--work", required=True)
    p.add_argument("--model", default="yolo11m.pt")
    p.add_argument("--imgsz", type=int, default=3840)
    p.add_argument("--conf", type=float, default=0.25)

    args = ap.parse_args()
    if args.cmd == "init":
        workspace.init(args.src, args.work)
        return
    ws = workspace.Workspace(args.work)
    if args.cmd == "stabilize":
        from . import stabilize
        stabilize.run(ws)
        stabilize.qa(ws)
    elif args.cmd == "register":
        from . import register, stabilize
        register.run(ws)
        stabilize.qa(ws)
    elif args.cmd == "detect":
        from . import detect
        detect.run(ws, model=args.model, imgsz=args.imgsz, conf=args.conf)
        detect.qa(ws)
    elif args.cmd == "world":
        from . import kinematics
        kinematics.run(ws)
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
    elif args.cmd == "plate":
        from . import plate
        plate.build(ws)
    elif args.cmd == "segment":
        from . import segment
        segment.run(ws)
    elif args.cmd == "speedqa":
        from . import viz
        viz.speed_qa(ws)
    elif args.cmd == "render":
        from . import render
        layers = ("roads", "cars") if args.roads else ("cars",)
        hot = {int(s) for s in args.highlight.split(",") if s.strip()}
        render.run(ws, layers=layers, highlight=hot)


if __name__ == "__main__":
    main()
