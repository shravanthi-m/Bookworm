"""Command line entry point:  python -m bookworm <command> [options]

Modes
    read                   read the book aloud, turning pages (Ctrl+C to stop)
    scan --out DIR         non-destructive scan: page images + text + PDF

Setup / calibration (writes config/local.yaml)
    check                  connect to Viam, list components, read arm pose + camera calibration
    pencil                 open gripper, insert pencil, grab
    save-pose NAME         store the arm's current pose (home, overview, scan_left, scan_right)
    calibrate-book         touch the eraser to 3 book points to define the book frame
    overlay POSE           move to POSE, draw the projected book outline -> runs/overlay.jpg

Development
    vision --images DIR    vision-only test bench on photos (accuracy report, debug panels)
    vision --live          same, on the live wrist camera — never moves the arm
    ocr IMAGE              vision pipeline on one photo; debug images in runs/debug/
    flip [--times N]       run the flip motion only (tuning)
    gui [--sim|--replay DIR|--mock]  local browser GUI (--sim / --replay work offline, no Viam)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2

import yaml

from .config import _merge, load_config

LOCAL_CFG = Path(__file__).resolve().parent.parent / "config" / "local.yaml"


def _save_local(updates: dict):
    """Merge only the calibrated keys into config/local.yaml (keeps other local overrides)."""
    local = yaml.safe_load(LOCAL_CFG.read_text()) if LOCAL_CFG.exists() else {}
    LOCAL_CFG.write_text(yaml.safe_dump(_merge(local or {}, updates), sort_keys=False))
    print("saved to", LOCAL_CFG)


# ---- modes -------------------------------------------------------------------
def _run_mode(cfg, args, mode, handler_factory):
    from .rig import build_robot
    from .session import BookSession
    from .vision import TesseractOcr

    robot, close = build_robot(cfg, images_dir=args.images, mock=args.mock)
    try:
        session = BookSession(robot, TesseractOcr(**cfg["ocr"]), handler_factory(), cfg, mode)
        print(session.run(max_spreads=args.max_spreads))
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        close()


def cmd_read(cfg, args):
    from .modes import ReadAloud
    from .speech import make_speaker

    if args.quiet:
        cfg["speech"]["engine"] = "print"
    _run_mode(cfg, args, "read", lambda: ReadAloud(make_speaker(cfg["speech"]), cfg))


def cmd_scan(cfg, args):
    import time

    from .modes import Scanner

    out = args.out or f"scans/{time.strftime('%Y%m%d-%H%M%S')}"
    _run_mode(cfg, args, "scan", lambda: Scanner(out, cfg))


# ---- setup / calibration ------------------------------------------------------
def cmd_check(cfg, args):
    import numpy as np

    from .hardware.viam_robot import ViamConnection, ViamRobot

    conn = ViamConnection()
    try:
        names = conn.resource_names()
        print("Connected to", conn.address)
        print("Resources:", ", ".join(sorted(set(names))))
        r = cfg["robot"]
        missing = [n for n in (r["arm"], r["gripper"], r["camera"]) if n not in names]
        if missing:
            print("MISSING:", missing, "-> set robot.arm/gripper/camera in config/local.yaml")
            return
        robot = ViamRobot(conn, **r)
        print("Arm flange pose:", {k: round(v, 1) for k, v in robot.end_pose().items()})
        frame = robot.capture()
        Path("runs").mkdir(exist_ok=True)
        cv2.imwrite("runs/check_frame.jpg", frame)
        print("Saved runs/check_frame.jpg", frame.shape)
        try:
            g = robot.camera_geometry()
            np.set_printoptions(precision=2, suppress=True)
            print("Intrinsics K:\n", g.K, "\nDistortion:", g.dist)
            print("Camera pose in world:\n", g.T_world_cam)
        except Exception as e:  # noqa: BLE001
            print("Camera geometry unavailable:", e)
    finally:
        conn.close()


def cmd_pencil(cfg, args):
    from .rig import build_robot

    robot, close = build_robot(cfg)
    try:
        robot.gripper("open")
        input("Gripper open. Insert the pencil (eraser down) in its holder, then press Enter...")
        robot.gripper("grab")
        print("Grabbed.")
    finally:
        close()


def cmd_save_pose(cfg, args):
    from .rig import build_robot

    robot, close = build_robot(cfg)
    try:
        pose = {k: round(float(v), 2) for k, v in robot.end_pose().items()}
    finally:
        close()
    print(args.name, "=", pose)
    _save_local({"poses": {args.name: pose}})


def cmd_calibrate_book(cfg, args):
    from .hardware import flange_to_tip
    from .rig import build_robot

    robot, close = build_robot(cfg)
    prompts = {
        "spine_bottom": "bottom end of the SPINE",
        "spine_top": "top end of the SPINE",
        "right_corner": "BOTTOM-OUTER corner of the RIGHT page",
    }
    points = {}
    try:
        for key, what in prompts.items():
            input(f"Jog the arm so the eraser tip just touches the {what}, then press Enter...")
            points[key] = [round(v, 2) for v in flange_to_tip(robot.end_pose(), cfg["robot"])]
            print(f"  {key} = {points[key]}")
    finally:
        close()
    _save_local({"book": points})


def cmd_overlay(cfg, args):
    from .hardware import BookFrame, page_quad
    from .rig import build_robot
    from .vision import book_quad_pixels

    book = BookFrame.from_config(cfg["book"])
    if book is None:
        sys.exit("run calibrate-book first")
    robot, close = build_robot(cfg)
    try:
        if cfg["poses"].get(args.pose):
            robot.move_to(cfg["poses"][args.pose])
        frame, geom = robot.capture(), robot.camera_geometry()
    finally:
        close()
    b = cfg["book"]
    for region, color in (("left", (255, 0, 0)), ("right", (0, 0, 255))):
        quad = book_quad_pixels(book, geom, page_quad(region, b["page_width_mm"], b["page_height_mm"]))
        cv2.polylines(frame, [quad.astype(int)], True, color, 3)
    Path("runs").mkdir(exist_ok=True)
    cv2.imwrite("runs/overlay.jpg", frame)
    print("saved runs/overlay.jpg — blue/red outlines should sit exactly on the left/right pages")


# ---- development --------------------------------------------------------------
def cmd_vision(cfg, args):
    """Vision-only test bench on photos or the live wrist camera. Never moves the arm."""
    import time

    from .hardware import BookFrame
    from .vision import FileCamera, TesseractOcr
    from .vision.bench import VisionBench

    out = Path(f"runs/vision-{time.strftime('%Y%m%d-%H%M%S')}")
    book = BookFrame.from_config(cfg["book"])
    bench = VisionBench(cfg, TesseractOcr(**cfg["ocr"]), out, args.region, book)
    if args.images:
        cam = FileCamera(args.images)
        for path in cam.paths:
            bench.process(path.name, cv2.imread(str(path)), truth_dir=Path(args.images))
    else:
        from .hardware.viam_robot import ViamConnection, ViamRobot

        conn = ViamConnection()
        try:
            robot = ViamRobot(conn, **cfg["robot"])
            for i in range(args.frames):
                frame = robot.capture()
                try:
                    geom = robot.camera_geometry()
                except Exception as e:  # noqa: BLE001 - still useful without geometry
                    print("no camera geometry:", e)
                    geom = None
                cv2.imwrite(str(out / f"live{i:02d}.jpg"), frame)
                bench.process(f"live{i:02d}.jpg", frame, geom)
                if i + 1 < args.frames:
                    input("Turn the page by hand (or leave it), then press Enter...") if args.manual \
                        else time.sleep(args.interval)
        finally:
            conn.close()
    bench.report()


def cmd_ocr(cfg, args):
    from .vision import TesseractOcr, process_shot, sharpness

    frame = cv2.imread(args.image)
    if frame is None:
        sys.exit(f"cannot read {args.image}")
    pages = process_shot(frame, "spread", cfg["vision"])
    ocr = TesseractOcr(**cfg["ocr"])
    debug = Path("runs/debug")
    debug.mkdir(parents=True, exist_ok=True)
    print(f"sharpness: {sharpness(frame):.1f}")
    for side, page in pages.items():
        cv2.imwrite(str(debug / f"{side}_color.png"), page.color)
        cv2.imwrite(str(debug / f"{side}_binary.png"), page.binary)
        res = ocr.read(page.binary)
        pn = ocr.page_number(page.binary, cfg["vision"]["page_number_region"])
        print(f"\n=== {side}: page {pn}, conf {res.mean_conf:.0f}, {res.word_count} words ===\n{res.text}")
    print(f"\ndebug images in {debug}/")


def cmd_flip(cfg, args):
    import time

    from .hardware import BookFrame, run_flip
    from .rig import build_robot

    book = BookFrame.from_config(cfg["book"])
    if book is None and not args.mock:
        sys.exit("run calibrate-book first")
    robot, close = build_robot(cfg, mock=args.mock)
    if args.mock:
        robot.verbose = True
    try:
        for i in range(args.times):
            print(f"flip {i + 1}/{args.times}")
            run_flip(robot, book or BookFrame.identity(), cfg, flip_index=i)
            if cfg["poses"].get("home"):
                robot.move_to(cfg["poses"]["home"])
            time.sleep(args.pause)
    finally:
        close()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "gui":          # own parser: python -m bookworm gui [--mock] [--port N]
        from .gui.__main__ import main as gui_main
        return gui_main(argv[1:])
    p = argparse.ArgumentParser(prog="bookworm", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="YAML overriding config/default.yaml (default: config/local.yaml if present)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    for name in ("read", "scan"):
        m = sub.add_parser(name)
        m.add_argument("--images", help="folder of spread photos instead of the arm (no Viam)")
        m.add_argument("--mock", action="store_true", help="fake robot, no Viam")
        m.add_argument("--max-spreads", type=int)
        if name == "read":
            m.add_argument("--quiet", action="store_true", help="print text instead of speaking")
        else:
            m.add_argument("--out", help="output folder (default scans/<timestamp>)")
    sub.add_parser("check")
    sub.add_parser("pencil")
    sp = sub.add_parser("save-pose")
    sp.add_argument("name")
    sub.add_parser("calibrate-book")
    ov = sub.add_parser("overlay")
    ov.add_argument("pose", nargs="?", default="overview")
    vb = sub.add_parser("vision", help="vision-only test bench (no arm motion)")
    src = vb.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", help="folder of photos (+ optional <stem>.left.txt/.right.txt ground truth)")
    src.add_argument("--live", action="store_true", help="Viam wrist camera at its current pose")
    vb.add_argument("--frames", type=int, default=1)
    vb.add_argument("--interval", type=float, default=2.0)
    vb.add_argument("--manual", action="store_true", help="wait for Enter between live frames (hand-flip test)")
    vb.add_argument("--region", default="spread", choices=["spread", "left", "right"])
    o = sub.add_parser("ocr")
    o.add_argument("image")
    f = sub.add_parser("flip")
    f.add_argument("--times", type=int, default=1)
    f.add_argument("--pause", type=float, default=2.0)
    f.add_argument("--mock", action="store_true")

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config or (LOCAL_CFG if LOCAL_CFG.exists() else None))
    handlers = {"read": cmd_read, "scan": cmd_scan, "check": cmd_check, "pencil": cmd_pencil,
                "save-pose": cmd_save_pose, "calibrate-book": cmd_calibrate_book,
                "overlay": cmd_overlay, "vision": cmd_vision, "ocr": cmd_ocr, "flip": cmd_flip}
    handlers[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
