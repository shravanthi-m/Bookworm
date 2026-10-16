"""Command line entry point:  python -m bookworm <command> [options]

Modes
    read                   read the book aloud, turning pages (Ctrl+C to stop)
    scan --out DIR         non-destructive scan: page images + text + searchable PDF
    pdf SCAN_DIR           (re)build the searchable PDF of a scan folder

Setup / calibration (writes config/local.yaml)
    check                  connect to Viam, list components, read arm pose + camera calibration
    pencil                 open gripper, insert pencil, grab
    save-pose NAME         store the arm's current pose (home, overview, scan_left, scan_right)
    calibrate-book         touch the eraser to 3 book points to define the book frame
    calibrate-tool-mask    mask the pencil out of the wrist camera (frames from a few views)
    calibrate-view         fixed camera (e.g. phone): click the spread's 4 corners once
    webcams                list local cameras on this computer (find the phone's index)
    overlay POSE           move to POSE, draw the projected book outline -> runs/overlay.jpg

UI
    ui                     local page: live feed + Flip / Read / Scan (+ combined) buttons

Development
    vision --images DIR    vision-only test bench on photos (accuracy report, debug panels)
    vision --live          same, on the live Viam camera — never moves the arm
    vision --webcam N      same, on a local camera (e.g. iPhone Continuity Camera), no Viam needed
    vision --video FILE    same, on a recorded video (one frame every --every seconds)
    watch-flip             fold detector: --record good|bad flips, then watch live (never moves the arm)
    ocr IMAGE              vision pipeline on one photo; debug images in runs/debug/
    flip [--times N]       run the flip motion only (tuning)
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
        s = session.run(max_spreads=args.max_spreads)
        print(f"{s.spreads} spreads, flips {s.flips}, stopped: {s.stopped_because}")
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        close()


def cmd_read(cfg, args):
    import time

    from .modes import Combined, ReadAloud, Scanner
    from .speech import make_speaker

    if args.quiet:
        cfg["speech"]["engine"] = "print"
    if args.max_words is not None:
        cfg["modes"]["read"]["max_words_per_page"] = args.max_words or None

    def handler():
        reader = ReadAloud(make_speaker(cfg["speech"]), cfg)
        if args.save_scan is None:
            return reader
        out = args.save_scan or f"scans/{time.strftime('%Y%m%d-%H%M%S')}"
        return Combined(reader, Scanner(out, cfg))

    _run_mode(cfg, args, "read", handler)


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


def cmd_calibrate_tool_mask(cfg, args):
    """Mask the pencil/gripper out of wrist-camera frames. Never moves the arm."""
    from .vision import FileCamera
    from .vision.tool_mask import build_tool_mask, preview

    if args.images:
        from .vision.camera import read_image

        frames = [read_image(p) for p in FileCamera(args.images).paths]
    else:
        from .hardware.viam_robot import ViamConnection, ViamRobot

        conn = ViamConnection()
        frames = []
        try:
            robot = ViamRobot(conn, **cfg["robot"])
            print("Capture views that look DIFFERENT (move the arm in the Viam app, or slide paper /"
                  " objects under the camera). The pencil must stay in the gripper.")
            for i in range(args.frames):
                input(f"[{i + 1}/{args.frames}] change the view, then press Enter...")
                frames.append(robot.capture())
        finally:
            conn.close()
    mask = build_tool_mask(frames)
    out = Path(__file__).resolve().parent.parent / "config" / "tool_mask.png"
    cv2.imwrite(str(out), mask)
    Path("runs").mkdir(exist_ok=True)
    cv2.imwrite("runs/tool_mask_preview.jpg", preview(frames[-1], mask))
    print(f"mask covers {100 * (mask > 0).mean():.1f}% of the frame -> {out}")
    print("check runs/tool_mask_preview.jpg: the pencil (and only the pencil) should be red")
    _save_local({"vision": {"tool_mask": str(out)}})


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
def _webcam_frame(index: int, cap=None):
    """One frame from a local camera (e.g. iPhone Continuity Camera on the Mac)."""
    own = cap is None
    if own:
        cap = cv2.VideoCapture(index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)       # ask for the max; the driver picks the nearest
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
    if not cap.isOpened():
        sys.exit(f"cannot open local camera {index} (try: python -m bookworm webcams)")
    for _ in range(10):                               # let exposure / autofocus settle
        cap.read()
    ok, frame = cap.read()
    if own:
        cap.release()
    if not ok:
        sys.exit(f"local camera {index} returned no frame")
    return frame


def cmd_webcams(cfg, args):
    """List local cameras (index, resolution) and save a thumbnail of each to runs/webcams/."""
    out = Path("runs/webcams")
    out.mkdir(parents=True, exist_ok=True)
    for i in range(6):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
        ok, frame = cap.read()
        cap.release()
        if ok:
            cv2.imwrite(str(out / f"cam{i}.jpg"), frame)
            print(f"index {i}: {frame.shape[1]}x{frame.shape[0]}  -> {out}/cam{i}.jpg")
    print("use the index of the phone with: vision --webcam N  /  calibrate-view --webcam N")


def _grab_view_frame(cfg, args):
    """Frame for calibrate-view: --webcam, --images, or the configured Viam camera."""
    if getattr(args, "webcam", None) is not None:
        return _webcam_frame(args.webcam)
    if getattr(args, "images", None):
        from .vision import FileCamera

        return FileCamera(args.images).capture()
    from .hardware.viam_robot import ViamConnection, ViamRobot

    conn = ViamConnection()
    try:
        return ViamRobot(conn, **cfg["robot"]).capture()
    finally:
        conn.close()


def cmd_calibrate_view(cfg, args):
    """Click the 4 corners of the open spread on a fixed camera's image -> vision.book_corners."""
    import numpy as np

    from .vision.preprocess import rotate_frame

    frame = rotate_frame(_grab_view_frame(cfg, args), cfg["vision"].get("rotate", 0))
    pts: list[list[int]] = []
    labels = ["TOP-LEFT", "TOP-RIGHT", "BOTTOM-RIGHT", "BOTTOM-LEFT"]
    view = frame.copy()

    def on_click(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append([x, y])
            cv2.circle(view, (x, y), 8, (0, 0, 255), -1)

    cv2.namedWindow("calibrate-view", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("calibrate-view", on_click)
    while len(pts) < 4:
        shown = view.copy()
        cv2.putText(shown, f"click the spread's {labels[len(pts)]} corner (q = quit)", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.imshow("calibrate-view", shown)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            return
    cv2.destroyAllWindows()
    p = np.array(pts, float)
    w = int(max(np.linalg.norm(p[1] - p[0]), np.linalg.norm(p[2] - p[3])))   # keep native resolution
    h = int(max(np.linalg.norm(p[3] - p[0]), np.linalg.norm(p[2] - p[1])))
    print("book_corners =", pts, " warp size =", [w, h])
    _save_local({"vision": {"book_corners": pts, "fallback_warp_size": [w, h]}})


def cmd_vision(cfg, args):
    """Vision-only test bench on photos, a local webcam/phone, or the live Viam camera. Never moves the arm."""
    import time

    from .hardware import BookFrame
    from .vision import FileCamera, TesseractOcr
    from .vision.bench import VisionBench
    from .vision.camera import read_image

    out = Path(f"runs/vision-{time.strftime('%Y%m%d-%H%M%S')}")
    if args.rotate is not None:
        cfg["vision"]["rotate"] = args.rotate
    book = BookFrame.from_config(cfg["book"])
    bench = VisionBench(cfg, TesseractOcr(**cfg["ocr"]), out, args.region, book)
    if args.images:
        cam = FileCamera(args.images)
        for path in cam.paths:
            bench.process(path.name, read_image(path), truth_dir=Path(args.images))
    elif args.video:
        from .vision.camera import video_frames

        (out / "frames").mkdir(parents=True, exist_ok=True)
        for t, frame in video_frames(args.video, args.every):
            name = f"t{t:07.2f}s.jpg"
            cv2.imwrite(str(out / "frames" / name), frame)        # reusable: vision --images <out>/frames
            bench.process(name, frame)
    elif args.webcam is not None:
        for i in range(args.frames):
            frame = _webcam_frame(args.webcam)
            cv2.imwrite(str(out / f"webcam{i:02d}.jpg"), frame)
            bench.process(f"webcam{i:02d}.jpg", frame)
            if i + 1 < args.frames:
                input("Turn the page by hand (or leave it), then press Enter...") if args.manual \
                    else time.sleep(args.interval)
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


def cmd_watch_flip(cfg, args):
    from .vision.watch import watch_flip

    watch_flip(cfg, args)


def cmd_pdf(cfg, args):
    from .archive import build_pdf

    build_pdf(args.scan_dir, args.out, searchable=not args.no_text, lang=cfg["ocr"]["lang"],
              psm=cfg["ocr"]["psm"], page_width_mm=cfg["book"]["page_width_mm"],
              ocr_input=cfg["vision"].get("ocr_input", "gray"))


def cmd_ui(cfg, args):
    import time

    from .ui.server import Controller, serve

    if args.quiet:
        cfg["speech"]["engine"] = "print"
    conn = None
    if args.images:
        from .vision import FileCamera

        camera = FileCamera(args.images)
    else:
        from .hardware.viam_robot import ViamConnection, ViamRobot

        conn = ViamConnection()
        camera = ViamRobot(conn, **cfg["robot"])
    try:
        out = args.out or f"scans/{time.strftime('%Y%m%d-%H%M%S')}"
        serve(Controller(cfg, camera, args.config, out), args.port)
    finally:
        if conn:
            conn.close()


def cmd_ocr(cfg, args):
    from .vision import TesseractOcr, process_shot, sharpness

    from .vision.camera import read_image

    frame = read_image(args.image)
    if frame is None:
        sys.exit(f"cannot read {args.image}")
    pages = process_shot(frame, "spread", cfg["vision"])
    ocr = TesseractOcr(**cfg["ocr"])
    debug = Path("runs/debug")
    debug.mkdir(parents=True, exist_ok=True)
    print(f"sharpness: {sharpness(frame):.1f}")
    for side, page in pages.items():
        cv2.imwrite(str(debug / f"{side}_color.png"), page.color)
        cv2.imwrite(str(debug / f"{side}_ocr.png"), page.ocr)
        res = ocr.read(page.ocr)
        pn = ocr.page_number(page.ocr, cfg["vision"]["page_number_region"])
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
            m.add_argument("--save-scan", nargs="?", const="", metavar="DIR",
                           help="also store the scan (pages, text, PDF); default dir scans/<timestamp>")
            m.add_argument("--max-words", type=int, metavar="N",
                           help="read only the first ~N words of each page (0 = whole page)")
        else:
            m.add_argument("--out", help="output folder (default scans/<timestamp>)")
    sub.add_parser("check")
    sub.add_parser("pencil")
    sp = sub.add_parser("save-pose")
    sp.add_argument("name")
    sub.add_parser("calibrate-book")
    cv_ = sub.add_parser("calibrate-view", help="click the spread corners for a fixed (uncalibrated) camera")
    cv_src = cv_.add_mutually_exclusive_group()
    cv_src.add_argument("--webcam", type=int, metavar="N")
    cv_src.add_argument("--images", help="use the first photo in this folder")
    sub.add_parser("webcams", help="list local cameras on this computer")
    tm = sub.add_parser("calibrate-tool-mask", help="mask the pencil out of wrist-camera frames")
    tm.add_argument("--images", help="folder of wrist-camera frames from different views")
    tm.add_argument("--frames", type=int, default=6)
    ov = sub.add_parser("overlay")
    ov.add_argument("pose", nargs="?", default="overview")
    vb = sub.add_parser("vision", help="vision-only test bench (no arm motion)")
    src = vb.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", help="folder of photos (+ optional <stem>.left.txt/.right.txt ground truth)")
    src.add_argument("--live", action="store_true", help="the configured Viam camera (robot.camera)")
    src.add_argument("--webcam", type=int, metavar="N", help="local camera N on this computer, e.g. iPhone "
                     "Continuity Camera (list with: bookworm webcams)")
    src.add_argument("--video", help="a recorded video file (.MOV/.mp4); samples one frame every --every s")
    vb.add_argument("--every", type=float, default=2.0, help="--video: seconds between sampled frames")
    vb.add_argument("--frames", type=int, default=1)
    vb.add_argument("--interval", type=float, default=2.0)
    vb.add_argument("--manual", action="store_true", help="wait for Enter between live frames (hand-flip test)")
    vb.add_argument("--region", default="spread", choices=["spread", "left", "right"])
    vb.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], help="override vision.rotate")
    wf = sub.add_parser("watch-flip", help="fold detector: watch a camera during a flip (never moves the arm)")
    wf_src = wf.add_mutually_exclusive_group(required=True)
    wf_src.add_argument("--webcam", type=int, metavar="N")
    wf_src.add_argument("--live", action="store_true", help="the configured Viam camera")
    wf_src.add_argument("--images", help="replay saved frames (e.g. runs/watch-*/frames)")
    wf_src.add_argument("--video", help="replay a recorded video of a flip")
    wf.add_argument("--record", choices=["good", "bad"], help="save frames as a GOOD/BAD reference instead of judging")
    wf.add_argument("--fps", type=float, default=8.0)
    wf.add_argument("--max-deviation", type=float, help="override vision.flip_monitor.max_deviation")
    wf.add_argument("--say", action="store_true", help="also say 'stop' out loud on STOP")
    pdf = sub.add_parser("pdf", help="(re)build a searchable PDF from a scan folder")
    pdf.add_argument("scan_dir")
    pdf.add_argument("--out", help="output file (default <scan_dir>/book.pdf)")
    pdf.add_argument("--no-text", action="store_true", help="images only, no OCR text layer")
    ui = sub.add_parser("ui", help="local control page: live feed + Flip / Read / Scan buttons")
    ui.add_argument("--port", type=int, default=8000)
    ui.add_argument("--images", help="use a folder of photos instead of the camera (Flip advances)")
    ui.add_argument("--out", help="scan folder (default scans/<timestamp>)")
    ui.add_argument("--quiet", action="store_true", help="print instead of speaking")
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
                "calibrate-tool-mask": cmd_calibrate_tool_mask, "calibrate-view": cmd_calibrate_view,
                "webcams": cmd_webcams, "watch-flip": cmd_watch_flip, "pdf": cmd_pdf, "ui": cmd_ui,
                "overlay": cmd_overlay, "vision": cmd_vision, "ocr": cmd_ocr, "flip": cmd_flip}
    handlers[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
