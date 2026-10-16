"""Command line entry point:  python -m bookworm <command> [options]

    check                  connect to Viam, list components, grab a test frame
    run                    read the book (Ctrl+C to stop)
    ocr IMAGE              run the vision pipeline on one photo, print text, save debug images
    flip [--times N]       exercise the flip sequence only (hardware tuning)
    calibrate              click the 4 book corners on a live frame; saves config/local.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2

from .config import load_config, save_config

LOCAL_CFG = Path(__file__).resolve().parent.parent / "config" / "local.yaml"


def _default_cfg_path():
    return LOCAL_CFG if LOCAL_CFG.exists() else None


def cmd_check(cfg, args):
    from .hardware.viam_link import ViamCamera, ViamConnection

    conn = ViamConnection()
    try:
        names = conn.resource_names()
        print("Connected to", conn.address)
        print("Resources:", ", ".join(sorted(names)))
        wanted = list(cfg["hardware"]["components"].values()) + [cfg["camera"]["viam_name"]]
        missing = [n for n in wanted if n not in names]
        print("Missing (rename in Viam app or config):", missing or "none")
        if cfg["camera"]["viam_name"] in names:
            frame = ViamCamera(conn, cfg["camera"]["viam_name"]).capture()
            cv2.imwrite("check_frame.jpg", frame)
            print("Saved check_frame.jpg", frame.shape)
    finally:
        conn.close()


def cmd_run(cfg, args):
    from .controller import BookReader
    from .rig import build_rig
    from .speech import make_speaker
    from .vision import TesseractOcr

    if args.quiet:
        cfg["speech"]["engine"] = "print"
    camera, act, close = build_rig(cfg, images_dir=args.images, mock=args.mock or bool(args.images))
    try:
        reader = BookReader(camera, act, TesseractOcr(**cfg["ocr"]), make_speaker(cfg["speech"]), cfg)
        print(reader.run(max_spreads=args.max_spreads))
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        close()


def cmd_ocr(cfg, args):
    from .vision import TesseractOcr, process_spread, sharpness

    frame = cv2.imread(args.image)
    if frame is None:
        sys.exit(f"cannot read {args.image}")
    out = process_spread(frame, cfg["vision"])
    ocr = TesseractOcr(**cfg["ocr"])
    debug = Path("runs/debug")
    debug.mkdir(parents=True, exist_ok=True)
    print(f"sharpness: {sharpness(frame):.1f}")
    for name in ("spread", "left", "right"):
        cv2.imwrite(str(debug / f"{name}.png"), out[name])
    for side in ("left", "right"):
        res = ocr.read(out[side])
        pn = ocr.page_number(out[side], cfg["vision"]["page_number_region"])
        print(f"\n=== {side}: page {pn}, conf {res.mean_conf:.0f}, {res.word_count} words ===\n{res.text}")
    print(f"\ndebug images in {debug}/")


def cmd_flip(cfg, args):
    import time

    from .hardware import run_sequence
    from .rig import build_rig

    camera, act, close = build_rig(cfg, mock=args.mock)
    try:
        act.home()
        for i in range(args.times):
            print(f"flip {i + 1}/{args.times}")
            run_sequence(act, cfg["flip_sequence"])
            time.sleep(args.pause)
    finally:
        close()


def cmd_calibrate(cfg, args):
    from .rig import build_rig

    camera, act, close = build_rig(cfg, images_dir=args.images, mock=True)
    try:
        frame = camera.capture()
    finally:
        close()
    pts: list[list[int]] = []
    labels = ["top-left", "top-right", "bottom-right", "bottom-left"]
    view = frame.copy()

    def on_click(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append([x, y])
            cv2.circle(view, (x, y), 8, (0, 0, 255), -1)

    cv2.namedWindow("calibrate", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("calibrate", on_click)
    while len(pts) < 4:
        shown = view.copy()
        cv2.putText(shown, f"click book {labels[len(pts)]} corner (q to quit)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imshow("calibrate", shown)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            return
    cv2.destroyAllWindows()
    cfg["vision"]["book_corners"] = pts
    save_config(cfg, LOCAL_CFG)
    print("saved corners", pts, "to", LOCAL_CFG)


def main(argv=None):
    p = argparse.ArgumentParser(prog="bookworm")
    p.add_argument("--config", default=None, help="YAML overriding config/default.yaml "
                   "(defaults to config/local.yaml if present)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check")
    r = sub.add_parser("run")
    r.add_argument("--images", help="folder of spread photos instead of a camera (implies --mock)")
    r.add_argument("--mock", action="store_true", help="fake servos")
    r.add_argument("--quiet", action="store_true", help="print text instead of speaking")
    r.add_argument("--max-spreads", type=int)
    o = sub.add_parser("ocr")
    o.add_argument("image")
    f = sub.add_parser("flip")
    f.add_argument("--times", type=int, default=1)
    f.add_argument("--pause", type=float, default=2.0)
    f.add_argument("--mock", action="store_true")
    c = sub.add_parser("calibrate")
    c.add_argument("--images", help="calibrate on a saved photo folder instead of the camera")

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config or _default_cfg_path())
    {"check": cmd_check, "run": cmd_run, "ocr": cmd_ocr, "flip": cmd_flip,
     "calibrate": cmd_calibrate}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
