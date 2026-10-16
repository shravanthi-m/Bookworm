"""`bookworm watch-flip`: run the flip monitor on a camera while someone else moves the arm.

    watch-flip --webcam 1 --record good      record a good flip as reference (Ctrl+C to end)
    watch-flip --webcam 1 --record bad       record a bad one (e.g. the page starting to fold)
    watch-flip --webcam 1                    watch: prints OK / anomalous / STOP per frame
    watch-flip --images runs/watch-.../frames  replay a saved run to tune the threshold

Sources: --webcam N (local camera / phone), --live (Viam camera), --images DIR.
Every watched frame and its scores are saved to runs/watch-<time>/ so any run can later be
added to the references (--images <that>/frames --record good|bad).
Never moves the arm.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2

from .camera import read_image
from .flip_monitor import FlipMonitor, ReferenceRecorder


class _Webcam:
    def __init__(self, index):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        if not self.cap.isOpened():
            sys.exit(f"cannot open local camera {index}")

    def frames(self):
        while True:
            ok, frame = self.cap.read()
            if not ok:
                return
            yield frame

    def close(self):
        self.cap.release()


class _Viam:
    def __init__(self, cfg):
        from ..hardware.viam_robot import ViamConnection, ViamRobot

        self.conn = ViamConnection()
        self.robot = ViamRobot(self.conn, **cfg["robot"])

    def frames(self):
        while True:
            yield self.robot.capture()

    def close(self):
        self.conn.close()


class _Folder:
    def __init__(self, folder):
        self.paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".heic"))

    def frames(self):
        for p in self.paths:
            yield read_image(p)

    def close(self):
        pass


def _alert(say: bool):
    print("\a", end="", flush=True)                       # terminal bell
    if say:
        import subprocess
        subprocess.Popen(["say", "stop"] if sys.platform == "darwin" else ["espeak-ng", "stop"])


def watch_flip(cfg: dict, args) -> None:
    mcfg = cfg["vision"]["flip_monitor"]
    if args.webcam is not None:
        src = _Webcam(args.webcam)
    elif args.images:
        src = _Folder(args.images)
    else:
        src = _Viam(cfg)
    realtime = args.images is None
    period = 1.0 / args.fps if realtime else 0.0

    recorder = ReferenceRecorder(mcfg["reference_dir"], args.record) if args.record else None
    monitor = None if recorder else FlipMonitor(mcfg["reference_dir"], args.max_deviation or mcfg["max_deviation"],
                                                mcfg["stop_after"])
    out = Path(f"runs/watch-{time.strftime('%Y%m%d-%H%M%S')}")
    (out / "frames").mkdir(parents=True, exist_ok=True)
    log = open(out / "scores.jsonl", "w")

    if recorder:
        print(f"Recording a {args.record.upper()} flip -> {recorder.dir}  (Ctrl+C when the flip is done)")
    else:
        c = monitor.counts()
        print(f"Watching. references: {c['good']} good / {c['bad']} bad frames   (Ctrl+C to stop)")
        if not c["good"]:
            print("  no GOOD references yet -> only recording. First run with --record good.")

    n = stops = 0
    first_stop = None
    try:
        for frame in src.frames():
            t0 = time.perf_counter()
            if realtime:
                cv2.imwrite(str(out / "frames" / f"{n:04d}.jpg"), frame)
            if recorder:
                recorder.add(frame)
                print(f"\r  frame {n:4d} recorded", end="", flush=True)
            else:
                s = monitor.score(frame)
                log.write(json.dumps({"frame": n, "dev_good": s.dev_good, "dev_bad": s.dev_bad,
                                      "anomalous": s.anomalous, "stop": s.stop, "reason": s.reason}) + "\n")
                status = "STOP " if s.stop else ("odd  " if s.anomalous else "ok   ")
                dg = "  -  " if s.dev_good is None else f"{s.dev_good:.3f}"
                db = "  -  " if s.dev_bad is None else f"{s.dev_bad:.3f}"
                print(f"frame {n:4d}  good {dg}  bad {db}  {status} {s.reason}")
                if s.stop:
                    stops += 1
                    if first_stop is None:
                        first_stop = n
                        print(f"\n>>> STOP at frame {n}: {s.reason}\n")
                        _alert(args.say)
            n += 1
            if realtime:
                time.sleep(max(0.0, period - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        src.close()
        log.close()

    print()
    if recorder:
        print(f"saved {recorder.n} frames as {args.record.upper()} reference -> {recorder.dir}")
    else:
        print(f"{n} frames, STOP first at frame {first_stop}" if first_stop is not None else f"{n} frames, no STOP")
        print(f"scores in {out}/scores.jsonl")
        if realtime:
            print(f"frames in {out}/frames  (add as reference: watch-flip --images {out}/frames --record good|bad)")
