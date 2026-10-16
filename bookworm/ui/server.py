"""Minimal local control page: live feed + Flip / Read / Scan buttons.

    python -m bookworm --config config/phone_mount.yaml ui        -> http://localhost:8000

Flip runs `ui.flip_command` (default: `python -m bookworm flip`), so a teammate's own flip
script can be plugged in without code changes. Stdlib HTTP server, no extra dependencies.
"""
from __future__ import annotations

import json
import shlex
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

from ..modes import Scanner, excerpt
from ..session import RunSummary, Spread, best_of_frames
from ..speech import make_speaker
from ..vision import make_ocr, ocr_image, process_shot
from ..vision.preprocess import rotate_frame

PAGE = Path(__file__).with_name("index.html")
ACTIONS = {"flip": ["flip"], "read": ["read"], "scan": ["scan"],
           "flip_read": ["flip", "read"], "flip_scan": ["flip", "scan"]}


class LazyViamCamera:
    """Connects to Viam in the background so the page opens immediately."""

    def __init__(self, cfg: dict):
        self.cfg, self.robot, self.conn, self.error = cfg, None, None, None
        threading.Thread(target=self._connect, daemon=True).start()

    def _connect(self):
        from ..hardware.viam_robot import ViamConnection, ViamRobot

        while self.robot is None:
            try:
                self.conn = ViamConnection()
                self.robot = ViamRobot(self.conn, **self.cfg["robot"])
            except Exception as e:  # noqa: BLE001 - keep retrying; shown on the page
                self.error = e
                time.sleep(3)

    def capture(self):
        if self.robot is None:
            raise RuntimeError("connecting to camera…")
        return self.robot.capture()

    def close(self):
        if self.conn:
            self.conn.close()


class Controller:
    def __init__(self, cfg: dict, camera, config_path: str | None, scan_dir: str):
        self.cfg, self.camera = cfg, camera
        self.ocr = make_ocr(cfg["ocr"])
        self.speaker = make_speaker(cfg["speech"])
        self.scanner = Scanner(scan_dir, cfg)
        default = [sys.executable, "-m", "bookworm"] + (["--config", config_path] if config_path else []) + ["flip"]
        cmd = cfg.get("ui", {}).get("flip_command")
        self.flip_cmd = shlex.split(cmd) if cmd else default
        self.frame = None                      # latest upright frame
        self.jpeg = b""
        self.lock = threading.Lock()
        self.state = {"busy": False, "status": "Ready", "text": "", "pages": 0, "pdf": False}
        self._proc = None
        self._stop = threading.Event()
        threading.Thread(target=self._feed_loop, daemon=True).start()

    # ---- camera ----------------------------------------------------------------
    def _feed_loop(self):
        while True:
            try:
                f = rotate_frame(self.camera.capture(), self.cfg["vision"].get("rotate", 0))
                ok, buf = cv2.imencode(".jpg", cv2.resize(f, (960, int(960 * f.shape[0] / f.shape[1]))),
                                       [cv2.IMWRITE_JPEG_QUALITY, 75])
                with self.lock:
                    self.frame, self.jpeg = f, buf.tobytes() if ok else self.jpeg
                if self.state["status"].startswith(("Camera:", "Connecting")):
                    self.state["status"] = "Ready"
            except Exception as e:  # noqa: BLE001 - keep the feed alive through network hiccups
                msg = str(e)
                self.state["status"] = "Connecting to camera…" if "connecting" in msg else f"Camera: {msg}"
                time.sleep(1)
            time.sleep(0.15)

    def _spread(self) -> Spread:
        time.sleep(0.4)                        # let the feed catch up after a flip
        frames = []
        for _ in range(max(1, int(self.cfg["ocr"].get("frames", 1)))):
            with self.lock:
                if self.frame is not None:
                    frames.append(self.frame.copy())
            time.sleep(self.cfg["ocr"].get("frame_gap_s", 0.3))
        if not frames:
            raise RuntimeError("no camera frame yet")
        v = dict(self.cfg["vision"], rotate=0)  # frames are already upright
        pages, results = best_of_frames(frames, "spread", v, self.ocr)
        frac = v["page_number_region"]
        left = self.ocr.page_number(ocr_image(self.ocr, pages["left"]), frac)
        right = self.ocr.page_number(ocr_image(self.ocr, pages["right"]), frac) if left is None else None
        return Spread(pages, [], left if left is not None else (right - 1 if right else None), results)

    # ---- actions ---------------------------------------------------------------
    def start(self, name: str) -> bool:
        if self.state["busy"] or name not in ACTIONS:
            return False
        self._stop.clear()
        self.state["busy"] = True
        threading.Thread(target=self._run, args=(ACTIONS[name],), daemon=True).start()
        return True

    def stop(self):
        self._stop.set()
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()

    def _run(self, steps):
        try:
            for step in steps:
                if self._stop.is_set():
                    self.state["status"] = "Stopped"
                    return
                getattr(self, f"_{step}")()
        except Exception as e:  # noqa: BLE001 - surface any failure on the page
            self.state["status"] = f"Error: {e}"
        finally:
            self.state["busy"] = False

    def _flip(self):
        self.state["status"] = "Flipping…"
        self._proc = subprocess.Popen(self.flip_cmd)
        if self._proc.wait() != 0 and not self._stop.is_set():
            raise RuntimeError(f"flip exited with {self._proc.returncode}")
        if hasattr(self.camera, "advance"):    # photo playback stands in for a real flip
            self.camera.advance()
        self.state["status"] = "Flipped"

    def _read(self):
        self.state["status"] = "Reading…"
        spread = self._spread()
        r = self.cfg["modes"]["read"]
        parts = [excerpt(spread.results[s].text, r.get("max_words_per_page"),
                         r.get("min_sentence_quality", 0.95)) for s in r["read_order"]]
        text = " ".join(p for p in parts if p) or "(nothing clear enough to read)"
        self.state["text"] = text
        if not text.startswith("("):
            self.speaker.say(text)
        self.state["status"] = f"Read page {spread.page_no}" if spread.page_no else "Read"

    def _scan(self):
        self.state["status"] = "Scanning…"
        spread = self._spread()
        results = spread.results
        self.scanner.on_spread(len(self.scanner.pages) // 2, spread, results)
        self.scanner.finish(RunSummary(spreads=len(self.scanner.pages) // 2, stopped_because="ui"))
        self.state.update(pages=len(self.scanner.pages), pdf=True,
                          status=f"Saved {len(self.scanner.pages)} pages")


def serve(ctrl: Controller, port: int = 8000):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def _send(self, code, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/state":
                self._send(200, json.dumps(ctrl.state).encode(), "application/json")
            elif self.path.startswith("/book.pdf") and (ctrl.scanner.out / "book.pdf").exists():
                self._send(200, (ctrl.scanner.out / "book.pdf").read_bytes(), "application/pdf")
            elif self.path == "/feed":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        with ctrl.lock:
                            jpg = ctrl.jpeg
                        if jpg:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                        time.sleep(0.2)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path == "/stop":
                ctrl.stop()
                self._send(200, b"{}", "application/json")
            elif self.path.startswith("/do/"):
                ok = ctrl.start(self.path[4:])
                self._send(200 if ok else 409, json.dumps({"ok": ok}).encode(), "application/json")
            else:
                self._send(404, b"not found", "text/plain")

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Bookworm UI on http://localhost:{port}  (Ctrl+C to quit)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
