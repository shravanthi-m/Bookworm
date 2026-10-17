"""Minimal local control page: live feed + Flip / Read / Scan buttons.

    python -m bookworm --config config/phone_mount.yaml ui        -> http://localhost:8000

Flip plays Midhat's recorded path over the page's own robot connection when `ui.flip_path`
is set (v1 / v2); otherwise it runs `ui.flip_command` (default: `python -m bookworm flip`), so a
teammate's own flip script can be plugged in without code changes. Stdlib HTTP server, no extra
dependencies.
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

from ..llm import LlmError, make_llm, summarize
from ..modes import Scanner, excerpt
from ..session import RunSummary, Spread, best_of_frames
from ..speech import make_speaker
from ..vision import make_ocr, ocr_image, process_shot
from ..vision.page_detect import find_book_quad
from ..vision.preprocess import rotate_frame

PAGE = Path(__file__).with_name("index.html")
ACTIONS = {"flip": ["flip"], "read": ["read"], "scan": ["scan"],
           "flip_read": ["flip", "read"], "flip_scan": ["flip", "scan"],
           "look": ["look"], "next": ["flip", "look"],   # look = refresh the on-screen pages, no OCR
           "summarize": ["summarize"], "flip_summarize": ["flip", "summarize"],
           "park": ["park"], "ready": ["ready"], "resume": ["resume"]}
ACKS = {"flip": "Turning the page.", "next": "Turning the page.", "read": "Okay, reading.",
        "scan": "Scanning.", "summarize": "Let me summarise.", "flip_read": "Turning the page, then reading.",
        "flip_scan": "Turning the page, then scanning.", "flip_summarize": "Turning the page, then summarising.",
        "park": "Moving away.", "ready": "Coming back.", "look": "Looking."}   # resume: no ack, it just carries on
GOTO_DIR = Path(__file__).resolve().parents[2] / "scripts"


class LazyViamCamera:
    """Connects to Viam in the background so the page opens immediately."""

    def __init__(self, cfg: dict):
        self.cfg, self.robot, self.conn, self.error = cfg, None, None, None
        threading.Thread(target=self._connect, daemon=True).start()

    def _connect(self):
        from ..hardware.viam_robot import ViamConnection, ViamRobot

        # Keep one connection and retry the camera lookup on it: every new connection starts
        # its own Rust runtime, and leaking one per retry ran macOS out of threads.
        while self.robot is None:
            try:
                self.conn = self.conn or ViamConnection()
                self.robot = ViamRobot(self.conn, **self.cfg["robot"])
            except Exception as e:  # noqa: BLE001 - keep retrying; shown on the page
                self.error = e
                print(f"[ui] camera not ready: {str(e)[:150]}", flush=True)
                time.sleep(10)

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
        self.llm = make_llm(cfg.get("llm"))
        self.scanner = Scanner(scan_dir, cfg)
        default = [sys.executable, "-m", "bookworm"] + (["--config", config_path] if config_path else []) + ["flip"]
        cmd = cfg.get("ui", {}).get("flip_command")
        self.flip_cmd = shlex.split(cmd) if cmd else default
        self.frame = None                      # latest upright frame
        self.jpeg = b""
        self.lock = threading.Lock()
        self.state = {"busy": False, "status": "Ready", "text": "", "pages": 0, "pdf": False,
                      # book view: page images version, flip timing so the page-turn animation
                      # lasts as long as the arm (measured after each flip; ui.flip_seconds until then)
                      "spread_v": 0, "page_no": None, "flipping": False,
                      "flip_s": float(cfg.get("ui", {}).get("flip_seconds", 31)), "flip_step": None}
        self.page_jpg = {"left": b"", "right": b""}
        self._proc = None
        self._stop = threading.Event()
        threading.Thread(target=self._feed_loop, daemon=True).start()

    # ---- camera ----------------------------------------------------------------
    def _feed_loop(self):
        while True:
            try:
                f = rotate_frame(self.camera.capture(), self.cfg["vision"].get("rotate", 0))
                view = f.copy()                                # full resolution: crisp on Retina screens
                if self.cfg["vision"].get("auto_corners"):
                    scale = 960 / f.shape[1]                   # detect on a small copy (fast)
                    quad = find_book_quad(cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
                    if quad is not None:
                        cv2.polylines(view, [(quad / scale).astype("int32")], True, (80, 140, 110), 3)
                ok, buf = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 90])
                with self.lock:
                    self.frame, self.jpeg = f, buf.tobytes() if ok else self.jpeg
                if self.state["status"].startswith(("Camera:", "Connecting")):
                    self.state["status"] = "Ready"
            except Exception as e:  # noqa: BLE001 - keep the feed alive through network hiccups
                msg = str(e)
                self.state["status"] = "Connecting to camera…" if "connecting" in msg else f"Camera: {msg}"
                time.sleep(1)
            time.sleep(0.05)

    def _spread(self, n_frames: int | None = None) -> Spread:
        time.sleep(0.4)                        # let the feed catch up after a flip
        frames = []
        for i in range(max(1, int(n_frames or self.cfg["ocr"].get("frames", 1)))):
            if i:
                time.sleep(self.cfg["ocr"].get("frame_gap_s", 0.3))   # only between frames
            with self.lock:
                if self.frame is not None:
                    frames.append(self.frame.copy())
        if not frames:
            raise RuntimeError("no camera frame yet")
        v = dict(self.cfg["vision"], rotate=0)  # frames are already upright
        pages, results = best_of_frames(frames, "spread", v, self.ocr, combine=self.cfg["ocr"].get("combine", "best"))
        frac = v["page_number_region"]
        left = self.ocr.page_number(ocr_image(self.ocr, pages["left"]), frac)
        right = self.ocr.page_number(ocr_image(self.ocr, pages["right"]), frac) if left is None else None
        page_no = left if left is not None else (right - 1 if right else None)
        self._show(pages, page_no)
        return Spread(pages, [], page_no, results)

    def _show(self, pages, page_no=None):
        """Publish the straightened left/right page images for the book view."""
        for side in ("left", "right"):
            if side in pages:
                ok, buf = cv2.imencode(".jpg", pages[side].color, [cv2.IMWRITE_JPEG_QUALITY, 88])
                if ok:
                    self.page_jpg[side] = buf.tobytes()
        self.state["page_no"] = page_no
        self.state["spread_v"] += 1

    def _look(self):
        """Refresh the on-screen pages from the current frame (no OCR, so it's quick)."""
        time.sleep(0.4)                        # let the feed catch up after a flip
        with self.lock:
            frame = None if self.frame is None else self.frame.copy()
        if frame is None:
            raise RuntimeError("no camera frame yet")
        v = dict(self.cfg["vision"], rotate=0)
        pages = process_shot(frame, "spread", v)
        page_no = None
        try:                                   # page number only (header/footer strip): quick
            frac = v["page_number_region"]
            left = self.ocr.page_number(ocr_image(self.ocr, pages["left"]), frac)
            right = self.ocr.page_number(ocr_image(self.ocr, pages["right"]), frac) if left is None else None
            page_no = left if left is not None else (right - 1 if right else None)
        except Exception as e:  # noqa: BLE001 - a missing number must never block the pages
            print(f"[look] page number not read: {e}", flush=True)
        prev = self.state.get("page_no")
        if page_no is None and getattr(self, "_flipped", False) and prev:
            page_no = prev + 2                 # one leaf turned: 200/201 -> 202/203
        self._flipped = False
        self._show(pages, page_no)
        if self.state["status"] in ("Ready", "Flipped"):
            self.state["status"] = "Ready"

    # ---- actions ---------------------------------------------------------------
    def start(self, name: str, voice: bool = False) -> bool:
        if name not in ACTIONS:
            return False
        if self.state["busy"]:
            if voice:
                self.speaker.say("Hang on, I'm still busy.", resumable=False)
            return False
        if voice and name in ACKS:             # spoken reply so the user knows they were heard
            self.speaker.say(ACKS[name], resumable=False)
        self._stop.clear()
        self.state["busy"] = True
        threading.Thread(target=self._run, args=(ACTIONS[name],), daemon=True).start()
        return True

    def stop(self):
        self._stop.set()
        self.speaker.stop()
        arm, conn = getattr(self, "_arm", None), getattr(self.camera, "conn", None)
        if arm is not None and conn is not None and self.state.get("busy"):
            threading.Thread(target=lambda: conn.call(arm.stop(), timeout=10), daemon=True).start()
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

    def _command(self, cmd, what):
        """Run an arm script; Stop terminates it (the scripts halt the arm on SIGTERM)."""
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        last = ""
        for line in self._proc.stdout:
            print(line, end="", flush=True)
            if line.startswith("STEP "):              # flip_path.py: the arm is heading to this point
                self.state["flip_step"] = line[5:].strip()
                continue
            last = line.strip() or last
        if self._proc.wait() != 0 and not self._stop.is_set():
            raise RuntimeError(f"{what}: {last or f'exited with {self._proc.returncode}'}")

    def _goto(self, target: str, doing: str, done: str):
        """Park / come back over the page's own Viam connection (no new dial per command)."""
        conn = getattr(self.camera, "conn", None)
        if conn is None:
            raise RuntimeError("not connected to the robot yet")
        if str(GOTO_DIR) not in sys.path:
            sys.path.insert(0, str(GOTO_DIR))
        from arm_goto import goto
        from viam.components.arm import Arm

        self.state["status"] = doing
        self._arm = conn.component(Arm, self.cfg["robot"].get("arm", "arm"))
        code, msg = goto(conn, self._arm, target)
        print(f"[arm] {msg}", flush=True)
        if self._stop.is_set():
            return
        if code != 0:
            raise RuntimeError(msg)            # shown in the status line
        self.state["status"] = done

    def _resume(self):
        self.state["status"] = "Continuing" if self.speaker.resume() else "Nothing to continue"

    def _park(self):
        self._goto("park", "Moving away…", "Parked")

    def _ready(self):
        self._goto("ready", "Coming back…", "Ready above the page")

    def _flip_path(self, version: str):
        """Play Midhat's recorded path (flip_path.py) over the page's own Viam connection,
        so a page turn starts right away instead of dialing the robot again (that dial is
        flaky: it once took ~80 s of 20 s timeouts before the arm moved)."""
        conn = getattr(self.camera, "conn", None)
        if conn is None:
            raise RuntimeError("not connected to the robot yet")
        if str(GOTO_DIR) not in sys.path:
            sys.path.insert(0, str(GOTO_DIR))
        import flip_path
        from viam.components.arm import Arm

        pts, path, between = flip_path.load(version)
        tag = flip_path.step_tag(version)

        t0 = time.time()

        def on_step(name):
            # +t = seconds since the flip started when the arm heads for this point (tunes the page animation)
            print(f"STEP {tag}{name}  +{time.time() - t0:.2f}s", flush=True)
            self.state["flip_step"] = f"{tag}{name}"   # the page turns the on-screen leaf in step with this

        self._arm = conn.component(Arm, self.cfg["robot"].get("arm", "arm"))   # Stop calls arm.stop() on it
        try:
            code, msg = flip_path.play(conn, self._arm, flip_path.make_plan(path, between), pts,
                                       on_step=on_step, stopped=self._stop.is_set,
                                       log=lambda m: print(m, flush=True))
        except Exception:
            if self._stop.is_set():            # Stop halted the arm mid-move; that's not an error
                return
            raise
        print(f"[flip {version}] {msg}", flush=True)
        if code != 0 and not self._stop.is_set():
            raise RuntimeError(f"flip: {msg}")

    def _flip(self):
        self.state.update(status="Flipping…", flipping=True, flip_step=None)
        self._flipped = True
        t0 = time.time()
        version = self.cfg.get("ui", {}).get("flip_path")
        try:
            if version:
                self._flip_path(str(version))
            else:
                self._command(self.flip_cmd, "flip")
        finally:
            self.state["flipping"] = False
        if not self._stop.is_set():
            self.state["flip_s"] = round(time.time() - t0, 1)   # next animation matches the real arm
        if hasattr(self.camera, "advance"):    # photo playback stands in for a real flip
            self.camera.advance()
        self.state["status"] = "Flipped"

    def _read(self):
        self.state["status"] = "Reading…"
        spread = self._spread(self.cfg["modes"]["read"].get("frames", 1))   # 1 frame: fast
        r = self.cfg["modes"]["read"]
        parts = [excerpt(spread.results[s].text, r.get("max_words_per_page"),
                         r.get("min_sentence_quality", 0.95)) for s in r["read_order"]]
        text = " ".join(p for p in parts if p)
        if not text:        # no clean sentence: read the start of the page anyway, better than silence
            raw = " ".join(spread.results[s].text for s in r["read_order"]).split()
            text = " ".join(raw[2:2 + (r.get("max_words_per_page") or 30)])   # skip the running header
        self.state["text"] = text or "(no text found on the page)"
        if text and not self._stop.is_set():
            self.speaker.say(text)
        self.state["status"] = f"Read page {spread.page_no}" if spread.page_no else "Read"

    def _summarize(self):
        self.state["status"] = "Summarising…"
        spread = self._spread(self.cfg["modes"]["read"].get("frames", 1))
        text = "\n\n".join(spread.results[s].text for s in self.cfg["modes"]["read"]["read_order"]).strip()
        if not text:
            raise RuntimeError("no text found on the page")
        try:
            summary = summarize(self.llm, text, self.cfg.get("llm", {}).get("summary_sentences", 2))
        except LlmError as e:
            raise RuntimeError(f"summary failed ({e})") from e
        self.state["text"] = summary
        if not self._stop.is_set():
            self.speaker.say(summary)
        self.state["status"] = f"Summarised page {spread.page_no}" if spread.page_no else "Summarised"

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
                state = dict(ctrl.state, speaking=bool(getattr(ctrl.speaker, "speaking", False)),
                             llm=ctrl.llm.ready)
                self._send(200, json.dumps(state).encode(), "application/json")
            elif self.path.startswith("/static/"):            # decor images that ship with the page
                f = PAGE.parent / Path(self.path.split("?")[0]).name
                types = {".png": "image/png", ".webp": "image/webp", ".svg": "image/svg+xml", ".jpg": "image/jpeg"}
                if f.suffix in types and f.is_file():
                    self._send(200, f.read_bytes(), types[f.suffix])
                else:
                    self._send(404, b"not found", "text/plain")
            elif self.path.split("?")[0] in ("/page/left.jpg", "/page/right.jpg"):
                jpg = ctrl.page_jpg[self.path.split("?")[0][6:-4]]
                if jpg:
                    self._send(200, jpg, "image/jpeg")
                else:
                    self._send(404, b"no page yet", "text/plain")
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
                        time.sleep(0.1)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path.startswith("/stop"):
                ctrl.stop()
                self._send(200, b"{}", "application/json")
            elif self.path.startswith("/heard"):            # voice transcripts, logged for debugging
                n = int(self.headers.get("Content-Length") or 0)
                print(f"[heard] {self.rfile.read(n).decode(errors='replace')[:200]}", flush=True)
                if "unknown" in self.path and not getattr(ctrl.speaker, "speaking", False):
                    ctrl.speaker.say("Sorry, I didn't catch that.", resumable=False)
                self._send(200, b"{}", "application/json")
            elif self.path.startswith("/do/"):
                name, _, query = self.path[4:].partition("?")
                ok = ctrl.start(name, voice="voice" in query)
                self._send(200 if ok else 409, json.dumps({"ok": ok}).encode(), "application/json")
            else:
                self._send(404, b"not found", "text/plain")

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Bookworm UI on http://localhost:{port}  (Ctrl+C to quit)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
