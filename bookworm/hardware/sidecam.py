"""Side-on webcam on this laptop (the Logitech Brio): a photo on demand, to see how deep the tool is into the page.

The camera is opened for each photo and released again, so nothing else holds it. Photos go to temp/side/.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
SIDE_DIR = ROOT / "temp" / "side"


def take_photo(index: int = 1, width: int = 1280, height: int = 720, warmup: int = 8, save: bool = True,
               label: str = ""):
    """Grab one frame from webcam ``index``. Returns (image, saved_path or None). Raises RuntimeError if it can't."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    for _ in range(3):                                   # right after another grab the driver can still be busy
        if cap.isOpened():
            break
        cap.release()
        time.sleep(0.8)
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"side camera {index} did not open (is another program using it?)")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        frame = None
        for _ in range(max(1, warmup)):                  # first frames are dark / still auto-exposing
            ok, f = cap.read()
            if ok:
                frame = f
        if frame is None:
            raise RuntimeError(f"side camera {index} gave no frames")
    finally:
        cap.release()
    path = None
    if save:
        SIDE_DIR.mkdir(parents=True, exist_ok=True)
        path = SIDE_DIR / (time.strftime("%Y%m%d_%H%M%S") + (f"_{label}" if label else "") + ".jpg")
        cv2.imwrite(str(path), frame)
    return frame, path


class SideCam:
    """Keep the side camera open for a run of photos (opening it for every photo costs seconds)::

        with SideCam(1) as cam:
            path = cam.photo("height3")
    """

    def __init__(self, index: int = 1, width: int = 1280, height: int = 720, warmup: int = 6):
        self.index, self.width, self.height, self.warmup = index, width, height, warmup
        self.cap = None

    def __enter__(self):
        cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        for _ in range(3):
            if cap.isOpened():
                break
            cap.release()
            time.sleep(0.8)
            cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"side camera {self.index} did not open (is another program using it?)")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        for _ in range(self.warmup):
            cap.read()
        self.cap = cap
        return self

    def __exit__(self, *exc):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        return False

    def photo(self, label: str = "") -> Path:
        frame = None
        for _ in range(3):                                   # drop the stale buffered frames, keep the fresh one
            ok, f = self.cap.read()
            if ok:
                frame = f
        if frame is None:
            raise RuntimeError(f"side camera {self.index} gave no frames")
        SIDE_DIR.mkdir(parents=True, exist_ok=True)
        path = SIDE_DIR / (time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}" + (f"_{label}" if label else "") + ".jpg")
        cv2.imwrite(str(path), frame)
        return path
