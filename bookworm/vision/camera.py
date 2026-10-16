"""Image sources: a live camera, or a folder of images for development without the rig."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class CameraError(RuntimeError):
    pass


class Camera:
    def __init__(self, index: int = 0, width: int = 1920, height: int = 1080, warmup_frames: int = 5):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise CameraError(f"could not open camera {index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.warmup_frames = warmup_frames

    def capture(self) -> np.ndarray:
        # Drop buffered frames so we get one taken *after* the last flip.
        for _ in range(self.warmup_frames):
            self.cap.grab()
        ok, frame = self.cap.read()
        if not ok:
            raise CameraError("camera read failed")
        return frame

    def close(self) -> None:
        self.cap.release()


class FileCamera:
    """Returns images from a folder in sorted order; each capture() advances one image.

    Useful for developing vision against photos of real books. ``advance()`` is
    called by the mock flipper so a "flip" moves to the next photo.
    """

    EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(self, folder: str | Path):
        self.paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in self.EXTS)
        if not self.paths:
            raise CameraError(f"no images in {folder}")
        self.i = 0

    def capture(self) -> np.ndarray:
        img = cv2.imread(str(self.paths[min(self.i, len(self.paths) - 1)]))
        if img is None:
            raise CameraError(f"could not read {self.paths[self.i]}")
        return img

    def advance(self) -> None:
        self.i += 1

    @property
    def exhausted(self) -> bool:
        return self.i >= len(self.paths)

    def close(self) -> None:
        pass
