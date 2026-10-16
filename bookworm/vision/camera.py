"""Photo playback for developing without the arm (the real camera is the Viam wrist camera)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class CameraError(RuntimeError):
    pass


class FileCamera:
    """Returns images from a folder in sorted order — one image per spread.

    ``advance()`` is called by the mock robot after a "flip" to move to the next photo.
    """

    EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(self, folder: str | Path):
        self.paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in self.EXTS)
        if not self.paths:
            raise CameraError(f"no images in {folder}")
        self.i = 0

    def capture(self) -> np.ndarray:
        path = self.paths[min(self.i, len(self.paths) - 1)]
        img = cv2.imread(str(path))
        if img is None:
            raise CameraError(f"could not read {path}")
        return img

    def advance(self) -> None:
        self.i += 1

    @property
    def exhausted(self) -> bool:
        return self.i >= len(self.paths)
