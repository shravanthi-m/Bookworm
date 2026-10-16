"""Photo playback for developing without the arm (the real camera is the Viam wrist camera)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class CameraError(RuntimeError):
    pass


def _is_heif(path: Path) -> bool:
    with open(path, "rb") as f:
        head = f.read(16)
    return head[4:8] == b"ftyp" and head[8:12] in (b"heic", b"heix", b"mif1", b"msf1", b"hevc")


def read_image(path) -> np.ndarray | None:
    """cv2.imread that also reads iPhone HEIC photos (even ones saved with a .png/.jpg name)."""
    path = Path(path)
    img = cv2.imread(str(path))
    if img is not None or not path.exists() or not _is_heif(path):
        return img
    import shutil
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "converted.jpg"
        if shutil.which("sips"):                                   # macOS built-in
            subprocess.run(["sips", "-s", "format", "jpeg", str(path), "--out", str(out)],
                           check=False, capture_output=True)
            img = cv2.imread(str(out))
        if img is None:
            try:                                                   # pip install pillow-heif
                import pillow_heif
                from PIL import Image

                pillow_heif.register_heif_opener()
                img = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
            except ImportError:
                raise CameraError(f"{path} is HEIC: convert to JPEG or `pip install pillow-heif`") from None
    return img


def video_frames(path, every_s: float = 1.0):
    """Yield (time_s, frame) from a video file, one frame every ``every_s`` seconds.

    Phone videos (.MOV/.MP4, HEVC) are decoded by OpenCV's FFmpeg backend; the phone's
    rotation tag is applied automatically.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise CameraError(f"cannot open video {path} (try: ffmpeg -i {path} -c:v libx264 out.mp4)")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(every_s * fps)))
    i = 0
    try:
        while True:
            if not cap.grab():
                return
            if i % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    yield i / fps, frame
            i += 1
    finally:
        cap.release()


class FileCamera:
    """Returns images from a folder in sorted order — one image per spread.

    ``advance()`` is called by the mock robot after a "flip" to move to the next photo.
    """

    EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".heic", ".heif"}

    def __init__(self, folder: str | Path):
        self.paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in self.EXTS)
        if not self.paths:
            raise CameraError(f"no images in {folder}")
        self.i = 0

    def capture(self) -> np.ndarray:
        path = self.paths[min(self.i, len(self.paths) - 1)]
        img = read_image(path)
        if img is None:
            raise CameraError(f"could not read {path}")
        return img

    def advance(self) -> None:
        self.i += 1

    @property
    def exhausted(self) -> bool:
        return self.i >= len(self.paths)
