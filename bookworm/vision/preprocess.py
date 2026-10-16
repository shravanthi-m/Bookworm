"""Turn a raw camera frame into two clean page images ready for OCR."""
from __future__ import annotations

import cv2
import numpy as np


def warp_to_book(img: np.ndarray, corners, size=(1600, 1100)) -> np.ndarray:
    """Perspective-correct the book region. ``corners`` = TL, TR, BR, BL in pixels."""
    if corners is None:
        return cv2.resize(img, tuple(size))
    w, h = size
    src = np.array(corners, dtype=np.float32)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, m, (w, h))


def sharpness(img: np.ndarray) -> float:
    """Variance of the Laplacian. Low = blurry (motion, focus, page still settling)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def find_gutter(img: np.ndarray, search_band: float = 0.3) -> int:
    """x-coordinate of the book spine: the darkest column near the middle.

    The fold casts a shadow, so column-mean brightness dips there.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    w = gray.shape[1]
    lo, hi = int(w * (0.5 - search_band / 2)), int(w * (0.5 + search_band / 2))
    profile = gray[:, lo:hi].mean(axis=0)
    profile = np.convolve(profile, np.ones(9) / 9, mode="same")  # smooth out text columns
    return lo + int(np.argmin(profile[4:-4])) + 4


def split_pages(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = find_gutter(img)
    return img[:, :x], img[:, x:]


def mask_rects(img: np.ndarray, rects, fill=255) -> np.ndarray:
    """White out normalized rectangles [x0, y0, x1, y1] (e.g. clamp fingers)."""
    out = img.copy()
    h, w = out.shape[:2]
    for x0, y0, x1, y1 in rects or []:
        out[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = fill
    return out


def mask_color(img: np.ndarray, hsv_range) -> np.ndarray:
    """White out pixels in an HSV colour range (clamp fingers printed in a marker colour)."""
    if not hsv_range or img.ndim != 3:
        return img
    lo, hi = (np.array(v, dtype=np.uint8) for v in hsv_range)
    mask = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), lo, hi)
    mask = cv2.dilate(mask, np.ones((15, 15), np.uint8))  # cover finger edges/shadow
    out = img.copy()
    out[mask > 0] = 255
    return out


def prepare_for_ocr(page: np.ndarray) -> np.ndarray:
    """Grayscale + contrast normalisation + binarisation; tolerant of uneven lighting."""
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY) if page.ndim == 3 else page
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.medianBlur(gray, 3)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)


def process_spread(frame: np.ndarray, vcfg: dict) -> dict:
    """Full pipeline: raw frame -> {'spread', 'left', 'right'} (pages binarised)."""
    spread = warp_to_book(frame, vcfg.get("book_corners"), vcfg.get("warp_size", (1600, 1100)))
    clean = mask_color(spread, vcfg.get("clamp_color_hsv"))
    clean = mask_rects(clean, vcfg.get("clamp_masks"))
    left, right = split_pages(clean)
    return {"spread": spread, "left": prepare_for_ocr(left), "right": prepare_for_ocr(right)}
