"""Turn a raw wrist-camera frame into clean page images for OCR and scanning."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Page:
    color: np.ndarray    # perspective-corrected colour page (archival scan)
    binary: np.ndarray   # binarised, clamps masked (OCR input)


def warp_quad(img: np.ndarray, quad_px, size) -> np.ndarray:
    """Perspective-warp a quad (TL, TR, BR, BL pixels) to an upright ``size`` = (w, h) image."""
    w, h = (int(round(v)) for v in size)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    m = cv2.getPerspectiveTransform(np.asarray(quad_px, np.float32), dst)
    return cv2.warpPerspective(img, m, (w, h), flags=cv2.INTER_CUBIC)


def sharpness(img: np.ndarray) -> float:
    """Variance of the Laplacian. Low = blurry (arm still vibrating, page still settling)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def find_gutter(img: np.ndarray, search_band: float = 0.3) -> int:
    """x-coordinate of the spine: the darkest column near the middle (fold shadow)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    w = gray.shape[1]
    lo, hi = int(w * (0.5 - search_band / 2)), int(w * (0.5 + search_band / 2))
    profile = gray[:, lo:hi].mean(axis=0)
    profile = np.convolve(profile, np.ones(9) / 9, mode="same")  # smooth out text columns
    return lo + int(np.argmin(profile[4:-4])) + 4


def mask_rects(img: np.ndarray, rects, fill=255) -> np.ndarray:
    """White out normalised rectangles [x0, y0, x1, y1]."""
    out = img.copy()
    h, w = out.shape[:2]
    for x0, y0, x1, y1 in rects or []:
        out[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = fill
    return out


def mask_color(img: np.ndarray, hsv_range) -> np.ndarray:
    """White out pixels in an HSV range (page holders printed in a marker colour)."""
    if not hsv_range or img.ndim != 3:
        return img
    lo, hi = (np.array(v, dtype=np.uint8) for v in hsv_range)
    mask = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), lo, hi)
    mask = cv2.dilate(mask, np.ones((15, 15), np.uint8))  # cover edges/shadow
    out = img.copy()
    out[mask > 0] = 255
    return out


def flatten_illumination(gray: np.ndarray) -> np.ndarray:
    """Divide out the lighting gradient (shadows, arm shadow, one-sided lamp).

    A grayscale closing with a kernel larger than the text strokes erases the text and
    leaves the paper's brightness map; dividing by it makes the paper uniformly white.
    """
    small = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    bg = cv2.GaussianBlur(bg, (0, 0), 3)
    bg = cv2.resize(bg, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    return cv2.divide(gray, np.maximum(bg, 1), scale=255)


def glare_fraction(img: np.ndarray) -> float:
    """Share of pixels blown out by specular glare (very bright, colourless). Text there is lost."""
    if img.ndim != 3:
        return float((img >= 250).mean())
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return float(((hsv[..., 2] >= 250) & (hsv[..., 1] <= 25)).mean())


_mask_cache: dict = {}


def apply_tool_mask(frame: np.ndarray, mask_path: str | None) -> np.ndarray:
    """Paint over the pencil/gripper (fixed in wrist-camera pixels) using a calibrated mask.

    Inpainting keeps colour scans looking clean; OCR sees plain paper there.
    """
    if not mask_path:
        return frame
    key = (mask_path, frame.shape[:2])
    if key not in _mask_cache:
        m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(f"tool mask not found: {mask_path} (run: python -m bookworm calibrate-tool-mask)")
        _mask_cache[key] = cv2.resize(m, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.inpaint(frame, _mask_cache[key], 5, cv2.INPAINT_TELEA)


def prepare_for_ocr(page: np.ndarray) -> np.ndarray:
    """Grayscale -> flatten lighting -> denoise -> binarise."""
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY) if page.ndim == 3 else page
    gray = flatten_illumination(gray)
    gray = cv2.medianBlur(gray, 3)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)


def process_shot(frame: np.ndarray, region: str, vcfg: dict, quad_px=None, size=None) -> dict[str, Page]:
    """Raw frame -> {side: Page} for the pages covered by ``region`` ('spread' | 'left' | 'right').

    With camera geometry, ``quad_px``/``size`` come from projecting the calibrated book
    into the image, so the warp is exact. Without it (photos, mock), the frame is treated
    as a whole spread using the clicked ``book_corners`` (or the full frame).
    """
    frame = apply_tool_mask(frame, vcfg.get("tool_mask"))
    if quad_px is not None:
        img = warp_quad(frame, quad_px, size)
        is_spread = region == "spread"
        band = 0.1                          # spine is known to be near the middle
    else:
        corners = vcfg.get("book_corners")
        size = vcfg.get("fallback_warp_size", (1600, 1100))
        img = warp_quad(frame, corners, size) if corners else cv2.resize(frame, tuple(size))
        is_spread, band = True, 0.3

    clean = mask_color(img, vcfg.get("clamp_color_hsv"))
    if not is_spread:
        return {region: Page(img, prepare_for_ocr(clean))}

    clean = mask_rects(clean, vcfg.get("clamp_masks"))
    x = find_gutter(clean, band)
    pages = {"left": Page(img[:, :x], prepare_for_ocr(clean[:, :x])),
             "right": Page(img[:, x:], prepare_for_ocr(clean[:, x:]))}
    return pages if region == "spread" else {region: pages[region]}
