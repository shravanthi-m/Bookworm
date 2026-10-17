"""Turn a raw wrist-camera frame into clean page images for OCR and scanning."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Page:
    color: np.ndarray    # perspective-corrected colour page (archival scan)
    ocr: np.ndarray      # single-channel OCR input: masked, lighting-flattened (see prepare_for_ocr)


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


def _ink(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (3, 3), 0), 1, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV, 25, 15).astype(np.float32)


def gutter_seam(img: np.ndarray, search_band: float = 0.3, rows_per_step: int = 6) -> np.ndarray:
    """x of the split between the pages for every row: the path down the middle that crosses
    the least text.

    A straight cut fails when the book fills the frame: the spine is slanted and slightly
    curved, and text runs right up to it, so every vertical column clips some line ends.
    The seam may drift one pixel sideways every ``rows_per_step`` rows (dynamic programming,
    like seam carving), enough to follow the fold but too little to weave between text lines.
    Text-free gaps ("least ink") are preferred over the fold's dark shadow, which cut words
    off when the darkest column was used. Small pull towards the centre breaks ties.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape
    lo, hi = int(w * (0.5 - search_band / 2)), int(w * (0.5 + search_band / 2))
    # Follow the fold: the darkest path that runs the full height of the page. Letters are dark
    # too, but only for a few pixels vertically, so a heavy vertical blur leaves the fold's
    # shadow as the darkest structure. Text-free gaps beside it stay as a tie-breaker.
    dark = cv2.blur(255 - gray.astype(np.float32), (3, 41))[:, lo:hi]
    ink = cv2.blur(_ink(gray), (3, 1))[:, lo:hi]
    ink = ink * 60 - dark           # cost: darkness good, text marks a little bad
    ink[: int(h * 0.04)] = ink[int(h * 0.97):] = 0          # ignore the page edges
    ink -= ink.min()
    n = h // rows_per_step
    cost = ink[: n * rows_per_step].reshape(n, rows_per_step, -1).sum(axis=1)
    cost += np.abs(np.arange(lo, hi) - w / 2) * (rows_per_step * 255 * 0.2 / w)
    acc = cost.copy()
    back = np.zeros(cost.shape, np.int8)
    for i in range(1, n):
        prev = acc[i - 1]
        options = np.stack([np.r_[np.inf, prev[:-1]], prev, np.r_[prev[1:], np.inf]])
        k = options.argmin(axis=0)
        acc[i] += options[k, np.arange(prev.size)]
        back[i] = k - 1
    xs = np.empty(n, int)
    xs[-1] = int(acc[-1].argmin())
    for i in range(n - 1, 0, -1):
        xs[i - 1] = xs[i] + back[i, xs[i]]
    per_row = np.repeat(xs, rows_per_step)
    per_row = np.r_[per_row, np.full(h - per_row.size, per_row[-1])]
    return per_row + lo


def find_gutter(img: np.ndarray, search_band: float = 0.3) -> int:
    """Single x for the split (the seam's median) — for callers that need one number."""
    return int(np.median(gutter_seam(img, search_band)))


def split_at_seam(img: np.ndarray, seam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Left/right page crops along a seam; the other page's pixels inside each crop are blanked
    to the paper colour so its line ends can't leak in."""
    x0, x1 = int(seam.min()), int(seam.max())
    cols = np.arange(img.shape[1])[None, :]
    paper = np.percentile(img.reshape(-1, *img.shape[2:]), 90, axis=0)
    left, right = img[:, :x1 + 1].copy(), img[:, x0:].copy()
    left[cols[:, :x1 + 1] >= seam[:, None]] = paper
    right[cols[:, x0:] < seam[:, None]] = paper
    return left, right


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


def unsharp(gray: np.ndarray, amount: float = 1.5, sigma: float = 2.0) -> np.ndarray:
    """Unsharp mask: partially undoes the wrist camera's soft focus at close range."""
    return cv2.addWeighted(gray, 1 + amount, cv2.GaussianBlur(gray, (0, 0), sigma), -amount, 0)


def prepare_for_ocr(page: np.ndarray, mode: str = "gray") -> np.ndarray:
    """OCR input image.

    mode "gray" (default): flatten lighting + unsharp, and let Tesseract binarise internally.
        On real wrist-camera frames this read ~3.5x more confident words than "binary".
    mode "binary": our own adaptive threshold — only better on very clean, sharp images.
    """
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY) if page.ndim == 3 else page
    gray = flatten_illumination(gray)
    if mode == "gray":
        return unsharp(gray)
    gray = cv2.medianBlur(gray, 3)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)


_ROTATIONS = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def rotate_frame(frame: np.ndarray, degrees_cw: int) -> np.ndarray:
    """Rotate a fixed camera's frame upright (vision.rotate: 0 | 90 | 180 | 270, clockwise)."""
    d = int(degrees_cw or 0) % 360
    return frame if d == 0 else cv2.rotate(frame, _ROTATIONS[d])


def detect_rotation(img: np.ndarray) -> int | None:
    """Clockwise rotation (0/90/180/270) that makes the text upright, via Tesseract OSD. None if unsure."""
    import pytesseract

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    scale = 1600 / max(gray.shape)
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    try:
        osd = pytesseract.image_to_osd(gray, output_type=pytesseract.Output.DICT)
    except pytesseract.TesseractError:
        return None
    return int(osd["rotate"]) if float(osd.get("orientation_conf", 0)) > 1.0 else None


def process_shot(frame: np.ndarray, region: str, vcfg: dict, quad_px=None, size=None) -> dict[str, Page]:
    """Raw frame -> {side: Page} for the pages covered by ``region`` ('spread' | 'left' | 'right').

    With camera geometry, ``quad_px``/``size`` come from projecting the calibrated book
    into the image, so the warp is exact. Without it (photos, mock), the frame is treated
    as a whole spread using the clicked ``book_corners`` (or the full frame).
    """
    frame = apply_tool_mask(frame, vcfg.get("tool_mask"))
    if quad_px is None:                     # geometry warps handle orientation themselves
        frame = rotate_frame(frame, vcfg.get("rotate", 0))
    if quad_px is not None:
        img = warp_quad(frame, quad_px, size)
        is_spread = region == "spread"
        band = 0.1                          # spine is known to be near the middle
    else:
        corners = vcfg.get("book_corners")
        if vcfg.get("auto_corners"):          # camera may have been bumped: find the book each frame
            from .page_detect import find_book_quad

            found = find_book_quad(frame)
            if found is not None:
                corners = found
        if corners is not None and len(corners):
            c = np.asarray(corners, np.float32)
            size = (max(np.linalg.norm(c[1] - c[0]), np.linalg.norm(c[2] - c[3])),   # native resolution
                    max(np.linalg.norm(c[3] - c[0]), np.linalg.norm(c[2] - c[1])))
            img = warp_quad(frame, c, size)
        else:
            img = frame                     # no corners: keep native resolution (don't throw away pixels)
        is_spread, band = True, 0.3

    mode = vcfg.get("ocr_input", "gray")
    clean = mask_color(img, vcfg.get("clamp_color_hsv"))
    if not is_spread:
        return {region: Page(img, prepare_for_ocr(clean, mode))}

    clean = mask_rects(clean, vcfg.get("clamp_masks"))
    seam = gutter_seam(clean, band)
    (li, ri), (lc, rc) = split_at_seam(img, seam), split_at_seam(clean, seam)
    pages = {"left": Page(li, prepare_for_ocr(lc, mode)), "right": Page(ri, prepare_for_ocr(rc, mode))}
    return pages if region == "spread" else {region: pages[region]}
