"""Find the book's outline in an image and locate the corner the eraser should press."""
from __future__ import annotations

import cv2
import numpy as np

from .geometry import pixel_to_plane


def find_paper_corners(img: np.ndarray, min_area_frac: float = 0.15) -> np.ndarray | None:
    """Corner pixels (N x 2) of the largest bright, roughly convex region (the open book).

    Pages are the brightest large region under typical lighting; Otsu separates them
    from the table. Returns None if nothing plausible is found.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))  # fill text holes
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area_frac * gray.size:
        return None
    hull = cv2.convexHull(c)
    approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
    return approx.reshape(-1, 2).astype(float)


def locate_right_corner(img, geom, book, bcfg: dict, search_mm: float) -> tuple[float, float] | None:
    """Bottom-outer corner of the right page in book-frame mm, or None to use calibration.

    Detected outline corners are projected onto the page plane; the one closest to the
    calibrated corner wins, if it's within ``search_mm``.
    """
    if geom is None:
        return None
    px = find_paper_corners(img)
    if px is None:
        return None
    world = pixel_to_plane(px, geom.K, geom.T_world_cam, book.origin[2], geom.dist)
    pts = np.array([book.from_world(p)[:2] for p in world])
    nominal = np.array([bcfg["page_width_mm"], 0.0])
    d = np.linalg.norm(pts - nominal, axis=1)
    i = int(np.argmin(d))
    return (float(pts[i, 0]), float(pts[i, 1])) if d[i] <= search_mm else None


def find_book_quad(frame: np.ndarray, margin: float = 0.06) -> np.ndarray | None:
    """Corners (TL, TR, BR, BL pixels) of the open book, found from its printed text.

    For a fixed camera that may get bumped. Looking for "bright paper" fails when the
    table is whiter than the pages, so this finds the block of text-sized dark marks
    instead (big dark blobs — clamps, the arm — and specks are ignored), fits a rotated
    rectangle around it and pads it by ``margin`` to reach the page edges.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    marks = cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (3, 3), 0), 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY_INV, 25, 15)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(marks, 8)
    sizes_ok = ((stats[:, cv2.CC_STAT_HEIGHT] >= 3) & (stats[:, cv2.CC_STAT_HEIGHT] <= h * 0.05)
                & (stats[:, cv2.CC_STAT_WIDTH] <= w * 0.1) & (stats[:, cv2.CC_STAT_AREA] >= 4))
    sizes_ok[0] = False                                    # background
    text = np.where(sizes_ok[labels], 255, 0).astype(np.uint8)
    k = max(15, w // 60)                                   # merge letters/lines into one block
    blob = cv2.morphologyEx(text, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    blob = cv2.morphologyEx(blob, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k // 2, k // 2)))
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    areas = [cv2.contourArea(c) for c in contours]
    big = [c for c, a in zip(contours, areas) if a >= 0.15 * max(areas)]   # text blocks, not stray marks
    if sum(cv2.contourArea(c) for c in big) < 0.02 * h * w:
        return None
    (cx, cy), (rw, rh), angle = cv2.minAreaRect(np.vstack(big))
    box = cv2.boxPoints(((cx, cy), (rw * (1 + 2 * margin), rh * (1 + 2 * margin)), angle))
    s, d = box.sum(1), np.diff(box, axis=1).ravel()
    quad = np.array([box[np.argmin(s)], box[np.argmin(d)], box[np.argmax(s)], box[np.argmax(d)]])
    return np.clip(quad, [0, 0], [w - 1, h - 1]).astype(np.float32)
