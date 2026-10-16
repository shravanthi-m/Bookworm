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
