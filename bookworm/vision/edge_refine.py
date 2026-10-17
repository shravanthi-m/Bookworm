"""Snap a coarse hint (a VLM point or a click) to the real page edge, then lift it to world coordinates.

Classical vision, no network: segment the pale page from the table (low saturation, bright), take the
page blob under the hint, fit a line to the outline segment nearest the hint, and report the corner,
the fitted edge and a grasp point on it.  The world position comes from intersecting camera rays with
a horizontal plane (``pixel_to_plane``), so it needs the page height, not a depth image.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import pixel_to_plane


@dataclass
class Refined:
    corner: tuple[float, float]
    edge: tuple[tuple[float, float], tuple[float, float]]    # fitted segment endpoints, pixels
    grasp: tuple[float, float]                               # pinch point on the edge, pixels
    score: float                                             # 0..1, how much of the segment is real outline


def page_mask(img_bgr: np.ndarray, s_max: int = 38, v_min: int = 135, hue: tuple[int, int] | None = None) -> np.ndarray:
    """Pale, unsaturated pixels = paper (carpet is dark; ``hue`` can also exclude orange wood). Text holes are closed."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    m = (hsv[..., 1] <= s_max) & (hsv[..., 2] >= v_min)
    if hue is not None:
        m &= (hsv[..., 0] >= hue[0]) & (hsv[..., 0] <= hue[1])
    m = m.astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))


def _dist_to_segment(p, a, b) -> float:
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1)
    return float(np.linalg.norm(p - (a + t * ab)))


def refine_edge(img_bgr: np.ndarray, hint, search_px: int = 90, corner_px: float = 45,
                min_edge_px: float = 40, inlier_px: float = 4, **mask_kw) -> Refined | None:
    """``hint`` = (x, y) near the page edge you want. Returns None if no plausible edge is found."""
    h = np.asarray(hint, float)
    mask = page_mask(img_bgr, **mask_kw)

    # the page blob under the hint: largest component touching a disc around it
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    disc = np.zeros(mask.shape, np.uint8)
    cv2.circle(disc, (int(h[0]), int(h[1])), search_px, 1, -1)
    ids = [i for i in np.unique(labels[disc > 0]) if i != 0]
    if not ids:
        return None
    best = max(ids, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    if stats[best, cv2.CC_STAT_AREA] < 2000:
        return None
    cnts, _ = cv2.findContours((labels == best).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    pts = c.reshape(-1, 2).astype(float)
    poly = cv2.approxPolyDP(c, 0.008 * cv2.arcLength(c, True), True).reshape(-1, 2).astype(float)
    if len(poly) < 3:
        return None

    # the polygon edge closest to the hint (ignoring the image border and short bits)
    H, W = mask.shape
    best_seg, best_d = None, 1e9
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        if np.linalg.norm(b - a) < min_edge_px:
            continue
        on_border = all(p[0] <= 2 or p[0] >= W - 3 or p[1] <= 2 or p[1] >= H - 3 for p in (a, b)) and \
            (abs(a[0] - b[0]) < 3 or abs(a[1] - b[1]) < 3) and (a[0] in (0, W - 1) or a[1] in (0, H - 1))
        if on_border:
            continue
        d = _dist_to_segment(h, a, b)
        if d < best_d:
            best_seg, best_d = (a, b), d
    if best_seg is None or best_d > search_px:
        return None
    a, b = best_seg

    # refit the line on the real outline pixels along that segment
    ab = b - a
    L = np.linalg.norm(ab)
    u = ab / L
    rel = pts - a
    along = rel @ u
    perp = np.abs(rel[:, 0] * u[1] - rel[:, 1] * u[0])
    keep = (perp < inlier_px * 2) & (along > -0.05 * L) & (along < 1.05 * L)
    if keep.sum() < 10:
        return None
    vx, vy, x0, y0 = cv2.fitLine(pts[keep].astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    d_ = np.array([vx, vy])
    p0 = np.array([x0, y0])
    if np.dot(d_, u) < 0:
        d_ = -d_
    t = (pts[keep] - p0) @ d_
    inl = np.abs((pts[keep] - p0)[:, 0] * d_[1] - (pts[keep] - p0)[:, 1] * d_[0]) < inlier_px
    if inl.sum() < 10:
        return None
    # keep only the unbroken run of outline pixels around the hint (a pale strip can bridge to far-off things)
    ts = np.sort(t[inl])
    tg0 = float((h - p0) @ d_)
    k0 = int(np.argmin(np.abs(ts - tg0)))
    lo = hi = k0
    while lo > 0 and ts[lo] - ts[lo - 1] < 15:
        lo -= 1
    while hi < len(ts) - 1 and ts[hi + 1] - ts[hi] < 15:
        hi += 1
    t_lo, t_hi = float(ts[lo]), float(ts[hi])
    e0, e1 = p0 + t_lo * d_, p0 + t_hi * d_
    score = float(min(1.0, (hi - lo + 1) / max(t_hi - t_lo, 1.0)))

    # corner: the polygon vertex nearest the hint if close, else the nearer end of the fitted edge
    vd = np.linalg.norm(poly - h, axis=1)
    corner = poly[int(np.argmin(vd))] if vd.min() <= corner_px else (e0 if np.linalg.norm(e0 - h) < np.linalg.norm(e1 - h) else e1)

    # grasp point: the hint projected onto the fitted segment (kept off the very ends)
    tg = float(np.clip((h - p0) @ d_, t_lo + 0.1 * (t_hi - t_lo), t_hi - 0.1 * (t_hi - t_lo)))
    g = p0 + tg * d_
    f = lambda p: (float(p[0]), float(p[1]))  # noqa: E731
    return Refined(f(corner), (f(e0), f(e1)), f(g), score)


def to_world(r: Refined, geom, plane_z: float) -> dict:
    """World (x, y, z) in mm for the refined points, on the horizontal plane z = ``plane_z``."""
    px = np.array([r.corner, r.edge[0], r.edge[1], r.grasp])
    w = pixel_to_plane(px, geom.K, geom.T_world_cam, plane_z, geom.dist)
    return {k: [round(float(v), 1) for v in p] for k, p in zip(("corner", "edge_a", "edge_b", "grasp"), w)}


def draw_refined(img_bgr: np.ndarray, r: Refined, hint=None) -> np.ndarray:
    out = img_bgr.copy()
    ip = lambda p: (int(round(p[0])), int(round(p[1])))  # noqa: E731
    cv2.line(out, ip(r.edge[0]), ip(r.edge[1]), (0, 220, 0), 3)
    cv2.circle(out, ip(r.corner), 9, (0, 255, 255), -1)
    cv2.drawMarker(out, ip(r.grasp), (255, 0, 255), cv2.MARKER_CROSS, 34, 3)
    if hint is not None:
        cv2.circle(out, ip(hint), 7, (255, 255, 255), 2)
    return out
