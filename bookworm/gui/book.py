"""Where the book is: six points found in the first capture from home, lifted to the arm's world frame.

The points are pixels of the home photo; with the camera geometry of that photo and the depth height map, each
becomes a world (x, y, z) on the page surface (the camera ray is intersected with the measured surface).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..hardware.base import CameraGeometry
from ..vision.edge_refine import page_mask
from ..vision.geometry import pixel_to_plane, world_to_pixel

# Slot names are in IMAGE terms as seen at home (the book lies rotated), except the two page centres.
SLOTS = [("cl", "Left page centre"), ("cr", "Right page centre"), ("tl", "Top-left corner"),
         ("tr", "Top-right corner"), ("br", "Bottom-right corner"), ("bl", "Bottom-left corner")]
KEYS = [k for k, _ in SLOTS]
LABEL = dict(SLOTS)
CORNERS = ("tl", "tr", "br", "bl")


@dataclass
class Book:
    image: np.ndarray | None = None
    geom: CameraGeometry | None = None                # camera geometry of the home photo
    home_pose: dict | None = None
    taken_at: str | None = None
    px: dict = field(default_factory=dict)            # slot -> (u, v) in the home photo
    height_fn: object = None                          # (world x, y) -> page surface z in the arm frame, or None
    fallback_z: float = 35.0                          # surface z used before there is a depth map

    def surface_z(self, x: float, y: float) -> float:
        if self.height_fn is None:
            return self.fallback_z
        h = self.height_fn(float(x), float(y))
        return self.fallback_z if h is None else float(h)

    def world3(self, key: str) -> list | None:
        """World (x, y, z) of a point: the camera ray through its pixel, intersected with the measured surface."""
        if key not in self.px or self.geom is None:
            return None
        return self.pixel_world(self.px[key], self.geom)

    def pixel_world(self, px, g: CameraGeometry) -> list:
        """World (x, y, z) where the camera ray through pixel ``px`` (of a photo with geometry ``g``) meets the surface."""
        z = self.fallback_z
        w = pixel_to_plane([px], g.K, g.T_world_cam, z, g.dist)[0]
        for _ in range(6):
            z2 = self.surface_z(w[0], w[1])
            w = pixel_to_plane([px], g.K, g.T_world_cam, z2, g.dist)[0]
            if abs(z2 - z) < 0.2:
                z = z2
                break
            z = z2
        return [round(float(w[0]), 1), round(float(w[1]), 1), round(float(z), 1)]

    def centre(self) -> np.ndarray | None:
        a, b = self.world3("cl"), self.world3("cr")
        return None if a is None or b is None else (np.array(a[:2]) + np.array(b[:2])) / 2

    def demo_xy(self, key: str, inset_mm: float) -> list | None:
        """Where the demo touches for a point: the page centres as they are; corners moved ``inset_mm`` in towards the
        middle of the spread, so the tip lands on the page and not on its edge (the depth map is blank at the edge)."""
        w, c = self.world3(key), self.centre()
        if w is None:
            return None
        xy = np.array(w[:2], float)
        if key in CORNERS and c is not None and inset_mm > 0:
            d = c - xy
            n = float(np.linalg.norm(d))
            if n > 1e-6:
                xy = xy + d / n * min(inset_mm, n / 2)
        return [round(float(xy[0]), 1), round(float(xy[1]), 1)]

    # ---- drawing
    def _project(self, geom: CameraGeometry, pts):
        return world_to_pixel(np.array(pts, float), geom.K, geom.T_world_cam, geom.dist)

    def draw(self, img: np.ndarray, geom: CameraGeometry, results: dict | None = None, tip=None) -> np.ndarray:
        """Points (and the outline of the spread) drawn on an image from camera ``geom``. Points are world positions,
        so they stay on the book when the wrist camera moves. ``results`` colours demo points: green on target, amber
        met the page early, red refused or skipped. ``tip``: world (x, y, z) of the tool tip, drawn as a magenta cross."""
        img = img.copy()
        results = results or {}
        h, w_ = img.shape[:2]
        T_cw = np.linalg.inv(geom.T_world_cam)
        world = {k: self.world3(k) for k in KEYS}
        seen = {k: v for k, v in world.items() if v is not None}
        if not seen:
            return img
        px = dict(zip(seen, self._project(geom, list(seen.values()))))
        ok = {k: (T_cw @ np.array([*seen[k], 1.0]))[2] > 10 and 0 <= px[k][0] < w_ and 0 <= px[k][1] < h for k in seen}
        ring = [k for k in CORNERS if k in px and ok[k]]
        if len(ring) == 4:
            cv2.polylines(img, [np.array([px[k] for k in ring], np.int32).reshape(-1, 1, 2)], True, (255, 200, 0), 2)
        colour = {"on": (80, 220, 80), "early": (40, 170, 255), "skip": (60, 60, 255)}
        for k in seen:
            if not ok[k]:
                continue
            c = (int(px[k][0]), int(px[k][1]))
            col = colour.get((results.get(k) or {}).get("verdict"), (0, 220, 255))
            cv2.circle(img, c, 10, col, 2)
            cv2.putText(img, k, (c[0] + 12, c[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        if tip is not None and (T_cw @ np.array([*tip, 1.0]))[2] > 10:
            u, v = self._project(geom, [tip])[0]
            if 0 <= u < w_ and 0 <= v < h:
                cv2.drawMarker(img, (int(u), int(v)), (255, 0, 255), cv2.MARKER_CROSS, 36, 3)
                cv2.putText(img, "tip", (int(u) + 14, int(v) + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        return img


def auto_targets(img: np.ndarray) -> dict | None:
    """Outer corners of the open spread (the biggest pale blob near the middle) and the centre of each page, as pixels."""
    mask = page_mask(img, s_max=60, hue=(75, 135))          # bluish-white paper; orange wood and dark carpet drop out
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        return None
    H, W = mask.shape
    best, best_score = None, 0.0
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 0.04 * H * W:
            continue
        if x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2:          # at the border: sweater, gripper, etc....
            b = labels == i
            if int(b[0].sum() + b[-1].sum() + b[:, 0].sum() + b[:, -1].sum()) > 60:   # ...unless it only grazes it
                continue
        if area > best_score:
            best, best_score = i, area
    if best is None:
        return None
    cnts, _ = cv2.findContours((labels == best).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
    eps = 0.01
    quad = cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True)
    while len(quad) > 4 and eps < 0.2:
        eps += 0.005
        quad = cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True)
    if len(quad) != 4:
        return None
    pts = quad.reshape(-1, 2).astype(float)
    s, d = pts.sum(1), pts[:, 0] - pts[:, 1]
    tl, br = pts[np.argmin(s)], pts[np.argmax(s)]
    tr, bl = pts[np.argmax(d)], pts[np.argmin(d)]
    mt, mb = (tl + tr) / 2, (bl + br) / 2                                   # spine ends
    cl = (tl + mt + mb + bl) / 4                                            # centroid of the image-left page
    cr = (mt + tr + br + mb) / 4                                            # ... and of the image-right page
    r = lambda p: (round(float(p[0]), 1), round(float(p[1]), 1))  # noqa: E731
    return {"tl": r(tl), "tr": r(tr), "br": r(br), "bl": r(bl), "cl": r(cl), "cr": r(cr)}
