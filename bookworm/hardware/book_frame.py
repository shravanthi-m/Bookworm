"""Book coordinate frame: maps book-relative millimetres to world coordinates.

origin at the bottom of the spine, x toward the right page's outer edge,
y up the spine, z = world up with 0 on the right page surface.
"""
from __future__ import annotations

import numpy as np


class BookFrame:
    def __init__(self, spine_bottom, spine_top, right_corner):
        o = np.asarray(spine_bottom, float)
        ey = np.asarray(spine_top, float) - o
        ey[2] = 0
        ey /= np.linalg.norm(ey)
        ex = np.asarray(right_corner, float) - o
        ex[2] = 0
        ex -= (ex @ ey) * ey
        ex /= np.linalg.norm(ex)
        self.ex, self.ey, self.ez = ex, ey, np.array([0.0, 0.0, 1.0])
        self.origin = np.array([o[0], o[1], float(right_corner[2])])

    @classmethod
    def from_config(cls, bcfg: dict) -> "BookFrame | None":
        pts = [bcfg.get(k) for k in ("spine_bottom", "spine_top", "right_corner")]
        return cls(*pts) if all(p is not None for p in pts) else None

    @classmethod
    def identity(cls) -> "BookFrame":
        """For mock runs: book frame == world frame."""
        return cls([0, 0, 0], [0, 1, 0], [1, 0, 0])

    def to_world(self, x, y, z=0.0) -> np.ndarray:
        return self.origin + x * self.ex + y * self.ey + z * self.ez

    def from_world(self, p) -> np.ndarray:
        d = np.asarray(p, float) - self.origin
        return np.array([d @ self.ex, d @ self.ey, d @ self.ez])


def page_quad(region: str, w: float, h: float) -> np.ndarray:
    """Book-frame corners (TL, TR, BR, BL as seen by a reader) of a region on the page plane."""
    x0, x1 = {"spread": (-w, w), "left": (-w, 0), "right": (0, w)}[region]
    return np.array([[x0, h], [x1, h], [x1, 0], [x0, 0]], float)
