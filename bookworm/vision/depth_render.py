"""Pictures of a height map without any plotting library: a top-down colour map and a rotatable 3D surface.

Axes follow the camera image: image right is world -y, image up is world +x. Heights are mm.
"""
from __future__ import annotations

import cv2
import numpy as np

TABLE_GREY = (58, 58, 58)


def _colors(h: np.ndarray, lo: float, hi: float) -> np.ndarray:
    idx = (np.clip((h - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(idx.reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(*h.shape, 3)


def _colorbar(img: np.ndarray, lo: float, hi: float, x: int, y0: int, y1: int, label: str = "mm") -> None:
    n = y1 - y0
    vals = np.linspace(hi, lo, n)
    img[y0:y1, x:x + 14] = _colors(vals, lo, hi).reshape(n, 1, 3)
    for t in np.arange(np.ceil(lo / 10) * 10, hi + 0.1, 10):
        yy = int(y1 - (t - lo) / (hi - lo) * n)
        cv2.line(img, (x + 14, yy), (x + 19, yy), (230, 230, 230), 1)
        cv2.putText(img, f"{t:.0f}", (x + 22, yy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(img, label, (x, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)


def clean(z: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Drop readings far outside the plausible range and knock out single-cell spikes (mixed pixels at edges)."""
    z = np.where((z > lo - 8) & (z < hi + 30), z, np.nan).astype(np.float32)
    f = np.where(np.isfinite(z), z, -999.0).astype(np.float32)
    med = cv2.medianBlur(f, 3)
    out = np.where(np.isfinite(z) & (med > -900), med, np.nan)
    # a cell that disagrees with its median by a lot is a spike; keep the median
    return out.astype(np.float32)


def to_image_axes(x, y, z, book=None):
    """Grid arrays indexed [x, y] -> (u right, v up, z[v, u]) so that drawing matches the camera view."""
    z2 = z[:, ::-1]
    return -y[::-1], x, z2, None if book is None else book[:, ::-1]


def render_top(x, y, z, book, lo, hi, points=(), scale: int = 6) -> np.ndarray:
    u, v, z2, b2 = to_image_axes(x, y, clean(z, lo, hi), book)
    h, w = z2.shape
    img = _colors(np.nan_to_num(z2, nan=lo), lo, hi)
    img[~np.isfinite(z2)] = TABLE_GREY
    img = np.ascontiguousarray(img[::-1])                                  # row 0 = highest x (image up)
    img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    if b2 is not None:
        m = cv2.resize(np.ascontiguousarray(b2[::-1]).astype(np.uint8), (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
        cnt, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnt, -1, (255, 255, 255), 2)
    du, dv = (u[-1] - u[0]) / max(w - 1, 1), (v[-1] - v[0]) / max(h - 1, 1)
    for p in points:
        c = (int((p["u"] - u[0]) / du * scale + scale / 2), int((v[-1] - p["x"]) / dv * scale + scale / 2))
        if p.get("h") is None or not (0 <= c[0] < img.shape[1] and 0 <= c[1] < img.shape[0]):
            continue
        cv2.circle(img, c, 5, (255, 255, 255), -1); cv2.circle(img, c, 5, (0, 0, 0), 1)
        cv2.putText(img, f"{p['h']:.0f}", (c[0] + 8, c[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"{p['h']:.0f}", (c[0] + 8, c[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    pad = np.full((img.shape[0], 78, 3), (24, 24, 24), np.uint8)
    out = np.hstack([img, pad])
    _colorbar(out, lo, hi, img.shape[1] + 10, 24, img.shape[0] - 14)
    cv2.putText(out, "up = +x", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, "right = -y", (out.shape[1] - 150, out.shape[0] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def render_surface(x, y, z, lo, hi, az: float = 35.0, el: float = 40.0, exaggerate: float = 3.0,
                   size=(900, 620), book=None) -> np.ndarray:
    """Painter's-algorithm render of the surface as coloured, shaded quads. ``az`` turns it about the vertical
    axis (0 = looking up the image), ``el`` is the viewing elevation above the table (90 = straight down)."""
    u, v, z2, b2 = to_image_axes(x, y, clean(z, lo, hi), book)
    h, w = z2.shape
    U, V = np.meshgrid(u - u.mean(), v - v.mean())
    H = np.nan_to_num(z2, nan=0.0) * exaggerate
    a, e = np.radians(az), np.radians(el)
    xr = U * np.cos(a) + V * np.sin(a)                       # right on screen
    yd = -U * np.sin(a) + V * np.cos(a)                      # away from the viewer
    sx, sy = xr, -(H * np.cos(e) + yd * np.sin(e))           # screen coords (y grows downward)
    depth = yd * np.cos(e) - H * np.sin(e)                   # larger = farther
    W_, H_ = size
    margin = 70
    # frame the table-level footprint plus the tallest point so rotation never clips
    ext_x = max(abs(xr).max(), 1.0)
    ext_y = max(abs(sy).max(), 1.0)
    sc = min((W_ - margin * 2) / (2 * ext_x), (H_ - margin * 1.2) / (2 * ext_y))
    cx, cy = (W_ - 40) / 2, H_ / 2 + 10
    P = np.stack([cx + sx * sc, cy + sy * sc], -1)
    img = np.full((H_, W_, 3), (24, 24, 24), np.uint8)
    # table-level footprint
    fp = np.array([[P[0, 0]], [P[0, -1]], [P[-1, -1]], [P[-1, 0]]]).astype(np.int32)
    base = np.stack([cx + xr * sc, cy - (yd * np.sin(e)) * sc], -1)
    fp = np.array([base[0, 0], base[0, -1], base[-1, -1], base[-1, 0]], np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(img, [fp], (44, 44, 44)); cv2.polylines(img, [fp], True, (90, 90, 90), 1, cv2.LINE_AA)
    valid = np.isfinite(z2)
    quads = []
    for i in range(h - 1):
        for j in range(w - 1):
            if valid[i, j] and valid[i + 1, j] and valid[i, j + 1] and valid[i + 1, j + 1]:
                quads.append((i, j))
    quads = np.array(quads)
    if len(quads):
        i, j = quads[:, 0], quads[:, 1]
        d = (depth[i, j] + depth[i + 1, j] + depth[i, j + 1] + depth[i + 1, j + 1]) / 4
        order = np.argsort(-d)
        hm = (z2[i, j] + z2[i + 1, j] + z2[i, j + 1] + z2[i + 1, j + 1]) / 4
        col = _colors(hm, lo, hi).astype(np.float32)
        # simple shading from the local slope
        dzdx = (z2[i, j + 1] + z2[i + 1, j + 1] - z2[i, j] - z2[i + 1, j]) / 2
        dzdy = (z2[i + 1, j] + z2[i + 1, j + 1] - z2[i, j] - z2[i, j + 1]) / 2
        shade = np.clip(1.0 - 0.025 * (dzdx * 0.6 - dzdy * 0.8), 0.6, 1.25)
        col = np.clip(col * shade[:, None], 0, 255).astype(np.uint8)
        for k in order:
            ii, jj = i[k], j[k]
            poly = np.array([P[ii, jj], P[ii, jj + 1], P[ii + 1, jj + 1], P[ii + 1, jj]], np.int32)
            c = tuple(int(t) for t in col[k])
            cv2.fillConvexPoly(img, poly, c, cv2.LINE_AA)
            cv2.polylines(img, [poly.reshape(-1, 1, 2)], True, c, 1)
    _colorbar(img, lo, hi, W_ - 62, 40, H_ - 40)
    cv2.putText(img, f"height x{exaggerate:g} exaggerated", (10, H_ - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return img
