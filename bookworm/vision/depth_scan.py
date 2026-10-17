"""Height map of the book from several depth views taken from different arm positions.

Each view is levelled to the TABLE (a plane fitted to the wood pixels), so heights are 'above the table' and the
pose-dependent depth offset of the arm drops out. Views are then fused on a world-frame grid. The book is the raised
region; the table around it gives the drop-off. ``arm_offset_mm`` (table height in the arm's frame, from one touch)
turns heights above the table into the z values the arm uses.
"""
from __future__ import annotations

import cv2
import numpy as np

from .depth_height import world_heights


def wood_mask(color: np.ndarray) -> np.ndarray:
    """Orange wood table: hue ~5-25, decently saturated. The page (bluish-white) and carpet fall out."""
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = (h >= 5) & (h <= 25) & (s >= 60) & (v >= 110)
    m = cv2.erode(m.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    m[: int(0.30 * m.shape[0])] = False                      # far table edge / floor beyond it
    return m


def fit_plane(pts: np.ndarray, iters: int = 4, clip: float = 6.0) -> np.ndarray:
    """z = a + b x + c y by least squares with outlier clipping; returns (a, b, c)."""
    A = np.c_[np.ones(len(pts)), pts[:, 0], pts[:, 1]]
    keep = np.ones(len(pts), bool)
    for _ in range(iters):
        coef, *_ = np.linalg.lstsq(A[keep], pts[keep, 2], rcond=None)
        res = pts[:, 2] - A @ coef
        keep = np.abs(res) < clip
        if keep.sum() < 50:
            break
    return coef


def table_plane(depth, color, geom, step: int = 2):
    """Table plane z = a + b x + c y in the arm's world frame, fitted to the wood pixels of one view.
    Returns (a, b, c, n_wood_pixels)."""
    W = world_heights(depth, geom, step)
    wm = wood_mask(color)[::step, ::step] & np.isfinite(W[..., 2])
    pts = W[wm]
    if len(pts) < 500:
        raise ValueError("too few table pixels to fit the table plane")
    a, b, c = fit_plane(pts)
    return float(a), float(b), float(c), int(wm.sum())


def tilt_deg(b: float, c: float) -> float:
    return float(np.degrees(np.arctan(np.hypot(b, c))))


def level_view(depth, color, geom, step: int = 2, slope=None):
    """World xyz points of one view (HxWx3, NaN = none), with the table plane subtracted from z.
    ``slope`` = (b, c) from the home snapshot: only the plane's height is refitted, which is steadier than
    refitting the tilt from every view. Returns (points, (a, b, c), wood pixel count)."""
    W = world_heights(depth, geom, step)
    wm = wood_mask(color)[::step, ::step] & np.isfinite(W[..., 2])
    pts = W[wm]
    if len(pts) < 500:
        raise ValueError("too few table pixels to level this view")
    if slope is None:
        a, b, c = fit_plane(pts)
    else:
        b, c = float(slope[0]), float(slope[1])
        r = pts[:, 2] - b * pts[:, 0] - c * pts[:, 1]
        a = float(np.median(r[np.abs(r - np.median(r)) < 8]))
    out = W.copy()
    out[..., 2] = W[..., 2] - (a + b * W[..., 0] + c * W[..., 1])
    return out, (a, b, c), int(wm.sum())


def fuse(views, cell: float = 4.0, bounds=(250, 640, -330, 200), edge_px: float = 0.06, slope=None):
    """views: list of (depth, color, geom). Returns dict(x, y, z (nx x ny, NaN = unseen), count, planes)."""
    x0, x1, y0, y1 = bounds
    nx, ny = int((x1 - x0) / cell), int((y1 - y0) / cell)
    S, C = np.zeros(nx * ny), np.zeros(nx * ny)
    planes = []
    for depth, color, geom in views:
        P, plane, _ = level_view(depth, color, geom, slope=slope)
        planes.append(plane)
        h, w = P.shape[:2]
        v, u = np.mgrid[0:h, 0:w]
        ok = np.isfinite(P[..., 2]) & (u > edge_px * w) & (u < (1 - edge_px) * w) & (v > edge_px * h) & (v < (1 - edge_px) * h)
        # drop depth-discontinuity pixels (mixed pixels along edges read as flying points)
        z = np.where(ok, P[..., 2], np.nan).astype(np.float32)
        gx, gy = np.abs(np.diff(np.nan_to_num(z), axis=1, append=0)), np.abs(np.diff(np.nan_to_num(z), axis=0, append=0))
        ok &= (gx < 8) & (gy < 8)
        p = P[ok]
        gi, gj = ((p[:, 0] - x0) / cell).astype(int), ((p[:, 1] - y0) / cell).astype(int)
        m = (gi >= 0) & (gi < nx) & (gj >= 0) & (gj < ny)
        k = gi[m] * ny + gj[m]
        S += np.bincount(k, p[m, 2], nx * ny)
        C += np.bincount(k, None, nx * ny)
    Z = np.where(C >= 2, S / np.maximum(C, 1), np.nan).reshape(nx, ny)
    xs = x0 + (np.arange(nx) + 0.5) * cell
    ys = y0 + (np.arange(ny) + 0.5) * cell
    return {"x": xs, "y": ys, "z": Z, "count": C.reshape(nx, ny), "planes": planes, "cell": cell}


def book_region(z: np.ndarray, cell: float, raised_mm: float = 6.0):
    """Boolean mask of the raised region (largest blob above ``raised_mm`` over the table), lightly smoothed."""
    m = np.nan_to_num(z, nan=-99) > raised_mm
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return np.zeros_like(m, bool)
    best = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return cv2.morphologyEx((lab == best).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0


def analyze(F, arm_offset_mm: float = 0.0, margin_mm: float = 60.0):
    """Book outline, key heights and cross-sections from a fused map. None if no raised region was found.
    Coordinates in the returned dicts follow the camera image: u = -y (right), x = up."""
    Z, x, y, cell = F["z"], F["x"], F["y"], float(F["cell"])
    book = book_region(Z, cell)
    if book.sum() < 50:
        return None
    ii, jj = np.nonzero(book)
    m = int(margin_mm / cell)
    i0, i1 = max(ii.min() - m, 0), min(ii.max() + m, len(x) - 1)
    j0, j1 = max(jj.min() - m, 0), min(jj.max() + m, len(y) - 1)
    z = Z[i0:i1 + 1, j0:j1 + 1].copy()
    z[np.abs(np.nan_to_num(z, nan=999)) > 120] = np.nan
    xs, ys, bk = x[i0:i1 + 1], y[j0:j1 + 1], book[i0:i1 + 1, j0:j1 + 1]
    zz, u = z[:, ::-1], -ys[::-1]                               # image orientation
    bb = bk[:, ::-1]

    def med(r, c, rad=2):
        p = zz[max(int(r) - rad, 0):int(r) + rad + 1, max(int(c) - rad, 0):int(c) + rad + 1]
        p = p[np.isfinite(p)]
        return float(np.median(p)) if len(p) else None

    bi, bj = np.nonzero(bb)
    rect = cv2.minAreaRect(np.c_[bj, bi].astype(np.float32))
    cen = np.array(rect[0])
    box = cen + (cv2.boxPoints(rect) - cen) * 0.8
    srt = sorted(box.tolist(), key=lambda p: (-p[1], p[0]))
    top, bot = sorted(srt[:2], key=lambda p: p[0]), sorted(srt[2:], key=lambda p: p[0])
    pts = []
    for name, p in zip(("top-left", "top-right", "bottom-left", "bottom-right"), (top[0], top[1], bot[0], bot[1])):
        pts.append({"name": name, "u": round(float(u[0] + p[0] * cell), 1), "x": round(float(xs[0] + p[1] * cell), 1),
                    "h": None if (h := med(p[1], p[0])) is None else round(h, 1)})
    h = med(cen[1], cen[0])
    pts.append({"name": "centre", "u": round(float(u[0] + cen[0] * cell), 1), "x": round(float(xs[0] + cen[1] * cell), 1),
                "h": None if h is None else round(h, 1)})
    v = zz[bb & np.isfinite(zz)]
    ring = ((cv2.dilate(bb.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0)
            & ~(cv2.dilate(bb.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0) & np.isfinite(zz) & (np.abs(np.nan_to_num(zz)) < 60))
    stats = {"cell_mm": cell, "book_w_mm": int(round(rect[1][0] * cell)), "book_h_mm": int(round(rect[1][1] * cell)),
             "area_cm2": int(round(bb.sum() * cell * cell / 100)),
             "p5": round(float(np.percentile(v, 5)), 1), "p50": round(float(np.percentile(v, 50)), 1),
             "p95": round(float(np.percentile(v, 95)), 1),
             "table": round(float(np.median(zz[ring])), 1) if ring.any() else None,
             "table_sd": round(float(zz[ring].std()), 1) if ring.any() else None, "arm_offset_mm": arm_offset_mm}
    ci, cj = int(round(bi.mean())), int(round(bj.mean()))
    nz = lambda a: [None if not np.isfinite(t) else round(float(t), 1) for t in a]  # noqa: E731
    across = [{"name": f"x = {xs[i]:.0f} mm", "at": nz(zz[i]) } for i in (ci - 12, ci, ci + 12) if 0 <= i < zz.shape[0]]
    along = [{"name": f"-y = {u[j]:.0f} mm", "at": nz(zz[:, j])} for j in (cj - 14, cj, cj + 14) if 0 <= j < zz.shape[1]]
    return {"x": xs, "y": ys, "z": z, "book": bk, "points": pts, "stats": stats,
            "profiles": {"across": {"axis": [round(float(t), 1) for t in u], "lines": across},
                         "along": {"axis": [round(float(t), 1) for t in xs], "lines": along}}}
