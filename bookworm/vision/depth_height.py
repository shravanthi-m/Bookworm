"""Page height from the wrist camera's depth image, so the robot needs no external camera to know where the book is.

Depth pixels are lifted to world points with the camera pose from the frame system (the same geometry the colour image
uses; the RealSense depth stream is aligned to colour). ``bias_mm`` is ONE number measured by touching the page once:
(height the arm reports at contact) - (height depth reports there). It absorbs tool-length and depth offsets together,
so everything stays consistent as long as depth is always read from the home pose.
"""
from __future__ import annotations

import struct

import numpy as np

from ..hardware.base import CameraGeometry


def decode_depth(data: bytes) -> np.ndarray:
    """Viam 'image/vnd.viam.dep' -> HxW uint16 millimetres (0 = no reading)."""
    if data[:8] != b"DEPTHMAP":
        raise ValueError("not a Viam depth image")
    w, h = struct.unpack(">QQ", data[8:24])
    return np.frombuffer(data[24:], ">u2").reshape(h, w)


def world_heights(depth_mm: np.ndarray, geom: CameraGeometry, step: int = 1) -> np.ndarray:
    """HxWx3 world xyz for every ``step``-th depth pixel; NaN where there is no valid reading."""
    d = depth_mm[::step, ::step].astype(np.float64)
    v, u = np.mgrid[0:depth_mm.shape[0]:step, 0:depth_mm.shape[1]:step]
    K = geom.K
    cam = np.stack([(u - K[0, 2]) / K[0, 0] * d, (v - K[1, 2]) / K[1, 1] * d, d, np.ones_like(d)])
    w = (geom.T_world_cam @ cam.reshape(4, -1))[:3].T.reshape(*d.shape, 3)
    w[(d < 200) | (d > 1500)] = np.nan
    return w


def page_height_at(depth_mm: np.ndarray, geom: CameraGeometry, px, bias_mm: float = 0.0, radius: int = 6) -> float | None:
    """Page-surface height (world z, arm frame) at image pixel ``px``: median over a small patch, plus ``bias_mm``.
    None if the patch has too few readings (edges and shadows read as holes)."""
    u, v = int(round(px[0])), int(round(px[1]))
    h, w = depth_mm.shape
    r = radius
    sub = depth_mm[max(v - r, 0):v + r + 1, max(u - r, 0):u + r + 1]
    ys, xs = np.nonzero(sub)
    if len(ys) < 0.3 * sub.size:
        return None
    full = world_heights(depth_mm, geom)
    z = full[max(v - r, 0):v + r + 1, max(u - r, 0):u + r + 1, 2]
    z = z[np.isfinite(z)]
    if len(z) < 0.3 * sub.size:
        return None
    return float(np.median(z)) + bias_mm
