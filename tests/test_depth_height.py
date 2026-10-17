import struct

import numpy as np

from bookworm.hardware import CameraGeometry
from bookworm.vision.depth_height import decode_depth, page_height_at, world_heights


def _geom():
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = np.diag([1, -1, -1]), [0, 0, 480]          # straight down from 480 mm
    return CameraGeometry(np.array([[900, 0, 64], [0, 900, 36], [0, 0, 1.0]]), None, T)


def test_decode_roundtrip():
    d = (np.arange(12, dtype=">u2") + 400).reshape(3, 4)
    raw = b"DEPTHMAP" + struct.pack(">QQ", 4, 3) + d.astype(">u2").tobytes()
    assert (decode_depth(raw) == d).all()


def test_flat_page_height_and_bias():
    depth = np.full((72, 128), 463, np.uint16)                      # a surface 17 mm above z=0 seen from 480
    g = _geom()
    assert page_height_at(depth, g, (64, 36)) == np.float64(17.0)
    assert abs(page_height_at(depth, g, (30, 50), bias_mm=12.5) - 29.5) < 0.1
    assert np.isnan(world_heights(np.zeros((4, 4), np.uint16), g)).all()


def test_holes_give_none():
    depth = np.zeros((72, 128), np.uint16)
    assert page_height_at(depth, _geom(), (64, 36)) is None
