"""Pixel <-> world mapping for the calibrated wrist camera.

Camera frame convention (OpenCV / Viam pinhole): x right, y down, z forward out of the lens.
The book lies on a horizontal plane z = plane_z in the world frame.
"""
from __future__ import annotations

import cv2
import numpy as np


def pixel_to_plane(uv, K, T_world_cam, plane_z: float, dist=None) -> np.ndarray:
    """Intersect the camera ray through pixel(s) ``uv`` (N x 2) with the plane z = plane_z.

    Returns N x 3 world points.
    """
    uv = np.asarray(uv, float).reshape(-1, 1, 2)
    norm = cv2.undistortPoints(uv, K, dist).reshape(-1, 2)        # normalised image coords
    rays = np.column_stack([norm, np.ones(len(norm))]) @ T_world_cam[:3, :3].T
    origin = T_world_cam[:3, 3]
    s = (plane_z - origin[2]) / rays[:, 2]
    if np.any(s <= 0):
        raise ValueError("plane is behind the camera — check camera pose / frame config")
    return origin + s[:, None] * rays


def world_to_pixel(points, K, T_world_cam, dist=None) -> np.ndarray:
    """Project N x 3 world points into the image. Returns N x 2 pixels."""
    T_cam_world = np.linalg.inv(T_world_cam)
    rvec, _ = cv2.Rodrigues(T_cam_world[:3, :3])
    px, _ = cv2.projectPoints(np.asarray(points, float).reshape(-1, 3), rvec, T_cam_world[:3, 3], K, dist)
    return px.reshape(-1, 2)


def book_quad_pixels(book, geom, quad_mm) -> np.ndarray:
    """Pixel corners of a book-frame quad (from hardware.page_quad) in the current image."""
    world = np.array([book.to_world(x, y, 0.0) for x, y in quad_mm])
    return world_to_pixel(world, geom.K, geom.T_world_cam, geom.dist)
