"""Build the pencil/gripper mask for the wrist camera.

The pencil is held by the gripper and the camera is bolted to the arm, so the pencil
occupies the *same pixels* in every frame, whatever the arm pose. Given frames taken
from several different views, pixels whose colour barely changes across them — and
that connect to the image border, where the tool enters the view — are the tool.
No detector or training needed.
"""
from __future__ import annotations

import cv2
import numpy as np


def build_tool_mask(frames: list[np.ndarray], std_thresh: float = 12.0, min_area_frac: float = 0.002,
                    dilate_px: int = 15) -> np.ndarray:
    """Return a uint8 mask (255 = tool) from >= 4 frames of *different* scenes."""
    if len(frames) < 4:
        raise ValueError("need at least 4 frames from different views")
    h, w = frames[0].shape[:2]
    stack = np.stack([cv2.GaussianBlur(cv2.resize(f, (w, h)), (9, 9), 0).astype(np.float32) for f in frames])
    std = stack.std(axis=0).max(axis=2)                     # worst channel
    static = (std < std_thresh).astype(np.uint8)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(static, connectivity=8)
    mask = np.zeros((h, w), np.uint8)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        touches_border = x == 0 or y == 0 or x + bw == w or y + bh == h
        if touches_border and area >= min_area_frac * h * w:
            mask[labels == i] = 255
    if dilate_px:
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px)))
    return mask


def preview(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Frame with the mask tinted red, for checking."""
    out = frame.copy()
    red = np.zeros_like(out)
    red[..., 2] = 255
    sel = mask > 0
    out[sel] = (0.5 * out[sel] + 0.5 * red[sel]).astype(np.uint8)
    return out
