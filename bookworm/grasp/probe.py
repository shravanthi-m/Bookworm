"""Tool-tip geometry: where the tip is for a flange pose, and the flange pose that puts the tip somewhere.

The tip is computed from the arm's real orientation (the end-effector z axis), not assumed vertical; the tool length
is the one manual setting (gui.probe.tool_length_mm), changed when a different tool such as a pencil is fitted.
"""
from __future__ import annotations

import numpy as np

from ..hardware.base import POSE_KEYS


def tool_axis(pose: dict) -> np.ndarray:
    """Unit vector the tool points along (the end-effector z axis, from the orientation vector)."""
    o = np.array([pose["o_x"], pose["o_y"], pose["o_z"]], float)
    return o / np.linalg.norm(o)


def tip_of(pose: dict, tool_len: float) -> np.ndarray:
    return np.array([pose["x"], pose["y"], pose["z"]], float) + tool_len * tool_axis(pose)


def flange_for_tip(tip, like: dict, tool_len: float) -> dict:
    """The flange pose that puts the tool tip at ``tip``, keeping the orientation of ``like``."""
    f = np.asarray(tip, float) - tool_len * tool_axis(like)
    out = {k: float(like[k]) for k in POSE_KEYS}
    out["x"], out["y"], out["z"] = (float(v) for v in f)
    return out
