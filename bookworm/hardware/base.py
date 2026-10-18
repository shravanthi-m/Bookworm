"""Robot interface. Sessions and modes only talk to this, so Viam and the mock are swappable."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

POSE_KEYS = ("x", "y", "z", "o_x", "o_y", "o_z", "theta")


class HardwareError(RuntimeError):
    pass


@dataclass
class CameraGeometry:
    """Everything needed to map pixels <-> world for one captured frame."""
    K: np.ndarray                 # 3x3 intrinsics, scaled to the captured image size
    dist: np.ndarray | None       # distortion coefficients (OpenCV order) or None
    T_world_cam: np.ndarray       # 4x4 camera pose in the world frame at capture time


class Robot(ABC):
    @abstractmethod
    def move_to(self, pose: dict) -> None:
        """Move the arm flange to a world pose {x, y, z, o_x, o_y, o_z, theta} (mm, deg)."""

    @abstractmethod
    def end_pose(self) -> dict:
        """Current flange pose in the world frame."""

    @abstractmethod
    def gripper(self, action: str) -> None:
        """'open' or 'grab'."""

    @abstractmethod
    def capture(self) -> np.ndarray:
        """BGR image from the wrist camera."""

    def camera_geometry(self) -> CameraGeometry | None:
        """Intrinsics + current camera pose, or None if unknown (mock / photos)."""
        return None

    def stop(self) -> None:
        """Halt arm and gripper motion immediately."""

    def joint_positions(self) -> list[float]:
        """Joint angles in degrees."""
        raise HardwareError("this robot backend can't read joint positions")

    def move_joints(self, degrees: list[float]) -> None:
        """Move to joint angles (degrees). Predictable, unlike orientation changes near vertical."""
        raise HardwareError("this robot backend can't move by joints")

    def set_speed(self, deg_per_s: float) -> None:
        """Set the arm's joint speed (deg/s). Raises HardwareError if the backend can't."""
        raise HardwareError("this robot backend can't set the arm speed")

    def gripper_position(self, value: float) -> None:
        """Move the gripper to a specific opening (0 = closed .. 850 = fully open)."""
        raise HardwareError("this robot backend can't set a gripper position")

    def gripper_get(self) -> float | None:
        """Current gripper opening, or None if unknown."""
        return None

    def capture_depth(self) -> np.ndarray:
        """Depth image aligned to the colour frame, HxW uint16 millimetres (0 = no reading)."""
        raise HardwareError("this robot backend has no depth camera")

    def joint_load(self) -> list[float] | None:
        """Per-joint load (torque) as the arm reports it, or None if this backend has no such reading."""
        return None

    def clear_error(self) -> None:
        """Clear a fault the arm raised (e.g. a collision stop) so it accepts motion commands again. No-op by default."""

    def close(self) -> None:
        pass


def tip_to_flange(tip_xyz, rcfg: dict) -> dict:
    """Flange pose that puts the eraser tip at ``tip_xyz`` with the tool pointing straight down."""
    x, y, z = (float(v) for v in tip_xyz)
    return {"x": x, "y": y, "z": z + rcfg["tool_length_mm"], **rcfg["tip_orientation"]}


def flange_to_tip(pose: dict, rcfg: dict) -> list[float]:
    """Inverse of tip_to_flange (assumes the tool points straight down)."""
    return [pose["x"], pose["y"], pose["z"] - rcfg["tool_length_mm"]]
