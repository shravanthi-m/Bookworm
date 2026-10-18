"""The robot as the calibration workshop (bookworm/gui) and the dynamic flip path use it.

Main's robot classes (base.py, mock.py, viam_robot.py) are left exactly as they are. The workshop needs a few more
things from the arm, so it gets its own subclasses here:

  * calls that retry once a dropped Viam session reconnects (SESSION_EXPIRED, context canceled, ...);
  * joint moves, speed, joint load (the touch detector), clearing a collision stop, a gripper opening in counts;
  * the wrist camera's depth image (depth scan).

``build_workshop_robot`` is the workshop's version of ``rig.build_robot``.
"""
from __future__ import annotations

import asyncio
import time

import numpy as np

from .base import POSE_KEYS, HardwareError
from .mock import MockRobot
from .viam_robot import ViamConnection, ViamRobot

# Errors that mean "this call was cut off", not "this call is wrong". SESSION_EXPIRED: the safety heartbeat lapsed, so
# viam-server stopped the arm and rejected the call; the SDK resets the session and the next call opens a new one.
TRANSIENT = ("session_expired", "context canceled", "channel closed", "streamterminated", "connection lost", "connection reset")


def is_transient(e: BaseException) -> bool:
    msg = f"{type(e).__name__}: {e}".lower()
    return any(t in msg for t in TRANSIENT)


class WorkshopMixin:
    """Defaults for what the workshop asks of any robot backend."""

    def stop(self) -> None:
        """Halt arm and gripper motion immediately."""

    def joint_positions(self) -> list[float]:
        raise HardwareError("this robot backend can't read joint positions")

    def move_joints(self, degrees: list[float]) -> None:
        raise HardwareError("this robot backend can't move by joints")

    def set_speed(self, deg_per_s: float) -> None:
        raise HardwareError("this robot backend can't set the arm speed")

    def gripper_position(self, value: float) -> None:
        """Move the gripper to an opening (0 = closed .. 850 = fully open)."""
        raise HardwareError("this robot backend can't set a gripper position")

    def gripper_get(self) -> float | None:
        return None

    def capture_depth(self) -> np.ndarray:
        """Depth image aligned to the colour frame, HxW uint16 millimetres (0 = no reading)."""
        raise HardwareError("this robot backend has no depth camera")

    def joint_load(self) -> list[float] | None:
        """Per-joint load (torque) as the arm reports it, or None."""
        return None

    def clear_error(self) -> None:
        """Clear a fault the arm raised (e.g. a collision stop). No-op by default."""


class WorkshopConnection(ViamConnection):
    def call_retry(self, make, timeout: float = 60.0, attempts: int = 3, say=None, what: str = "robot call"):
        """``make()`` builds the coroutine, so a call cut off by a connection hiccup can be sent again. Only for
        idempotent calls - every call here is (moves go to an absolute target, reads read, set_speed sets)."""
        for k in range(attempts):
            try:
                return self.call(make(), timeout)
            except Exception as e:  # noqa: BLE001
                if k == attempts - 1 or not is_transient(e):
                    raise
                if say:
                    say(f"{what}: connection hiccup ({str(e)[:60]}) - retrying ({k + 2} of {attempts})")
                time.sleep(0.4 * (k + 1))


class WorkshopViamRobot(WorkshopMixin, ViamRobot):
    def __init__(self, conn: WorkshopConnection, *a, **k):
        self.say = None                      # set by the workshop GUI so retries show up in its log
        super().__init__(conn, *a, **k)

    def _c(self, make, timeout: float = 60.0, attempts: int = 3, what: str = "robot call"):
        return self.conn.call_retry(make, timeout, attempts, say=self.say, what=what)

    def move_to(self, pose):
        from viam.proto.common import Pose

        self._c(lambda: self.arm.move_to_position(Pose(**{k: float(pose[k]) for k in POSE_KEYS})))

    def end_pose(self):
        p = self._c(lambda: self.arm.get_end_position())
        return {k: getattr(p, k) for k in POSE_KEYS}

    def gripper(self, action):
        if action == "open":
            self._c(lambda: self.grip.open())
        elif action == "grab":
            self._c(lambda: self.grip.grab())
        else:
            raise HardwareError(f"unknown gripper action {action!r}")

    def joint_positions(self):
        return list(self._c(lambda: self.arm.get_joint_positions()).values)

    def move_joints(self, degrees):
        from viam.proto.component.arm import JointPositions

        self._c(lambda: self.arm.move_to_joint_positions(JointPositions(values=[float(v) for v in degrees])))

    def set_speed(self, deg_per_s):
        r = self._c(lambda: self.arm.do_command({"set_speed": float(deg_per_s)}))
        if r and r.get("error"):
            raise HardwareError(f"set_speed failed: {r['error']}")

    def joint_load(self):
        """Per-joint torque from the xArm module's 'load' DoCommand; None if the module does not answer with a list."""
        r = self._c(lambda: self.arm.do_command({"load": ""}))
        v = (r or {}).get("load")
        try:
            return [float(x) for x in v] if v else None
        except (TypeError, ValueError):
            return None

    def clear_error(self):
        """Clear the xArm's error state (needed after a collision stop, per the module's README)."""
        r = self._c(lambda: self.arm.do_command({"clear_error": True}))
        if r and r.get("error"):
            raise HardwareError(f"clear_error failed: {r['error']}")

    def gripper_position(self, value):
        self._c(lambda: self.grip.do_command({"set": float(value)}))

    def gripper_get(self):
        return self._c(lambda: self.grip.do_command({"get": True})).get("pos")

    def stop(self):
        """Stop the arm and the gripper at once (both calls issued concurrently)."""
        async def _stop():
            await asyncio.gather(self.arm.stop(), self.grip.stop(), return_exceptions=True)

        self._c(lambda: _stop(), timeout=10)

    def capture_depth(self):
        from ..vision.depth_height import decode_depth

        images, _ = self._c(lambda: self.cam.get_images(filter_source_names=["depth"]))
        dep = [im for im in images if im.mime_type.startswith("image/vnd.viam.dep")]
        if not dep:
            raise HardwareError("wrist camera returned no depth image")
        return decode_depth(dep[0].data)


class WorkshopMock(WorkshopMixin, MockRobot):
    """main's MockRobot plus joints, speed and a gripper opening, recorded like its other calls."""

    def joint_positions(self):
        return list(getattr(self, "_joints", [0.0] * 6))

    def move_joints(self, degrees):
        self._record("move_joints", [round(float(v), 2) for v in degrees])
        self._joints = [float(v) for v in degrees]
        if hasattr(self, "_home_pose_for_joints"):          # tests: joint moves land on a known pose
            self._pose = dict(self._home_pose_for_joints)

    def set_speed(self, deg_per_s):
        self._record("speed", deg_per_s)

    def gripper_position(self, value):
        self._record("gripper_position", value)
        self._grip = float(value)

    def gripper_get(self):
        return getattr(self, "_grip", 850.0)

    def stop(self):
        self._record("stop")


def build_workshop_robot(cfg: dict, mock: bool = False):
    """(robot, close_fn): the Viam machine with the workshop's extras, or WorkshopMock with ``mock``."""
    if mock or cfg["robot"]["backend"] != "viam":
        cfg["robot"]["backend"] = "mock"
        robot = WorkshopMock(None, verbose=False)
        return robot, robot.close
    conn = WorkshopConnection()
    try:
        robot = WorkshopViamRobot(conn, **cfg["robot"])
    except Exception:
        conn.close()
        raise

    def close():
        try:
            robot.close()
        finally:
            conn.close()

    return robot, close
