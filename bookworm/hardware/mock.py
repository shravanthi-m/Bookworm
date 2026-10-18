"""Fake robot for developing without the arm. Images come from a folder of photos."""
from __future__ import annotations

import numpy as np

from .base import POSE_KEYS, Robot


class MockRobot(Robot):
    def __init__(self, camera=None, verbose: bool = False):
        self.camera = camera          # anything with capture(); FileCamera also has advance()
        self.verbose = verbose
        self.log: list[tuple] = []
        self._pose = dict.fromkeys(POSE_KEYS, 0.0)

    def _record(self, *entry):
        self.log.append(entry)
        if self.verbose:
            print("[mock robot]", *entry)

    def move_to(self, pose):
        self._record("move", {k: round(float(pose[k]), 1) for k in ("x", "y", "z")})
        self._pose = dict(pose)

    def end_pose(self):
        return dict(self._pose)

    def gripper(self, action):
        self._record("gripper", action)

    def capture(self):
        if self.camera is None:
            return np.full((1100, 1600, 3), 255, np.uint8)
        return self.camera.capture()

    def simulate_page_turn(self):
        """Called by the session after a flip so photo playback moves to the next spread."""
        self._record("page_turned")
        if hasattr(self.camera, "advance"):
            self.camera.advance()
