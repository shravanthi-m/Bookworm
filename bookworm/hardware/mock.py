"""Fake hardware for developing/testing software without the rig."""
from __future__ import annotations

from .base import Actuators


class MockActuators(Actuators):
    def __init__(self, on_flip=None, verbose: bool = False):
        self.log: list[tuple] = []
        self.on_flip = on_flip      # e.g. FileCamera.advance, called when the arm sweeps left
        self.verbose = verbose
        self._angle = 0.0

    def _record(self, *entry):
        self.log.append(entry)
        if self.verbose:
            print("[mock hw]", *entry)

    def clamp(self, side, engaged):
        self._record("clamp", side, engaged)

    def lift(self, down):
        self._record("lift", down)

    def arm(self, angle):
        self._record("arm", angle)
        if self._angle < 90 <= angle and self.on_flip:
            self.on_flip()
        self._angle = angle

    def home(self):
        self._record("home")
        self._angle = 0.0
