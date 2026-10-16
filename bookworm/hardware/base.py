"""Hardware abstraction. The controller only talks to this interface, so the
mechanism (clamps, eraser arm) can change without touching vision or control code."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod


class HardwareError(RuntimeError):
    pass


class Actuators(ABC):
    @abstractmethod
    def clamp(self, side: str, engaged: bool) -> None:
        """side: 'left' | 'right'."""

    @abstractmethod
    def lift(self, down: bool) -> None:
        """Lower the pen eraser onto the page (True) or raise it (False)."""

    @abstractmethod
    def arm(self, angle: float) -> None:
        """Rotate the flipper arm. 0 = over right page, 180 = over left page."""

    @abstractmethod
    def home(self) -> None:
        """Safe pose: eraser up, arm at 0, both clamps engaged."""

    def close(self) -> None:
        pass


def run_sequence(act: Actuators, steps: list[dict]) -> None:
    """Execute a flip_sequence from config, e.g. [{'arm': {'angle': 10}}, {'wait': {'s': 0.3}}]."""
    for step in steps:
        (name, args), = step.items()
        args = args or {}
        if name == "clamp":
            act.clamp(args["side"], bool(args["engaged"]))
        elif name == "lift":
            act.lift(bool(args["down"]))
        elif name == "arm":
            act.arm(float(args["angle"]))
        elif name == "wait":
            time.sleep(float(args["s"]))
        else:
            raise HardwareError(f"unknown flip step: {name}")
