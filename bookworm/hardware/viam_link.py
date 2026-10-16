"""Viam backend: servos and camera on a machine running viam-server.

Credentials come from environment variables (loaded from .env):
    VIAM_ADDRESS, VIAM_API_KEY_ID, VIAM_API_KEY

The Viam SDK is async; the rest of Bookworm is synchronous, so ViamConnection
runs an event loop on a background thread and exposes a blocking ``call()``.
"""
from __future__ import annotations

import asyncio
import os
import threading

import cv2
import numpy as np

from .base import Actuators, HardwareError


class ViamConnection:
    def __init__(self, address: str | None = None, api_key_id: str | None = None, api_key: str | None = None):
        from dotenv import load_dotenv

        load_dotenv()
        self.address = address or os.getenv("VIAM_ADDRESS")
        key_id = api_key_id or os.getenv("VIAM_API_KEY_ID")
        key = api_key or os.getenv("VIAM_API_KEY")
        if not (self.address and key_id and key):
            raise HardwareError("VIAM_ADDRESS / VIAM_API_KEY_ID / VIAM_API_KEY not set — copy .env.example to .env")

        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        self.machine = self.call(self._connect(key_id, key))

    async def _connect(self, key_id, key):
        from viam.robot.client import RobotClient

        opts = RobotClient.Options.with_api_key(api_key=key, api_key_id=key_id)
        return await RobotClient.at_address(self.address, opts)

    def call(self, coro, timeout: float = 15.0):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def resource_names(self) -> list[str]:
        return [r.name for r in self.machine.resource_names]

    def close(self) -> None:
        try:
            self.call(self.machine.close())
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)


class ViamActuators(Actuators):
    """Maps Bookworm primitives onto four Viam servo components (names in config)."""

    def __init__(self, conn: ViamConnection, components: dict, angles: dict, **_):
        from viam.components.servo import Servo

        self.conn = conn
        self.angles = angles
        try:
            self.servos = {role: Servo.from_robot(conn.machine, name) for role, name in components.items()}
        except Exception as e:  # noqa: BLE001 - SDK raises various types for missing resources
            raise HardwareError(f"servo lookup failed ({e}); machine has: {conn.resource_names()}") from e

    def _move(self, role: str, angle: float) -> None:
        self.conn.call(self.servos[role].move(int(round(max(0, min(180, angle))))))

    def clamp(self, side, engaged):
        a = self.angles["clamp_engaged" if engaged else "clamp_released"]
        self._move(f"clamp_{side}", a[side] if isinstance(a, dict) else a)

    def lift(self, down):
        self._move("lift", self.angles["lift_down" if down else "lift_up"])

    def arm(self, angle):
        self._move("arm", angle)

    def home(self):
        self.lift(False)
        self.arm(0)
        self.clamp("left", True)
        self.clamp("right", True)

    def close(self):
        self.home()


class ViamCamera:
    def __init__(self, conn: ViamConnection, name: str, **_):
        from viam.components.camera import Camera

        self.conn = conn
        self.cam = Camera.from_robot(conn.machine, name)

    def capture(self) -> np.ndarray:
        img = self.conn.call(self.cam.get_image(mime_type="image/jpeg"))
        frame = cv2.imdecode(np.frombuffer(img.data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise HardwareError("could not decode image from Viam camera")
        return frame

    def close(self) -> None:
        pass
