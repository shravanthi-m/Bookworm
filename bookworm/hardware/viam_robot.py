"""Viam backend: uFactory arm + gripper + calibrated wrist camera.

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

from .base import POSE_KEYS, CameraGeometry, HardwareError, Robot


class ViamConnection:
    def __init__(self):
        from dotenv import load_dotenv

        load_dotenv()
        self.address, key_id, key = ((os.getenv(k) or "").strip()
                                     for k in ("VIAM_ADDRESS", "VIAM_API_KEY_ID", "VIAM_API_KEY"))
        if not (self.address and key_id and key):
            raise HardwareError("VIAM_ADDRESS / VIAM_API_KEY_ID / VIAM_API_KEY not set — copy .env.example to .env")

        self._key_id, self._key = key_id, key
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        self._dial()

    def _dial(self) -> None:
        # Dialing the cloud machine is flaky (some routes time out after ~20 s each), so
        # give each attempt time and retry instead of failing on the first slow dial.
        last = None
        for _ in range(3):
            try:
                self.machine = self.call(self._connect(self._key_id, self._key), timeout=90)
                return
            except Exception as e:  # noqa: BLE001 - TimeoutError / connection errors from the SDK
                last = e
        raise HardwareError(f"could not connect to {self.address}: {last!r}")

    async def _connect(self, key_id, key):
        from viam.robot.client import RobotClient

        opts = RobotClient.Options.with_api_key(api_key=key, api_key_id=key_id)
        return await RobotClient.at_address(self.address, opts)

    def call(self, coro, timeout: float = 60.0):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def resource_names(self) -> list[str]:
        return [r.name for r in self.machine.resource_names]

    def component(self, cls, name: str, attempts: int = 10):
        """``cls.from_robot`` with retries: the SDK populates resource clients in the background
        after connecting, so lookups made immediately can miss."""
        import time

        last = None
        for _ in range(attempts):
            try:
                return cls.from_robot(self.machine, name)
            except Exception as e:  # noqa: BLE001 - SDK raises various types for missing resources
                last = e
                try:
                    self.call(self.machine.refresh(), timeout=20)
                except Exception:  # noqa: BLE001
                    time.sleep(1)
        raise HardwareError(f"component '{name}' not available ({last}); machine has: {self.resource_names()}")

    def reconnect(self) -> None:
        """Drop a broken connection and dial again (long sessions over flaky Wi-Fi do break)."""
        try:
            self.call(self.machine.close(), timeout=10)
        except Exception:  # noqa: BLE001 - it's already broken
            pass
        self._dial()

    def close(self) -> None:
        try:
            self.call(self.machine.close())
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)


class ViamRobot(Robot):
    def __init__(self, conn: ViamConnection, arm: str, gripper: str, camera: str,
                 camera_calibrated: bool = True, **_):
        self.conn, self.camera_name, self.camera_calibrated = conn, camera, camera_calibrated
        self._names = (arm, gripper, camera)
        self._lookup()
        self._intrinsics = None

    def _lookup(self) -> None:
        from viam.components.arm import Arm
        from viam.components.camera import Camera
        from viam.components.gripper import Gripper

        arm, gripper, camera = self._names
        self.arm = self.conn.component(Arm, arm)
        self.grip = self.conn.component(Gripper, gripper)
        self.cam = self.conn.component(Camera, camera)

    def reconnect(self) -> None:
        self.conn.reconnect()
        self._lookup()

    # ---- arm / gripper ------------------------------------------------------
    def move_to(self, pose):
        from viam.proto.common import Pose

        self.conn.call(self.arm.move_to_position(Pose(**{k: float(pose[k]) for k in POSE_KEYS})))

    def end_pose(self):
        p = self.conn.call(self.arm.get_end_position())
        return {k: getattr(p, k) for k in POSE_KEYS}

    def gripper(self, action):
        if action == "open":
            self.conn.call(self.grip.open())
        elif action == "grab":
            self.conn.call(self.grip.grab())
        else:
            raise HardwareError(f"unknown gripper action {action!r}")

    # ---- camera -------------------------------------------------------------
    def capture(self):
        try:
            return self._capture()
        except Exception:  # noqa: BLE001 - dropped connection: redial once, then let errors through
            self.reconnect()
            return self._capture()

    def _capture(self):
        # RealSense returns 'color' (jpeg, ~70 KB) and 'depth' (~1.8 MB). Asking for colour only
        # cut capture from ~485 ms to ~31 ms on the rig.
        try:
            images, _ = self.conn.call(self.cam.get_images(filter_source_names=["color"]))
        except Exception:  # noqa: BLE001 - e.g. a webcam has no source called "color"
            images = []
        if not images:
            images, _ = self.conn.call(self.cam.get_images())
        color = [im for im in images if not im.mime_type.startswith("image/vnd.viam.dep")]
        if not color:
            raise HardwareError("wrist camera returned no colour image")
        frame = cv2.imdecode(np.frombuffer(color[0].data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise HardwareError(f"could not decode {color[0].mime_type} from wrist camera")
        self._last_size = frame.shape[1], frame.shape[0]
        return frame

    def _load_intrinsics(self):
        props = self.conn.call(self.cam.get_properties())
        ip = props.intrinsic_parameters
        if not ip.focal_x_px:
            raise HardwareError("wrist camera has no intrinsic_parameters in its Viam config")
        K = np.array([[ip.focal_x_px, 0, ip.center_x_px], [0, ip.focal_y_px, ip.center_y_px], [0, 0, 1]])
        dist = None
        dp = props.distortion_parameters
        if dp.model == "brown_conrady" and len(dp.parameters) == 5:
            k1, k2, k3, p1, p2 = dp.parameters     # Viam order -> OpenCV order (k1, k2, p1, p2, k3)
            dist = np.array([k1, k2, p1, p2, k3])
        self._intrinsics = (K, dist, (ip.width_px, ip.height_px))

    def camera_pose(self) -> np.ndarray:
        """4x4 world_from_camera, by transforming camera-frame points through Viam's frame system."""
        from viam.proto.common import Pose, PoseInFrame

        def to_world(x, y, z):
            q = PoseInFrame(reference_frame=self.camera_name, pose=Pose(x=x, y=y, z=z, o_z=1))
            p = self.conn.call(self.conn.machine.transform_pose(q, "world")).pose
            return np.array([p.x, p.y, p.z])

        o = to_world(0, 0, 0)
        R = np.column_stack([(to_world(*(100 * e)) - o) / 100 for e in np.eye(3)])
        u, _, vt = np.linalg.svd(R)            # re-orthonormalise against rounding
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = u @ vt, o
        return T

    def camera_geometry(self):
        if not self.camera_calibrated:
            return None     # static uncalibrated camera (e.g. phone): use clicked book_corners instead
        if self._intrinsics is None:
            self._load_intrinsics()
        K, dist, (w, h) = self._intrinsics
        fw, fh = getattr(self, "_last_size", (w, h))
        K = K.copy()
        if w and h and (fw, fh) != (w, h):     # stream resolution differs from calibration
            K[0] *= fw / w
            K[1] *= fh / h
        return CameraGeometry(K, dist, self.camera_pose())
