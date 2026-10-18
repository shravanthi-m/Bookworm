"""Simulated rig for working offline: the arm and the wrist RGB-D camera, with no Viam.

The scene is the real one, measured on armfarm22 on 2026-10-16: a wood table and an open spread of
"The Myth of Sisyphus" at the corners the calibration found, the left (title) page thin and the right page
thick, with a dip at the gutter. The camera has the real intrinsics and the real camera-to-flange offset,
so pixels, depth and world coordinates behave like the rig's. Like the real RealSense, the depth has a
~4 degree tilt against the table, which the Home & level stage has to find and remove.

Everything is ray-cast from wherever the arm is, so camera-pass depth scans, probes and the table-height
search all work.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from .base import CameraGeometry
from .workshop import WorkshopMock

# measured on the rig (data/calibration/20261016_184629, config/local.yaml)
HOME = {"x": 389.667, "y": -56.869, "z": 480.904, "o_x": 0.0, "o_y": 0.0, "o_z": -1.0, "theta": -5.636}
HOME_JOINTS = [-8.3215, -17.1494, -63.922, 0.0116, 80.9934, -13.9593]
K = np.array([[905.526, 0.0, 657.051], [0.0, 905.255, 350.014], [0.0, 0.0, 1.0]])
T_WORLD_CAM_HOME = np.array([
    [0.117079, -0.992693, -0.029216, 470.915],
    [-0.993115, -0.117145, 0.000537, -34.786],
    [-0.003956, 0.028952, -0.999573, 463.016],
    [0.0, 0.0, 0.0, 1.0]])
CORNERS = {"tl": (530.0, 21.8), "tr": (497.1, -213.2), "br": (326.2, -180.7), "bl": (342.2, 38.6)}
SIZE = (1280, 720)


class SimRobot(WorkshopMock):
    """``table_z`` is the truth the table-height stage should find (the table top's z in the arm frame)."""

    sim = True
    hard_table = True      # the tip cannot go below the table top (the page, being paper, gives way)
    fault_on_block = False  # True: a blocked move raises the xArm module's collision error and needs clear_error

    def __init__(self, tool_length: float = 155.0, table_z: float = 21.3, left_mm: float = 15.0, right_mm: float = 24.0,
                 depth_tilt=(-0.069, -0.008), noise_mm: float = 0.8, seed: int = 0):
        super().__init__(camera=None)
        self.tool_length, self.table_z = float(tool_length), float(table_z)
        self.left_mm, self.right_mm = float(left_mm), float(right_mm)
        self.depth_tilt, self.noise_mm = depth_tilt, float(noise_mm)
        self.rng = np.random.default_rng(seed)
        self._pose, self._joints = dict(HOME), list(HOME_JOINTS)
        self._home_pose_for_joints = dict(HOME)
        self.T_flange_cam = np.linalg.inv(self._flange_T(HOME)) @ T_WORLD_CAM_HOME
        self.pressed_mm = 0.0                                          # how far the tip went below the page
        self._push_mm, self._faulted = 0.0, False
        c = np.array(list(CORNERS.values()))
        tl, tr, br, bl = (np.array(CORNERS[k]) for k in ("tl", "tr", "br", "bl"))
        self.centre = c.mean(0)
        u = (tr - tl) + (br - bl)                                      # left -> right across the spread
        self.u = u / np.linalg.norm(u)
        self.v = np.array([-self.u[1], self.u[0]])                     # bottom -> top
        if np.dot(self.v, (tl + tr) / 2 - (bl + br) / 2) < 0:
            self.v = -self.v
        self.half_w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 4
        self.half_h = (np.linalg.norm(tl - bl) + np.linalg.norm(tr - br)) / 4
        self._wood = self._wood_texture()

    # ---- geometry --------------------------------------------------------------------------------------
    @staticmethod
    def _flange_T(p: dict) -> np.ndarray:
        """Flange pose matrix for a pose pointing straight down, rotated by theta about z."""
        t = np.radians(p["theta"])
        c, s = np.cos(t), np.sin(t)
        T = np.eye(4)
        T[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ np.diag([1, -1, -1])
        T[:3, 3] = [p["x"], p["y"], p["z"]]
        return T

    def _book(self, x, y):
        """Book coordinates: s across (negative = left page), r along the spine."""
        dx, dy = x - self.centre[0], y - self.centre[1]
        return dx * self.u[0] + dy * self.u[1], dx * self.v[0] + dy * self.v[1]

    def surface(self, x, y):
        """Top surface z (arm frame) at world x, y: table, or page."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        s, r = self._book(x, y)
        inside = (np.abs(s) <= self.half_w) & (np.abs(r) <= self.half_h)
        a = np.clip(np.abs(s) / self.half_w, 0, 1)                     # 0 at the spine, 1 at the outer edge
        base = np.where(s < 0, self.left_mm, self.right_mm)
        dome = 3.0 * np.sin(np.pi * a)                                 # pages bow up between spine and edge
        gutter = 7.0 * np.clip(1 - np.abs(s) / 14.0, 0, 1) ** 2         # and dive into the gutter
        page = base + dome - gutter - 2.0 * a ** 4                     # and roll off at the outer edge
        return self.table_z + np.where(inside, np.maximum(page, 1.0), 0.0)

    def camera_geometry(self):
        return CameraGeometry(K.copy(), None, self._flange_T(self._pose) @ self.T_flange_cam)

    def _cast(self):
        """Ray-cast every pixel (cached per pose: the live view asks many times while the arm stands still)."""
        key = tuple(round(float(self._pose[k]), 2) for k in ("x", "y", "z", "theta"))
        if getattr(self, "_cast_key", None) != key:
            self._cast_val, self._cast_key, self._img = self._cast_now(), key, None
        return self._cast_val

    def _cast_now(self):
        """World x, y, z of the surface each pixel sees, and the camera-z depth (mm)."""
        T = self._flange_T(self._pose) @ self.T_flange_cam
        W, H = SIZE
        uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        d = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu)], -1) @ T[:3, :3].T.astype(np.float32)
        C = T[:3, 3]
        down = -d[..., 2]
        t = (C[2] - self.table_z) / down
        for _ in range(8):                                             # fixed point onto the height field
            px, py = C[0] + t * d[..., 0], C[1] + t * d[..., 1]
            t = 0.5 * t + 0.5 * (C[2] - self.surface(px, py)) / down
        px, py = C[0] + t * d[..., 0], C[1] + t * d[..., 1]
        return px, py, C[2] - t * down, t, down

    # ---- cameras -----------------------------------------------------------------------------------------
    def _wood_texture(self):
        g = np.random.default_rng(1)
        n = cv2.resize(g.normal(0, 1, (40, 400)).astype(np.float32), (1600, 1600), interpolation=cv2.INTER_CUBIC)
        return n

    def capture(self):
        px, py, pz, _, _ = self._cast()
        if self._img is None:
            self._img = self._render(px, py, pz)
        noise = self.rng.integers(-2, 3, self._img.shape, dtype=np.int16)
        return np.clip(self._img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    def _render(self, px, py, pz):
        on_book = pz > self.table_z + 0.5
        s, r = self._book(px, py)
        ix = np.clip(((px - 100) * 2).astype(int), 0, 1599)
        iy = np.clip(((py + 400) * 2).astype(int), 0, 1599)
        grain = self._wood[iy, ix][..., None]
        img = np.array([55, 135, 205], np.float32) + grain * np.array([8, 14, 18], np.float32)   # orange wood (BGR)
        paper = np.array([236, 233, 229], np.float32)
        a = np.clip(np.abs(s) / self.half_w, 0, 1)
        text = ((np.abs(r) < self.half_h - 18) & (a > 0.15) & (a < 0.85) & ((np.floor(r / 4.5) % 2) == 0)
                & ((np.floor(s / 2.2) % 5) != 0))
        shade = 1 - 0.25 * np.clip(1 - np.abs(s) / 14.0, 0, 1)        # gutter shadow
        page = paper * shade[..., None]
        page = np.where(text[..., None], page * 0.82, page)
        img = np.where(on_book[..., None], page, img)
        return np.clip(img, 0, 255).astype(np.uint8)

    def capture_depth(self):
        px, py, _, t, down = self._cast()
        a, b = self.depth_tilt
        err = a * (px - HOME["x"]) + b * (py - HOME["y"])               # the RealSense's apparent tilt
        t = t - err / down + self.rng.normal(0, self.noise_mm, t.shape)
        return np.clip(np.round(t), 0, 65535).astype(np.uint16)

    # ---- arm -------------------------------------------------------------------------------------------
    def move_to(self, pose):
        if getattr(self, "_faulted", False):
            raise RuntimeError("arm is in an error state and will not accept motion commands: clear the error")
        super().move_to(pose)
        tip = np.array([pose["x"], pose["y"], pose["z"]]) + self.tool_length * np.array([pose["o_x"], pose["o_y"], pose["o_z"]]) \
            / np.linalg.norm([pose["o_x"], pose["o_y"], pose["o_z"]])
        short_cmd = max(0.0, self.table_z - float(tip[2]))
        if self.hard_table and tip[2] < self.table_z:                    # the table is hard: the arm stops there, the step is unfinished
            short = self.table_z - float(tip[2])
            self._pose["z"] += short
            tip[2] += short
            self._record("sim: tip stopped by the table", round(short, 1))
        self.pressed_mm = max(0.0, float(self.surface(tip[0], tip[1])) - float(tip[2]))
        self._push_mm = short_cmd if self.hard_table else 0.0
        if self.pressed_mm > 0:
            self._record("sim: tip pressed into the page", round(self.pressed_mm, 1))
        if self.fault_on_block and short_cmd > 0:
            self._faulted = True
            raise RuntimeError("collision caused overcurrent: ensure robot is clear of obstacles and clear error "
                               "through UFACTORY Studio or clear_error do command")
        time.sleep(0.05)

    def joint_load(self):
        """Like the real xArm6 (2026-10-17 run, Nm): joints 2 and 3 carry gravity with a slight trend in height, and a
        push into the table adds ~7.5 and ~4.5 Nm per mm on them; the wrist joint 5 reading flips between two levels."""
        tip_z = self._pose["z"] - self.tool_length
        push = max(self._push_mm, self.pressed_mm * 0.6)        # a page on the book block gives a little, then pushes back
        n = self.rng.normal(0, 0.005, 6)
        j5 = -1.862 if self.rng.random() < 0.6 else -3.8
        return [0.0, -21.0 - 0.026 * (43.3 - tip_z) + 7.5 * push + n[1], -14.09 + 0.034 * (43.3 - tip_z) + 4.5 * push + n[2],
                -0.12 + n[3], j5, 0.0]

    def clear_error(self):
        self._faulted = False
        self._record("sim: error cleared")

    def move_joints(self, degrees):
        super().move_joints(degrees)
        if max(abs(a - b) for a, b in zip(degrees, HOME_JOINTS)) > 0.5:
            self._record("sim: joint move away from home is not simulated; pose unchanged")
        time.sleep(0.1)
