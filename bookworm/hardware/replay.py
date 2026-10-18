"""Record the real camera's views (colour + depth + geometry + arm pose), and replay them offline.

When online, the GUI saves every depth view it takes to data/depth/raw/<session>/view_NNN/. Offline,
``python -m bookworm.gui --replay data/depth/raw/<session>`` serves those views back: the arm "moves"
like the mock, and the cameras return the recorded view taken nearest to where the arm now is.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from .base import CameraGeometry
from .mock import MockRobot


def save_view(root: Path, frame: np.ndarray, depth: np.ndarray, geom: CameraGeometry, pose: dict | None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    d = root / f"view_{len(list(root.glob('view_*'))) + 1:03d}"
    d.mkdir()
    cv2.imwrite(str(d / "color.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(d / "depth.png"), np.asarray(depth, np.uint16))          # 16-bit, mm
    (d / "meta.json").write_text(json.dumps({
        "taken": time.strftime("%Y-%m-%d %H:%M:%S"), "pose": pose,
        "K": np.asarray(geom.K).tolist(), "dist": None if geom.dist is None else np.asarray(geom.dist).tolist(),
        "T_world_cam": np.asarray(geom.T_world_cam).tolist()}, indent=1), encoding="utf-8")
    return d


class ReplayRobot(MockRobot):
    replay = True

    def __init__(self, folder: str | Path, home_pose: dict | None = None, home_joints=None):
        super().__init__(camera=None)
        self.folder = Path(folder)
        self.views = []
        for d in sorted(self.folder.glob("view_*")):
            m = json.loads((d / "meta.json").read_text(encoding="utf-8"))
            self.views.append((d, m))
        if not self.views:
            raise FileNotFoundError(f"no recorded views (view_*/meta.json) in {self.folder}")
        first = self.views[0][1].get("pose")
        self._pose = dict(home_pose or first or {"x": 0, "y": 0, "z": 0, "o_x": 0, "o_y": 0, "o_z": -1, "theta": 0})
        if home_joints:
            self._joints = list(home_joints)
            self._home_pose_for_joints = dict(home_pose or self._pose)
        self.last_miss_mm = 0.0

    def _nearest(self):
        p = np.array([self._pose[k] for k in ("x", "y", "z")])
        best = min(self.views, key=lambda v: np.inf if not v[1].get("pose") else
                   np.linalg.norm(np.array([v[1]["pose"][k] for k in ("x", "y", "z")]) - p))
        if best[1].get("pose"):
            self.last_miss_mm = float(np.linalg.norm(np.array([best[1]["pose"][k] for k in ("x", "y", "z")]) - p))
        return best

    def capture(self):
        return cv2.imread(str(self._nearest()[0] / "color.jpg"))

    def capture_depth(self):
        return cv2.imread(str(self._nearest()[0] / "depth.png"), cv2.IMREAD_UNCHANGED)

    def camera_geometry(self):
        m = self._nearest()[1]
        return CameraGeometry(np.array(m["K"], float), None if m["dist"] is None else np.array(m["dist"], float),
                              np.array(m["T_world_cam"], float))
