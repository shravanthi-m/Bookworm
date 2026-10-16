"""Build the robot from config: the Viam machine, or a mock that replays photos."""
from __future__ import annotations

from .hardware import MockRobot


def build_robot(cfg: dict, images_dir: str | None = None, mock: bool = False):
    """Returns (robot, close_fn). ``images_dir`` or ``mock`` => no Viam connection."""
    if images_dir or mock or cfg["robot"]["backend"] != "viam":
        from .vision import FileCamera

        cfg["robot"]["backend"] = "mock"
        robot = MockRobot(FileCamera(images_dir) if images_dir else None, verbose=False)
        return robot, robot.close

    from .hardware.viam_robot import ViamConnection, ViamRobot

    conn = ViamConnection()
    try:
        robot = ViamRobot(conn, **cfg["robot"])
    except Exception:
        conn.close()
        raise

    def close():
        try:
            robot.close()
        finally:
            conn.close()

    return robot, close
