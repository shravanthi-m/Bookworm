"""Execute the pencil-eraser page flip from the waypoint list in config."""
from __future__ import annotations

import time

from .base import HardwareError, Robot, tip_to_flange
from .book_frame import BookFrame


def run_flip(robot: Robot, book: BookFrame, cfg: dict, corner_xy=None, flip_index: int = 0) -> None:
    """``corner_xy``: right page bottom-outer corner in book mm (from vision); defaults to calibration."""
    bcfg = cfg["book"]
    cx, cy = corner_xy if corner_xy is not None else (bcfg["page_width_mm"], 0.0)
    # The right stack loses a leaf per flip, so its surface drops a little each time.
    dz = -flip_index * bcfg["z_drop_per_flip_mm"]

    for step in cfg["flip"]["sequence"]:
        (name, args), = step.items()
        args = args or {}
        if name == "move":
            ox, oy = (cx, cy) if args.get("from", "spine") == "corner" else (0.0, 0.0)
            tip = book.to_world(ox + args["x"], oy + args["y"], args["z"] + dz)
            robot.move_to(tip_to_flange(tip, cfg["robot"]))
        elif name == "gripper":
            robot.gripper(args["action"])
        elif name == "wait":
            time.sleep(float(args["s"]))
        else:
            raise HardwareError(f"unknown flip step: {name}")
