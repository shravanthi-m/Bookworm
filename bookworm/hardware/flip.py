"""Execute the pencil-eraser page flip from the waypoint list in config."""
from __future__ import annotations

import logging
import time

from .base import HardwareError, Robot, tip_to_flange
from .book_frame import BookFrame

log = logging.getLogger("bookworm")


def surface_z(book: BookFrame, bcfg: dict, surface: str, flip_index: int) -> float:
    """Book-frame z of a page stack's top after ``flip_index`` flips this run.

    Each flip moves one leaf from the right stack to the left, so the right surface
    drops and the left surface rises by ``z_drop_per_flip_mm``.
    """
    drop = flip_index * bcfg["z_drop_per_flip_mm"]
    if surface == "right":
        return -drop
    if surface == "left":
        return (book.left_dz or 0.0) + drop
    raise HardwareError(f"unknown surface {surface!r} (use right | left)")


def check_tip(bxyz, z_rel: float, width: float, bcfg: dict, fcfg: dict) -> None:
    """Refuse waypoints that would press too hard or leave the book area.

    The pencil is held rigidly in the gripper, so a bad waypoint or a bad vision
    corner goes straight into the book. Limits are in config ``flip.limits``.
    """
    lim = fcfg["limits"]
    x, y, _ = bxyz
    if z_rel < -lim["max_press_mm"]:
        raise HardwareError(f"waypoint presses {-z_rel:.1f} mm into the page (max {lim['max_press_mm']})")
    m = lim["margin_mm"]
    if not (-width - m <= x <= width + m and -m <= y <= bcfg["page_height_mm"] + m):
        raise HardwareError(f"waypoint ({x:.0f}, {y:.0f}) mm is outside the book area (+{m} mm margin)")


def plan_flip(book: BookFrame, cfg: dict, corner_xy=None, flip_index: int = 0) -> list[tuple]:
    """Resolve the whole flip into concrete steps and check every one BEFORE anything moves.

    Returns [("move", tip_world, book_xyz, surface) | ("gripper", action) | ("wait", s)].
    ``corner_xy``: right page bottom-outer corner in book mm (from vision); defaults to calibration.
    """
    bcfg, fcfg = cfg["book"], cfg["flip"]
    width = book.right_width or bcfg["page_width_mm"]
    cx, cy = corner_xy if corner_xy is not None else (width, 0.0)
    steps = []
    for step in fcfg["sequence"]:
        (name, args), = step.items()
        args = args or {}
        if name == "move":
            surface = args.get("surface", "right")
            ox, oy = (cx, cy) if args.get("from", "spine") == "corner" else (0.0, 0.0)
            bxyz = (ox + args["x"], oy + args["y"], surface_z(book, bcfg, surface, flip_index) + args["z"])
            check_tip(bxyz, args["z"], width, bcfg, fcfg)
            steps.append(("move", book.to_world(*bxyz), bxyz, surface))
        elif name == "gripper":
            steps.append(("gripper", args["action"]))
        elif name == "wait":
            steps.append(("wait", float(args["s"])))
        else:
            raise HardwareError(f"unknown flip step: {name}")
    return steps


def run_flip(robot: Robot, book: BookFrame, cfg: dict, corner_xy=None, flip_index: int = 0) -> None:
    for step in plan_flip(book, cfg, corner_xy, flip_index):
        if step[0] == "move":
            robot.move_to(tip_to_flange(step[1], cfg["robot"]))
        elif step[0] == "gripper":
            robot.gripper(step[1])
        else:
            time.sleep(step[1])


def retreat(robot: Robot, cfg: dict) -> None:
    """Safe stop: halt the arm, lift the pencil off the page, then go home if a home pose is set.

    Used on Ctrl+C or any error mid-run, so the pencil is never left pressing on the book.
    Never raises: it runs while another error is already being reported.
    """
    try:
        robot.stop()
        pose = robot.end_pose()
        pose["z"] = pose["z"] + cfg["robot"]["safe_height_mm"]
        robot.move_to(pose)
        if cfg["poses"].get("home"):
            robot.move_to(cfg["poses"]["home"])
    except Exception:  # noqa: BLE001
        log.exception("safe retreat failed: lift the pencil by hand / use the E-stop")
