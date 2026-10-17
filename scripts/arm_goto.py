"""Move the arm between Midhat's recorded resting spots (config/waypoints.yaml) — for voice
commands like "Bookworm, move away".

    .venv/bin/python scripts/arm_goto.py park     -> out_of_the_way (clear of the book and camera)
    .venv/bin/python scripts/arm_goto.py ready    -> "0" (40 mm above the flip start)
    .venv/bin/python scripts/arm_goto.py where    which recorded spot the arm is at, if any

Plans no paths: it only does the single joint move flip_path.py already does between those
two spots (0 <-> out_of_the_way). If the arm is anywhere else (e.g. stopped mid-flip, near
the page) it refuses rather than sweep across the book; move it with the Viam CONTROL tab.

Exit codes: 0 = there, 3 = refused (not at a known spot), 2 = joint error, 130 = stopped.
"""
from __future__ import annotations

import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from flip_path import MAX_JOINT_ERROR_DEG, load  # noqa: E402 - Midhat's waypoint loader

TARGETS = {"park": "out_of_the_way", "ready": "0"}
SAFE_FROM = {"out_of_the_way": {"0", "6b", "out_of_the_way"},   # the moves flip_path already makes
             "0": {"out_of_the_way", "0", "start"}}
AT_TOLERANCE_DEG = 2.0


def nearest(joints, pts) -> tuple[str, float]:
    name = min(pts, key=lambda n: max(abs(a - b) for a, b in zip(joints, pts[n])))
    return name, max(abs(a - b) for a, b in zip(joints, pts[name]))


def goto(conn, arm, target: str, pts=None) -> tuple[int, str]:
    """Move ``arm`` to 'park' / 'ready' over an open connection. Returns (exit code, message).
    The control page calls this directly so it reuses its connection (a fresh Viam dial per
    command was slow and often timed out)."""
    from viam.proto.component.arm import JointPositions

    pts = pts or load()[0]
    here, off = nearest(list(conn.call(arm.get_joint_positions()).values), pts)
    at = here if off <= AT_TOLERANCE_DEG else None
    if target == "where":
        return 0, f"at {here}" if at else f"not at a recorded spot (nearest {here}, {off:.1f} deg off)"
    goal = TARGETS[target]
    if at == goal:
        return 0, f"already at {goal}"
    if at not in SAFE_FROM[goal]:
        return 3, (f"REFUSED: arm is {'at ' + at if at else f'not at a recorded spot ({off:.1f} deg from {here})'};"
                   f" won't move to {goal} from there. Use the Viam CONTROL tab.")
    conn.call(arm.move_to_joint_positions(JointPositions(values=pts[goal])), timeout=120)
    err = max(abs(a - b) for a, b in zip(conn.call(arm.get_joint_positions()).values, pts[goal]))
    if err > MAX_JOINT_ERROR_DEG:
        conn.call(arm.stop())
        return 2, f"STOPPED short of {goal}: joint error {err:.2f} deg"
    return 0, f"at {goal}"


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1 or argv[0] not in (*TARGETS, "where"):
        print(__doc__)
        return 1

    from viam.components.arm import Arm

    from bookworm.hardware.viam_robot import ViamConnection

    conn = ViamConnection()
    arm = conn.component(Arm, "arm")

    def on_stop(*_):                       # SIGTERM (e.g. a Stop button) halts the arm
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, on_stop)

    try:
        code, msg = goto(conn, arm, argv[0])
        print(msg, flush=True)
        return code
    except KeyboardInterrupt:
        try:
            conn.call(arm.stop(), timeout=10)
        finally:
            print("stopped: arm halted where it is", flush=True)
        return 130
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
