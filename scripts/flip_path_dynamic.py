"""The recorded page flip (config/waypoints.yaml), moved with the book. The dynamic twin of scripts/flip_path.py.

    .venv/bin/python scripts/flip_path_dynamic.py --set-reference      the book is where the path works: remember it
    .venv/bin/python scripts/flip_path_dynamic.py --path v2 --dry-run  print the moved points, don't connect
    .venv/bin/python scripts/flip_path_dynamic.py --path v2            one page, moved with the book
    .venv/bin/python scripts/flip_path_dynamic.py --times 3            three pages

scripts/flip_path.py (main) is not changed: it plays the recorded joints exactly as taught, and this script uses its
path loading and its player for everything that hasn't moved. Here every point is carried by how far the book has
moved and turned since the path was taught (bookworm/hardware/dynamic_path.py):

  reference   where the book was when the path works: set once with --set-reference
  now         where the book is now: data/book_frame.json, written by the calibration workshop's First capture + Z link

With no reference, or a book that hasn't moved, the recorded joints play exactly as taught. A book moved further than
120 mm or turned more than 25 deg is refused (move it back, or teach the path again).

Exit codes: 0 = done, 1 = no book frame / book moved too far, 2 = a point was missed (arm stopped), 130 = stopped.
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import flip_path  # noqa: E402  (main's recorded-path player, unchanged)

from bookworm.hardware import dynamic_path as dp  # noqa: E402

MAX_POSE_ERROR_MM = 3.0


def dynamic_plan(version="v2", frame_now=None, reference=None, log=print) -> dict:
    """dynamic_path.plan for this path: {"mode", "why", "motion", "poses"} (poses empty = play the recorded joints)."""
    d = yaml.safe_load(flip_path.WAYPOINTS.read_text())
    p = dp.plan(d, flip_path.PATHS[version], reference if reference is not None else dp.load_reference(),
                frame_now if frame_now is not None else dp.load_frame())
    log(f"flip path {version}: {p['mode']} ({p['why']})")
    return p


def play(conn, arm, plan, pts, poses, on_step=None, stopped=lambda: False, log=print):
    """Like flip_path.play, but the points in ``poses`` are reached as poses (through the motion planner) and checked
    to MAX_POSE_ERROR_MM. With no poses it is flip_path.play itself."""
    if not poses:
        return flip_path.play(conn, arm, plan, pts, on_step=on_step, stopped=stopped, log=log)
    from viam.proto.common import Pose
    from viam.proto.component.arm import JointPositions

    page, t_page = 1, time.time()
    for step in plan:
        if stopped():
            return 130, "stopped: arm halted where it is"
        if step.startswith("wait "):
            pause = float(step.split()[1])
            log(f"page {page} done in {time.time() - t_page:.1f} s, parked; pausing {pause:g} s")
            end = time.time() + pause
            while time.time() < end and not stopped():
                time.sleep(min(0.1, end - time.time()))
            page, t_page = page + 1, time.time()
            continue
        if on_step:
            on_step(step)
        if step in poses:
            want = poses[step]
            conn.call(arm.move_to_position(Pose(**{k: float(want[k]) for k in dp.POSE_KEYS})), timeout=120)
            if stopped():
                return 130, "stopped: arm halted where it is"
            got = conn.call(arm.get_end_position())
            err = ((got.x - want["x"]) ** 2 + (got.y - want["y"]) ** 2 + (got.z - want["z"]) ** 2) ** 0.5
            if err > MAX_POSE_ERROR_MM:
                conn.call(arm.stop())
                return 2, f"STOPPED at {step}: {err:.1f} mm from the moved point"
        else:
            conn.call(arm.move_to_joint_positions(JointPositions(values=pts[step])), timeout=120)
            if stopped():
                return 130, "stopped: arm halted where it is"
    return 0, "done"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", choices=sorted(flip_path.PATHS), default="v2", help="which recorded path (default v2)")
    ap.add_argument("--times", type=int, default=1, help="pages to turn (default 1)")
    ap.add_argument("--no-park", action="store_true", help="end at the path's last point instead of parking")
    ap.add_argument("--dry-run", action="store_true", help="print the moved points without connecting")
    ap.add_argument("--set-reference", action="store_true",
                    help="remember the current book frame as where the path works, then exit")
    args = ap.parse_args(argv)

    if args.set_reference:
        now = dp.load_frame()
        if now is None:
            print("no current book frame: run the calibration workshop's First capture and Z link first", flush=True)
            return 1
        dp.set_reference(now)
        print(f"reference set: centre {now['center']}, {now['angle_deg']:.1f} deg, top z {now['top_z']}", flush=True)
        return 0

    pts, path, between = flip_path.load(args.path)
    plan = flip_path.make_plan(path, between, args.times, park=not args.no_park)
    try:
        poses = dynamic_plan(args.path, log=lambda m: print(m, flush=True))["poses"]
    except ValueError as e:
        print(e, flush=True)
        return 1
    if args.dry_run:
        print(" -> ".join(plan))
        for name, p in poses.items():
            print(f"  {name}: " + ", ".join(f"{k} {v:.2f}" for k, v in p.items()))
        return 0

    from viam.components.arm import Arm

    from bookworm.hardware.viam_robot import ViamConnection

    conn = ViamConnection()
    arm = conn.component(Arm, "arm")

    def on_stop(*_):                       # SIGTERM (e.g. a Stop button) halts the arm
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, on_stop)

    tag = flip_path.step_tag(args.path)
    try:
        code, msg = play(conn, arm, plan, pts, poses, on_step=lambda s: print(f"STEP {tag}{s}", flush=True),
                         log=lambda m: print(m, flush=True))
        if code or args.no_park:
            print(msg if code else f"done: {args.times} page(s)", flush=True)
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
