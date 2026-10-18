"""Play Midhat's recorded page flip (config/waypoints.yaml) on the arm.

    .venv/bin/python scripts/flip_path.py              one page (v1 path), then park + pause
    .venv/bin/python scripts/flip_path.py --path v2    use the v2 path (flip_path_v2)
    .venv/bin/python scripts/flip_path.py --times 3    three pages
    .venv/bin/python scripts/flip_path.py --dry-run    print the moves, don't connect
    .venv/bin/python scripts/flip_path.py --dynamic    follow the book: move the path with it (see below)
    .venv/bin/python scripts/flip_path.py --set-reference   the book is where the path works: remember it

Dynamic: the path was taught with the book in one place. With --dynamic (or `ui.flip_dynamic: true`)
every point is carried by how far the book has moved and turned since then
(bookworm/hardware/dynamic_path.py). The book's position now comes from the workshop GUI's first
capture + Z link (data/book_frame.json); the reference is set once with --set-reference while the
book sits where the recorded path works. If the book hasn't moved, the recorded joints play as taught.

Paths: v1 = flip_path (original), v2 = flip_path_v2 (rubber-covered eraser, re-taught,
numbered 1-11). v1 is the fallback if v2 misbehaves.

The control page's Next page / Flip button calls `play()` directly over its own open Viam
connection (`ui.flip_path` in config/phone_mount.yaml picks v1 or v2), so it doesn't dial
the robot again for every page. Run from the command line, this script dials its own.
Every point is a saved joint position, so the arm repeats exactly what was recorded.
After each page it parks at `between_pages.park` (out of the camera's view) and waits
`between_pages.pause_s` seconds, so the page can be read.

Exit codes: 0 = done, 2 = a joint was more than 1 deg off (arm stopped), 130 = stopped.
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
WAYPOINTS = ROOT / "config" / "waypoints.yaml"
MAX_JOINT_ERROR_DEG = 1.0
MAX_POSE_ERROR_MM = 3.0             # dynamic points are reached as poses


PATHS = {"v1": "flip_path", "v2": "flip_path_v2"}


def load(version="v1"):
    d = yaml.safe_load(WAYPOINTS.read_text())
    key = PATHS[version]
    pts = {str(p["name"]): p["joints"] for p in d[key] if "joints" in p}
    for name in ("start", "out_of_the_way"):
        pts[name] = d[name]["joints"]
    path = [str(p["name"]) for p in d[key]]
    return pts, path, d.get("between_pages", {})


def make_plan(path, between, times=1, park=True):
    """The move list: the path `times` times, each followed by park + 'wait <s>'."""
    spot, pause = between.get("park"), float(between.get("pause_s", 0))
    plan = []
    for _ in range(times):
        plan += path
        if spot and park:
            plan += [spot, f"wait {pause:g}"]
    return plan


def step_tag(version):
    """v2 step names are numbers, so they're tagged 'v2:' for the control page's page-turn animation."""
    return "" if version == "v1" else f"{version}:"


def dynamic_poses(version="v1", frame_now=None, log=print):
    """Poses that follow the book (bookworm/hardware/dynamic_path.py), or {} to play the recorded joints.
    ``frame_now``: the current book frame (default: data/book_frame.json from the workshop's calibration)."""
    from bookworm.hardware import dynamic_path as dp

    d = yaml.safe_load(WAYPOINTS.read_text())
    p = dp.plan(d, PATHS[version], dp.load_reference(), frame_now or dp.load_frame())
    log(f"flip path: {p['mode']} ({p['why']})")
    return p["poses"]


def play(conn, arm, plan, pts, on_step=None, stopped=lambda: False, log=print, poses=None):
    """Play ``plan`` over an open connection. Returns (exit code, message).

    on_step(name) is called before each move; stopped() is checked before each move and during
    pauses (the caller halts the arm itself, e.g. the control page's Stop calls arm.stop()).
    ``poses`` ({name: pose}, from dynamic_poses): those points are reached as poses that follow the book,
    through the motion planner, instead of the recorded joints."""
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
        if poses and step in poses:
            from viam.proto.common import Pose

            want = poses[step]
            conn.call(arm.move_to_position(Pose(**{k: float(want[k]) for k in ("x", "y", "z", "o_x", "o_y", "o_z", "theta")})),
                      timeout=120)
            if stopped():
                return 130, "stopped: arm halted where it is"
            got = conn.call(arm.get_end_position())
            err = ((got.x - want["x"]) ** 2 + (got.y - want["y"]) ** 2 + (got.z - want["z"]) ** 2) ** 0.5
            if err > MAX_POSE_ERROR_MM:
                conn.call(arm.stop())
                return 2, f"STOPPED at {step}: {err:.1f} mm from the moved point"
            continue
        target = pts[step]
        conn.call(arm.move_to_joint_positions(JointPositions(values=target)), timeout=120)
        if stopped():
            return 130, "stopped: arm halted where it is"
        now = list(conn.call(arm.get_joint_positions()).values)
        err = max(abs(a - b) for a, b in zip(now, target))
        if err > MAX_JOINT_ERROR_DEG:
            conn.call(arm.stop())
            return 2, f"STOPPED at {step}: joint error {err:.2f} deg"
    return 0, "done"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", choices=sorted(PATHS), default="v1", help="which recorded path (default v1)")
    ap.add_argument("--times", type=int, default=1, help="pages to turn (default 1)")
    ap.add_argument("--no-park", action="store_true", help="end at 0 instead of parking and pausing")
    ap.add_argument("--dry-run", action="store_true", help="print the moves without connecting")
    ap.add_argument("--dynamic", action="store_true", help="move the path with the book (data/book_frame.json)")
    ap.add_argument("--set-reference", action="store_true",
                    help="remember the current book frame as where the path was taught, then exit")
    args = ap.parse_args(argv)

    if args.set_reference:
        from bookworm.hardware import dynamic_path as dp

        now = dp.load_frame()
        if now is None:
            print("no current book frame: run the workshop GUI's first capture and Z link first", flush=True)
            return 1
        dp.set_reference(now)
        print(f"reference book position set: centre {now['center']}, {now['angle_deg']:.1f} deg, top z {now['top_z']}")
        return 0

    pts, path, between = load(args.path)
    plan = make_plan(path, between, args.times, park=not args.no_park)
    poses = dynamic_poses(args.path) if args.dynamic else {}
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

    tag = step_tag(args.path)
    try:
        code, msg = play(conn, arm, plan, pts,
                         on_step=lambda s: print(f"STEP {tag}{s}", flush=True),
                         log=lambda m: print(m, flush=True), poses=poses)
        if code or args.no_park:
            print(msg if code else f"done: {args.times} page(s), ended at 0", flush=True)
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
