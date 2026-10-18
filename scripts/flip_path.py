"""Play Midhat's recorded page flip (config/waypoints.yaml) on the arm.

    .venv/bin/python scripts/flip_path.py              one page (v1 path), then park + pause
    .venv/bin/python scripts/flip_path.py --path v2    use the v2 path (flip_path_v2)
    .venv/bin/python scripts/flip_path.py --times 3    three pages
    .venv/bin/python scripts/flip_path.py --dry-run    print the moves, don't connect

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


def play(conn, arm, plan, pts, on_step=None, stopped=lambda: False, log=print):
    """Play ``plan`` over an open connection. Returns (exit code, message).

    on_step(name) is called before each move; stopped() is checked before each move and during
    pauses (the caller halts the arm itself, e.g. the control page's Stop calls arm.stop())."""
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
        target = pts[step]
        if on_step:
            on_step(step)
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
    args = ap.parse_args(argv)

    pts, path, between = load(args.path)
    plan = make_plan(path, between, args.times, park=not args.no_park)
    if args.dry_run:
        print(" -> ".join(plan))
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
                         log=lambda m: print(m, flush=True))
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
