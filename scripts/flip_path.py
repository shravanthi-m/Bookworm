"""Play Midhat's recorded page flip (config/waypoints.yaml -> flip_path) on the arm.

    .venv/bin/python scripts/flip_path.py              one page, then park + pause
    .venv/bin/python scripts/flip_path.py --times 3    three pages
    .venv/bin/python scripts/flip_path.py --dry-run    print the moves, don't connect

The control page's Flip button runs this (ui.flip_command in config/phone_mount.yaml).
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


def load():
    d = yaml.safe_load(WAYPOINTS.read_text())
    pts = {str(p["name"]): p["joints"] for p in d["flip_path"] if "joints" in p}
    for name in ("start", "out_of_the_way"):
        pts[name] = d[name]["joints"]
    path = [str(p["name"]) for p in d["flip_path"]]
    return pts, path, d.get("between_pages", {})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--times", type=int, default=1, help="pages to turn (default 1)")
    ap.add_argument("--no-park", action="store_true", help="end at 0 instead of parking and pausing")
    ap.add_argument("--dry-run", action="store_true", help="print the moves without connecting")
    args = ap.parse_args(argv)

    pts, path, between = load()
    park, pause = between.get("park"), float(between.get("pause_s", 0))
    plan = []
    for _ in range(args.times):
        plan += path
        if park and not args.no_park:
            plan += [park, f"wait {pause:g}"]
    if args.dry_run:
        print(" -> ".join(plan))
        return 0

    from viam.components.arm import Arm
    from viam.proto.component.arm import JointPositions

    from bookworm.hardware.viam_robot import ViamConnection

    conn = ViamConnection()
    arm = conn.component(Arm, "arm")

    def on_stop(*_):                       # the UI's Stop button sends SIGTERM
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, on_stop)

    try:
        page, t_page = 1, time.time()
        for step in plan:
            if step.startswith("wait "):
                print(f"page {page} done in {time.time() - t_page:.1f} s, parked; pausing {pause:g} s", flush=True)
                time.sleep(pause)
                page, t_page = page + 1, time.time()
                continue
            target = pts[step]
            conn.call(arm.move_to_joint_positions(JointPositions(values=target)), timeout=120)
            now = list(conn.call(arm.get_joint_positions()).values)
            err = max(abs(a - b) for a, b in zip(now, target))
            if err > MAX_JOINT_ERROR_DEG:
                conn.call(arm.stop())
                print(f"STOPPED at {step}: joint error {err:.2f} deg", flush=True)
                return 2
        if args.no_park:
            print(f"done: {args.times} page(s), ended at 0", flush=True)
        return 0
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
