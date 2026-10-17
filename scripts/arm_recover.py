"""Bring the arm from ANY position back to Midhat's park spot, with Viam's motion planner.

    .venv/bin/python scripts/arm_recover.py          dry run: show where the arm is and the plan
    .venv/bin/python scripts/arm_recover.py --go     do it (hand on the e-stop the first times)

Steps (Ctrl+C or SIGTERM halts the arm at any point):
  1. lift straight up to a safe height, orientation held (always away from the book)
  2. plan to the park pose with the Viam motion service, avoiding the table, the book
     (config/workspace.yaml from measure_workspace.py, else hand-measured fallback) and a
     sphere around the gripper standing in for the pencil
  3. fine-align onto the exact recorded park joints (planners may land on slightly
     different joint angles); refuse if any joint would need > 10 deg, then check <= 1 deg
From park, arm_goto.py / the flip take over with the recorded joint moves.

If the arm is already at a recorded spot it can reach park from directly, it just does the
recorded joint move (same as arm_goto.py park).
"""
from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from arm_goto import AT_TOLERANCE_DEG, SAFE_FROM, nearest  # noqa: E402
from flip_path import MAX_JOINT_ERROR_DEG  # noqa: E402

WAYPOINTS = ROOT / "config" / "waypoints.yaml"
WORKSPACE = ROOT / "config" / "workspace.yaml"
GOAL = "out_of_the_way"
LIFT_ABOVE_BOOK_MM = 200          # step 1 target height above the book top
BOOK_MARGIN_MM = 40               # keep-out margin around the book on every side
PENCIL_MM = 135                   # pencil sticks out this far past the gripper, along the tool axis
PENCIL_RADIUS_MM = 25             # capsule radius (pencil + margin)
BOOK_MOVED_MM = 30                # warn if a fresh scan finds the book this far away (scan noise is ~20 mm)
MAX_ALIGN_DEG = 10.0


def load_points():
    d = yaml.safe_load(WAYPOINTS.read_text())
    joints = {str(p["name"]): p["joints"] for p in d["flip_path"] if "joints" in p}
    poses = {str(p["name"]): p["pose"] for p in d["flip_path"] if "pose" in p}
    for name in ("start", "out_of_the_way"):
        joints[name], poses[name] = d[name]["joints"], d[name]["pose"]
    return joints, poses


def workspace(fresh: dict | None = None) -> dict:
    """Obstacles from a fresh depth scan (``fresh``, measure_workspace.measure()), else the saved file."""
    if fresh or WORKSPACE.exists():
        w = fresh or yaml.safe_load(WORKSPACE.read_text())
        b = w["book"]
        hand = w.get("book_hand_measured_mm", [274, 197, 30])
        return {"table_z": w["table_z"], "center": b["center"], "angle": b["angle_deg"], "top_z": b["top_z"],
                "length": max(b["length"], hand[0]), "width": max(b["width"], hand[1]),
                "obstacles": w.get("obstacles", []), "voxel": w.get("obstacle_voxel_mm", 50.0),
                "source": "fresh depth scan" if fresh else f"{WORKSPACE.name} ({w.get('measured', '?')})"}
    # fallback: depth-camera reading of 2026-10-17 (book partly out of view) + hand-measured size
    return {"table_z": 2.0, "center": [435.0, 403.0], "angle": 0.0, "top_z": 44.0,
            "length": 274.0, "width": 197.0, "obstacles": [], "voxel": 50.0,
            "source": "fallback numbers (run measure_workspace.py --save)"}


def world_state(ws: dict):
    from viam.proto.common import (Capsule, Geometry, GeometriesInFrame, Pose, PoseInFrame, RectangularPrism,
                                   Transform, Vector3, WorldState)

    tz = ws["table_z"]
    h = ws["top_z"] - tz + BOOK_MARGIN_MM
    book = Geometry(center=Pose(x=ws["center"][0], y=ws["center"][1], z=tz + h / 2, o_z=1, theta=ws["angle"]),
                    box=RectangularPrism(dims_mm=Vector3(x=ws["length"] + 2 * BOOK_MARGIN_MM,
                                                         y=ws["width"] + 2 * BOOK_MARGIN_MM, z=h)),
                    label="book")
    # table top as a thin slab just under the measured surface (the arm base stands on it)
    table = Geometry(center=Pose(x=0, y=0, z=tz - 30, o_z=1), box=RectangularPrism(dims_mm=Vector3(x=3000, y=3000, z=40)),
                     label="table")
    # depth-camera blocks: cables, the phone-mount bar, the page holder, hands…
    s = ws["voxel"]
    blocks = [Geometry(center=Pose(x=x, y=y, z=z, o_z=1), box=RectangularPrism(dims_mm=Vector3(x=s, y=s, z=s)),
                       label=f"seen{i}") for i, (x, y, z) in enumerate(ws["obstacles"])]
    # the pencil, as a capsule along the tool axis from the gripper, so the planner keeps it clear too
    pencil = Transform(reference_frame="pencil",
                       pose_in_observer_frame=PoseInFrame(reference_frame="gripper", pose=Pose(z=PENCIL_MM / 2, o_z=1)),
                       physical_object=Geometry(capsule=Capsule(radius_mm=PENCIL_RADIUS_MM,
                                                                length_mm=PENCIL_MM + 2 * PENCIL_RADIUS_MM),
                                                label="pencil"))
    return WorldState(obstacles=[GeometriesInFrame(reference_frame="world", geometries=[table, book, *blocks])],
                      transforms=[pencil])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--go", action="store_true", help="actually move (default: dry run)")
    ap.add_argument("--no-scan", action="store_true", help="skip the fresh depth scan; use config/workspace.yaml")
    args = ap.parse_args(argv)

    joints, poses = load_points()

    from viam.components.arm import Arm
    from viam.proto.common import Pose, PoseInFrame
    from viam.proto.component.arm import JointPositions
    from viam.proto.service.motion import Constraints, LinearConstraint
    from viam.services.motion import MotionClient

    from bookworm.hardware.viam_robot import ViamConnection

    conn = ViamConnection()
    arm = conn.component(Arm, "arm")
    motion = conn.component(MotionClient, "builtin")

    def on_stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, on_stop)

    try:
        saved = workspace()
        fresh = None
        if not args.no_scan:                 # look before moving: plan around what is there right now
            try:
                from measure_workspace import capture, measure

                fresh = measure(*capture(conn))
            except (Exception, SystemExit) as e:  # noqa: BLE001 - camera can't see the table from here
                print(f"depth scan unusable from this pose ({e}); using {saved['source']}")
        ws = workspace(fresh) if fresh else saved
        if fresh and WORKSPACE.exists():
            moved = ((ws["center"][0] - saved["center"][0]) ** 2 + (ws["center"][1] - saved["center"][1]) ** 2) ** 0.5
            print(f"book is {moved:.0f} mm from where it was measured"
                  + (f"  !! more than {BOOK_MOVED_MM} mm: the recorded flip may miss the page" if moved > BOOK_MOVED_MM else ""))
        safe_z = ws["top_z"] + LIFT_ABOVE_BOOK_MM

        now_j = list(conn.call(arm.get_joint_positions()).values)
        p = conn.call(arm.get_end_position())
        here, off = nearest(now_j, joints)
        at = here if off <= AT_TOLERANCE_DEG else None
        goal = poses[GOAL]
        print(f"arm: end at ({p.x:.0f}, {p.y:.0f}, {p.z:.0f}) mm; "
              + (f"at recorded spot '{at}'" if at else f"not at a recorded spot (nearest '{here}', {off:.1f} deg off)"))
        print(f"obstacles from {ws['source']}: table top z={ws['table_z']:.0f}; book "
              f"{ws['length']:.0f}x{ws['width']:.0f} mm at ({ws['center'][0]:.0f}, {ws['center'][1]:.0f}), "
              f"top z={ws['top_z']:.0f}, +{BOOK_MARGIN_MM} mm margin; {len(ws['obstacles'])} other blocks; "
              f"pencil capsule {PENCIL_MM} mm")

        if at == GOAL:
            print("already parked")
            return 0
        if at in SAFE_FROM[GOAL]:
            print(f"plan: recorded joint move {at} -> {GOAL}")
            steps = [("joints", None)]
        else:
            steps = []
            if p.z < safe_z:
                steps.append(("lift", None))
            steps += [("plan", None), ("align", None)]
            print("plan: " + " -> ".join({"lift": f"lift straight up to z={safe_z:.0f}",
                                           "plan": f"motion-plan to {GOAL} ({goal['x']:.0f}, {goal['y']:.0f}, {goal['z']:.0f})",
                                           "align": "fine-align joints"}[s] for s, _ in steps))
        if not args.go:
            print("dry run: nothing moved. Add --go to do it.")
            return 0

        wsd = world_state(ws)
        for step, _ in steps:
            if step == "joints":
                conn.call(arm.move_to_joint_positions(JointPositions(values=joints[GOAL])), timeout=120)
            elif step == "lift":
                print(f"lifting to z={safe_z:.0f}…", flush=True)
                up = Pose(x=p.x, y=p.y, z=safe_z, o_x=p.o_x, o_y=p.o_y, o_z=p.o_z, theta=p.theta)
                conn.call(motion.move("arm", PoseInFrame(reference_frame="world", pose=up), wsd,
                                      Constraints(linear_constraint=[LinearConstraint(line_tolerance_mm=15,
                                                                                      orientation_tolerance_degs=5)])),
                          timeout=120)
            elif step == "plan":
                print("planning to park…", flush=True)
                conn.call(motion.move("arm", PoseInFrame(reference_frame="world", pose=Pose(**goal)), wsd), timeout=180)
            elif step == "align":
                cur = list(conn.call(arm.get_joint_positions()).values)
                worst = max(abs(a - b) for a, b in zip(cur, joints[GOAL]))
                if worst > MAX_ALIGN_DEG:
                    print(f"STOPPED: at the park pose but a joint is {worst:.1f} deg from the recorded angles "
                          f"(planner chose another arm configuration); not forcing it", flush=True)
                    return 2
                conn.call(arm.move_to_joint_positions(JointPositions(values=joints[GOAL])), timeout=60)
        err = max(abs(a - b) for a, b in zip(conn.call(arm.get_joint_positions()).values, joints[GOAL]))
        if err > MAX_JOINT_ERROR_DEG:
            print(f"STOPPED short of {GOAL}: joint error {err:.2f} deg", flush=True)
            return 2
        print(f"at {GOAL} (within {err:.2f} deg)", flush=True)
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
