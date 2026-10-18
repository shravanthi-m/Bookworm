"""The recorded flip path moved with the book (bookworm/hardware/dynamic_path.py, scripts/flip_path_dynamic.py),
with no robot. scripts/flip_path.py (main) is used unchanged."""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from bookworm.hardware import dynamic_path as dp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import flip_path  # noqa: E402
import flip_path_dynamic  # noqa: E402

REF = {"center": [400.0, 0.0], "angle_deg": 90.0, "top_z": 35.0}
WP = yaml.safe_load((ROOT / "config" / "waypoints.yaml").read_text())


def test_frame_from_points_finds_centre_angle_and_top():
    pts = {"tl": [500, 120], "bl": [300, 120], "tr": [500, -120], "br": [300, -120]}
    f = dp.frame_from_points(pts, 36.0)
    assert f["center"] == [400.0, 0.0] and f["top_z"] == 36.0
    assert f["angle_deg"] == pytest.approx(-90.0)                       # left page -> right page is world -y


@pytest.mark.parametrize("o", [(0.76, -0.57, -0.30, -88.44), (0.0, 0.0, -1.0, -5.6), (0.34, -0.81, -0.48, -97.02)])
def test_orientation_vectors_round_trip(o):
    back = dp.matrix_to_ov(dp.ov_to_matrix(*o))
    n = np.linalg.norm(o[:3])
    assert np.allclose(back[:3], np.array(o[:3]) / n, atol=1e-9)
    assert (back[3] - o[3] + 180) % 360 - 180 == pytest.approx(0, abs=1e-6)


def test_a_book_that_has_not_moved_plays_the_recorded_joints():
    p = dp.plan(WP, "flip_path_v2", REF, dict(REF, center=[400.5, 0.3]))
    assert p["mode"] == "recorded" and p["poses"] == {}
    assert dp.plan(WP, "flip_path_v2", None, REF)["mode"] == "recorded"


def test_a_shifted_and_turned_book_moves_every_point_rigidly():
    now = {"center": [430.0, -20.0], "angle_deg": 100.0, "top_z": 38.0}
    p = dp.plan(WP, "flip_path_v2", REF, now)
    assert p["mode"] == "dynamic" and set(p["poses"]) >= {str(n) for n in range(1, 12)} | {"out_of_the_way"}
    taught = {str(q["name"]): q["pose"] for q in WP["flip_path_v2"] if "pose" in q}
    a, b = taught["1"], taught["10"]
    pa, pb = p["poses"]["1"], p["poses"]["10"]
    # distances between points are kept (a rigid move), heights all shift by the book's change in height
    assert math.dist((a["x"], a["y"]), (b["x"], b["y"])) == pytest.approx(math.dist((pa["x"], pa["y"]), (pb["x"], pb["y"])))
    assert pa["z"] - a["z"] == pytest.approx(3.0)
    # the point turned 10 deg about the book's centre
    ang = lambda q, c: math.degrees(math.atan2(q["y"] - c[1], q["x"] - c[0]))                  # noqa: E731
    assert (ang(pa, now["center"]) - ang(a, REF["center"]) - 10 + 180) % 360 - 180 == pytest.approx(0, abs=1e-6)
    # the tool turned with it: same tilt, its direction turned 10 deg
    assert pa["o_z"] == pytest.approx(a["o_z"] / np.linalg.norm([a["o_x"], a["o_y"], a["o_z"]]), abs=1e-9)
    assert (math.degrees(math.atan2(pa["o_y"], pa["o_x"]) - math.atan2(a["o_y"], a["o_x"])) - 10 + 180) % 360 - 180 \
        == pytest.approx(0, abs=1e-6)


def test_a_book_moved_too_far_is_refused():
    with pytest.raises(ValueError, match="teach the path again"):
        dp.plan(WP, "flip_path_v2", REF, dict(REF, center=[600.0, 0.0]))


def test_reference_is_kept_in_local_yaml(tmp_path):
    cfg = tmp_path / "local.yaml"
    cfg.write_text("gui: {probe: {tool_length_mm: 155}}\n")
    dp.set_reference(REF, cfg)
    assert dp.load_reference(cfg) == REF and yaml.safe_load(cfg.read_text())["gui"]["probe"]["tool_length_mm"] == 155


class PoseArm:
    """A fake arm that records joint moves and pose moves."""

    def __init__(self):
        self.joints, self.poses, self.now = [], [], SimpleNamespace(x=0.0, y=0.0, z=0.0)

    def move_to_joint_positions(self, jp):
        self.joints.append(list(jp.values))

    def get_joint_positions(self):
        return SimpleNamespace(values=self.joints[-1])

    def move_to_position(self, pose):
        self.poses.append(pose)
        self.now = SimpleNamespace(x=pose.x, y=pose.y, z=pose.z)

    def get_end_position(self):
        return self.now

    def stop(self):
        pass


def test_play_reaches_moved_points_as_poses():
    conn = SimpleNamespace(call=lambda v, timeout=None: v)
    pts, path, between = flip_path.load("v2")
    poses = dp.plan(WP, "flip_path_v2", REF, {"center": [410.0, 5.0], "angle_deg": 92.0, "top_z": 35.0})["poses"]
    arm = PoseArm()
    code, msg = flip_path_dynamic.play(conn, arm, flip_path.make_plan(path, between), pts, poses, log=lambda m: None)
    assert code == 0, msg
    assert len(arm.poses) == 12 and arm.joints == []                     # 11 path points + park, all moved
    assert arm.poses[0].x == pytest.approx(poses["1"]["x"])


def test_no_poses_is_mains_player_unchanged():
    conn = SimpleNamespace(call=lambda v, timeout=None: v)
    pts, path, between = flip_path.load("v2")
    arm = PoseArm()
    code, _ = flip_path_dynamic.play(conn, arm, flip_path.make_plan(path, between), pts, {}, log=lambda m: None)
    assert code == 0 and arm.poses == [] and arm.joints[0] == pts["1"]


def test_dry_run_without_a_reference_plays_as_taught(capsys):
    assert flip_path_dynamic.main(["--path", "v2", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "recorded" in out and "1 -> 2" in out
