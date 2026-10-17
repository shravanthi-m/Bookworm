"""The GUI's basic routes on a plain mock robot: token, STOP, jog limits, poses, gripper, side camera, the tool setting."""
import re
import time

import numpy as np
import pytest
import yaml

from bookworm.gui import server
from bookworm.hardware import MockRobot, sidecam

REAL_SIDECAM = sidecam.SideCam
HOME = {"x": 390.0, "y": -57.0, "z": 480.0, "o_x": 0.0, "o_y": 0.0, "o_z": -1.0, "theta": -5.6}


@pytest.fixture(autouse=True)
def _no_real_webcam(monkeypatch):
    """Never open the laptop's real webcam from a test."""
    class NoCam:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            raise RuntimeError("no side camera in tests")

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(sidecam, "SideCam", NoCam)


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "LOCAL_CFG", tmp_path / "local.yaml")
    monkeypatch.setattr(server, "ROOT", tmp_path)
    robot = MockRobot()
    robot._pose = dict(HOME)
    cfg = {"poses": {"home": dict(HOME)}, "home_joints": [0, -17, -64, 0, 81, -14],
           "gui": {"limits": {"x": [100, 700], "y": [-400, 400], "z": [100, 600]}, "probe": {"tool_length_mm": 155}}}
    a = server.create_app(robot, cfg=cfg)
    c = a.test_client()
    tok = re.search(r'TOKEN = "([^"]+)"', c.get("/").get_data(as_text=True)).group(1)
    time.sleep(0.2)
    return c, (lambda url, **b: c.post(url, json=b, headers={"X-Token": tok})), robot, tmp_path


def _idle(c):
    for _ in range(50):
        if not c.get("/api/status").get_json()["moving"]:
            return
        time.sleep(0.05)


def test_post_needs_the_page_token(app):
    c, post, robot, _ = app
    assert c.post("/api/stop", json={}).status_code == 403
    assert post("/api/stop").status_code == 200


def test_stop_blocks_motion_until_reset(app):
    c, post, robot, _ = app
    post("/api/stop")
    assert c.get("/api/status").get_json()["estop"]
    assert post("/api/jog", axis="z", delta=5).status_code == 409
    assert post("/api/flow/stage", id="level", confirm=True).status_code == 409
    post("/api/reset")
    assert post("/api/jog", axis="z", delta=5).status_code == 200


def test_jog_respects_limits_and_the_tip_floor(app):
    c, post, robot, _ = app
    assert post("/api/jog", axis="x", delta=80).status_code == 400                       # step too big
    assert post("/api/jog", axis="z", delta=50).status_code == 200
    _idle(c)
    assert robot.end_pose()["z"] == 530
    robot._pose = dict(HOME, z=175.0)                                                     # tip at z 20: under the 25 mm fallback floor
    r = post("/api/jog", axis="z", delta=-5)
    assert r.status_code == 409 and "floor" in r.get_json()["error"]


def test_poses_save_and_go(app):
    c, post, robot, tmp = app
    assert post("/api/pose/save", name="bad name!").status_code == 400
    assert post("/api/pose/save", name="over").status_code == 200
    assert yaml.safe_load((tmp / "local.yaml").read_text())["poses"]["over"]["z"] == 480
    assert "over" in c.get("/api/status").get_json()["saved_poses"]
    robot._pose = dict(HOME, x=300.0)
    assert post("/api/pose/go", name="over").status_code == 200
    _idle(c)
    assert robot.end_pose()["x"] == 390


def test_gripper_is_clamped(app):
    c, post, robot, _ = app
    assert post("/api/gripper", position=2000).status_code == 200
    assert robot.gripper_get() == 850


def test_side_photo_endpoint(app, monkeypatch, tmp_path):
    c, post, robot, _ = app
    monkeypatch.setattr(sidecam, "SideCam", REAL_SIDECAM)
    monkeypatch.setattr(sidecam, "SIDE_DIR", tmp_path / "side")
    monkeypatch.setattr(sidecam.cv2, "VideoCapture", lambda *a: type("C", (), {
        "isOpened": lambda s: True, "set": lambda s, *a: True, "release": lambda s: None,
        "read": lambda s: (True, np.full((72, 128, 3), 90, np.uint8))})())
    r = post("/api/side")
    assert r.status_code == 200, r.get_json()
    assert c.get("/side/" + r.get_json()["name"]).status_code == 200
    assert c.get("/side/nope.jpg").status_code == 404


def test_the_tool_length_is_the_one_setting(app):
    c, post, robot, tmp = app
    assert post("/api/tool", tool_length_mm=10).status_code == 400
    assert post("/api/tool", tool_length_mm="x").status_code == 400
    assert post("/api/tool", tool_length_mm=190).status_code == 200                     # a pencil
    assert yaml.safe_load((tmp / "local.yaml").read_text())["gui"]["probe"]["tool_length_mm"] == 190
    assert c.get("/api/flow").get_json()["tool_length_mm"] == 190


def test_stages_need_confirmation_and_prerequisites(app):
    c, post, robot, _ = app
    assert post("/api/flow/stage", id="level").status_code == 400
    assert post("/api/flow/stage", id="nope", confirm=True).status_code == 400
    assert post("/api/demo/plan").status_code == 409                                     # no capture, no depth
    assert post("/api/flow/stage", id="demo", confirm=True).status_code == 200
    for _ in range(50):
        f = c.get("/api/flow").get_json()
        if not f["running"]:
            break
        time.sleep(0.05)
    demo = next(s for s in f["stages"] if s["id"] == "demo")
    assert demo["state"] == "fail" and "first capture" in demo["msg"]


def test_run_all_leaves_out_the_manual_xy_marking(app):
    c, post, robot, _ = app
    ids = post("/api/flow/run", confirm=True).get_json()["stages"]
    assert "xycal" not in ids and ids[-1] == "demo"
    assert post("/api/xycal/next").status_code == 409                                    # nothing is waiting
    assert post("/api/xycal/mark", key="tl", u=1, v=1).status_code == 409               # no photo yet


def test_contact_press_only_while_a_search_waits(app):
    c, post, robot, _ = app
    assert post("/api/flow/contact", step=1).status_code == 409


def test_zlink_spot_lock_and_unlock(app):
    c, post, robot, tmp = app
    assert post("/api/zlink/spot", x="a").status_code == 400
    assert post("/api/zlink/spot", x=300, y=150).get_json()["lock"] == [300, 150]
    assert c.get("/api/flow").get_json()["z_link_spot_lock"] == [300, 150]
    assert post("/api/zlink/spot", here=True).get_json()["lock"] == [390.0, -57.0]       # tip straight below the flange
    post("/api/zlink/spot", clear=True)
    assert c.get("/api/flow").get_json()["z_link_spot_lock"] is None


def test_page_has_only_the_new_setup(app):
    c, post, robot, _ = app
    html = c.get("/").get_data(as_text=True)
    for gone in ("Moondream", "Save calibration", "apply saved offset", "Probe all six", ">Record<"):
        assert gone not in html
    for there in ("Touch demo", "Z link", "Book model", "XY marking", "tool length", "Lock the arm's spot"):
        assert there in html
