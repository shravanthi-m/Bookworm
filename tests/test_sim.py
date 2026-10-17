"""The whole Setup flow offline on the simulated rig, and recorded views replaying."""
import json
import re
import time

import numpy as np
import pytest
import yaml

from bookworm.gui import server
from bookworm.hardware import sidecam
from bookworm.hardware.replay import ReplayRobot, save_view
from bookworm.hardware.sim import HOME, HOME_JOINTS, SimRobot
from bookworm.vision.depth_height import world_heights


# ---- the simulator on its own ---------------------------------------------------------------------------------
def test_sim_depth_lifts_to_the_scene_heights():
    r = SimRobot(depth_tilt=(0.0, 0.0), noise_mm=0.0)
    w = world_heights(r.capture_depth(), r.camera_geometry(), step=8)
    x, y, z = w[..., 0], w[..., 1], w[..., 2]
    ok = np.isfinite(z)
    assert np.median(np.abs(z[ok] - r.surface(x[ok], y[ok]))) < 1.0


def test_sim_tip_contact_load_and_side_view(tmp_path):
    r = SimRobot(side_dir=tmp_path)
    x, y = 403.0, -17.0
    page = float(r.surface(x, y))
    r.move_to(dict(HOME, x=x, y=y, z=page + 5 + r.tool_length))
    free = r.joint_load()
    r.move_to(dict(HOME, x=x, y=y, z=page - 2 + r.tool_length))
    assert r.pressed_mm == pytest.approx(2, abs=0.01)
    assert r.joint_load()[1] - free[1] > 5                                         # pressing shows on joint 2
    with r.side_cam() as cam:
        assert cam.photo("t").parent == tmp_path


def test_real_views_are_recorded_and_replay(tmp_path):
    r = SimRobot()
    for dx in (0, 80):
        r.move_to(dict(HOME, x=HOME["x"] + dx))
        save_view(tmp_path, r.capture(), r.capture_depth(), r.camera_geometry(), r.end_pose())
    rp = ReplayRobot(tmp_path, home_pose=HOME, home_joints=HOME_JOINTS)
    rp.move_to(dict(HOME, x=HOME["x"] + 75))                                      # nearest is the second view
    assert rp.camera_geometry().T_world_cam[0, 3] == pytest.approx(r.camera_geometry().T_world_cam[0, 3], abs=1e-6)
    assert rp.capture_depth().dtype == np.uint16 and rp.last_miss_mm == pytest.approx(5, abs=0.01)


# ---- the GUI on the simulator -----------------------------------------------------------------------------------
@pytest.fixture
def sim(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "LOCAL_CFG", tmp_path / "local.yaml")
    monkeypatch.setattr(server, "ROOT", tmp_path)
    monkeypatch.setattr(sidecam, "SIDE_DIR", tmp_path / "side")
    robot = SimRobot(side_dir=tmp_path / "side")
    cfg = {"poses": {"home": dict(HOME)}, "home_joints": HOME_JOINTS,
           "gui": {"tip_min_z_mm": 25, "limits": {"x": [100, 700], "y": [-400, 400], "z": [100, 600]},
                   "probe": {"tool_length_mm": 155},
                   "depth": {"height_step_wait_s": 0.05, "height_settle_s": 0.0}}}
    app = server.create_app(robot, cfg=cfg)
    c = app.test_client()
    tok = re.search(r'TOKEN = "([^"]+)"', c.get("/").get_data(as_text=True)).group(1)
    time.sleep(0.3)
    return c, (lambda url, **b: c.post(url, json=b, headers={"X-Token": tok})), robot, tmp_path


def _wait(c, secs=240):
    for _ in range(int(secs * 10)):
        time.sleep(0.1)
        f = c.get("/api/flow").get_json()
        if not f["running"] and not f["moving"]:
            return {s["id"]: s for s in f["stages"]}, f
    raise AssertionError("flow did not finish")


def _setup(sim):
    c, post, robot, tmp = sim
    assert post("/api/flow/run", confirm=True, skip_touch=True).status_code == 200
    st, f = _wait(c)
    assert [st[k]["state"] for k in ("level", "capture", "depth")] == ["warn", "ok", "ok"], {k: st[k]["msg"] for k in st}
    assert st["zlink"]["state"] == st["demo"]["state"] == "idle" and robot.pressed_mm == 0      # nothing touched
    return c, post, robot, tmp


def _zlink(c, post):
    assert post("/api/flow/stage", id="zlink", confirm=True).status_code == 200
    st, f = _wait(c)
    return st["zlink"], f


def _model(c, post):
    assert post("/api/flow/stage", id="model", confirm=True).status_code == 200
    st, f = _wait(c)
    assert st["model"]["state"] == "ok", st["model"]["msg"]
    return f["model"]


def test_stop_before_touching_runs_stages_1_to_3_only(sim):
    c, post, robot, tmp = _setup(sim)
    f = c.get("/api/flow").get_json()
    assert len([p for p in f["points"] if p["world"]]) == 6
    assert list((tmp / "data" / "depth").glob("*/scan.json"))                      # every scan is kept


def test_zlink_finds_the_table_by_the_load_and_locks_the_spot(sim):
    c, post, robot, tmp = _setup(sim)
    n0 = len(robot.log)
    z, f = _zlink(c, post)
    d = z["data"]
    assert z["state"] in ("ok", "warn"), z["msg"]
    assert d["table_found_by"] in ("load", "stall") and abs(d["arm_offset_mm"] - robot.table_z) < 1.5
    assert d["retreat"] == "previous"                                              # straight back one step, not home
    pushes = [e[1] for e in robot.log[n0:] if e[0] == "sim: tip stopped by the table"]
    assert pushes and max(pushes) <= 3.0 + 1e-6                                   # never more than one step into the table
    assert f["z_link_spot_lock"] == d["spot"] or f["z_link_spot_lock"] == [pytest.approx(v, abs=1) for v in d["spot"]]
    assert f["table_measured"] and f["z_link_tool_length_mm"] == 155
    assert f["live"]["traces"]["table"] and json.loads(next((tmp / "data" / "depth" / "zlink").glob("*.json")).read_text())["traces"]["table"]


def test_when_the_load_stays_high_after_backing_off_the_arm_goes_home(sim, monkeypatch):
    c, post, robot, tmp = _setup(sim)
    real = robot.joint_load
    stuck = {"on": False}

    def load():
        v = real()
        if robot._push_mm > 0:
            stuck["on"] = True                                                     # caught on something: stays loaded
        return [v[0], v[1] + 15 * stuck["on"], v[2] + 9 * stuck["on"], *v[3:]]

    monkeypatch.setattr(robot, "joint_load", load)
    z, f = _zlink(c, post)
    assert z["state"] == "warn" and z["data"]["retreat"] == "home" and "went home" in z["msg"]
    assert max(abs(robot.end_pose()[k] - HOME[k]) for k in "xyz") < 1                 # at home


def test_zlink_is_reused_until_the_tool_changes(sim):
    c, post, robot, tmp = _setup(sim)
    _zlink(c, post)
    post("/api/flow/run", confirm=True)
    st, f = _wait(c)
    assert "using the saved z link" in st["zlink"]["msg"]
    assert post("/api/tool", tool_length_mm=172).status_code == 200
    robot.tool_length = 172.0                                                      # the new tool really is longer: 17 mm more than the search below the old table
    st = {s["id"]: s for s in c.get("/api/flow").get_json()["stages"]}
    assert st["zlink"]["state"] == "stale"
    post("/api/flow/run", confirm=True, skip_touch=False)
    st, f = _wait(c)
    assert "using the saved" not in st["zlink"]["msg"] and abs(st["zlink"]["data"]["arm_offset_mm"] - robot.table_z) < 1.5


def test_demo_dry_run_lists_every_point_with_its_page_z(sim):
    c, post, robot, tmp = _setup(sim)
    q = post("/api/demo/plan").get_json()
    assert not q["table_measured"] and all("skip" in p for p in q["points"])          # no book model yet
    _zlink(c, post)
    _model(c, post)
    q = post("/api/demo/plan").get_json()
    assert [p["key"] for p in q["points"]] == ["tbl", "tl", "tr", "cr", "br", "bl", "cl"]
    assert all("aim" in p and p["z"] > robot.table_z for p in q["points"][1:]), q


def test_demo_touches_every_point_at_the_expected_height(sim):
    c, post, robot, tmp = _setup(sim)
    _zlink(c, post)
    _model(c, post)
    n0 = len(robot.log)
    assert post("/api/flow/stage", id="demo", confirm=True).status_code == 200
    st, f = _wait(c, 400)
    res = f["demo"]["results"]
    assert st["demo"]["state"] in ("ok", "warn"), st["demo"]["msg"]
    assert set(res) == {"tbl", "tl", "tr", "cr", "br", "bl", "cl"} and all(r["verdict"] in ("on", "early") for r in res.values()), res
    assert all(r["early_mm"] is None or abs(r["early_mm"]) <= 3 for r in res.values())     # the map is right to a few mm
    presses = [e[1] for e in robot.log[n0:] if e[0] == "sim: tip pressed into the page"]
    assert not presses or max(presses) <= 3.0                                           # touched, not pushed into the book
    assert all(r["photo"] for r in res.values())                                        # a side photo at every touch
    assert max(abs(robot.end_pose()[k] - HOME[k]) for k in "xyz") < 1                     # finished at home
    assert list((tmp / "data" / "demo").glob("*.json"))


def test_demo_reports_a_page_higher_than_the_map_and_backs_off(sim):
    c, post, robot, tmp = _setup(sim)
    _zlink(c, post)
    _model(c, post)
    robot.left_mm += 6.0                                                           # the left page rose after the scan
    n0 = len(robot.log)
    post("/api/flow/stage", id="demo", confirm=True)
    st, f = _wait(c, 400)
    res = f["demo"]["results"]
    assert res["cl"]["verdict"] == "early" and 3 < res["cl"]["early_mm"] < 9, res["cl"]
    assert res["tr"]["verdict"] == "on" or abs(res["tr"]["early_mm"]) <= 3                   # the right page did not move
    presses = [e[1] for e in robot.log[n0:] if e[0] == "sim: tip pressed into the page"]
    assert max(presses) <= 3.0 + 1e-6                                                     # backed off within a step
    assert st["demo"]["state"] == "warn" and "early" in st["demo"]["msg"]
    m = f["model"]                                                                        # the touches went into the model
    assert m["version"] == 2 and m["dir"].endswith("v2") and m["correction"]["kind"] == "plane"
    assert m["correction"]["rms_after"] < m["correction"]["rms_before"]
    z = m["touches"]
    left = [t for t in z if t["kind"] == "demo" and t["key"] in ("tl", "cl", "bl")]
    assert all(t["residual"] > 2 for t in left if t["verdict"] == "early")              # the left page measured higher


def test_demo_touches_the_far_table_first_and_travels_high_between_points(sim):
    c, post, robot, tmp = _setup(sim)
    z, f = _zlink(c, post)
    spot = z["data"]["spot"]
    away = np.hypot(spot[0] - robot.centre[0], spot[1] - robot.centre[1])
    assert away > 200, spot                                                       # the far edge, not next to the book
    _model(c, post)
    moves = []
    real = robot.move_to
    robot.move_to = lambda p: (moves.append(dict(p)), real(p))[1]
    post("/api/flow/stage", id="demo", confirm=True)
    st, f = _wait(c, 400)
    res = f["demo"]["results"]
    assert f["demo"]["order"][0] == "tbl" and res["tbl"]["verdict"] in ("on", "early"), res.get("tbl")
    assert res["tbl"]["expected_z"] == pytest.approx(robot.table_z, abs=1.5)
    # every sideways move (x or y changes by more than 5 mm) happens at half the home height or above
    side = [(a, b) for a, b in zip(moves, moves[1:]) if np.hypot(b["x"] - a["x"], b["y"] - a["y"]) > 5]
    assert side and all(min(a["z"], b["z"]) >= HOME["z"] / 2 - 1 for a, b in side), [(a["z"], b["z"]) for a, b in side]


def test_xy_marking_hovers_waits_and_saves_the_correction(sim):
    from bookworm.vision.geometry import world_to_pixel

    c, post, robot, tmp = _setup(sim)
    _zlink(c, post)
    _model(c, post)
    assert post("/api/flow/stage", id="xycal", confirm=True).status_code == 200
    hovers = []
    for _ in range(600):
        time.sleep(0.1)
        f = c.get("/api/flow").get_json()
        k = f["xycal"]["waiting"]
        if k:
            tip_z = robot.end_pose()["z"] - robot.tool_length
            hovers.append(tip_z - float(robot.surface(robot.end_pose()["x"], robot.end_pose()["y"])))
            assert post("/api/xycal/next", skip=(k == "cr")).status_code == 200
            while c.get("/api/flow").get_json()["xycal"]["waiting"] == k:
                time.sleep(0.05)
        elif not f["running"] and not f["moving"]:
            break
    assert len(hovers) == 6 and all(0.5 < h < 6 for h in hovers), hovers               # just above the page, never on it
    st = {s["id"]: s for s in f["stages"]}
    assert st["xycal"]["state"] == "warn" and set(f["xycal"]["hovered"]) == {"tl", "tr", "br", "bl", "cl"}
    g = robot.camera_geometry()                                                        # home: the photo's geometry
    shift = np.array([4.0, -6.0])                                                      # the arm landed 4, -6 mm off
    for k, hv in f["xycal"]["hovered"].items():
        p = np.array(hv["aim"]) + shift
        u, v = world_to_pixel(np.array([[*p, float(robot.surface(*p))]]), g.K, g.T_world_cam, g.dist)[0]
        assert post("/api/xycal/mark", key=k, u=float(u), v=float(v)).status_code == 200
    s = c.get("/api/flow").get_json()["xycal"]["summary"]
    assert s["n"] == 5 and s["new_offset_mm"] == [pytest.approx(4.0, abs=1.0), pytest.approx(-6.0, abs=1.0)], s
    assert post("/api/xycal/save").status_code == 200
    saved = yaml.safe_load((tmp / "local.yaml").read_text())["gui"]["calibration"]
    assert saved["offset_mm"] == s["new_offset_mm"] and (tmp / saved["saved_in"] / "marked.jpg").exists()
    q = post("/api/demo/plan").get_json()                                              # the demo now aims with it
    p = next(p for p in q["points"] if p["key"] == "tl")
    assert p["aim"] == [pytest.approx(p["xy"][0] - s["new_offset_mm"][0], abs=0.11), pytest.approx(p["xy"][1] - s["new_offset_mm"][1], abs=0.11)]


def test_book_model_merges_depth_and_touches_and_is_saved(sim):
    from bookworm.gui import book_model as bmod

    c, post, robot, tmp = _setup(sim)
    _zlink(c, post)
    m = _model(c, post)
    kinds = sorted(t["kind"] for t in m["touches"])
    assert kinds == ["page_check", "table"] and m["version"] == 1
    assert m["correction"]["kind"] in ("constant", "none") and set(m["pages"]) == {"left", "right"}
    disk = bmod.load(m["dir"])
    assert c.get("/book_model.json").get_json()["version"] == 1
    g = disk["grid"]
    X, Y = np.meshgrid(g["x"], g["y"], indexing="ij")
    ok = g["book"] & np.isfinite(g["z_arm"])
    err = np.abs(g["z_arm"][ok] - robot.surface(X[ok], Y[ok]))
    assert np.median(err) < 1.5                                                          # the file is the book in arm z
    assert disk["table_z"] == pytest.approx(robot.table_z, abs=1.5) and disk["tool_length_mm"] == 155


def test_the_arms_own_collision_stop_is_cleared_and_taken_as_the_contact(sim):
    c, post, robot, tmp = _setup(sim)
    robot.fault_on_block = True                                                    # collision_sensitivity > 0 on the real arm
    z, f = _zlink(c, post)
    assert z["data"]["table_found_by"] == "fault" and abs(z["data"]["arm_offset_mm"] - robot.table_z) < 3
    assert ("sim: error cleared",) in [tuple(e) for e in robot.log] and not robot._faulted

