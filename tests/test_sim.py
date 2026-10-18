"""The whole Setup flow offline on the simulated rig, and recorded views replaying."""
import json
import re
import time

import numpy as np
import pytest
import yaml

from bookworm.gui import server
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


def test_sim_tip_contact_and_load():
    r = SimRobot()
    x, y = 403.0, -17.0
    page = float(r.surface(x, y))
    r.move_to(dict(HOME, x=x, y=y, z=page + 5 + r.tool_length))
    free = r.joint_load()
    r.move_to(dict(HOME, x=x, y=y, z=page - 2 + r.tool_length))
    assert r.pressed_mm == pytest.approx(2, abs=0.01)
    assert r.joint_load()[1] - free[1] > 5                                         # pressing shows on joint 2


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
    robot = SimRobot()
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
    pts ={p["key"]: p["world"] for p in f["points"]}
    tl, bl = np.array(pts["tl"][:2]), np.array(pts["bl"][:2])
    m, half = (tl + bl) / 2, np.linalg.norm(bl - tl) / 2
    s = np.array(d["spot"], float)                       # (corner xy shift a little once the table z is known)
    assert 0.8 * half < np.linalg.norm(s - m) < 1.3 * half                                   # the right-angle apex off TL-BL
    assert abs(np.linalg.norm(s - tl) - np.linalg.norm(s - bl)) < 10
    assert np.linalg.norm(s - robot.centre) > np.linalg.norm(m - robot.centre)               # on the side away from the book
    assert f["z_link_spot_lock"] is None                                                   # follows the book, not locked
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


def test_demo_dry_run_is_book_points_only_at_book_height_plus_units(sim):
    c, post, robot, tmp = _setup(sim)
    q = post("/api/demo/plan").get_json()
    assert all("skip" in p for p in q["points"])                                      # no book model yet
    _zlink(c, post)
    m = _model(c, post)
    assert m["driven"] == "touch" and m["level"]["kind"] == "constant"
    q = post("/api/demo/plan").get_json()
    assert [p["key"] for p in q["points"]] == ["tl", "tr", "cr", "br", "bl", "cl"]      # no table point
    assert all(p["units"] == 3 and p["z"] == pytest.approx(p["page_z"] + 6, abs=0.01) for p in q["points"]), q
    q = post("/api/demo/plan", points=["bl", "cl"], units={"bl": 1}).get_json()
    assert [(p["key"], p["units"]) for p in q["points"]] == [("bl", 1), ("cl", 3)]
    assert q["points"][0]["z"] == pytest.approx(q["points"][0]["page_z"] + 2, abs=0.01)


def test_demo_light_touches_the_book_points_and_merges_early_contacts(sim):
    c, post, robot, tmp = _setup(sim)
    _zlink(c, post)
    _model(c, post)
    n0 = len(robot.log)
    moves, real = [], robot.move_to
    robot.move_to = lambda p: (moves.append(dict(p)), real(p))[1]
    assert post("/api/flow/stage", id="demo", confirm=True).status_code == 200
    st, f = _wait(c, 400)
    res = f["demo"]["results"]
    assert set(res) == {"tl", "tr", "cr", "br", "bl", "cl"} and all(r["verdict"] in ("on", "early") for r in res.values()), res
    assert not any(e[0] == "sim: tip stopped by the table" for e in robot.log[n0:])    # no table touch in the demo
    assert all(r["expected_z"] == pytest.approx(r["page_z"] + 6, abs=0.01) for r in res.values())
    assert res["bl"]["verdict"] == "on"                                                   # the page the model was built from
    early = [k for k, r in res.items() if r["verdict"] == "early"]
    assert early                                                                          # the right page is higher than BL
    presses = [e[1] for e in robot.log[n0:] if e[0] == "sim: tip pressed into the page"]
    assert not presses or max(presses) <= 3.0 + 1e-6                                      # backed off within a step
    m = f["model"]                                                                        # early contacts were measurements
    assert m["version"] == 2 and m["dir"].endswith("v2") and m["driven"] == "touch"
    assert sorted(t["key"] for t in m["touches"] if t["kind"] == "demo") == sorted(early)
    assert max(abs(robot.end_pose()[k] - HOME[k]) for k in "xyz") < 1                       # finished at home
    # every sideways move (x or y changes by more than 5 mm) happens at half the home height or above
    side = [(a, b) for a, b in zip(moves, moves[1:]) if np.hypot(b["x"] - a["x"], b["y"] - a["y"]) > 5]
    assert side and all(min(a["z"], b["z"]) >= HOME["z"] / 2 - 1 for a, b in side), [(a["z"], b["z"]) for a, b in side]


def test_touch_calibration_goes_straight_to_the_points_at_the_zlink_height_plus_units(sim):
    from bookworm.vision.geometry import world_to_pixel

    c, post, robot, tmp = _setup(sim)
    assert post("/api/flow/stage", id="xycal", confirm=True, options={"points": ["bl"]}).status_code == 200
    _wait(c)
    f = c.get("/api/flow").get_json()                                                  # needs the Z link first
    assert {s["id"]: s for s in f["stages"]}["xycal"]["state"] == "fail"
    z, f = _zlink(c, post)
    page_z = yaml.safe_load((tmp / "local.yaml").read_text())["gui"]["depth"]["z_link_page_tip_z"]

    real, refused = robot.move_to, []

    def move_to(p):                                                                    # the planner refuses one big lift
        pr = (c.get("/api/flow").get_json().get("progress") or {}).get("text", "")
        if pr.startswith("back to home") and p["z"] - robot.end_pose()["z"] > 40:       # only the lift after the last point
            refused.append(p["z"])
            raise RuntimeError("all IK solutions failed constraints. Failures: collision with obstacle")
        return real(p)

    robot.move_to = move_to
    n0 = len(robot.log)
    units = {"bl": 1, "cl": 2, "tr": 0}
    assert post("/api/flow/stage", id="xycal", confirm=True, options={"points": ["tr", "cl", "bl"], "units": units}).status_code == 200
    st, f = _wait(c, 400)
    res = f["xycal"]["results"]
    assert st["xycal"]["state"] == "warn", st["xycal"]["msg"]
    assert list(res) == ["bl", "cl", "tr"]                                             # BL first, no table touch, no 50 mm ladder
    assert not any(e[0] == "sim: tip stopped by the table" for e in robot.log[n0:])
    for k, u in units.items():
        r = res[k]
        assert r["book_z"] == pytest.approx(page_z, abs=0.01) or r["early_mm"] is not None
        assert r["touch_z"] == pytest.approx(r["book_z"] + 2 * u, abs=0.01)             # Z link height + units * 2 mm
    assert refused and max(abs(robot.end_pose()[k] - HOME[k]) for k in "xyz") < 1       # refused lift -> hops -> home
    assert set(f["xycal"]["hovered"]) == {"bl", "cl", "tr"} and f["xycal"]["has_photo"]
    assert c.get("/xycal/photo.jpg").status_code == 200 and post("/api/xycal/photo").status_code == 200
    g = robot.camera_geometry()                                                        # home: the photo's geometry
    shift = np.array([4.0, -6.0])                                                      # the arm landed 4, -6 mm off
    for k, hv in f["xycal"]["hovered"].items():
        p = np.array(hv["aim"]) + shift
        u, v = world_to_pixel(np.array([[*p, hv["hover_z"]]]), g.K, g.T_world_cam, g.dist)[0]
        assert post("/api/xycal/mark", key=k, u=float(u), v=float(v)).status_code == 200
    s = c.get("/api/flow").get_json()["xycal"]["summary"]
    assert s["n"] == 3 and s["new_offset_mm"] == [pytest.approx(4.0, abs=1.0), pytest.approx(-6.0, abs=1.0)], s
    assert post("/api/xycal/save").status_code == 200
    saved = yaml.safe_load((tmp / "local.yaml").read_text())["gui"]["calibration"]
    assert saved["offset_mm"] == s["new_offset_mm"] and (tmp / saved["saved_in"] / "marked.jpg").exists()


def test_book_model_is_driven_by_the_touches_with_depth_as_a_reference_and_is_saved(sim):
    from bookworm.gui import book_model as bmod

    c, post, robot, tmp = _setup(sim)
    _zlink(c, post)
    page_z = yaml.safe_load((tmp / "local.yaml").read_text())["gui"]["depth"]["z_link_page_tip_z"]
    m = _model(c, post)
    assert sorted(t["kind"] for t in m["touches"]) == ["page_check", "table"] and m["version"] == 1
    assert m["driven"] == "touch" and m["level"]["a"] == pytest.approx(page_z, abs=0.01) and set(m["pages"]) == {"left", "right"}
    bl = next(t for t in m["touches"] if t["kind"] == "page_check")
    assert bl["depth_z"] is not None and bl["depth_off"] is not None                    # the depth map is kept as a reference
    disk = bmod.load(m["dir"])
    assert c.get("/book_model.json").get_json()["version"] == 1
    assert bmod.z_at(disk, bl["x"], bl["y"]) == pytest.approx(page_z, abs=0.01)         # the file holds the touched height...
    assert np.ptp(disk["grid"]["z_arm"][disk["grid"]["book"]]) < 0.01                    # ...flat over the book, not the depth map
    assert disk["table_z"] == pytest.approx(robot.table_z, abs=1.5) and disk["tool_length_mm"] == 155


def test_the_arms_own_collision_stop_is_cleared_and_taken_as_the_contact(sim):
    c, post, robot, tmp = _setup(sim)
    robot.fault_on_block = True                                                    # collision_sensitivity > 0 on the real arm
    z, f = _zlink(c, post)
    assert z["data"]["table_found_by"] == "fault" and abs(z["data"]["arm_offset_mm"] - robot.table_z) < 3
    assert ("sim: error cleared",) in [tuple(e) for e in robot.log] and not robot._faulted

