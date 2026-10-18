"""Bookworm dev GUI: a guided setup that finds the book in the arm's own coordinates, then a touch demo.

    python -m bookworm.gui              # the real arm (credentials from .env)
    python -m bookworm.gui --sim        # offline: simulated arm and depth camera
    python -m bookworm.gui --replay DIR # offline: depth views recorded on the rig

Setup, in order (each stage can also be run on its own):
  1 Home & level   go home and fit the table plane in the depth image (the depth camera's tilt is removed in software)
  2 First capture  one photo from home; the book's outline gives six points (4 corners, both page centres)
  3 Depth scan     the page surface's height above the table, fused from views taken around home at home height
  4 Z link         lower the tip onto bare table (then onto the page, as a check) until the joint load jumps: the table's
                   z in the arm frame, which turns depth heights into arm z
  5 Touch demo     touch every point at exactly the height the depth map says, back to back

The one setting a person may change is the tool length (tool tip below the flange; e.g. when a pencil is fitted).
Everything else is measured. Every state-changing request needs the per-run token embedded in the page, so other web
pages open in the browser can't drive the arm through localhost.
"""
from __future__ import annotations

import collections
import json
import re
import secrets
import threading
import time
from pathlib import Path

import cv2
import yaml
from flask import Flask, Response, jsonify, request

from ..config import _merge

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LOCAL_CFG = ROOT / "config" / "local.yaml"

# Flange limits (world mm) used when config has no gui.limits: walls at x=740 and y=+-500, ceiling z=1050 in the Arm Farm.
DEFAULT_LIMITS = {"x": [-400, 650], "y": [-450, 450], "z": [120, 600]}
JOG_AXES = ("x", "y", "z", "theta")
MAX_STEP = {"x": 50.0, "y": 50.0, "z": 50.0, "theta": 15.0}

# Fixed motion choices (not settings): the touch ladders and the demo.
STEP_MM = 2.0              # ladder step
DEMO_FROM_MM = 16.0        # the demo ladder starts this far above the expected page (8 steps of trend before it)
TRAVEL_MM = 30.0           # travel between demo points this far above the highest of them
CORNER_INSET_MM = 15.0     # demo corners are touched this far in from the corner (the depth map is blank at the edge)
DEMO_ORDER = ["tl", "tr", "cr", "br", "bl", "cl"]            # book points only (no table touch in the demo)
PAGE_SEARCH_MM = 60.0       # the Z link's page touch searches from this far above the table (a thick book) down, by the load alone
DEMO_UNITS = 3             # default units (2 mm steps) above the book for a demo touch: 3 light-touched perfectly in the XY test
MARK_ORDER = ["tl", "tr", "br", "bl", "cl", "cr"]          # the order the touched points are listed in for marking


class State:
    def __init__(self, robot, cfg: dict):
        self.robot = robot
        self.cfg = cfg
        self.lock = threading.Lock()
        self.frame = None
        self.frame_t = 0.0
        self.fetch_ms = None
        self.fps = 0.0
        self.live = True
        self.estop = False
        self.moving = False          # an arm move is in flight; new arm moves are refused
        self.pose = None
        self.gripper = None
        self.log = collections.deque(maxlen=300)

    def say(self, msg: str) -> None:
        self.log.append(f"{time.strftime('%H:%M:%S')}  {msg}")

    @property
    def limits(self) -> dict:
        return {**DEFAULT_LIMITS, **((self.cfg.get("gui") or {}).get("limits") or {})}


def code_build() -> int:
    """Newest modification time of the Python code, at startup and on disk: the page warns when they differ."""
    return int(max(f.stat().st_mtime for f in HERE.parent.rglob("*.py")))


class StageFail(Exception):
    pass


class PlannerRefused(StageFail):
    """The motion planner found no way to make a move (nothing moved)."""


def create_app(robot, cfg: dict | None = None) -> Flask:
    import numpy as np

    from ..grasp.probe import flange_for_tip, tip_of
    from ..grasp.step_drop import StepDrop
    from . import book_model as bmod
    from .book import KEYS, LABEL, Book, auto_targets

    DEMO_LABEL = dict(LABEL)

    started_build = code_build()
    robot_mode = "sim" if getattr(robot, "sim", False) else "replay" if getattr(robot, "replay", False) else "real"
    RAW_DIR = ROOT / "data" / "depth" / "raw" / time.strftime("%Y%m%d_%H%M%S")      # this session's recorded views
    st = State(robot, cfg or {})
    if hasattr(robot, "_c"):
        robot.say = st.say                                  # connection retries show up in the log
    token = secrets.token_urlsafe(16)
    app = Flask(__name__)

    # ---- background workers ------------------------------------------------------------------------------
    def camera_loop():
        n, t0 = 0, time.time()
        while True:
            if not st.live:
                time.sleep(0.1)
                continue
            try:
                t = time.time()
                frame = robot.capture()
                with st.lock:
                    st.frame, st.frame_t, st.fetch_ms = frame, time.time(), round((time.time() - t) * 1000)
                n += 1
                time.sleep(0.02)
                if time.time() - t0 >= 2:
                    st.fps, n, t0 = n / (time.time() - t0), 0, time.time()
            except Exception as e:  # noqa: BLE001 - keep streaming; show the problem
                st.say(f"camera error: {e}")
                time.sleep(2)

    def state_loop():
        while True:
            try:
                st.pose = {k: round(v, 2) for k, v in robot.end_pose().items()}
                st.gripper = robot.gripper_get()
            except Exception as e:  # noqa: BLE001
                st.say(f"state error: {e}")
            time.sleep(1)

    lv = {"geom": None, "pose": None, "wanted": 0.0, "err_t": 0.0}

    def geom_loop():
        """Keep the camera pose fresh while someone watches the Setup camera view (points are redrawn from it)."""
        while True:
            if time.time() - lv["wanted"] > 3:
                time.sleep(0.3)
                continue
            try:
                lv["pose"] = robot.end_pose()
                lv["geom"] = robot.camera_geometry()
            except Exception as e:  # noqa: BLE001
                if time.time() - lv["err_t"] > 10:
                    st.say(f"live overlay: {e}")
                    lv["err_t"] = time.time()
                time.sleep(1)
            time.sleep(0.05)

    for fn in (camera_loop, state_loop, geom_loop):
        threading.Thread(target=fn, daemon=True).start()

    # ---- config ----------------------------------------------------------------------------------------------
    def gui_cfg() -> dict:
        return st.cfg.get("gui") or {}

    def depth_cfg() -> dict:
        return gui_cfg().get("depth") or {}

    def save_local(update: dict) -> None:
        """Merge ``update`` into config/local.yaml and the running config."""
        local = yaml.safe_load(LOCAL_CFG.read_text(encoding="utf-8")) if LOCAL_CFG.exists() else {}
        LOCAL_CFG.write_text(yaml.safe_dump(_merge(local or {}, update), sort_keys=False), encoding="utf-8")
        st.cfg.update(_merge(st.cfg, update))

    def save_depth_cfg(rec: dict) -> None:
        save_local({"gui": {"depth": rec}})

    def tool_len() -> float:
        """The one manual setting: how far the tool tip is below the flange (mm)."""
        return float((gui_cfg().get("probe") or {}).get("tool_length_mm", 155))

    def speeds() -> tuple[float, float]:
        p = gui_cfg().get("probe") or {}
        return float(p.get("slow_deg_s", 20)), float(p.get("restore_deg_s", 60))

    def xy_offset() -> list | None:
        """XY correction measured with the touch calibration of 2026-10-16 (where the arm lands vs where vision aims)."""
        o = (gui_cfg().get("calibration") or {}).get("offset_mm")
        return [float(o[0]), float(o[1])] if o else None

    def table_z() -> float:
        return float(depth_cfg().get("arm_offset_mm", 0.0))

    def table_measured() -> bool:
        return bool(depth_cfg().get("table_height_fitted"))

    def tip_floor() -> float:
        """Lowest z the tool tip may go to: the measured table + 1 mm, or the configured fallback before the z link."""
        if table_measured():
            return table_z() + 1.0
        return float(gui_cfg().get("tip_min_z_mm", 25))

    def in_limits(p: dict, floor: float | None = None) -> str | None:
        for axis, (lo, hi) in st.limits.items():
            if not lo <= p[axis] <= hi:
                return f"{axis} = {p[axis]:.0f} is outside the allowed {lo}..{hi} mm"
        tip_z = float(tip_of(p, tool_len())[2])                # the tool TIP must stay above the table
        floor = tip_floor() if floor is None else floor
        if tip_z < floor:
            return f"the tool tip would be at z = {tip_z:.0f} mm, below the {floor:g} mm floor"
        return None

    # ---- guards ----------------------------------------------------------------------------------------------
    @app.before_request
    def check_token():
        if request.method == "POST" and request.headers.get("X-Token") != token:
            return jsonify(error="bad token"), 403

    def blocked():
        if st.estop:
            return jsonify(error="STOPPED - press Reset to re-enable motion"), 409
        return None

    def arm_blocked():
        if (r := blocked()) is not None:
            return r
        if st.moving:
            return jsonify(error="the arm is still moving"), 409
        return None

    def run_move(target: dict, what: str) -> None:
        """One move in a worker thread so the UI (and STOP) stay responsive."""
        def work():
            try:
                if st.estop:
                    return
                st.say(f"moving: {what}")
                robot.move_to(target)
                st.say(f"arrived: {what}")
            except Exception as e:  # noqa: BLE001
                st.say(f"move failed ({what}): {e}")
            finally:
                st.moving = False

        st.moving = True
        threading.Thread(target=work, daemon=True).start()

    # ---- home ------------------------------------------------------------------------------------------------
    def home_pose():
        return (st.cfg.get("poses") or {}).get("home")

    def orientation_ok(now: dict, ref: dict) -> bool:
        """Same tool direction and wrist angle as the reference (within ~1 deg): nothing is twisted."""
        a = np.array([now["o_x"], now["o_y"], now["o_z"]], float)
        b = np.array([ref["o_x"], ref["o_y"], ref["o_z"]], float)
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        return cos > 0.99985 and abs(now["theta"] - ref["theta"]) < 2.0

    def at_home(pose: dict | None = None) -> bool:
        hp = home_pose()
        now = pose or robot.end_pose()
        if not hp or not now:
            return False
        return max(abs(now[k] - hp[k]) for k in ("x", "y", "z")) <= 3.0 and orientation_ok(now, hp)

    def slow_block(fn):
        slow, restore = speeds()
        robot.set_speed(slow)                                   # raises if unsupported -> nothing moves
        try:
            return fn()
        finally:
            try:
                robot.set_speed(restore)
            except Exception as e:  # noqa: BLE001
                st.say(f"could not restore arm speed ({e}); reconfigure the machine to reset it")

    def go_home_blocking():
        joints = st.cfg.get("home_joints")
        if not joints or not home_pose():
            raise StageFail("no home saved in config/local.yaml (home_joints + poses.home)")
        st.say("going home (slow joint move)")
        slow_block(lambda: robot.move_joints(joints))
        time.sleep(0.6)
        if not at_home():
            raise StageFail("the arm did not reach the home pose")

    def lift_then_home():
        """From low over the book: straight up to home height first (a joint move from low can swing the tip down), then home."""
        hp = home_pose()
        now = robot.end_pose()
        # up to half the home height only (the planner accepts that); home itself is a joint move, which needs no planning
        if hp and now["z"] < hp["z"] / 2.0 - 5:
            target = float(hp["z"]) / 2.0
            lifted = False
            for hop in (None, 40.0, 20.0):                         # one move; if the planner refuses, hops of 40, then 20 mm
                try:
                    while robot.end_pose()["z"] < target - 1.0:
                        z = target if hop is None else min(target, robot.end_pose()["z"] + hop)
                        up = dict(robot.end_pose(), z=z)
                        if in_limits(up) is not None:
                            break
                        slow_block(lambda: guarded_move(up, "lift before home"))
                    lifted = robot.end_pose()["z"] >= target - 1.0
                    break
                except PlannerRefused:
                    st.say("lift before home: refused" + (", trying shorter hops" if hop != 20.0 else ""))
            if not lifted:
                st.say("lift before home: could not lift in a straight line - going home with a slow joint move from here")
        go_home_blocking()

    def ensure_home():
        if not at_home():
            go_home_blocking()
        time.sleep(0.4)

    # ---- the book and its depth map --------------------------------------------------------------------------
    book = Book(fallback_z=float(gui_cfg().get("page_plane_z_mm", 35.0)))
    dsc = {"views": [], "res": None, "taken": None, "mode": None, "skipped": []}

    def surface_fn():
        """(world x, y) -> page surface z in the ARM frame (depth height above the table + the table's z), from the
        fused depth map (median of the 3x3 cells around the point); None where the map has no reading."""
        res = dsc["res"]
        if res is None:
            return None
        xs, ys, z = res["x"], res["y"], res["z"]
        off, cell = table_z(), float(xs[1] - xs[0])

        def fn(x, y):
            i, j = (x - xs[0]) / cell, (y - ys[0]) / cell
            if not (0 <= i <= len(xs) - 1 and 0 <= j <= len(ys) - 1):
                return None
            ii, jj = int(round(i)), int(round(j))
            patch = z[max(ii - 1, 0):ii + 2, max(jj - 1, 0):jj + 2]
            patch = patch[np.isfinite(patch)]
            return None if patch.size < 3 else float(np.median(patch)) + off

        return fn

    bm = {"model": None, "dir": None, "stamp": None}                            # the book model (stage 5), tied to this scan and this z link

    def apply_height():
        m = bm["model"]
        book.height_fn = surface_fn() if m is None else (lambda x, y: bmod.z_at(m, x, y))

    def drop_model():
        bm["model"], bm["dir"] = None, None
        apply_height()

    def table_h_at(x, y):
        """Depth height above the table (offset-free) at a world point: median of the 5x5 cells around it, or None."""
        res = dsc["res"]
        if res is None:
            return None
        xs, ys, z = res["x"], res["y"], res["z"]
        cell = float(xs[1] - xs[0])
        i, j = int(round((x - xs[0]) / cell)), int(round((y - ys[0]) / cell))
        if not (0 <= i < len(xs) and 0 <= j < len(ys)):
            return None
        p = z[max(i - 2, 0):i + 3, max(j - 2, 0):j + 3]
        p = p[np.isfinite(p)]
        return float(np.median(p)) if p.size >= 5 else None

    def bare_table_spots(min_away: float, far: bool = False):
        """Candidate (x, y, height above table) spots of bare table from the depth map: flat, seen, ``min_away`` mm from
        the book, 12 mm from anything that is not flat table and 15 mm in from the map's edge. Nearest to the book first,
        or with ``far`` the farthest first (out at the edge of what the camera sees)."""
        res = dsc["res"]
        xs, ys, z, bk = res["x"], res["y"], res["z"], res["book"].astype(bool)
        cell = float(xs[1] - xs[0])
        flat = np.isfinite(z) & (np.abs(z) < 2.5) & ~bk
        clear = cv2.distanceTransform(flat.astype(np.uint8), cv2.DIST_L2, 3) * cell
        away = cv2.distanceTransform((~bk).astype(np.uint8), cv2.DIST_L2, 3) * cell
        edge = int(np.ceil(15.0 / cell))
        inner = np.zeros_like(flat)
        inner[edge:-edge, edge:-edge] = True
        cand = np.argwhere((clear >= 12) & (away >= min_away) & inner)   # far enough that an open finger can't reach the book
        if not len(cand):
            return []
        bi, bj = np.nonzero(bk)
        c = np.array([xs[int(bi.mean())], ys[int(bj.mean())]])
        dist = np.array([np.hypot(xs[i] - c[0], ys[j] - c[1]) for i, j in cand])
        order = np.argsort(-dist if far else dist)
        return [(float(xs[i]), float(ys[j]), float(np.nanmedian(z[max(i - 2, 0):i + 3, max(j - 2, 0):j + 3])))
                for i, j in cand[order[:400:8]]]

    def fit_home_level(frame, geom):
        """At home the camera sees the whole book and plenty of bare table. The table plane fitted in depth there is
        what every later depth view is levelled with."""
        from ..vision.depth_scan import table_plane, tilt_deg

        try:
            depth = robot.capture_depth()
            a, b, c, n = table_plane(depth, frame, geom)
        except Exception as e:  # noqa: BLE001
            st.say(f"home level: no table plane ({e})")
            return None
        rec = {"table_slope": [round(b, 5), round(c, 5)], "table_tilt_deg": round(tilt_deg(b, c), 2), "wood_px": n,
               "fitted": time.strftime("%Y-%m-%d %H:%M:%S")}
        st.say(f"home level: table plane tilts {rec['table_tilt_deg']} deg in the depth data ({n} table pixels)")
        return rec                                                   # saved by the stage only if it passes

    def grab_view():
        frame = robot.capture()
        depth = robot.capture_depth()
        geom = robot.camera_geometry()
        if geom is None:
            raise RuntimeError("no camera geometry")
        if robot_mode == "real" and depth_cfg().get("record_views", True):
            try:                                                   # raw copy for offline replay (--replay)
                from ..hardware.replay import save_view

                save_view(RAW_DIR, frame, depth, geom, robot.end_pose())
            except Exception as e:  # noqa: BLE001
                st.say(f"could not record the depth view: {e}")
        return depth, frame, geom

    def build_scan(mode: str):
        from ..vision.depth_render import render_surface, render_top
        from ..vision.depth_scan import analyze, fuse

        F = fuse(dsc["views"], slope=depth_cfg().get("table_slope"))
        res = analyze(F, table_z())
        dsc["res"], dsc["taken"], dsc["mode"] = res, time.strftime("%Y-%m-%d %H:%M:%S"), mode
        drop_model()                                               # a new scan: the old model is for the old map
        if res is None:
            st.say("depth scan: no raised book region found")
            return
        s_ = res["stats"]
        st.say(f"depth scan: book {s_['book_w_mm']} x {s_['book_h_mm']} mm, typical height {s_['p50']} mm, table {s_['table']} mm "
               f"({len(dsc['views'])} view(s))")
        try:                                                       # every scan is kept
            d = ROOT / "data" / "depth" / time.strftime("%Y%m%d_%H%M%S")
            d.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(d / "scan.npz", x=res["x"], y=res["y"], z=res["z"], book=res["book"])
            (d / "scan.json").write_text(json.dumps({"stats": s_, "points": res["points"], "taken": dsc["taken"], "mode": mode,
                                                     "views": len(dsc["views"])}, indent=2), encoding="utf-8")
            cv2.imwrite(str(d / "top.png"), render_top(res["x"], res["y"], res["z"], res["book"], -2, 62, res["points"]))
            cv2.imwrite(str(d / "surface.png"), render_surface(res["x"], res["y"], res["z"], -2, 62, book=res["book"]))
        except Exception as e:  # noqa: BLE001
            st.say(f"depth scan: could not save ({e})")

    def scan_pass_blocking():
        """The camera visits spots around the BOOK's midpoint (from the first capture; home if there is none yet) at home
        height (never lower), a depth view at each; back home; fuse."""
        hp, joints = home_pose(), st.cfg.get("home_joints")
        offsets = depth_cfg().get("pass_offsets_mm") or [[0, 0], [70, 0], [-70, 0], [0, 80], [0, -80]]
        cx, cy = 0.0, 0.0                                             # flange shift that puts the camera's centre over the book
        mid, g = book.centre(), robot.camera_geometry()
        if mid is not None and g is not None:
            seen = book.pixel_world([float(g.K[0, 2]), float(g.K[1, 2])], g)[:2]   # what the image centre looks at now
            cx, cy = float(mid[0] - seen[0]), float(mid[1] - seen[1])
            if np.hypot(cx, cy) > 200:
                st.say(f"depth scan: the book midpoint is {np.hypot(cx, cy):.0f} mm from the view centre - scanning around home")
                cx, cy = 0.0, 0.0
            else:
                st.say(f"depth scan: views centred on the book midpoint ({mid[0]:.0f}, {mid[1]:.0f}): shift {cx:+.0f}, {cy:+.0f} mm")
        views = []
        try:
            for dx, dy in offsets:
                if st.estop:
                    st.say("depth scan: stopped")
                    break
                dx, dy = dx + cx, dy + cy
                if abs(dx) > 0.5 or abs(dy) > 0.5:
                    tgt = {kk: float(hp[kk]) for kk in ("x", "y", "z", "o_x", "o_y", "o_z", "theta")}
                    tgt["x"] += dx
                    tgt["y"] += dy
                    if (why := in_limits(tgt)) is not None:
                        dsc["skipped"].append(f"({dx:.0f}, {dy:.0f}): {why}")
                        continue
                    try:
                        robot.move_to(tgt)
                    except Exception as e:  # noqa: BLE001
                        if "IK solutions" in str(e) or "constraint" in str(e):    # the planner said no: skip this view
                            dsc["skipped"].append(f"({dx:.0f}, {dy:.0f}): planner refused")
                            st.say(f"depth scan: view ({dx:.0f}, {dy:.0f}) refused by the planner, skipped")
                            continue
                        raise
                    time.sleep(0.8)
                views.append(grab_view())
                st.say(f"depth scan: view {len(views)} of {len(offsets)}")
                set_progress("depth", 0.1 + 0.7 * len(views) / len(offsets), f"depth view {len(views)} of {len(offsets)}")
        finally:
            robot.move_joints(joints)
        if views:
            dsc["views"] = views
            build_scan("camera pass")

    # ---- the touch ladder: shared by the z link and the demo ------------------------------------------------
    contact = {"ev": threading.Event(), "step": None}

    def make_step_drop():
        c = depth_cfg().get("step_drop") if isinstance(depth_cfg().get("step_drop"), dict) else {}
        return StepDrop(enabled=bool(c.get("enabled", True)), step_mm=STEP_MM, min_drop_mm=float(c.get("min_drop_mm", 1.0)),
                        mode=c.get("mode", "regression"), k=float(c.get("k", 5.0)), pos_floor_mm=float(c.get("pos_floor_mm", 0.3)),
                        load_k=float(c.get("load_k", 8.0)), use_load=bool(c.get("use_load", True)),
                        load_joints=c.get("load_joints", [1, 2]), load_floor=float(c.get("load_floor", 0.3)))

    def read_load(sd, missing: list):
        if not sd.use_load or missing:
            return None
        try:
            return robot.joint_load()
        except Exception as e:  # noqa: BLE001
            missing.append(1)
            st.say(f"touch: no joint load reading ({e}); position only")
            return None

    def is_collision(e) -> bool:
        """The xArm's own collision stop (collision_sensitivity > 0): 'collision caused overcurrent'."""
        m = str(e).lower()
        return "collision" in m and "overcurrent" in m

    def is_refusal(e) -> bool:
        """The Viam motion planner refusing a move (no IK solution inside its constraints). The arm did not move and
        nothing needs clearing; the same move will be refused again, so it is not retried."""
        m = str(e)
        return "IK solutions" in m or "constraint" in m.lower()

    def guarded_move(ps, what):
        """A move outside a touch step. The arm's collision stop: clear it and try once more. The planner refusing:
        raise PlannerRefused at once (callers choose another route)."""
        for attempt in (1, 2):
            try:
                robot.move_to(ps)
                return
            except Exception as e:  # noqa: BLE001
                if is_collision(e):
                    robot.clear_error()
                    st.say(f"{what}: the arm's collision stop fired; error cleared" + (", trying once more" if attempt == 1 else ""))
                    time.sleep(0.5)
                elif is_refusal(e):
                    st.say(f"{what}: the motion planner refused this move: {str(e)[:500]}")
                    raise PlannerRefused(f"{what}: the motion planner refused this move") from e
                else:
                    raise
        raise StageFail(f"{what}: the arm's collision stop fired twice - the error is cleared and the arm left where it stopped")

    def ladder(key, label, x, y, zs, *, press=False, lo=0.0, hi=1.0, stage="zlink"):
        """Lower the tip over (x, y) through the tip heights ``zs`` (highest first), one step at a time, slowly.

        After each step the real pose and the joint load go to the step drop test. On contact the arm goes STRAIGHT
        back to the previous step; if the load there is still off its trend, it backs out to the start height and
        home. With ``press`` the person's 'It's touching' press also ends the search (at the step it was pressed).
        Returns {"tip_z", "how" (load|stall|fault|press|bottom), "step", "retreat" (previous|home|None)}."""
        L = tool_len()
        now = robot.end_pose()
        poses = [flange_for_tip([x, y, z], now, L) for z in zs]
        for z, ps in zip(zs, poses):
            if (why := in_limits(ps, zs[-1] - 0.5)) is not None:
                raise StageFail(f"{label}: tip z {z:.0f}: {why}")
        sd = make_step_drop()
        contact["ev"].clear()
        contact["step"] = None
        live.update(active=True, key=key, label=label, step=0, of=len(zs), tip_z=None, stall=None, press=press,
                    k=sd.k, load_k=sd.load_k)
        live["traces"][key] = sd.trace
        live["expected"][key] = round(float(zs[-1] if stage == "demo" else zs[0] - 22.0), 1)
        out = {"tip_z": None, "how": None, "step": None, "retreat": None}
        missing = []
        st.say(f"{label}: lowering over ({x:.0f}, {y:.0f}) from tip z {zs[0]:.1f} to {zs[-1]:.1f} mm in {STEP_MM:g} mm steps")

        def back_off(i):
            """Contact at step i: straight back to step i - 1; still loaded there -> out to the start height and home."""
            guarded_move(poses[max(i - 1, 0)], f"{label}: backing off")
            time.sleep(float(depth_cfg().get("height_settle_s", 0.4)))
            ld = read_load(sd, missing)
            score = None if ld is None else sd.load_score(ld)
            if score is not None and score > sd.load_k:
                st.say(f"{label}: the load is still {score:.0f} std devs off its trend after backing up - backing out to home")
                guarded_move(poses[0], f"{label}: backing out")
                hp = home_pose()
                if hp:
                    up = dict(robot.end_pose(), z=float(hp["z"]))
                    if in_limits(up) is None:
                        guarded_move(up, f"{label}: backing out")
                if st.cfg.get("home_joints"):
                    robot.move_joints(st.cfg["home_joints"])
                out["retreat"] = "home"
            else:
                out["retreat"] = "previous"

        def descend():
            try:
                for i, ps in enumerate(poses):
                    if st.estop:
                        raise StageFail("stopped")
                    if press and contact["ev"].is_set():
                        break
                    try:
                        robot.move_to(ps)
                    except Exception as e:  # noqa: BLE001
                        if not is_collision(e):
                            raise
                        robot.clear_error()                         # the arm's own collision stop (sensitivity > 0)
                        out.update(tip_z=float(tip_of(robot.end_pose(), L)[2]), how="fault", step=i)
                        live["stall"] = f"the arm's collision stop fired at step {i + 1}; error cleared"
                        st.say(f"{label}: collision stop at step {i + 1} ({str(e)[:70]}); error cleared")
                        back_off(i)
                        return
                    time.sleep(float(depth_cfg().get("height_settle_s", 0.4)))
                    stall = sd.check(float(tip_of(robot.end_pose(), L)[2]), commanded_tip_z=zs[i], load=read_load(sd, missing))
                    if stall is not None:
                        # contact happened between the previous step and this one; a stalled tip sits on the surface
                        at = stall.tip_z if stall.reason != "load" else (stall.tip_z + stall.previous_tip_z) / 2
                        out.update(tip_z=float(at), how="load" if stall.reason == "load" else "stall", step=i)
                        live["stall"] = stall.text()
                        st.say(f"{label}: {stall.text()} - back to the previous step")
                        back_off(i)
                        return
                    live.update(step=i + 1, tip_z=round(zs[i], 1))
                    set_progress(stage, lo + (hi - lo) * (i + 1) / len(zs), f"{label}: tip z {zs[i]:.1f} mm")
                    if press and contact["ev"].wait(float(depth_cfg().get("height_step_wait_s", 1.2))):
                        break
                else:
                    out.update(tip_z=float(zs[-1]), how="bottom", step=len(zs) - 1)
            finally:
                if out["retreat"] != "home":
                    guarded_move(poses[0], f"{label}: back up")    # back up to the start height

        try:
            slow_block(descend)
        finally:
            live["active"] = False
        if st.estop:
            raise StageFail("stopped")
        k = contact["step"]
        if out["how"] in (None, "bottom") and press and isinstance(k, int) and 1 <= k <= len(zs):
            out.update(tip_z=zs[k - 1], how="press", step=k - 1)
        return out

    # ---- setup flow ------------------------------------------------------------------------------------------
    STAGES = [("level", "Home & level"), ("capture", "First capture"), ("depth", "Depth scan"),
              ("zlink", "Z link"), ("model", "Book model"), ("xycal", "Touch calibration"), ("demo", "Touch demo")]
    DEPENDS = {"level": ["depth", "zlink", "model", "demo"], "capture": ["model", "xycal", "demo"], "depth": ["zlink", "model", "demo"],
               "zlink": ["model", "demo"], "model": ["demo"]}
    MANUAL = ("xycal",)                                            # needs a person at every point: never part of Run all
    flow = {"stages": {k: {"state": "idle", "msg": "", "t": None, "data": None} for k, _ in STAGES}, "running": None, "auto": False,
            "progress": None}
    live = {"active": False, "traces": {}, "expected": {}}
    demo = {"results": {}, "order": [], "taken": None}

    def set_progress(k, sub, text=""):
        pr = flow.get("progress")
        if pr is not None:
            pr.update(stage=k, sub=max(0.0, min(1.0, float(sub))), text=text)

    def set_stage(k, state, msg="", data=None):
        flow["stages"][k] = {"state": state, "msg": msg, "t": time.strftime("%H:%M:%S"), "data": data}
        st.say(f"[{k}] {state}" + (f": {msg}" if msg else ""))

    def mark_stale(ks):
        for k in ks:
            if flow["stages"][k]["state"] in ("ok", "warn"):
                flow["stages"][k]["state"] = "stale"

    def stage_level(o):
        set_progress("level", 0.1, "going home if needed")
        ensure_home()
        set_progress("level", 0.4, "reading the camera and the depth image")
        frame, geom = robot.capture(), robot.camera_geometry()
        if geom is None:
            raise StageFail("no camera geometry")
        set_progress("level", 0.7, "fitting the table plane")
        rec = fit_home_level(frame, geom)
        if rec is None:
            raise StageFail("no table plane from depth - is the depth camera streaming and the table in view?")
        b, c = rec["table_slope"]
        n = np.array([-b, -c, 1.0])
        n /= np.linalg.norm(n)
        axis = geom.T_world_cam[:3, 2] / np.linalg.norm(geom.T_world_cam[:3, 2])
        cam_deg = float(np.degrees(np.arccos(min(1.0, abs(float(n @ axis))))))
        pose = robot.end_pose()
        ov = np.array([pose["o_x"], pose["o_y"], pose["o_z"]], float)
        tool_deg = float(np.degrees(np.arccos(min(1.0, -float(ov[2]) / float(np.linalg.norm(ov))))))
        tilt = rec["table_tilt_deg"]
        data = {"table_tilt_deg": tilt, "camera_vs_table_deg": round(cam_deg, 2), "tool_off_vertical_deg": round(tool_deg, 2)}
        if tilt <= 8.0:                                              # a bad fit must not level every later depth view
            save_depth_cfg({**rec, "camera_vs_table_deg": data["camera_vs_table_deg"]})
        if tilt <= 1.0 and cam_deg <= 1.0:
            return "ok", f"home is parallel to the table (table {tilt} deg off, camera {cam_deg:.1f} deg off)", data
        if tilt <= 8.0:
            return "warn", (f"the table is {tilt} deg off the arm's vertical in the depth data and the camera looks {cam_deg:.1f} deg "
                            f"off the table normal (the tool is {tool_deg:.1f} deg off vertical). Depth is levelled in software."), data
        raise StageFail(f"the table is {tilt} deg off the arm's vertical - re-teach the home pose so the camera looks straight down")

    def stage_capture(o):
        set_progress("capture", 0.1, "going home if needed")
        ensure_home()
        set_progress("capture", 0.4, "taking the photo")
        frame, geom = robot.capture(), robot.camera_geometry()
        if geom is None:
            raise StageFail("no camera geometry")
        set_progress("capture", 0.8, "finding the book outline")
        found = auto_targets(frame)
        if found is None:
            raise StageFail("the book outline was not found in the photo - is the whole open book in view from home?")
        book.image, book.geom, book.home_pose = frame, geom, dict(robot.end_pose())
        book.taken_at = time.strftime("%Y-%m-%d %H:%M:%S")
        book.px = dict(found)
        demo["results"] = {}
        save_book_frame()
        return "ok", "book found: 6 points (4 corners and both page centres)", {"points": found}

    def stage_depth(o):
        if not depth_cfg().get("table_slope"):
            raise StageFail("run Home & level first - it measures the camera tilt that depth needs")
        set_progress("depth", 0.05, "going home if needed")
        ensure_home()
        dsc["skipped"] = []
        if o.get("scan", depth_cfg().get("scan_mode", "pass")) == "pass":
            slow_block(scan_pass_blocking)
        else:
            dsc["views"] = [grab_view()]
            build_scan("home view")
        res = dsc["res"]
        if res is None:
            raise StageFail("no raised book region found in the depth data")
        s_ = res["stats"]
        msg = f"book {s_['book_w_mm']} x {s_['book_h_mm']} mm, typical height {s_['p50']} mm above the table ({len(dsc['views'])} view(s))"
        if dsc["skipped"]:
            msg += f"; skipped {len(dsc['skipped'])} view(s)"
        return "ok", msg, {"stats": s_}

    def save_zlink_trace(stamp, extra):
        dd = ROOT / "data" / "depth" / "zlink"
        dd.mkdir(parents=True, exist_ok=True)
        rec = {"taken": stamp, "traces": {k: live["traces"].get(k) for k in ("table", "page")},
               "expected_tip_z": {k: live["expected"].get(k) for k in ("table", "page")}, **extra}
        (dd / f"{stamp}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")

    def stage_zlink(o):
        """Table touch -> the table's z in the arm frame; then a page touch at the left-page centre -> the book's height.
        Both are found by the joint load alone: no depth map needed, and the page search covers a whole book's thickness."""
        if "cl" not in book.px:
            raise StageFail("run the first capture first - the left-page centre (CL) is the page-check point")
        d, L = depth_cfg(), tool_len()
        if table_measured() and not o.get("force") and d.get("z_link_tool_length_mm") == L:
            set_progress("zlink", 1.0, "using the saved value")
            return "ok", (f"using the saved z link: the table is {table_z():g} mm in the arm frame (measured {d['table_height_fitted']} "
                          f"with the {L:g} mm tool)"), {"saved": True}
        ensure_home()
        wx, wy = book.demo_xy("cl", 0)                                # the left-page centre (the BL route was refused by the planner)
        lock = d.get("z_link_spot_lock")
        if lock:
            spot = (float(lock[0]), float(lock[1]))
        else:
            ap = apex_table_spot()
            if ap is None:
                raise StageFail("run the first capture first - the table point is worked out from the TL and BL corners")
            spot = (ap[0], ap[1])
        if dsc["res"] is not None and (h_t := table_h_at(spot[0], spot[1])) is not None and abs(h_t) > 3.0:
            raise StageFail(f"the table point ({spot[0]:.0f}, {spot[1]:.0f}) is not bare table in the depth map ({h_t:.1f} mm "
                            "above it) - clear it, or jog over a clear spot and press 'Lock the arm's spot'")
        expected = table_z() if table_measured() else float(gui_cfg().get("tip_min_z_mm", 25))
        below = float(d.get("z_link_search_below_mm", 10))
        top, bottom = expected + 22.0, expected - below
        if table_measured() and d.get("z_link_tool_length_mm") is not None:
            # the tool length changed: a new tool (the table stays at the same tip z) or a corrected number (the table
            # moves by the difference in tip z). Search across both.
            moved = expected + float(d["z_link_tool_length_mm"]) - L
            top, bottom = max(top, moved + 22.0), min(bottom, moved - below)
        ends = [flange_for_tip([spot[0], spot[1], z], robot.end_pose(), L) for z in (top, bottom - 0.5)]
        if not all(in_limits(ps, bottom - 1.0) is None for ps in ends):
            raise StageFail("the table spot is outside the arm's limits")
        live["traces"], live["expected"] = {}, {}
        stamp = time.strftime("%Y%m%d_%H%M%S")
        zs = [float(z) for z in np.arange(top, bottom - 1e-6, -STEP_MM)]
        travel_above(spot[0], spot[1], top)                         # across high, then straight down over the spot
        r = ladder("table", "bare table", spot[0], spot[1], zs, press=True, lo=0.0, hi=0.5)
        if r["tip_z"] is None or r["how"] == "bottom":
            save_zlink_trace(stamp, {"result": "no contact"})
            raise StageFail(f"no contact with the table down to tip z {zs[-1]:.0f} mm. If the table is lower than {expected:.0f} mm in the "
                            "arm frame, raise gui.depth.z_link_search_below_mm and run it again")
        new_off = round(float(r["tip_z"]), 1)
        save_depth_cfg({"arm_offset_mm": new_off, "table_height_fitted": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "z_link_table_tip_z": new_off, "z_link_tool_length_mm": L,
                        "z_link_spot": [round(spot[0], 1), round(spot[1], 1)],     # not locked: it follows the book
                        "z_link_page_tip_z": None, "z_link_page_xy": None, "z_link_page_at": None})
        drop_model()
        by = {"stall": "the tip stopped going down", "load": "the joint load jumped", "fault": "the arm's collision stop fired",
              "press": "you pressed"}[r["how"]]
        msg = f"the table is {new_off:g} mm in the arm frame (bare table at ({spot[0]:.0f}, {spot[1]:.0f}); {by})."
        out = {"arm_offset_mm": new_off, "table_tip_z": new_off, "table_found_by": r["how"], "spot": [round(spot[0]), round(spot[1])],
               "retreat": r["retreat"]}
        if r["retreat"] == "home":
            save_zlink_trace(stamp, out)
            return "warn", msg + " The load stayed high after backing up, so the arm went home; the page touch was not done.", out
        # the book: from PAGE_SEARCH_MM above the table (a thick book) down to 3 mm above it, by the load alone
        zp = [float(z) for z in np.arange(new_off + PAGE_SEARCH_MM, new_off + 3.0 - 1e-6, -STEP_MM)]
        travel_above(wx, wy, zp[0] + TRAVEL_MM)                      # up off the table, across above the book, down over CL
        rp = ladder("page", "left page centre (CL)", wx, wy, zp, press=True, lo=0.5, hi=1.0)
        if rp["retreat"] != "home":
            lift_then_home()                                         # never leave the tip sitting just above the book
        if rp["tip_z"] is None or rp["how"] == "bottom":
            save_zlink_trace(stamp, out)
            return "warn", (msg + f" Page touch: no contact between {zp[0]:.0f} and {zp[-1]:.0f} mm over the left-page centre - is "
                            "the book under that point? The book height is not measured."), out
        page_z = round(float(rp["tip_z"]), 1)
        out.update(page_tip_z=page_z, page_above_table=round(page_z - new_off, 1), page_found_by=rp["how"])
        save_depth_cfg({"z_link_page_tip_z": page_z, "z_link_page_xy": [round(wx, 1), round(wy, 1)],     # a page touch for the
                        "z_link_page_at": time.strftime("%Y-%m-%d %H:%M:%S")})                          # book model, for this book position
        save_book_frame()
        save_zlink_trace(stamp, out)
        return "ok", msg + f" The book at the left-page centre is at z {page_z:g} mm, {page_z - new_off:.1f} mm above the table.", out

    def model_touches():
        """The touches the z link made: the table (sets the table's z) and the left-page check (a page touch)."""
        d = depth_cfg()
        out = []
        if d.get("z_link_table_tip_z") is not None and d.get("z_link_spot"):
            sx, sy = d["z_link_spot"]
            out.append({"key": "table", "kind": "table", "x": float(sx), "y": float(sy), "z": float(d["z_link_table_tip_z"])})
        fresh = bool(d.get("z_link_page_at")) and bool(book.taken_at) and str(d["z_link_page_at"]) >= str(book.taken_at)
        if not fresh and d.get("z_link_page_tip_z") is not None:     # touched before this capture: the book may have moved
            st.say("book model: the saved page check is older than the first capture - not used (run the Z link to redo it)")
        if fresh and d.get("z_link_page_tip_z") is not None and d.get("z_link_page_xy"):
            px, py = d["z_link_page_xy"]
            out.append({"key": "cl", "kind": "page_check", "x": float(px), "y": float(py), "z": float(d["z_link_page_tip_z"])})
        tc = d.get("touchcal") or {}
        if tc.get("at") and book.taken_at and str(tc["at"]) >= str(book.taken_at):   # touch calibration on this book position:
            for k, p in (tc.get("points") or {}).items():                            # only touches that MET the page (higher
                if p.get("contact_z") is not None:                                    # than the Z link height) are measurements
                    out.append({"key": k, "kind": "page_check", "x": float(p["xy"][0]), "y": float(p["xy"][1]),
                                "z": float(p["contact_z"])})
        return out

    def make_model(touches, version, stamp=None):
        pts = {k: [round(float(v), 1) for v in book.world3(k)[:2]] for k in KEYS if book.world3(k) is not None}
        m = bmod.build(dsc["res"], table_z(), touches, points=pts, meta={
            "version": version, "tool_length_mm": tool_len(), "xy_offset_mm": xy_offset(), "depth_scan_taken": dsc["taken"],
            "z_link_measured": depth_cfg().get("table_height_fitted"), "capture_taken": book.taken_at})
        stamp = stamp or time.strftime("%Y%m%d_%H%M%S")              # v1 in data/book/<stamp>/, later versions in .../v2, v3
        folder = ROOT / "data" / "book" / stamp
        folder = folder if version == 1 else folder / f"v{version}"
        bmod.save(m, folder)
        bm["model"], bm["dir"], bm["stamp"] = m, str(folder), stamp
        apply_height()
        return m

    def stage_model(o):
        """No motion: depth map + z link + every page touch so far -> the book's surface in arm z, saved to data/book/."""
        if not book.px:
            raise StageFail("run the first capture first")
        if dsc["res"] is None:
            raise StageFail("run the depth scan first")
        if not table_measured():
            raise StageFail("run the z link first - without it depth heights can't be turned into arm z")
        set_progress("model", 0.5, "building the model from the touches")
        m = make_model(model_touches(), 1)
        c = m["level"]
        if m["driven"] == "depth":
            return "warn", ("no page touch on this book yet (run the Z link after the first capture): the depth map is used as "
                            "measured, and it can be many mm off. Touch before relying on it."), {"model": bmod.summary(m)}
        offs = [t["depth_off"] for t in m["touches"] if t["kind"] in bmod.PAGE_KINDS and t.get("depth_off") is not None]
        msg = (f"driven by {c['n']} page touch(es): {'a plane' if c['kind'] == 'plane' else 'one level'} at {c['a']:.1f} mm"
               + (f" (touches {c['rms_after']:.1f} mm rms from it)" if c["n"] > 1 else ""))
        if offs:
            msg += f"; the depth map (reference only) was {sum(offs) / len(offs):+.1f} mm off at the touches"
        out = f"{msg}. Saved to {Path(bm['dir']).relative_to(ROOT).as_posix()}"
        bad = []                                                     # every point must sit on believable page
        for k in KEYS:
            xy = book.demo_xy(k, CORNER_INSET_MM)
            z = None if xy is None else bmod.z_at(m, xy[0], xy[1])
            if z is None or z < table_z() + 3.0:
                bad.append(f"{k} ({'outside the book outline' if z is None else f'{z:.1f}'})")
        if bad:
            return "warn", (out + f". CHECK: no book height at {', '.join(bad)} (table {table_z():g}): those points are outside "
                            "the book outline in the depth map - look at the top-down view"), \
                {"model": bmod.summary(m), "bad_points": bad}
        return "ok", out, {"model": bmod.summary(m)}

    def apex_table_spot():
        """The table touch point: off the book's TL-BL edge, where lines from TL and BL meet at a right angle (on the
        side away from the book): half the edge's length out from its middle."""
        a, b, c = book.world3("tl"), book.world3("bl"), book.centre()
        if a is None or b is None or c is None:
            return None
        a, b = np.array(a[:2], float), np.array(b[:2], float)
        m, e = (a + b) / 2, b - a
        n = np.array([-e[1], e[0]]) / np.linalg.norm(e)
        if np.dot(n, m - c) < 0:                                     # point away from the book
            n = -n
        p = m + n * np.linalg.norm(e) / 2
        return [round(float(p[0]), 1), round(float(p[1]), 1)]

    def travel_above(ax, ay, min_tip_z):
        """Between points: straight up to the travel height (half the home height, never lower than ``min_tip_z`` for the
        tip), across at that height, then straight down to it over (ax, ay) if coming from higher (home)."""
        L, now = tool_len(), robot.end_pose()
        hz = float(home_pose()["z"]) if home_pose() else now["z"]
        travel = max(hz / 2.0, min_tip_z + L)                       # flange z: half the home height, or higher if needed
        # go to exactly the travel height first (up from low, or straight down from home), then across, at that height
        level = dict(now, z=travel)
        across = flange_for_tip([ax, ay, travel - L], now, L)
        for ps in (level, across):
            if (why := in_limits(ps)) is not None:
                raise StageFail(f"travel: {why}")
        if abs(now["z"] - travel) > 1.0:
            slow_block(lambda: guarded_move(level, "travel to the travel height"))
        try:
            slow_block(lambda: guarded_move(across, "travel across"))
        except PlannerRefused:
            # The planner could not reach this pose with the usual wrist angle (its message names the wrist-vs-camera box and
            # the back wall). Turning the wrist about the vertical tool axis does not move the tip, but changes the arm's
            # bend: try a few wrist angles.
            for dth in (15.0, -15.0, 30.0, -30.0):
                try:
                    slow_block(lambda: guarded_move(dict(across, theta=across["theta"] + dth), "travel across"))
                    st.say(f"travel: reached the point with the wrist turned {dth:+.0f} deg")
                    return
                except PlannerRefused:
                    continue
            raise

    def demo_units(k, units=None):
        """Units (2 mm steps) above the book height for a demo touch: the person's choice, else what the touch calibration
        used for that point (it light-touched there), else DEMO_UNITS."""
        if units and k in units:
            return max(0, min(8, int(units[k])))
        tc = depth_cfg().get("touchcal") or {}
        if tc.get("at") and book.taken_at and str(tc["at"]) >= str(book.taken_at):
            u = ((tc.get("points") or {}).get(k) or {}).get("units")
            if u is not None:
                return int(u)
        return DEMO_UNITS

    def demo_plan(keys=None, units=None):
        """For each chosen book point: where the tip goes (x, y, after the XY correction) and the touch height: the book
        model's height there (driven by the touches) plus ``units`` x 2 mm. Nothing moves."""
        keys = [k for k in DEMO_ORDER if keys is None or k in keys]
        if bm["model"] is None:                                      # the demo touches where the book model puts the page
            return [{"key": k, "skip": "no book model yet (stage 5)"} for k in keys]
        off = xy_offset() or [0.0, 0.0]
        m = bm["model"]
        plan = []
        for k in keys:
            xy = book.demo_xy(k, CORNER_INSET_MM)
            if xy is None:
                plan.append({"key": k, "skip": "no point"})
                continue
            ax, ay = xy[0] - off[0], xy[1] - off[1]                   # aim so the arm LANDS on the point
            page = bmod.z_at(m, xy[0], xy[1])                         # the model is in the camera's coordinates, like the point
            if page is None:
                plan.append({"key": k, "skip": "outside the book outline"})
                continue
            u = demo_units(k, units)
            z = float(page) + u * STEP_MM
            if page < table_z() + 3.0 or z < tip_floor():             # a page at or under the table: the model is wrong there
                plan.append({"key": k, "skip": f"book height {page:.1f} is not believable (table {table_z():g})"})
                continue
            plan.append({"key": k, "xy": xy, "aim": [round(ax, 1), round(ay, 1)], "page_z": round(float(page), 1), "units": u,
                         "z": round(z, 1)})
        return plan

    def stage_demo(o):
        """Light-touch every chosen book point back to back, at the book height the touches gave plus ``units`` x 2 mm
        (no table touch). A touch that meets the page earlier (the load jumps first) backs off at once and is reported;
        that contact is a measured page height and goes into the model."""
        if not book.px:
            raise StageFail("run the first capture first")
        if bm["model"] is None:
            raise StageFail("build the book model first (stage 5)")
        keys = [k for k in (o.get("points") or DEMO_ORDER) if k in DEMO_ORDER]
        ensure_home()
        plan = demo_plan(keys, o.get("units"))
        todo = [p for p in plan if "skip" not in p]
        if not todo:
            raise StageFail("no chosen point has a believable book height")
        L = tool_len()
        min_tip = max(p["z"] for p in todo) + TRAVEL_MM
        demo["results"] = {p["key"]: {"verdict": "skip", "why": p["skip"]} for p in plan if "skip" in p}
        demo["order"], demo["taken"] = [p["key"] for p in plan], time.strftime("%Y-%m-%d %H:%M:%S")
        live["traces"], live["expected"] = {}, {}

        for n_, p in enumerate(todo):
            if st.estop:
                raise StageFail("stopped")
            k = p["key"]
            set_stage("demo", "running", f"touching {LABEL[k]} ({n_ + 1} of {len(todo)})")
            zs = [float(z) for z in np.arange(p["z"] + DEMO_FROM_MM, p["z"] - 1e-6, -STEP_MM)]
            try:
                travel_above(*p["aim"], min_tip)
                r = ladder(k, LABEL[k], p["aim"][0], p["aim"][1], zs, stage="demo",
                           lo=n_ / len(todo), hi=(n_ + 1) / len(todo))
            except StageFail as e:
                if "stopped" in str(e):
                    raise
                demo["results"][k] = {"verdict": "skip", "why": str(e), "expected_z": p["z"], "aim": p["aim"]}
                st.say(f"demo: {k} skipped - {e}")
                continue
            # contact within one step of the touch height is the touch we wanted; higher than that is early
            early = None if r["how"] == "bottom" else round(r["tip_z"] - p["z"], 1)
            verdict = "on" if early is None or early <= STEP_MM else "early"
            demo["results"][k] = {"verdict": verdict, "expected_z": p["z"], "page_z": p["page_z"], "units": p["units"],
                                  "aim": p["aim"], "xy": p["xy"], "how": r["how"],
                                  "contact_z": None if early is None else round(r["tip_z"], 1),
                                  "early_mm": early, "retreat": r["retreat"]}
            if r["retreat"] == "home":
                raise StageFail(f"{LABEL[k]}: the load stayed high after backing up, so the arm went home - the demo stopped there")
            if early is not None:
                st.say(f"demo: {k} met the page {early:+.1f} mm above its touch height ({r['how']}) - backed off")
            else:
                st.say(f"demo: {k} light touch at z {p['z']:.1f} mm ({p['units']} unit(s) above the book at {p['page_z']:.1f})")
        lift_then_home()
        res = demo["results"]
        on = [k for k in DEMO_ORDER if (res.get(k) or {}).get("verdict") == "on"]
        early = [k for k in DEMO_ORDER if (res.get(k) or {}).get("verdict") == "early"]
        skip = [k for k in DEMO_ORDER if (res.get(k) or {}).get("verdict") == "skip"]
        dd = ROOT / "data" / "demo"
        dd.mkdir(parents=True, exist_ok=True)
        (dd / f"{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(json.dumps(
            {"taken": demo["taken"], "tool_length_mm": L, "table_z": table_z(), "xy_offset": xy_offset(), "results": res,
             "traces": live["traces"]}, indent=1), encoding="utf-8")
        msg = f"{len(on)} of {len(todo)} touched at the planned height"
        if early:
            msg += "; met the page early: " + ", ".join(f"{k} {res[k]['early_mm']:+.1f} mm" for k in early)
        if skip:
            msg += "; skipped: " + ", ".join(f"{k} ({res[k]['why']})" for k in skip)
        # an early contact is a measured page height: merge those into the model (a touch that went all the way down is no
        # new measurement - the page is somewhere under it)
        old = bm["model"]
        new = [{"key": k, "kind": "demo", "x": float(r["xy"][0]), "y": float(r["xy"][1]), "verdict": "early",
                "z": float(r["contact_z"])} for k, r in res.items() if r.get("verdict") == "early" and r.get("xy")]
        data = {"results": res}
        if new:
            kept = [t for t in old["touches"] if t["kind"] != "demo"]
            m = make_model([{k: t[k] for k in ("key", "kind", "x", "y", "z")} for t in kept] + new,
                           int(old.get("version", 1)) + 1, bm["stamp"])
            c = m["level"]
            msg += f". Book model updated with {len(new)} measured page height(s) (v{m['version']}: {c['kind']})"
            data["model"] = bmod.summary(m)
        return ("ok" if not early and not skip else "warn"), msg, data

    # ---- XY marking calibration (manual, optional): where does the arm really land vs where vision aims? -------------
    xc = {"ev": threading.Event(), "action": None, "waiting": None, "hovered": {}, "clicks": {}, "photo": None, "geom": None,
          "applied": None, "taken": None, "saved_in": None, "results": {}}

    def xycal_summary():
        """Landing error per marked point (mark - target, world mm) and the XY correction it gives."""
        if xc["geom"] is None or not xc["clicks"]:
            return None
        rows, d = {}, []
        from ..vision.geometry import pixel_to_plane

        g = xc["geom"]
        for k in MARK_ORDER:
            if k in xc["clicks"] and k in xc["hovered"]:                # the mark is on the page at the touched height
                a = pixel_to_plane([xc["clicks"][k]], g.K, g.T_world_cam, xc["hovered"][k]["hover_z"], g.dist)[0][:2]
                t = xc["hovered"][k]["xy"]
                dd = [round(a[0] - t[0], 1), round(a[1] - t[1], 1)]
                rows[k] = {"target": t, "mark": [round(a[0], 1), round(a[1], 1)], "d_mm": dd}
                d.append(dd)
        if not d:
            return None
        D, ap = np.array(d, float), np.array(xc["applied"] or [0.0, 0.0], float)
        mean = D.mean(0)
        return {"n": len(d), "points": rows, "mean_mm": [round(float(v), 1) for v in mean],
                "rms_mm": round(float(np.sqrt((D ** 2).sum(1).mean())), 1),
                "spread_mm": round(float(np.sqrt(((D - mean) ** 2).sum(1).mean())), 1),
                "applied_mm": [round(float(v), 1) for v in ap], "new_offset_mm": [round(float(v), 1) for v in ap + mean]}

    def stage_xycal(o):
        """Touch calibration (XY), straight onto the book:
          for each ticked point (BL first) go directly over it, come down in 2 mm steps to the book height the Z link
          measured plus ``units`` steps (the light touch, no pressure), hold a second, and on to the next; then home and a
          fresh photo, on which the person clicks where each point was really touched -> the XY correction.
        If the load jumps before that height the page is higher there: that is reported and the touch is made above it."""
        if not book.px:
            raise StageFail("run the first capture first")
        d = depth_cfg()
        fresh = bool(d.get("z_link_page_at")) and str(d["z_link_page_at"]) >= str(book.taken_at or "")
        if d.get("z_link_page_tip_z") is None or not fresh:
            raise StageFail("run the Z link first (after the first capture): it measures the book height at CL that the touches "
                            "are held above")
        book_z = float(d["z_link_page_tip_z"])
        order = ["bl", "tl", "tr", "br", "cl", "cr"]
        keys = [k for k in order if k in (o.get("points") or ["bl"])]
        units = {k: int(v) for k, v in (o.get("units") or {}).items() if k in KEYS}
        if not keys:
            raise StageFail("choose at least one point")
        ensure_home()
        off = xy_offset() or [0.0, 0.0]
        xc.update(hovered={}, clicks={}, photo=None, geom=None, applied=list(off), taken=time.strftime("%Y-%m-%d %H:%M:%S"),
                  saved_in=None, results={})
        live["traces"], live["expected"] = {}, {}
        L, results = tool_len(), {}
        try:
            for n_, k in enumerate(keys):
                if st.estop:
                    raise StageFail("stopped")
                xy = book.demo_xy(k, CORNER_INSET_MM)
                if xy is None:
                    results[k] = {"skip": "no point"}
                    continue
                aim = [round(xy[0] - off[0], 1), round(xy[1] - off[1], 1)]
                u = max(0, min(5, units.get(k, 1)))
                hold = book_z + u * STEP_MM
                if hold < tip_floor():
                    results[k] = {"skip": f"touch z {hold:.1f} is below the tip floor {tip_floor():g}"}
                    continue
                set_progress("xycal", n_ / len(keys), f"{LABEL[k]}: moving over it")
                travel_above(aim[0], aim[1], hold + 40.0)
                zs = [float(v) for v in np.arange(hold + 14.0, hold - 1e-6, -STEP_MM)]   # straight down onto the touch height
                r = ladder("tc_" + k, LABEL[k], aim[0], aim[1], zs, stage="xycal",
                           lo=n_ / len(keys), hi=(n_ + 0.8) / len(keys))
                if r["retreat"] == "home":
                    raise StageFail(f"{LABEL[k]}: the load stayed high after backing off, so the arm went home")
                early = None
                if r["how"] != "bottom":                               # the page is higher here than the Z link height
                    early = round(float(r["tip_z"]) - book_z, 1)
                    hold = float(r["tip_z"]) + u * STEP_MM
                    st.say(f"touch calibration: {k}: met the page at {r['tip_z']:.1f}, {early:+.1f} mm above the Z link height - "
                           f"touching {u} unit(s) above that instead")
                ps = flange_for_tip([aim[0], aim[1], hold], robot.end_pose(), L)
                slow_block(lambda: guarded_move(ps, LABEL[k] + " light touch"))
                st.say(f"touch calibration: {k}: light touch at {hold:.1f} ({u} unit(s) above {hold - u * STEP_MM:.1f})")
                results[k] = {"xy": xy, "aim": aim, "book_z": round(hold - u * STEP_MM, 1), "units": u, "touch_z": round(hold, 1),
                              "early_mm": early, "contact_z": None if early is None else round(float(r["tip_z"]), 1)}
                xc["hovered"][k] = {"xy": xy, "aim": aim, "hover_z": round(hold, 1)}
                xc["results"] = dict(results)                           # shown as it goes
                time.sleep(1.0)                                         # a moment at the light touch, then on
        except StageFail as e:
            if "stopped" not in str(e):                                 # anything but STOP: back up and home, so it can be retried
                try:
                    lift_then_home()
                except Exception as e2:  # noqa: BLE001
                    st.say(f"touch calibration: could not get home after the failure: {e2}")
            raise
        finally:
            xc["waiting"] = None
            xc["results"] = results
        touched = {k: v for k, v in results.items() if "touch_z" in v}
        save_depth_cfg({"touchcal": {"at": time.strftime("%Y-%m-%d %H:%M:%S"), "book_z": book_z,
                                     "points": touched}})
        drop_model()
        set_progress("xycal", 0.97, "back to home for the marking photo")
        lift_then_home()
        time.sleep(0.5)
        xc["photo"], xc["geom"] = robot.capture(), robot.camera_geometry()
        pz = ", ".join(f"{k} {v['touch_z']:.1f} (Z link {v['book_z']:.1f} + {v['units']})" for k, v in touched.items())
        return "warn", (f"light touches at {pz or 'none'}. Back at home: in the viewer's 'XY marks' click where each point was "
                        "really touched and press 'Save the XY correction'."), {"results": results, "book_z": book_z}

    STAGE_FN = {"level": stage_level, "capture": stage_capture, "depth": stage_depth, "zlink": stage_zlink, "model": stage_model,
                "xycal": stage_xycal, "demo": stage_demo}

    def run_stages(ids, opts, auto):
        t0 = time.time()

        def work():
            try:
                for n, k in enumerate(ids):
                    if st.estop:
                        set_stage(k, "fail", "stopped")
                        break
                    flow["running"] = k
                    flow["progress"] = {"stage": k, "n": n, "of": len(ids), "sub": 0.0, "text": "", "t0": t0, "ids": list(ids)}
                    set_stage(k, "running")
                    try:
                        state, msg, data = STAGE_FN[k](opts)
                    except StageFail as e:
                        set_stage(k, "fail", str(e))
                        break
                    except Exception as e:  # noqa: BLE001
                        from ..hardware.workshop import is_transient

                        if is_transient(e):
                            set_stage(k, "fail", f"the connection to the robot dropped and did not come back after 3 tries ({e}). "
                                                 "Viam stops the arm when that happens. Check the network, then run the stage again.")
                        else:
                            set_stage(k, "fail", f"{e} - the arm was left where it is")
                        break
                    set_stage(k, state, msg, data)
                    mark_stale([d for d in DEPENDS.get(k, []) if d not in ids])
                    if k == "model" and state == "warn" and len(ids) > 1:     # never touch the book on a doubtful model
                        st.say("Run all stopped after the book model: it needs checking before anything touches the book")
                        break
            finally:
                flow["running"], flow["auto"] = None, False
                live["active"] = False
                if flow.get("progress"):
                    flow["progress"]["finished"] = time.time()
                st.moving = False

        flow["auto"] = auto
        st.moving = True
        threading.Thread(target=work, daemon=True).start()

    # ---- routes: page, status, safety ----------------------------------------------------------------------
    @app.get("/")
    def index():
        return Response((HERE / "index.html").read_text(encoding="utf-8").replace("__TOKEN__", token)
                        .replace("__DISK_BUILD__", str(code_build())).replace("__SERVER_BUILD__", str(started_build)),
                        mimetype="text/html")

    @app.get("/api/status")
    def status():
        return jsonify(estop=st.estop, moving=st.moving, live=st.live, fps=round(st.fps, 1), fetch_ms=st.fetch_ms,
                       frame_age_s=None if not st.frame_t else round(time.time() - st.frame_t, 1),
                       pose=st.pose, gripper=st.gripper, limits=st.limits, max_step=MAX_STEP,
                       saved_poses=sorted(k for k, v in (st.cfg.get("poses") or {}).items() if v),
                       log=list(st.log)[-60:], build=started_build, robot_mode=robot_mode)

    @app.post("/api/stop")
    def stop():
        st.estop = True                      # refuse new motion first, then halt what's moving
        st.say("STOP pressed")

        def do_stop():
            try:
                robot.stop()
                st.say("arm and gripper stopped")
            except Exception as e:  # noqa: BLE001
                st.say(f"STOP FAILED: {e} - use the pendant / e-stop!")

        threading.Thread(target=do_stop, daemon=True).start()
        return jsonify(ok=True)

    @app.post("/api/reset")
    def reset():
        st.estop = False
        st.say("motion re-enabled")
        return jsonify(ok=True)

    # ---- routes: monitor (camera, gripper, jog, saved poses) ---------------------------------
    def jpeg(img, q=80):
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        return Response(buf.tobytes(), mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/frame.jpg")
    def frame_jpg():
        with st.lock:
            frame = None if st.frame is None else st.frame.copy()
        return Response(status=204) if frame is None else jpeg(frame)

    @app.post("/api/live")
    def live_toggle():
        st.live = bool(request.get_json(force=True)["on"])
        return jsonify(ok=True)

    @app.post("/api/gripper")
    def gripper():
        if (r := blocked()) is not None:
            return r
        pos = max(0.0, min(850.0, float(request.get_json(force=True)["position"])))
        try:
            robot.gripper_position(pos)
            st.say(f"gripper -> {pos:.0f}")
            return jsonify(ok=True)
        except Exception as e:  # noqa: BLE001
            st.say(f"gripper error: {e}")
            return jsonify(error=str(e)), 500

    @app.post("/api/jog")
    def jog():
        if (r := arm_blocked()) is not None:
            return r
        body = request.get_json(force=True)
        axis, delta = body.get("axis"), float(body.get("delta", 0))
        if axis not in JOG_AXES or delta == 0 or abs(delta) > MAX_STEP[axis]:
            return jsonify(error=f"bad jog (axis {JOG_AXES}, |step| <= {MAX_STEP})"), 400
        try:
            target = dict(robot.end_pose())
        except Exception as e:  # noqa: BLE001
            return jsonify(error=f"can't read the arm pose: {e}"), 500
        target[axis] = target[axis] + delta
        if (why := in_limits(target)) is not None:
            return jsonify(error=f"refused: {why}"), 409
        run_move(target, f"jog {axis} {delta:+g}")
        return jsonify(ok=True)

    @app.post("/api/pose/save")
    def pose_save():
        name = str(request.get_json(force=True).get("name", "")).strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,24}", name):
            return jsonify(error="pose name: letters, digits, _ or - (max 24)"), 400
        pose = {k: round(float(v), 3) for k, v in robot.end_pose().items()}
        save_local({"poses": {name: pose}})
        st.say(f"saved pose '{name}' to config/local.yaml")
        return jsonify(ok=True)

    @app.post("/api/pose/go")
    def pose_go():
        if (r := arm_blocked()) is not None:
            return r
        name = str(request.get_json(force=True).get("name", ""))
        if name == "home":                                      # home is a slow joint move, like the Setup tab's
            if not st.cfg.get("home_joints"):
                return jsonify(error="no home joints saved"), 409

            def work():
                try:
                    lift_then_home()                             # straight up first if low (over the book)
                except Exception as e:  # noqa: BLE001
                    st.say(f"go home failed: {e}")
                finally:
                    st.moving = False

            st.moving = True
            threading.Thread(target=work, daemon=True).start()
            return jsonify(ok=True)
        pose = (st.cfg.get("poses") or {}).get(name)
        if not pose:
            return jsonify(error=f"no saved pose '{name}'"), 404
        if (why := in_limits(pose)) is not None:
            return jsonify(error=f"refused: pose '{name}' {why}"), 409
        run_move(dict(pose), f"go to '{name}'")
        return jsonify(ok=True)

    # ---- routes: setup ---------------------------------------------------------------------------------------
    @app.get("/setup/view.jpg")
    def setup_view():
        """Live wrist camera with the points (and the tool tip) redrawn from the camera's pose; or the first capture."""
        demo_res = demo["results"]
        if request.args.get("mode") == "capture":
            if book.image is None:
                return Response(status=204)
            return jpeg(book.draw(book.image, book.geom, demo_res), 90)
        lv["wanted"] = time.time()
        with st.lock:
            frame = None if st.frame is None else st.frame.copy()
        if frame is None:
            return Response(status=204)
        g, pose = lv["geom"], lv["pose"]
        if g is not None and pose is not None and book.px:
            tip = tip_of(pose, tool_len())
            tz = book.surface_z(tip[0], tip[1])
            frame = book.draw(frame, g, demo_res, tip=[float(tip[0]), float(tip[1]), tz])
        return jpeg(frame)

    @app.get("/api/flow")
    def flow_state():
        d = depth_cfg()
        pts = []
        if book.geom is not None:
            for k in KEYS:
                w = book.world3(k)
                pts.append({"key": k, "label": LABEL[k], "world": w})
        try:
            ah = at_home(st.pose) if st.pose else False
        except Exception:  # noqa: BLE001
            ah = False
        return jsonify(
            stages=[{"id": k, "label": lab, **flow["stages"][k]} for k, lab in STAGES], running=flow["running"], auto=flow["auto"],
            moving=st.moving, estop=st.estop, progress=flow["progress"], now=time.time(),
            tool_length_mm=tool_len(), table_z=table_z(), table_measured=table_measured(), table_measured_at=d.get("table_height_fitted"),
            z_link_tool_length_mm=d.get("z_link_tool_length_mm"), z_link_spot_lock=d.get("z_link_spot_lock"),
            xy_offset=xy_offset(), path=path_status(), at_home=ah, has_home=bool(home_pose()) and bool(st.cfg.get("home_joints")),
            capture_taken=book.taken_at, points=pts, live={k: v for k, v in live.items()},
            demo={"results": demo["results"], "order": demo["order"], "taken": demo["taken"]},
            model=None if bm["model"] is None else {**bmod.summary(bm["model"]), "dir": bm["dir"]},
            xycal={"waiting": xc["waiting"], "hovered": xc["hovered"], "clicks": xc["clicks"], "has_photo": xc["photo"] is not None,
                   "results": xc["results"],
                   "summary": xycal_summary(), "saved_in": xc["saved_in"],
                   "calibration": gui_cfg().get("calibration")})

    @app.post("/api/flow/stage")
    def flow_stage():
        if (r := arm_blocked()) is not None:
            return r
        b = request.get_json(force=True)
        if b.get("confirm") is not True:
            return jsonify(error="needs confirmation"), 400
        if b.get("id") not in STAGE_FN:
            return jsonify(error="unknown stage"), 400
        opts = dict(b.get("options") or {})
        if b["id"] == "zlink":
            opts["force"] = True                                # the stage's own button always measures
        run_stages([b["id"]], opts, auto=False)
        return jsonify(ok=True)

    @app.post("/api/flow/run")
    def flow_run():
        if (r := arm_blocked()) is not None:
            return r
        b = request.get_json(force=True)
        if b.get("confirm") is not True:
            return jsonify(error="needs confirmation"), 400
        ids = [k for k, _ in STAGES if k not in MANUAL]
        if b.get("skip_touch"):                                 # nothing that lowers the tool onto anything
            ids = [k for k in ids if k not in ("zlink", "model", "demo")]
        run_stages(ids, dict(b.get("options") or {}), auto=True)
        return jsonify(ok=True, stages=ids)

    @app.post("/api/flow/contact")
    def flow_contact():
        """The person says the tip is touching, at the step shown (0 = skip this touch)."""
        if not live.get("active") or not live.get("press"):
            return jsonify(error="no touch search is waiting for a press"), 409
        step = request.get_json(force=True).get("step")
        if not isinstance(step, int) or not 0 <= step <= live.get("step", 0):
            return jsonify(error="the arm has not reached that step yet"), 400
        contact["step"] = step
        contact["ev"].set()
        return jsonify(ok=True)

    @app.post("/api/tool")
    def set_tool():
        """The one manual setting: the tool tip's distance below the flange. Changing it makes the z link out of date."""
        try:
            v = float(request.get_json(force=True)["tool_length_mm"])
        except (KeyError, TypeError, ValueError):
            return jsonify(error="tool_length_mm must be a number"), 400
        if not 50 <= v <= 400:
            return jsonify(error="tool length must be 50..400 mm"), 400
        save_local({"gui": {"probe": {"tool_length_mm": round(v, 1)}}})
        mark_stale(["zlink", "model", "demo"])
        st.say(f"tool length set to {v:g} mm - run the z link again")
        return jsonify(ok=True)

    @app.post("/api/zlink/spot")
    def zlink_spot():
        """Lock the z link's table spot: {here: true} = the tool tip's x, y now, {x, y}, or {clear: true}."""
        b = request.get_json(force=True)
        if b.get("clear"):
            save_depth_cfg({"z_link_spot_lock": None})
            st.say("z link: table spot unlocked - the stage picks one clear of the book")
            return jsonify(ok=True, lock=None)
        if b.get("here"):
            tip = tip_of(robot.end_pose(), tool_len())
            x, y = float(tip[0]), float(tip[1])
        else:
            try:
                x, y = float(b["x"]), float(b["y"])
            except (KeyError, TypeError, ValueError):
                return jsonify(error="give x and y in mm"), 400
        if (h := table_h_at(x, y)) is not None and abs(h) > 3.0:
            return jsonify(error=f"({x:.0f}, {y:.0f}) is not bare table in the depth map ({h:.1f} mm above it)"), 400
        save_depth_cfg({"z_link_spot_lock": [round(x, 1), round(y, 1)]})
        st.say(f"z link: table spot locked at ({x:.0f}, {y:.0f})")
        return jsonify(ok=True, lock=[round(x, 1), round(y, 1)])

    # ---- the recorded flip path, following the book ------------------------------------------------------------
    def book_frame_now() -> dict | None:
        """Where the book is now, as the dynamic flip path wants it (bookworm/hardware/dynamic_path.py): from the first
        capture's corners (moved by the XY correction, i.e. where the arm really lands) and the Z link's book height.
        None until both are there and the Z link was measured after the capture."""
        from ..hardware import dynamic_path as dp

        d = depth_cfg()
        pts = {k: book.world3(k) for k in ("tl", "tr", "br", "bl")}
        fresh = bool(d.get("z_link_page_at")) and bool(book.taken_at) and str(d["z_link_page_at"]) >= str(book.taken_at)
        if any(v is None for v in pts.values()) or d.get("z_link_page_tip_z") is None or not fresh:
            return None
        off = xy_offset() or [0.0, 0.0]
        pts = {k: [v[0] - off[0], v[1] - off[1]] for k, v in pts.items()}
        return dp.frame_from_points(pts, float(d["z_link_page_tip_z"]), source=f"workshop capture {book.taken_at}")

    def save_book_frame() -> None:
        from ..hardware import dynamic_path as dp

        f = book_frame_now()
        if f is not None:
            dp.save_frame(f, ROOT / "data" / "book_frame.json")
            st.say(f"book frame saved for the flip path: centre ({f['center'][0]:.0f}, {f['center'][1]:.0f}), "
                   f"{f['angle_deg']:.1f} deg, top z {f['top_z']:.1f}")

    def path_status() -> dict:
        from ..hardware import dynamic_path as dp

        ref, now = dp.load_reference(LOCAL_CFG), book_frame_now()
        return {"reference": ref, "now": now, "motion": dp.motion(ref, now) if ref and now else None}

    @app.post("/api/path/reference")
    def path_reference():
        """The book is where the recorded path works: remember this book frame as the path's reference."""
        from ..hardware import dynamic_path as dp

        f = book_frame_now()
        if f is None:
            return jsonify(error="needs the first capture and then the Z link"), 409
        dp.set_reference(f, LOCAL_CFG)
        dp.save_frame(f, ROOT / "data" / "book_frame.json")
        st.say("flip path reference set: the recorded path is moved from here on")
        return jsonify(ok=True, reference=f)

    def run_path_flip(version: str) -> None:
        """Midhat's recorded path (config/waypoints.yaml), moved with the book; the recorded joints if it hasn't moved."""
        import sys

        from ..hardware import dynamic_path as dp

        scripts = str(HERE.parents[1] / "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        import flip_path

        pts, path, between = flip_path.load(version)
        wp = yaml.safe_load(flip_path.WAYPOINTS.read_text())
        p = dp.plan(wp, flip_path.PATHS[version], dp.load_reference(LOCAL_CFG), book_frame_now())
        st.say(f"flip path {version}: {p['mode']} ({p['why']})")
        for name in flip_path.make_plan(path, between):
            if st.estop:
                raise StageFail("stopped")
            if name.startswith("wait "):
                continue
            st.say(f"flip path: {name}")
            if name in p["poses"]:
                guarded_move(p["poses"][name], f"flip path {name}")
            else:
                robot.move_joints(pts[name])

    @app.post("/api/flip/run")
    def flip_run():
        """``motion`` "path": the recorded path (config/waypoints.yaml) moved with the book. "tape" (default): the
        taped-gripper flip (bookworm/gui/flip.py) at 2x the teaching speed, from home, back home; ``tuck`` also slides
        the flipped page toward the holder (experimental)."""
        from . import flip

        if (r := arm_blocked()) is not None:
            return r
        body = request.get_json(silent=True) or {}
        if body.get("motion") == "path":
            version = str(body.get("version") or "v2")

            def work_path():
                try:
                    run_path_flip(version)
                    st.say("flip path done")
                except Exception as e:  # noqa: BLE001
                    st.say(f"flip path stopped: {e}")
                finally:
                    st.moving = False

            st.moving = True
            threading.Thread(target=work_path, daemon=True).start()
            return jsonify(ok=True, motion="path")
        pts = {k: book.world3(k) for k in ("bl", "tl", "cl", "cr")}
        page_z = depth_cfg().get("z_link_page_tip_z")
        if any(v is None for v in pts.values()) or page_z is None:
            return jsonify(error="needs the first capture and the Z link (book height)"), 409
        f = flip.frame(pts["bl"], pts["tl"], pts["cl"], pts["cr"], xy_offset() or [0.0, 0.0], page_z, tool_len())
        steps = flip.plan(f, tuck=bool(body.get("tuck")))

        def work():
            t0 = time.time()
            try:
                ensure_home()
                robot.set_speed(flip.FAST_DEG_S)
                flip.run(robot, steps, say=st.say, stopped=lambda: st.estop)
                robot.set_speed(flip.HOME_DEG_S)
                robot.move_joints(st.cfg["home_joints"])
                st.say(f"page flip done in {time.time() - t0:.0f} s")
            except Exception as e:  # noqa: BLE001
                st.say(f"page flip stopped: {e}")
            finally:
                try:
                    robot.set_speed(speeds()[1])
                except Exception:  # noqa: BLE001
                    pass
                st.moving = False

        st.moving = True
        threading.Thread(target=work, daemon=True).start()
        return jsonify(ok=True, steps=len(steps))

    @app.post("/api/demo/plan")
    def demo_plan_route():
        """Dry run of the touch demo: where each point would be touched and at what z. Nothing moves."""
        if not book.px or dsc["res"] is None:
            return jsonify(error="run the first capture and the depth scan first"), 409
        b = request.get_json(force=True, silent=True) or {}
        return jsonify(points=demo_plan(b.get("points"), b.get("units")), xy_offset=xy_offset(), table_z=table_z(),
                       table_measured=table_measured(), driven=None if bm["model"] is None else bm["model"].get("driven"))

    @app.post("/api/xycal/next")
    def xycal_next():
        """The person marked the paper under the tip ({skip: true}: this point could not be marked)."""
        if xc["waiting"] is None:
            return jsonify(error="the arm is not waiting over a point"), 409
        xc["action"] = "skip" if request.get_json(force=True).get("skip") else "marked"
        xc["ev"].set()
        return jsonify(ok=True)

    @app.post("/api/xycal/mark")
    def xycal_mark():
        """Where a mark is in the photo: {key, u, v} (pixels of the photo), or {key, clear: true}."""
        b = request.get_json(force=True)
        k = b.get("key")
        if xc["photo"] is None:
            return jsonify(error="no photo of the marks yet - run the XY marking stage first"), 409
        if k not in xc["hovered"]:
            return jsonify(error="that point was not marked"), 400
        if b.get("clear"):
            xc["clicks"].pop(k, None)
        else:
            try:
                u, v = float(b["u"]), float(b["v"])
            except (KeyError, TypeError, ValueError):
                return jsonify(error="give u and v in pixels"), 400
            h, w = xc["photo"].shape[:2]
            if not (0 <= u < w and 0 <= v < h):
                return jsonify(error="outside the photo"), 400
            xc["clicks"][k] = [round(u, 1), round(v, 1)]
        return jsonify(ok=True, summary=xycal_summary())

    @app.post("/api/xycal/photo")
    def xycal_retake():
        """A fresh photo for marking, from home (the arm must be at home and still)."""
        if (r := arm_blocked()) is not None:
            return r
        if not xc["hovered"]:
            return jsonify(error="run the touch calibration first"), 409
        if not at_home():
            return jsonify(error="the arm is not at home - the marks are read from the home view"), 409
        xc["photo"], xc["geom"] = robot.capture(), robot.camera_geometry()
        return jsonify(ok=True)

    @app.get("/xycal/photo.jpg")
    def xycal_photo():
        return Response(status=204) if xc["photo"] is None else jpeg(xycal_drawn(), 85)

    def xycal_drawn():
        """The photo of the marks with each target (yellow ring) and each clicked mark (red x)."""
        img, g = xc["photo"].copy(), xc["geom"]
        for k, hv in xc["hovered"].items():
            t = book._project(g, [[*hv["xy"], hv["hover_z"]]])[0]
            c = (int(t[0]), int(t[1]))
            cv2.circle(img, c, 10, (0, 220, 255), 2)
            cv2.putText(img, k, (c[0] + 12, c[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
            if k in xc["clicks"]:
                a = tuple(int(v) for v in xc["clicks"][k])
                cv2.drawMarker(img, a, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 22, 3)
                cv2.line(img, c, a, (255, 255, 255), 1)
        return img

    @app.post("/api/xycal/save")
    def xycal_save():
        """Save the marks and make their mean landing error part of the XY correction the demo aims with."""
        s_ = xycal_summary()
        if s_ is None:
            return jsonify(error="click at least one mark on the photo first"), 409
        d = ROOT / "data" / "calibration" / time.strftime("%Y%m%d_%H%M%S")
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / "home.jpg"), xc["photo"])
        cv2.imwrite(str(d / "marked.jpg"), xycal_drawn())
        g = xc["geom"]
        (d / "calibration.json").write_text(json.dumps({
            "taken_at": xc["taken"], "kind": "xy marking", "tool_length_mm": tool_len(),
            "K": g.K.tolist(), "T_world_cam": g.T_world_cam.tolist(), "hovered": xc["hovered"], "clicks": xc["clicks"],
            "summary": s_}, indent=2), encoding="utf-8")
        save_local({"gui": {"calibration": {"offset_mm": s_["new_offset_mm"], "rms_mm": s_["rms_mm"], "spread_mm": s_["spread_mm"],
                                            "points": s_["n"], "tool_length_mm": tool_len(),
                                            "saved_in": str(d.relative_to(ROOT)), "measured": xc["taken"]}}})
        xc["saved_in"] = str(d.relative_to(ROOT))
        set_stage("xycal", "ok", f"XY correction is now ({s_['new_offset_mm'][0]:+.1f}, {s_['new_offset_mm'][1]:+.1f}) mm "
                                 f"({s_['n']} marks, {s_['rms_mm']} mm rms, {s_['spread_mm']} mm spread)", {"summary": s_})
        mark_stale(["demo"])
        return jsonify(ok=True, summary=s_)

    @app.get("/book_model.json")
    def book_model_file():
        """The current book model's json (the grid itself is in book_model.npz next to it)."""
        if bm["dir"] is None:
            return jsonify(error="no book model yet"), 404
        return Response((Path(bm["dir"]) / "book_model.json").read_text(encoding="utf-8"), mimetype="application/json")

    # ---- routes: depth views -----------------------------------------------------------------------------------
    @app.get("/api/depth")
    def depth_state():
        res = dsc["res"]
        out = dict(has_scan=res is not None, taken=dsc["taken"], mode=dsc["mode"], views=len(dsc["views"]), skipped=dsc["skipped"],
                   table_z=table_z(), table_tilt_deg=depth_cfg().get("table_tilt_deg"))
        if res is not None:
            out.update(stats=res["stats"], points=res["points"], profiles=res["profiles"])
        return jsonify(out)

    def depth_range():
        off = float(request.args.get("ref") == "arm") * table_z()
        return -2 + off, 62 + off, off

    @app.get("/depth/top.png")
    def depth_top():
        from ..vision.depth_render import render_top

        res = dsc["res"]
        if res is None:
            return Response(status=204)
        lo, hi, off = depth_range()
        pts = [{**p, "h": None if p["h"] is None else p["h"] + off} for p in res["points"]]
        ok, buf = cv2.imencode(".png", render_top(res["x"], res["y"], res["z"] + off, res["book"], lo, hi, pts))
        return Response(buf.tobytes(), mimetype="image/png", headers={"Cache-Control": "no-store"})

    @app.get("/depth/surface.png")
    def depth_surface():
        from ..vision.depth_render import render_surface

        res = dsc["res"]
        if res is None:
            return Response(status=204)
        lo, hi, off = depth_range()
        az = float(request.args.get("az", 35))
        el = min(max(float(request.args.get("el", 40)), 5), 90)
        ok, buf = cv2.imencode(".png", render_surface(res["x"], res["y"], res["z"] + off, lo, hi, az, el, book=res["book"]))
        return Response(buf.tobytes(), mimetype="image/png", headers={"Cache-Control": "no-store"})

    return app
