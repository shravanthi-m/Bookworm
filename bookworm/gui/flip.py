"""Page flip, taught on the real arm on 2026-10-17 (temp/flip/poses.json has the session's notes).

All poses are in the BOOK FRAME, so they follow the book and every recalibration:
  origin  the TL-BL edge midpoint (map coords) minus the XY correction = where the arm aims at the left page's outer edge
  u       along BL -> TL;  v  across, + out over the table, - toward the spine;  w  tip height above the Z link page z
  dtheta  wrist angle minus the edge angle;  the tool stays vertical.

The motion (taped finger = the gripper's spine-side finger at wrist dtheta -8.25):
  1 pick     gripper 175, firm touch 3.5 mm above the page at the edge, open fully (the taped finger slides ~42 mm toward
             the spine and peels the top page up), lift to w 23.5
  2 arc      gripper still open: the taped finger goes round the spine on a circle through where it is now (R ~ 88);
             a circle any bigger pulls outward and peels the tape off
  3 pinch    at 110 deg the page body lies between the fingers: close to 50
  4 carry    pinched, round to 160 deg on R 90 (fingers centred), short of the holder's face
  5 release  open to 400, lift clear, home
"""
from __future__ import annotations

import math
import time

PICK_U, PICK_V, DTHETA = 40.0, -10.0, -8.25        # v -10: deeper into the page (at the edge several pages came up);
                                                    # u +40: beside the holder (it spans u ~ -64..-9), never in line with it
TUCK_W, TUCK_LAY, TUCK_FLAT = 8.0, 0.7, 0.95        # tuck: pinched page laid bowed at w 8, then flattened (never pulled
                                                    # past its length: it is bound at the spine)
HOVER_W, FIRM_W, LIFT_W = 11.5, 5.5, 23.5          # light touch (5.5 = the soft touch first taught; stronger tape now)
GRIP_START, GRIP_PEEL, GRIP_PINCH, GRIP_RELEASE = 175, 850, 50, 400
FINGER_OFF = 42.0                                   # taped finger, toward the spine from the flange, gripper at 850
ARC_DEG = (30, 50, 70, 90, 110)
CARRY_SLACK, CARRY_DEG = 6.0, (130, 145, 160)
CARRY_R = 0.0                                       # >0: carry on this fixed radius instead
CLEAR_W = 70.0
FAST_DEG_S, HOME_DEG_S = 120, 40                     # 2x the speeds it was taught at (60, and the 20 deg/s home move)


def frame(bl, tl, cl, cr, offset, page_z, tool_len):
    """The book frame from the first-capture points (map coords), the XY correction and the Z link page height."""
    a = math.atan2(tl[1] - bl[1], tl[0] - bl[0])
    ox, oy = (bl[0] + tl[0]) / 2 - offset[0], (bl[1] + tl[1]) / 2 - offset[1]
    sx, sy = (cl[0] + cr[0]) / 2 - (bl[0] + tl[0]) / 2, (cl[1] + cr[1]) / 2 - (bl[1] + tl[1]) / 2
    spine_v = -math.sin(a) * sx + math.cos(a) * sy             # the spine's v (negative: in from the edge)
    return {"a": a, "ox": ox, "oy": oy, "page_z": float(page_z), "L": float(tool_len), "spine_v": spine_v}


def to_arm(f, u, v, w, dtheta=DTHETA):
    a = f["a"]
    return {"x": f["ox"] + u * math.cos(a) - v * math.sin(a), "y": f["oy"] + u * math.sin(a) + v * math.cos(a),
            "z": w + f["L"] + f["page_z"], "o_x": 0.0, "o_y": 0.0, "o_z": -1.0, "theta": dtheta + math.degrees(a)}


def plan(f, tuck=False) -> list[dict]:
    """The whole flip as steps: {"what", "pose"} or {"what", "grip"}. ``tuck``: lay the page down pinched and slide its
    edge into the holder's slit before letting go (released at 160 deg the page springs back over the spine)."""
    s = f["spine_v"]
    steps = [{"what": "gripper to start", "grip": GRIP_START},
             {"what": "hover", "pose": to_arm(f, PICK_U, PICK_V, HOVER_W + 30)},
             {"what": "hover low", "pose": to_arm(f, PICK_U, PICK_V, HOVER_W)},
             {"what": "firm touch", "pose": to_arm(f, PICK_U, PICK_V, FIRM_W), "dwell": 0.3},   # let the tape bond
             {"what": "peel (open)", "grip": GRIP_PEEL, "dwell": 0.4},
             {"what": "lift", "pose": to_arm(f, PICK_U, PICK_V, LIFT_W)}]
    r = math.hypot(PICK_V - FINGER_OFF - s, LIFT_W)         # the taped finger's distance from the spine now
    for d in ARC_DEG:
        t = math.radians(d)
        steps.append({"what": f"arc {d}", "pose": to_arm(f, PICK_U, s + r * math.cos(t) + FINGER_OFF, r * math.sin(t))})
    steps.append({"what": "pinch", "grip": GRIP_PINCH})
    t = math.radians(ARC_DEG[-1])                              # after the pinch the page is held at the flange: carry it
    cr = math.hypot(r * math.cos(t) + FINGER_OFF, r * math.sin(t)) + CARRY_SLACK   # on a circle through there (+ a little)
    if CARRY_R:                                                # or a fixed circle (90 worked in the first runs)
        cr = CARRY_R
    for d in CARRY_DEG:
        t = math.radians(d)
        steps.append({"what": f"carry {d}", "pose": to_arm(f, PICK_U, s + cr * math.cos(t), cr * math.sin(t))})
    if tuck:   # still pinched, ~d of paper between the spine and the fingers: lay it down bowed (pinch at TUCK_LAY*d from
        d = cr - CARRY_SLACK                       # the spine), then flatten it (pinch out to TUCK_FLAT*d) so the free
        steps += [{"what": "tuck: lay", "pose": to_arm(f, PICK_U, s - TUCK_LAY * d, TUCK_W)},    # edge slides out into
                  {"what": "tuck: flatten", "pose": to_arm(f, PICK_U, s - TUCK_FLAT * d, TUCK_W)}]   # the holder's slit
    steps.append({"what": "release", "grip": GRIP_RELEASE})
    last = steps[-2]["pose"]
    steps.append({"what": "clear", "pose": dict(last, z=f["L"] + f["page_z"] + CLEAR_W)})
    return steps


def run(robot, steps, say=print, stopped=lambda: False) -> None:
    """Runs the steps in order; stops (raises) as soon as ``stopped()`` is true."""
    for i, st in enumerate(steps):
        if stopped():
            raise RuntimeError(f"stopped before '{st['what']}'")
        say(f"flip {i + 1}/{len(steps)}: {st['what']}")
        if "grip" in st:
            robot.gripper_position(st["grip"])
        else:
            robot.move_to({k: round(float(v), 2) for k, v in st["pose"].items()})
        if st.get("dwell"):
            time.sleep(st["dwell"])


if __name__ == "__main__":                                     # python -m bookworm.gui.flip [runs]  (GUI must be up)
    import json
    import sys
    import time
    import urllib.request
    from pathlib import Path

    import yaml

    from ..config import load_config
    from ..rig import build_robot

    root = Path(__file__).resolve().parents[2]
    flow = json.load(urllib.request.urlopen("http://localhost:8765/api/flow", timeout=5))
    pts = {p["key"]: p["world"] for p in flow.get("points") or [] if p.get("world")} or None
    loc = yaml.safe_load((root / "config" / "local.yaml").read_text(encoding="utf-8"))["gui"]
    if pts is None:
        raise SystemExit("no first-capture points in the GUI")
    page_z = next((float(a[7:]) for a in sys.argv if a.startswith("page_z=")), loc["depth"]["z_link_page_tip_z"])
    if page_z is None:
        raise SystemExit("no book height: run the Z link or pass page_z=<tip z of the page>")
    f = frame(pts["bl"], pts["tl"], pts["cl"], pts["cr"], loc["calibration"]["offset_mm"], page_z,
              loc["probe"]["tool_length_mm"])
    import threading

    import cv2

    full = yaml.safe_load((root / "config" / "local.yaml").read_text(encoding="utf-8"))
    cfg = load_config()
    robot, close = build_robot(cfg)

    def film(folder, done):                                    # side camera at ~4 fps while the flip runs
        folder.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        k = 0
        while not done.is_set():
            ok, img = cap.read()
            if ok:
                cv2.imwrite(str(folder / f"{k:03d}.jpg"), cv2.resize(img, (640, 360)))
                k += 1
            time.sleep(0.2)
        cap.release()

    for a in sys.argv[1:]:                                     # NAME=value overrides, e.g. PICK_U=-4 CARRY_R=90
        k, _, v = a.partition("=")
        if k.isupper() and k in globals():
            globals()[k] = type(globals()[k])(eval(v)) if not isinstance(globals()[k], tuple) else tuple(eval(v))
            print("override", k, "=", globals()[k], flush=True)
    start = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 0   # resume from step index ``start``
    try:
        for n in range(int(sys.argv[1]) if len(sys.argv) > 1 else 1):
            t0, done = time.time(), threading.Event()
            cam = threading.Thread(target=film, args=(root / "temp" / "flip" / f"run_{time.strftime('%H%M%S')}", done))
            cam.start()
            robot.set_speed(FAST_DEG_S)
            try:
                upto = next((int(a[5:]) for a in sys.argv if a.startswith("upto=")), None)   # stop after step index
                run(robot, plan(f, tuck="tuck" in sys.argv)[start:None if upto is None else upto + 1])
                start = 0
                if upto is not None:
                    print("stopped after step", upto, "- arm left there", flush=True)
                    break
                robot.set_speed(HOME_DEG_S)
                robot.move_joints(full["home_joints"])
            finally:
                robot.set_speed(60)
                done.set()
                cam.join()
            print(f"RUN {n + 1} done in {time.time() - t0:.0f} s", flush=True)
    finally:
        close()
