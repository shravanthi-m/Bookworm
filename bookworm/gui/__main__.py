"""python -m bookworm.gui [--sim | --replay DIR | --mock] [--port 8765]

--sim      offline: simulated arm + RGB-D camera + side camera over the measured book scene
--replay   offline: serve depth views recorded on the rig (data/depth/raw/<session>)
Both keep their config and data in temp/sim/ (seeded from config/local.yaml), so the rig's files are never changed.
"""
import argparse
import logging
import shutil
import webbrowser
from pathlib import Path

from ..config import load_config
from ..rig import build_robot
from .server import create_app

LOCAL_CFG = Path(__file__).resolve().parents[2] / "config" / "local.yaml"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Bookworm local dev GUI")
    ap.add_argument("--mock", action="store_true", help="fake robot, no Viam")
    ap.add_argument("--sim", action="store_true", help="offline simulated rig (arm, RGB-D camera, side camera)")
    ap.add_argument("--replay", metavar="DIR", help="offline replay of recorded depth views, e.g. data/depth/raw/<session>")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("127.0.0.1", args.port)) == 0:
            raise SystemExit(f"Port {args.port} is already in use - another Bookworm GUI is probably still running "
                             f"(open http://127.0.0.1:{args.port}/ to check). Stop it first, or use --port.")

    from . import server

    local = LOCAL_CFG
    if args.sim or args.replay:
        sim_root = LOCAL_CFG.parents[1] / "temp" / "sim"
        sim_root.mkdir(parents=True, exist_ok=True)
        local = sim_root / "local.yaml"
        if not local.exists() and LOCAL_CFG.exists():
            shutil.copy(LOCAL_CFG, local)                     # start from the rig's settings
        server.ROOT, server.LOCAL_CFG = sim_root, local
        print(f"offline mode: settings and data go to {sim_root} (config/local.yaml is not touched)")
    cfg = load_config(local if local.exists() else None)
    if args.sim:
        (cfg.get("gui") or {}).pop("calibration", None)    # the rig's XY correction is for its real camera, not the simulated one
        from ..hardware.sim import SimRobot

        robot = SimRobot(tool_length=float(((cfg.get("gui") or {}).get("probe") or {}).get("tool_length_mm", 155)))
        close = robot.close
    elif args.replay:
        from ..hardware.replay import ReplayRobot

        robot = ReplayRobot(args.replay, home_pose=(cfg.get("poses") or {}).get("home"), home_joints=cfg.get("home_joints"))
        close = robot.close
    else:
        robot, close = build_robot(cfg, mock=args.mock)
    app = create_app(robot, cfg=cfg)
    url = f"http://127.0.0.1:{args.port}/"
    print("Bookworm GUI:", url, "(Ctrl+C to quit)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        app.run(host="127.0.0.1", port=args.port, threaded=True)
    finally:
        close()


if __name__ == "__main__":
    main()
