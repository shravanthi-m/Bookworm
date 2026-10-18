"""The control page with the dynamic flip path: python -m bookworm.ui.dynamic --config config/phone_mount.yaml

The same page as `python -m bookworm ui` (bookworm/ui/server.py, unchanged), with one difference: Next page / Flip
plays the recorded path (`ui.flip_path`) moved with the book (scripts/flip_path_dynamic.py) instead of the
recorded joints. Until a reference is set, or while the book hasn't moved, that is the recorded joints exactly.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

from ..config import load_config
from .server import GOTO_DIR, Controller, LazyViamCamera, serve


class DynamicController(Controller):
    def _flip_path(self, version: str):
        """Controller._flip_path, with each recorded point carried by how far the book moved since it was taught."""
        conn = getattr(self.camera, "conn", None)
        if conn is None:
            raise RuntimeError("not connected to the robot yet")
        if str(GOTO_DIR) not in sys.path:
            sys.path.insert(0, str(GOTO_DIR))
        import flip_path
        import flip_path_dynamic
        from viam.components.arm import Arm

        pts, path, between = flip_path.load(version)
        poses = flip_path_dynamic.dynamic_plan(version, log=lambda m: print(m, flush=True))["poses"]
        tag = flip_path.step_tag(version)
        t0 = time.time()

        def on_step(name):
            print(f"STEP {tag}{name}  +{time.time() - t0:.2f}s", flush=True)
            self.state["flip_step"] = f"{tag}{name}"   # the page turns the on-screen leaf in step with this

        self._arm = conn.component(Arm, self.cfg["robot"].get("arm", "arm"))   # Stop calls arm.stop() on it
        try:
            code, msg = flip_path_dynamic.play(conn, self._arm, flip_path.make_plan(path, between), pts, poses,
                                               on_step=on_step, stopped=self._stop.is_set,
                                               log=lambda m: print(m, flush=True))
        except Exception:
            if self._stop.is_set():            # Stop halted the arm mid-move; that's not an error
                return
            raise
        print(f"[flip {version} dynamic] {msg}", flush=True)
        if code != 0 and not self._stop.is_set():
            raise RuntimeError(f"flip: {msg}")


def main(argv=None) -> None:
    from ..__main__ import LOCAL_CFG

    ap = argparse.ArgumentParser(description="Bookworm control page, with the flip path following the book")
    ap.add_argument("--config", help="YAML overriding config/default.yaml (default: config/local.yaml if present)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--images", help="use a folder of photos instead of the camera (Flip advances)")
    ap.add_argument("--out", help="scan folder (default scans/<timestamp>)")
    ap.add_argument("--quiet", action="store_true", help="print instead of speaking")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config or (LOCAL_CFG if LOCAL_CFG.exists() else None))
    if args.quiet:
        cfg["speech"]["engine"] = "print"
    if args.images:
        from ..vision import FileCamera

        camera = FileCamera(args.images)
    else:
        camera = LazyViamCamera(cfg)
    try:
        out = args.out or f"scans/{time.strftime('%Y%m%d-%H%M%S')}"
        serve(DynamicController(cfg, camera, args.config, out), args.port)
    finally:
        if hasattr(camera, "close"):
            camera.close()


if __name__ == "__main__":
    main()
