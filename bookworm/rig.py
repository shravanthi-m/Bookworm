"""Build camera / actuators from config."""
from __future__ import annotations

from .hardware import MockActuators


def build_rig(cfg: dict, images_dir: str | None = None, mock: bool = False):
    """Returns (camera, actuators, close_fn).

    ``images_dir`` replays a folder of photos instead of a live camera;
    ``mock`` fakes the servos. Either way, Viam is only contacted if needed.
    """
    viam_cam = not images_dir and cfg["camera"]["source"] == "viam"
    viam_hw = not mock and cfg["hardware"]["backend"] == "viam"
    conn = None
    if viam_cam or viam_hw:
        from .hardware.viam_link import ViamConnection

        conn = ViamConnection()

    if images_dir:
        from .vision import FileCamera

        camera = FileCamera(images_dir)
    elif viam_cam:
        from .hardware.viam_link import ViamCamera

        camera = ViamCamera(conn, cfg["camera"]["viam_name"])
    else:
        from .vision import Camera

        c = cfg["camera"]
        camera = Camera(c["index"], c["width"], c["height"], c["warmup_frames"])

    if viam_hw:
        from .hardware.viam_link import ViamActuators

        act = ViamActuators(conn, **cfg["hardware"])
    else:
        act = MockActuators(on_flip=getattr(camera, "advance", None), verbose=True)

    def close():
        try:
            act.close()
            camera.close()
        finally:
            if conn:
                conn.close()

    return camera, act, close
