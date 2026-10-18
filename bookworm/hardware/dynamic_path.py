"""The recorded page-flip path (config/waypoints.yaml), made dynamic: it follows the book.

The path was taught by hand with the book in one place. Every pose in it is a fixed point in the arm's world, so it
only works while the book stays exactly there. Here the path is moved with the book instead:

    reference book frame   where the book was when the path was taught (set once, `set_reference`)
    current book frame     where the book is now (the workshop's first capture + Z link)

and each recorded pose is carried by the rigid motion from one to the other: turned about world z by the change in
the book's angle, shifted by the change in its centre, raised or lowered by the change in its top height. The tool's
orientation turns with it, so the eraser meets the page at the same angle as when it was taught.

A book frame is {center: [x, y], angle_deg, top_z}: the centre of the open book, the direction from the left
page's outer edge toward the right page's outer edge (degrees from world +x), and the page-top z, all in the arm's
world frame (mm). When the book has not moved (within `still_mm` / `still_deg`) the recorded joints are played
unchanged, exactly as taught.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FRAME_FILE = ROOT / "data" / "book_frame.json"          # the current book frame, written by the workshop GUI
POSE_KEYS = ("x", "y", "z", "o_x", "o_y", "o_z", "theta")


# ---- book frames ------------------------------------------------------------------------------------------------
def frame_from_points(points: dict, top_z: float, source: str = "") -> dict:
    """The book frame from the six first-capture points (world x, y[, z]): tl, bl = left page's outer corners,
    tr, br = right page's. ``top_z``: the page-top z (the Z link's book height)."""
    p = {k: np.asarray(points[k][:2], float) for k in ("tl", "tr", "br", "bl")}
    centre = (p["tl"] + p["tr"] + p["br"] + p["bl"]) / 4
    across = (p["tr"] + p["br"]) / 2 - (p["tl"] + p["bl"]) / 2
    return {"center": [round(float(centre[0]), 2), round(float(centre[1]), 2)],
            "angle_deg": round(math.degrees(math.atan2(across[1], across[0])), 3), "top_z": round(float(top_z), 2),
            "measured": time.strftime("%Y-%m-%d %H:%M:%S"), "source": source}


def save_frame(frame: dict, path: Path = FRAME_FILE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(frame, indent=1), encoding="utf-8")
    return path


def load_frame(path: Path = FRAME_FILE) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


LOCAL_CFG = ROOT / "config" / "local.yaml"             # the reference is rig-specific: kept with the rig's calibration


def load_reference(path: Path = LOCAL_CFG) -> dict | None:
    import yaml

    try:
        d = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except OSError:
        return None
    return (d.get("dynamic_path") or {}).get("reference")


def set_reference(frame: dict, path: Path = LOCAL_CFG) -> None:
    """Remember ``frame`` as where the book was when the path was taught (the flip works with the book there)."""
    import yaml

    path = Path(path)
    d = (yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else None) or {}
    d.setdefault("dynamic_path", {})["reference"] = dict(frame)
    path.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")


def motion(ref: dict, now: dict) -> dict:
    """How the book moved from ``ref`` to ``now``: {dx, dy, dz, dtheta_deg, shift_mm}."""
    d = (now["angle_deg"] - ref["angle_deg"] + 180.0) % 360.0 - 180.0
    dx, dy = now["center"][0] - ref["center"][0], now["center"][1] - ref["center"][1]
    return {"dx": dx, "dy": dy, "dz": now["top_z"] - ref["top_z"], "dtheta_deg": d, "shift_mm": math.hypot(dx, dy)}


# ---- Viam orientation vectors <-> rotation matrices (Viam's convention: R = Rz(lon) Ry(lat) Rz(theta)) -------------
def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def ov_to_matrix(o_x, o_y, o_z, theta_deg) -> np.ndarray:
    v = np.array([o_x, o_y, o_z], float)
    v /= np.linalg.norm(v)
    lat = math.acos(max(-1.0, min(1.0, v[2])))
    lon = math.atan2(v[1], v[0]) if 1 - abs(v[2]) > 1e-4 else 0.0
    return _rz(lon) @ _ry(lat) @ _rz(math.radians(theta_deg))


def matrix_to_ov(R: np.ndarray) -> tuple[float, float, float, float]:
    v = R[:, 2]
    lat = math.acos(max(-1.0, min(1.0, v[2])))
    lon = math.atan2(v[1], v[0]) if 1 - abs(v[2]) > 1e-4 else 0.0
    M = _ry(-lat) @ _rz(-lon) @ R
    return float(v[0]), float(v[1]), float(v[2]), math.degrees(math.atan2(M[1, 0], M[0, 0]))


def move_pose(pose: dict, ref: dict, now: dict) -> dict:
    """``pose`` (a flange pose taught with the book at ``ref``) carried to the book at ``now``."""
    m = motion(ref, now)
    Rz = _rz(math.radians(m["dtheta_deg"]))
    c0 = np.array([ref["center"][0], ref["center"][1], 0.0])
    c1 = np.array([now["center"][0], now["center"][1], 0.0])
    p = Rz @ (np.array([pose["x"], pose["y"], pose["z"]], float) - c0) + c1
    o = matrix_to_ov(Rz @ ov_to_matrix(pose["o_x"], pose["o_y"], pose["o_z"], pose["theta"]))
    return {"x": float(p[0]), "y": float(p[1]), "z": float(p[2] + m["dz"]),
            "o_x": o[0], "o_y": o[1], "o_z": o[2], "theta": o[3]}


# ---- the plan ---------------------------------------------------------------------------------------------------
def plan(waypoints: dict, path_key: str, ref: dict | None, now: dict | None, *, still_mm: float = 1.5,
         still_deg: float = 0.3, max_mm: float = 120.0, max_deg: float = 25.0) -> dict:
    """{"mode": "recorded"|"dynamic", "why", "motion", "poses": {name: pose}} for the recorded points of ``path_key``
    plus `start` and `out_of_the_way`. "recorded" (play the taught joints as they are) when there is no reference
    or no current frame, or the book is where it was taught. Raises ValueError when the book moved further than
    ``max_mm`` / ``max_deg`` (too far for a taught path to still make sense: teach it again)."""
    if ref is None or now is None:
        return {"mode": "recorded", "why": "no reference book position" if ref is None else "no current book frame",
                "motion": None, "poses": {}}
    m = motion(ref, now)
    if m["shift_mm"] <= still_mm and abs(m["dtheta_deg"]) <= still_deg and abs(m["dz"]) <= still_mm:
        return {"mode": "recorded", "why": "the book is where the path was taught", "motion": m, "poses": {}}
    if m["shift_mm"] > max_mm or abs(m["dtheta_deg"]) > max_deg:
        raise ValueError(f"the book moved {m['shift_mm']:.0f} mm and turned {m['dtheta_deg']:+.1f} deg since the path "
                         f"was taught (limit {max_mm:g} mm, {max_deg:g} deg): move it back or teach the path again")
    pts = {str(p["name"]): p["pose"] for p in waypoints[path_key] if "pose" in p}
    for name in ("start", "out_of_the_way"):
        if name in waypoints and "pose" in waypoints[name]:
            pts[name] = waypoints[name]["pose"]
    return {"mode": "dynamic", "why": f"book moved {m['shift_mm']:.1f} mm, turned {m['dtheta_deg']:+.2f} deg, "
                                      f"{m['dz']:+.1f} mm in height", "motion": m,
            "poses": {k: move_pose(v, ref, now) for k, v in pts.items()}}
