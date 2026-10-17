"""Measure the table and the book with the wrist depth camera, for motion planning.

    .venv/bin/python scripts/measure_workspace.py            capture from the RealSense (`cam`)
    .venv/bin/python scripts/measure_workspace.py --pcd runs/cam.pcd --pose runs/cam_pose.npy

Read-only: never moves the arm. Jog the arm so the wrist camera looks down at the whole book
first. The point cloud is moved into the world frame (the arm base) through Viam's frame
system; the table is the dominant flat level, and the book is the biggest block standing on
it. Results go to config/workspace.yaml (used by arm_recover.py as obstacles) and a top-down
height map to runs/workspace_topdown.png for checking by eye.

Known book size (measured by hand): 274 x 197 mm open spread, under 30 mm tall.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "config" / "workspace.yaml"
BOOK_MM = (274, 197, 30)            # hand-measured: spread length, page width, max height
CELL = 4.0                          # mm per top-down grid cell


def parse_pcd(data: bytes) -> np.ndarray:
    """Viam PCD (binary, x y z rgb, metres, camera frame) -> Nx3 mm, invalid points dropped."""
    i = data.index(b"DATA binary\n") + len(b"DATA binary\n")
    a = np.frombuffer(data[i:], dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<u4")]))
    p = np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float64) * 1000.0
    return p[np.isfinite(p).all(1) & (p[:, 2] > 50)]


def capture(conn=None) -> tuple[np.ndarray, np.ndarray]:
    """Point cloud (camera frame, mm) + world_from_camera. Reuses ``conn`` if given."""
    from bookworm.config import load_config
    from bookworm.hardware.viam_robot import ViamConnection, ViamRobot

    cfg = load_config(None)
    own = conn is None
    conn = conn or ViamConnection()
    try:
        robot = ViamRobot(conn, **dict(cfg["robot"], camera="cam", camera_calibrated=True))
        data, _ = conn.call(robot.cam.get_point_cloud(), timeout=60)
        T = robot.camera_pose()               # world_from_camera, mm
    finally:
        if own:
            conn.close()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (ROOT / "runs").mkdir(exist_ok=True)
    (ROOT / "runs" / f"cam-{stamp}.pcd").write_bytes(data)
    np.save(ROOT / "runs" / f"cam-{stamp}-pose.npy", T)
    print(f"saved runs/cam-{stamp}.pcd (+ -pose.npy) for re-running offline")
    return parse_pcd(data), T


def measure(pts_cam: np.ndarray, T: np.ndarray) -> dict:
    pts_cam = pts_cam[pts_cam[:, 2] > 250]        # nearer than 25 cm = the arm's own gripper/pencil
    w = pts_cam @ T[:3, :3].T + T[:3, 3]
    w = w[(np.abs(w[:, 0]) < 1200) & (np.abs(w[:, 1]) < 1200) & (w[:, 2] > -400) & (w[:, 2] < 600)]
    if len(w) < 1000:
        raise SystemExit("too few points in the workspace: is the camera looking at the table?")

    # table = the lowest big flat level (5 mm histogram peak holding >= 15% of the biggest one)
    z = w[:, 2]
    bins = np.arange(z.min(), z.max() + 5, 5)
    hist, edges = np.histogram(z, bins)
    peaks = [k for k in range(len(hist)) if hist[k] >= 0.15 * hist.max()
             and hist[k] >= hist[max(k - 1, 0)] and hist[k] >= hist[min(k + 1, len(hist) - 1)]]
    k = peaks[0]
    # fit a (possibly tilted) plane z = ax + by + c to the table, refining away the book
    sel = (z > edges[k] - 15) & (z < edges[k + 1] + 15)
    for _ in range(4):
        A = np.c_[w[sel, 0], w[sel, 1], np.ones(sel.sum())]
        coef, *_ = np.linalg.lstsq(A, z[sel], rcond=None)
        resid = z - (w[:, 0] * coef[0] + w[:, 1] * coef[1] + coef[2])
        sel = np.abs(resid) < 6
    tilt = float(np.degrees(np.arctan(np.hypot(coef[0], coef[1]))))
    z = resid                                       # from here on: height above the table plane
    table_z = float(coef[2] + coef[0] * np.median(w[sel, 0]) + coef[1] * np.median(w[sel, 1]))

    # book = largest connected block 4-80 mm above the table, seen from above
    above = w[(z > 6) & (z < 80)]
    above_h = z[(z > 6) & (z < 80)]
    x0, y0 = w[:, 0].min(), w[:, 1].min()
    shape = (int((w[:, 1].max() - y0) / CELL) + 1, int((w[:, 0].max() - x0) / CELL) + 1)
    # Count points per cell: the book is densely covered, depth noise (flying pixels at
    # edges) is sparse. Keep dense cells, drop specks, then close small holes.
    counts = np.zeros(shape, np.int32)
    np.add.at(counts, (((above[:, 1] - y0) / CELL).astype(int), ((above[:, 0] - x0) / CELL).astype(int)), 1)
    occ = (counts >= max(3, np.percentile(counts[counts > 0], 50))).astype(np.uint8) * 255
    occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    # strip thin things lying across the view (cables, the phone-mount bar): the book is
    # ~200 mm wide, a bar ~20-30 mm, so a 40 mm opening keeps only the book
    k = int(40 / CELL)
    occ = cv2.morphologyEx(occ, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(occ)
    if n < 2:
        raise SystemExit(f"table found at z={table_z:.0f} mm, but nothing standing on it in view")
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.nonzero(lab == big)
    (cx, cy), (a, b), ang = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32))
    cx, cy, a, b = x0 + cx * CELL, y0 + cy * CELL, a * CELL, b * CELL
    in_book = (np.abs(above[:, 0] - cx) < max(a, b) / 2) & (np.abs(above[:, 1] - cy) < max(a, b) / 2)
    top = table_z + (float(np.percentile(above_h[in_book], 98)) if in_book.any() else BOOK_MM[2])

    # everything else standing above the table (cables, mounts, hands) -> 50 mm voxel obstacles
    vox = 50.0
    rest = w[(z > 8)]
    rx, ry = rest[:, 0] - cx, rest[:, 1] - cy
    t = np.radians(ang if a >= b else ang + 90)
    u, v = rx * np.cos(t) + ry * np.sin(t), -rx * np.sin(t) + ry * np.cos(t)
    rest = rest[~((np.abs(u) < max(a, b) / 2 + 15) & (np.abs(v) < min(a, b) / 2 + 15) & (rest[:, 2] < top + 15))]
    keys, n_in = np.unique(np.floor(rest / vox).astype(int), axis=0, return_counts=True)
    obstacles = [[round(float(c), 1) for c in (k3 + 0.5) * vox] for k3, cnt in zip(keys, n_in) if cnt >= 40]

    # top-down height map for a visual check
    hm = np.zeros(shape, np.float32)
    hm[((w[:, 1] - y0) / CELL).astype(int), ((w[:, 0] - x0) / CELL).astype(int)] = np.clip(z, 0, 60)
    img = cv2.applyColorMap((hm / 60 * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    box = cv2.boxPoints((((cx - x0) / CELL, (cy - y0) / CELL), (a / CELL, b / CELL), ang)).astype(np.int32)
    cv2.polylines(img, [box], True, (0, 0, 255), 2)
    cv2.imwrite(str(ROOT / "runs" / "workspace_topdown.png"), cv2.flip(img, 0))   # +y up

    length, width = max(a, b), min(a, b)
    return {"table_z": round(table_z, 1), "table_tilt_deg": round(tilt, 2),
            "book": {"center": [round(float(cx), 1), round(float(cy), 1)], "length": round(float(length), 1),
                     "width": round(float(width), 1), "angle_deg": round(float(ang if a >= b else ang + 90), 1),
                     "top_z": round(float(top), 1)},
            "obstacle_voxel_mm": vox, "obstacles": obstacles,
            "points": int(len(w))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pcd", help="re-use a saved point cloud instead of capturing")
    ap.add_argument("--pose", help="the matching saved world_from_camera .npy")
    ap.add_argument("--save", action="store_true", help=f"write {OUT.relative_to(ROOT)}")
    args = ap.parse_args(argv)

    if args.pcd:
        pts, T = parse_pcd(Path(args.pcd).read_bytes()), np.load(args.pose)
    else:
        pts, T = capture()
    m = measure(pts, T)
    b = m["book"]
    print(f"table top     z = {m['table_z']:.0f} mm (world frame = arm base), tilted {m['table_tilt_deg']:.1f} deg")
    print(f"book          centre ({b['center'][0]:.0f}, {b['center'][1]:.0f}) mm, "
          f"{b['length']:.0f} x {b['width']:.0f} mm at {b['angle_deg']:.0f} deg, top z = {b['top_z']:.0f} mm "
          f"({b['top_z'] - m['table_z']:.0f} mm tall)")
    print(f"hand-measured {BOOK_MM[0]} x {BOOK_MM[1]} mm, < {BOOK_MM[2]} mm tall")
    print(f"other obstacles: {len(m['obstacles'])} blocks of {m['obstacle_voxel_mm']:.0f} mm (cables, mounts, ...)")
    print("check runs/workspace_topdown.png (red box = book)")
    if args.save:
        m["measured"] = time.strftime("%Y-%m-%d %H:%M")
        m["book_hand_measured_mm"] = list(BOOK_MM)
        OUT.write_text("# Written by scripts/measure_workspace.py (wrist depth camera). Obstacles for arm_recover.py.\n"
                       + yaml.safe_dump(m, sort_keys=False))
        print(f"saved {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
