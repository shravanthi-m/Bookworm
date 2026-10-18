"""The book in the arm's space, driven by where the tip has actually touched it.

    page z (arm frame) at map (x, y) = level(x, y), inside the book outline; the table's z outside it

``level`` is fitted to the page touches (their touched z, pressure contacts):
  * 1-2 page touches -> their mean, as a constant over the book
  * 3+ page touches  -> a plane a + b*x + c*y through them (only when they are spread out in both x and y)
The depth map is only a reference: it supplies the book outline, and each touch records what the depth map said there (so the
disagreement is visible) but never moves the model. With no page touch at all the model falls back to the depth map
(``driven`` = "depth").
The table touch is not a page touch: it defines the table's z.

x, y are in the depth map's (camera's) coordinates. The arm lands at (x, y) - xy_offset (kept in the file).

On disk (one folder per model): book_model.json (everything but the grid) + book_model.npz (x, y, z_arm, height, book).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

PAGE_KINDS = ("page_check", "demo")


def _median_at(xs, ys, z, x, y, r=1, min_n=3):
    cell = float(xs[1] - xs[0])
    i, j = int(round((x - xs[0]) / cell)), int(round((y - ys[0]) / cell))
    if not (0 <= i < len(xs) and 0 <= j < len(ys)):
        return None
    p = z[max(i - r, 0):i + r + 1, max(j - r, 0):j + r + 1]
    p = p[np.isfinite(p)]
    return float(np.median(p)) if p.size >= min_n else None


def fit_level(touches: list[dict]) -> dict:
    """The page level from the page touches' z: {kind: none|constant|plane, a, b, c, n, spread, rms_after}; z = a + b*x + c*y."""
    pts = [t for t in touches if t.get("kind") in PAGE_KINDS]
    if not pts:
        return {"kind": "none", "a": 0.0, "b": 0.0, "c": 0.0, "n": 0}
    x = np.array([t["x"] for t in pts], float)
    y = np.array([t["y"] for t in pts], float)
    z = np.array([t["z"] for t in pts], float)
    out = {"kind": "constant", "a": float(z.mean()), "b": 0.0, "c": 0.0, "n": len(pts)}
    if len(pts) >= 3:
        A = np.c_[np.ones_like(x), x - x.mean(), y - y.mean()]
        sv = np.linalg.svd(A[:, 1:], compute_uv=False)
        if sv.min() / np.sqrt(len(pts)) > 20.0:                     # spread at least ~20 mm across both directions
            a, b, c = np.linalg.lstsq(A, z, rcond=None)[0]
            out = {"kind": "plane", "a": float(a - b * x.mean() - c * y.mean()), "b": float(b), "c": float(c), "n": len(pts)}
    fit = out["a"] + out["b"] * x + out["c"] * y
    out["spread"] = round(float(np.sqrt(np.mean((z - z.mean()) ** 2))), 2)
    out["rms_after"] = round(float(np.sqrt(np.mean((z - fit) ** 2))), 2)
    return out


def fit_plane(x, y, z) -> dict | None:
    """Least-squares plane z = a + b*x + c*y; its tilt from level and how far the surface is from it."""
    ok = np.isfinite(z)
    if ok.sum() < 20:
        return None
    x, y, z = x[ok], y[ok], z[ok]
    A = np.c_[np.ones_like(x), x, y]
    a, b, c = np.linalg.lstsq(A, z, rcond=None)[0]
    d = z - A @ [a, b, c]
    n = np.array([-b, -c, 1.0]) / np.sqrt(b * b + c * c + 1.0)
    return {"a": float(a), "b": float(b), "c": float(c), "normal": [round(float(v), 4) for v in n],
            "tilt_deg": round(float(np.degrees(np.arccos(n[2]))), 2), "rms_mm": round(float(np.sqrt(np.mean(d ** 2))), 2),
            "max_off_mm": round(float(np.abs(d).max()), 1), "z_mid": round(float(np.median(z)), 1)}


def build(res: dict, table_z: float, touches: list[dict], *, points: dict | None = None, meta: dict | None = None) -> dict:
    """``res``: the fused depth map (x, y, z = height above the table, book mask). ``touches``: [{key, kind, x, y, z}] with
    x, y in map coordinates and z the touched tip z in the arm frame. ``points``: {key: [x, y]} the six book points."""
    xs, ys, h, bk = res["x"], res["y"], res["z"], res["book"].astype(bool)
    tt = []
    for t in touches:
        d = _median_at(xs, ys, h, t["x"], t["y"], r=1, min_n=3)       # what the depth map says there (reference only)
        ref = None if d is None else d + table_z
        tt.append({**t, "depth_z": None if ref is None else round(ref, 2),
                   "depth_off": None if ref is None or t["kind"] == "table" else round(float(t["z"]) - ref, 2)})
    lvl = fit_level(tt)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    if lvl["kind"] == "none":                                         # no touch on the page yet: the depth map, as measured
        z_arm, driven = h + table_z, "depth"
    else:
        z_arm, driven = np.where(bk, lvl["a"] + lvl["b"] * X + lvl["c"] * Y, table_z), "touch"
    for t in tt:
        if t["kind"] in PAGE_KINDS and lvl["kind"] != "none":
            t["after"] = round(float(t["z"]) - (lvl["a"] + lvl["b"] * t["x"] + lvl["c"] * t["y"]), 2)
    pages = {}
    if points and "cl" in points and "cr" in points:              # split the book at the line half way between the centres
        cl, cr = np.array(points["cl"][:2], float), np.array(points["cr"][:2], float)
        u = (cr - cl) / np.linalg.norm(cr - cl)
        s = (X - (cl[0] + cr[0]) / 2) * u[0] + (Y - (cl[1] + cr[1]) / 2) * u[1]
        for name, m in (("left", bk & (s < -10)), ("right", bk & (s > 10))):   # 10 mm clear of the spine on each side
            pl = fit_plane(X[m], Y[m], z_arm[m])
            if pl is not None:
                pages[name] = pl
    whole = fit_plane(X[bk], Y[bk], z_arm[bk])
    return {"version": 1, "built": time.strftime("%Y-%m-%d %H:%M:%S"), "frame": "arm world, mm; x, y in depth-map coordinates",
            "driven": driven, "table_z": round(float(table_z), 2), "level": lvl, "touches": tt, "points": points or {},
            "pages": pages, "book_plane": whole, **(meta or {}),
            "grid": {"x": xs, "y": ys, "z_arm": z_arm, "height": h, "book": bk}}


def z_at(model: dict, x: float, y: float) -> float | None:
    """Page z in the arm frame at map (x, y): median of the 3x3 cells around it."""
    g = model["grid"]
    return _median_at(g["x"], g["y"], g["z_arm"], x, y, r=1, min_n=3)


def save(model: dict, folder: Path) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    g = model["grid"]
    np.savez_compressed(folder / "book_model.npz", x=g["x"], y=g["y"], z_arm=g["z_arm"], height=g["height"], book=g["book"])
    rec = {k: v for k, v in model.items() if k != "grid"}
    rec["grid"] = {"file": "book_model.npz", "cell_mm": float(g["x"][1] - g["x"][0]), "shape": list(g["z_arm"].shape),
                   "x_range": [float(g["x"][0]), float(g["x"][-1])], "y_range": [float(g["y"][0]), float(g["y"][-1])]}
    (folder / "book_model.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return folder


def load(folder: Path) -> dict:
    folder = Path(folder)
    model = json.loads((folder / "book_model.json").read_text(encoding="utf-8"))
    g = np.load(folder / "book_model.npz")
    model["grid"] = {k: g[k] for k in ("x", "y", "z_arm", "height")} | {"book": g["book"].astype(bool)}
    return model


def summary(model: dict) -> dict:
    """Everything but the grid, for the GUI."""
    return {k: v for k, v in model.items() if k != "grid"}
