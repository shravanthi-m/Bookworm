"""Grab colour frames from the Viam wrist camera at a steady rate. Read-only: never moves the arm.

    .venv\\Scripts\\python scripts\\capture_burst.py --seconds 10 --fps 5 --out temp\\burst1
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bookworm.hardware.viam_robot import ViamConnection  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seconds", type=float, default=10)
ap.add_argument("--fps", type=float, default=5, help="target rate; actual rate is limited by the connection")
ap.add_argument("--camera", default="cam")
ap.add_argument("--out", default="temp/burst1")
args = ap.parse_args()

from viam.components.camera import Camera  # noqa: E402

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
t0 = time.time()
conn = ViamConnection()
cam = conn.component(Camera, args.camera)
print(f"connected in {time.time() - t0:.1f}s")

log, period, start = [], 1.0 / args.fps, time.time()
try:
    while time.time() - start < args.seconds:
        t = time.time()
        try:
            images, _ = conn.call(cam.get_images(filter_source_names=["color"]))
        except TypeError:                     # older SDK without the filter argument
            images, _ = conn.call(cam.get_images())
        color = next(im for im in images if not im.mime_type.startswith("image/vnd.viam.dep"))
        name = f"frame_{len(log):03d}.jpg"
        (out / name).write_bytes(color.data)
        log.append({"file": name, "t": round(t - start, 3), "fetch_s": round(time.time() - t, 3)})
        time.sleep(max(0.0, period - (time.time() - t)))
finally:
    conn.close()

(out / "frames.json").write_text(json.dumps(log, indent=1))
fetch = sorted(f["fetch_s"] for f in log)
print(f"{len(log)} frames in {args.seconds:.0f}s -> {out}")
if fetch:
    print(f"fetch time per frame: median {fetch[len(fetch) // 2]:.3f}s, max {fetch[-1]:.3f}s")
