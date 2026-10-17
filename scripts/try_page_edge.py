"""Run the Moondream edge finder on saved frames and write annotated copies. Never touches the arm.

    .venv\\Scripts\\python scripts\\try_page_edge.py temp\\armfarm22_color.jpg temp\\burst\\frame_1.jpg
"""
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bookworm.vision.page_edge import MoondreamEdgeFinder, draw  # noqa: E402

finder = MoondreamEdgeFinder()
for path in map(Path, sys.argv[1:]):
    frame = cv2.imread(str(path))
    res = finder.find(frame)
    out = path.with_name(path.stem + "_edges.jpg")
    cv2.imwrite(str(out), draw(frame, res))
    print(path.name, "->", out.name)
    print("  boxes :", {k: [tuple(round(v) for v in b) for b in bs] for k, bs in res.boxes.items()})
    print("  points:", {k: [tuple(round(v) for v in p) for p in ps] for k, ps in res.points.items()})
