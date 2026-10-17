"""Check edge refinement against the hand labels: start from a deliberately off hint, see where it lands.

    .venv\\Scripts\\python scripts\\eval_refine.py        # writes temp/refine_check/*.jpg
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bookworm.vision.edge_refine import draw_refined, refine_edge  # noqa: E402

out = Path("temp/refine_check")
out.mkdir(parents=True, exist_ok=True)
OFFSET = np.array([22.0, -16.0])        # pretend the VLM / click was this far off

for path, lab in json.loads(Path("data/labels/labels.json").read_text()).items():
    if path.startswith("_"):
        continue
    img = cv2.imread(path)
    hint = np.array(lab["grasp_point"], float) + OFFSET
    r = refine_edge(img, hint)
    name = Path(path).stem
    if r is None:
        print(f"{name:18s} no edge found")
        cv2.imwrite(str(out / f"{name}.jpg"), img)
        continue
    a, b = (np.array(p, float) for p in lab["free_edge"])
    u = (b - a) / np.linalg.norm(b - a)
    perp = lambda p: abs((np.array(p) - a)[0] * u[1] - (np.array(p) - a)[1] * u[0])  # noqa: E731
    g_err = np.linalg.norm(np.array(r.grasp) - np.array(lab["grasp_point"]))
    print(f"{name:18s} score {r.score:.2f}  grasp off label by {g_err:5.1f}px  "
          f"edge ends off label line by {perp(r.edge[0]):4.1f} / {perp(r.edge[1]):4.1f}px  "
          f"(hint was {np.linalg.norm(OFFSET):.0f}px off)")
    cv2.imwrite(str(out / f"{name}.jpg"), draw_refined(img, r, hint))
