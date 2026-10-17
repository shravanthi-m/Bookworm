"""Draw data/labels/labels.json onto the frames -> temp/labels_check/. Run from the repo root."""
import json
from pathlib import Path

import cv2

out = Path("temp/labels_check")
out.mkdir(parents=True, exist_ok=True)
for path, lab in json.loads(Path("data/labels/labels.json").read_text()).items():
    if path.startswith("_"):
        continue
    img = cv2.imread(path)
    x0, y0, x1, y1 = lab["book_box"]
    cv2.rectangle(img, (x0, y0), (x1, y1), (255, 160, 0), 2)
    for p in lab["page_corners"]:
        cv2.circle(img, tuple(p), 8, (0, 0, 255), -1)
    a, b = lab["free_edge"]
    cv2.line(img, tuple(a), tuple(b), (0, 220, 0), 3)
    cv2.drawMarker(img, tuple(lab["grasp_point"]), (255, 0, 255), cv2.MARKER_CROSS, 30, 3)
    cv2.putText(img, f"lifted={lab['leaves_lifted']}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imwrite(str(out / (Path(path).stem + ".jpg")), img)
print("wrote", out)
