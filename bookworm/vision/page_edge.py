"""Locate the book and page edge in a wrist-camera frame with a VLM (Moondream Cloud).

The VLM gives a coarse target; refine it with depth/edge analysis before moving the arm.
Needs MOONDREAM_API_KEY in .env.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import numpy as np

DEFAULT_QUERIES = {
    "book": "open book",
    "page_corner": "top corner of the page edge",
    "page_edge": "edge of the top page",
}


@dataclass
class EdgeResult:
    """Pixel coordinates in the input frame."""
    boxes: dict[str, list[tuple[float, float, float, float]]] = field(default_factory=dict)  # x0, y0, x1, y1
    points: dict[str, list[tuple[float, float]]] = field(default_factory=dict)


class MoondreamEdgeFinder:
    def __init__(self, api_key: str | None = None, queries: dict[str, str] | None = None):
        import moondream as md
        from dotenv import load_dotenv

        load_dotenv()
        key = (api_key or os.getenv("MOONDREAM_API_KEY") or "").strip()
        if not key:
            raise RuntimeError("MOONDREAM_API_KEY not set — add it to .env")
        self.model = md.vl(api_key=key)
        self.queries = queries or DEFAULT_QUERIES

    def find(self, frame_bgr: np.ndarray, which: tuple[str, ...] | None = None, max_side: int = 768) -> EdgeResult:
        """One API call per name in ``which`` (default: all queries). Frames are downscaled to ``max_side``
        before upload; results are normalized, so they map back to the original size unchanged."""
        from PIL import Image

        h, w = frame_bgr.shape[:2]
        small = frame_bgr
        if max(h, w) > max_side:
            s = max_side / max(h, w)
            small = cv2.resize(frame_bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        image = self.model.encode_image(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
        out = EdgeResult()
        for name, prompt in self.queries.items():
            if which is not None and name not in which:
                continue
            if name == "book":      # a region, not a point
                objs = self.model.detect(image, prompt)["objects"]
                out.boxes[name] = [(o["x_min"] * w, o["y_min"] * h, o["x_max"] * w, o["y_max"] * h)
                                   for o in objs]
            else:
                pts = self.model.point(image, prompt)["points"]
                out.points[name] = [(p["x"] * w, p["y"] * h) for p in pts]
        return out


def draw(frame_bgr: np.ndarray, res: EdgeResult) -> np.ndarray:
    img = frame_bgr.copy()
    for name, boxes in res.boxes.items():
        for x0, y0, x1, y1 in boxes:
            cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (255, 160, 0), 2)
            cv2.putText(img, name, (int(x0), max(14, int(y0) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 160, 0), 2)
    colors = [(0, 0, 255), (0, 200, 0), (255, 0, 255)]
    for i, (name, pts) in enumerate(res.points.items()):
        for x, y in pts:
            cv2.circle(img, (int(x), int(y)), 9, colors[i % 3], -1)
            cv2.putText(img, name, (int(x) + 12, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i % 3], 2)
    return img
