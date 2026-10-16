"""Vision test bench: run the full vision pipeline on photos or the live wrist camera,
with no arm motion. Produces debug images and an accuracy report.

    python -m bookworm vision --images photos/          # offline, any photos of open books
    python -m bookworm vision --live --frames 5         # Viam wrist camera at its current pose

Ground truth (optional): next to ``photo.jpg`` put ``photo.left.txt`` / ``photo.right.txt``
with the true page text; the report then includes word accuracy per page.
Images are processed in sorted order and consecutive pairs are also run through
flip verification, so a folder of consecutive spreads doubles as a flip-check test.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from .geometry import book_quad_pixels
from .page_detect import find_paper_corners, locate_right_corner
from .preprocess import glare_fraction, process_shot, sharpness
from .verify import classify_flip


def word_accuracy(truth: str, hyp: str) -> float:
    """1 - word error rate (Levenshtein over lowercase words), floored at 0."""
    t, h = truth.lower().split(), hyp.lower().split()
    if not t:
        return 1.0 if not h else 0.0
    prev = list(range(len(h) + 1))
    for i, tw in enumerate(t, 1):
        cur = [i] + [0] * len(h)
        for j, hw in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (tw != hw))
        prev = cur
    return max(0.0, 1 - prev[-1] / len(t))


def _panel(frame, pages, outline_px=None, quad_px=None, h=500):
    """Side-by-side debug image: raw frame with detections | colour pages | OCR-input pages."""
    raw = frame.copy()
    if outline_px is not None:
        cv2.polylines(raw, [outline_px.astype(np.int32)], True, (0, 255, 255), 4)   # detected paper
    if quad_px is not None:
        cv2.polylines(raw, [quad_px.astype(np.int32)], True, (255, 0, 255), 4)      # calibrated book
    tiles = [raw] + [p.color for p in pages.values()] + [cv2.cvtColor(p.ocr, cv2.COLOR_GRAY2BGR)
                                                         for p in pages.values()]
    tiles = [cv2.resize(t, (max(1, int(t.shape[1] * h / t.shape[0])), h)) for t in tiles]
    return np.hstack(tiles)


class VisionBench:
    def __init__(self, cfg: dict, ocr, out_dir: Path, region: str = "spread", book=None):
        self.cfg, self.ocr, self.out, self.region, self.book = cfg, ocr, out_dir, region, book
        self.out.mkdir(parents=True, exist_ok=True)
        self.results: list[dict] = []
        self._prev = None

    def process(self, name: str, frame, geom=None, truth_dir: Path | None = None) -> dict:
        v, b = self.cfg["vision"], self.cfg["book"]
        t0 = time.perf_counter()
        quad = size = None
        if geom is not None and self.book is not None:
            from ..hardware import page_quad

            ppm = self.cfg["modes"]["read"]["px_per_mm"]
            quad_mm = page_quad(self.region, b["page_width_mm"], b["page_height_mm"])
            quad = book_quad_pixels(self.book, geom, quad_mm)
            size = ((quad_mm[1, 0] - quad_mm[0, 0]) * ppm, b["page_height_mm"] * ppm)
        pages = process_shot(frame, self.region, v, quad, size)
        t_pre = time.perf_counter() - t0

        rec = {"image": name, "sharpness": round(sharpness(frame), 1),
               "sharp_ok": sharpness(frame) >= v["min_sharpness"],
               "glare_pct": round(100 * glare_fraction(frame), 2),
               "geometry": quad is not None, "pages": {}}
        outline = find_paper_corners(frame)
        rec["outline_found"] = outline is not None
        if geom is not None and self.book is not None:
            rec["corner_mm"] = locate_right_corner(frame, geom, self.book, b, v["corner_search_mm"])

        t1 = time.perf_counter()
        for side, page in pages.items():
            res = self.ocr.read(page.ocr)
            pr = {"words": res.word_count, "conf": round(res.mean_conf, 1),
                  "page_no": self.ocr.page_number(page.ocr, v["page_number_region"]),
                  "readable": res.mean_conf >= self.cfg["ocr"]["min_page_conf"]}
            truth = truth_dir / f"{Path(name).stem}.{side}.txt" if truth_dir else None
            if truth and truth.exists():
                pr["word_accuracy"] = round(word_accuracy(truth.read_text(), res.text), 3)
            (self.out / f"{Path(name).stem}.{side}.txt").write_text(res.text + "\n")
            rec["pages"][side] = pr
        rec["ms_preprocess"] = round(t_pre * 1000)
        rec["ms_ocr"] = round((time.perf_counter() - t1) * 1000)

        side = next(iter(pages))
        page_no = rec["pages"][side]["page_no"]
        if self._prev is not None:
            prev_bin, prev_no = self._prev
            rec["flip_vs_previous"] = classify_flip(prev_bin, pages[side].ocr, prev_no, page_no,
                                                    self.cfg["verify"]["change_threshold"]).value
        self._prev = (pages[side].ocr, page_no)

        cv2.imwrite(str(self.out / f"{Path(name).stem}.panel.jpg"), _panel(frame, pages, outline, quad))
        self.results.append(rec)
        self._print(rec)
        return rec

    @staticmethod
    def _print(rec):
        pages = "  ".join(f"{s}: {p['words']}w conf {p['conf']} p#{p['page_no']}"
                          + (f" acc {p['word_accuracy']:.0%}" if "word_accuracy" in p else "")
                          for s, p in rec["pages"].items())
        flip = f"  flip={rec['flip_vs_previous']}" if "flip_vs_previous" in rec else ""
        print(f"{rec['image']}: sharp {rec['sharpness']}{'' if rec['sharp_ok'] else ' (BLURRY)'}  "
              f"glare {rec['glare_pct']}%  "
              f"{pages}{flip}  [{rec['ms_preprocess']}+{rec['ms_ocr']} ms]")

    def report(self) -> dict:
        pages = [p for r in self.results for p in r["pages"].values()]
        accs = [p["word_accuracy"] for p in pages if "word_accuracy" in p]
        flips = [r["flip_vs_previous"] for r in self.results if "flip_vs_previous" in r]
        summary = {
            "images": len(self.results),
            "sharp_ok": sum(r["sharp_ok"] for r in self.results),
            "pages": len(pages),
            "readable_pages": sum(p["readable"] for p in pages),
            "page_number_found": sum(p["page_no"] is not None for p in pages),
            "mean_conf": round(float(np.mean([p["conf"] for p in pages])), 1) if pages else None,
            "mean_word_accuracy": round(float(np.mean(accs)), 3) if accs else None,
            "flip_outcomes": {k: flips.count(k) for k in sorted(set(flips))},
            "mean_ms_per_image": round(float(np.mean([r["ms_preprocess"] + r["ms_ocr"] for r in self.results])))
            if self.results else None,
        }
        (self.out / "results.json").write_text(json.dumps({"summary": summary, "images": self.results},
                                                          indent=2, default=str))
        print("\nSUMMARY", json.dumps(summary, indent=2))
        print(f"debug panels + text in {self.out}/")
        return summary
