"""Scan mode: non-destructive digitisation. Saves page images, text, a manifest and a PDF.

Output layout:
    <out>/pages/0001.png  0001.txt  ...     one file pair per page, in reading order
    <out>/book.txt                          all text with page markers
    <out>/book.pdf                          page images as a PDF
    <out>/manifest.json                     page numbers, OCR confidence, flip problems
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2


class Scanner:
    def __init__(self, out_dir: str | Path, cfg: dict):
        self.out = Path(out_dir)
        (self.out / "pages").mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.pages: list[dict] = []
        self.events: list[dict] = []

    def on_spread(self, index, spread, results):
        save_color = self.cfg["modes"]["scan"].get("save_color", True)
        for side in ("left", "right"):
            if side not in spread.pages:
                continue
            n = len(self.pages) + 1
            page, res = spread.pages[side], results[side]
            stem = self.out / "pages" / f"{n:04d}"
            cv2.imwrite(str(stem.with_suffix(".png")), page.color if save_color else page.ocr)
            stem.with_suffix(".txt").write_text(res.text + "\n")
            printed = None
            if spread.page_no is not None:
                printed = spread.page_no + (0 if side == "left" else 1)
            self.pages.append({"file": f"pages/{n:04d}.png", "spread": index, "side": side,
                               "printed_page": printed, "ocr_conf": round(res.mean_conf, 1),
                               "words": res.word_count})
            print(f"[scan] page {n} ({side}, printed {printed}, {res.word_count} words)")

    def on_event(self, event):
        self.events.append({"after_page": len(self.pages), "event": event})
        print(f"[scan] WARNING: {event} after page {len(self.pages)}")

    def finish(self, summary):
        texts = []
        for p in self.pages:
            txt = (self.out / p["file"]).with_suffix(".txt").read_text()
            label = p["printed_page"] if p["printed_page"] is not None else "?"
            texts.append(f"----- page {label} ({p['file']}) -----\n{txt}")
        (self.out / "book.txt").write_text("\n".join(texts))
        self._write_pdf()
        manifest = {"pages": self.pages, "events": self.events,
                    "spreads": summary.spreads, "stopped_because": summary.stopped_because}
        (self.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"[scan] {len(self.pages)} pages saved to {self.out}/")

    def _write_pdf(self):
        if not self.pages:
            return
        from PIL import Image

        imgs = [Image.open(self.out / p["file"]).convert("RGB") for p in self.pages]
        imgs[0].save(self.out / "book.pdf", save_all=True, append_images=imgs[1:])
