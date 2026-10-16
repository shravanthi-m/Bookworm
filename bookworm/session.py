"""Core page loop shared by both modes:  capture -> OCR -> handler -> flip -> verify -> repeat.

A mode (read aloud / scan) is a handler object with:
    on_spread(index, spread, ocr_results)   called once per spread
    on_event(message)                        flip problems, end of book
    finish(summary)                          after the loop
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from .hardware import BookFrame, HardwareError, Robot, page_quad, run_flip
from .vision import (FlipOutcome, Page, book_quad_pixels, classify_flip, locate_right_corner,
                     process_shot, sharpness)

log = logging.getLogger("bookworm")


@dataclass
class Shot:
    region: str
    frame: object
    geom: object          # CameraGeometry or None


@dataclass
class Spread:
    pages: dict[str, Page]
    shots: list[Shot]
    page_no: int | None   # left-page number (None if not found)


@dataclass
class RunSummary:
    spreads: int = 0
    flips: dict = field(default_factory=dict)   # FlipOutcome.value -> count
    stopped_because: str = ""


class BookSession:
    def __init__(self, robot: Robot, ocr, handler, cfg: dict, mode: str):
        self.robot, self.ocr, self.handler, self.cfg = robot, ocr, handler, cfg
        self.mcfg = cfg["modes"][mode]
        self.book = BookFrame.from_config(cfg["book"])
        self.flips_done = 0
        self.run_dir = None
        if cfg["run"].get("save_dir"):
            self.run_dir = Path(cfg["run"]["save_dir"]) / f"{time.strftime('%Y%m%d-%H%M%S')}-{mode}"
            self.run_dir.mkdir(parents=True, exist_ok=True)

    # ---- capture ------------------------------------------------------------
    def _goto(self, pose_name: str) -> None:
        pose = self.cfg["poses"].get(pose_name)
        if pose:
            self.robot.move_to(pose)
            time.sleep(self.cfg["vision"]["settle_s"])
        elif self.cfg["robot"]["backend"] == "viam":
            raise HardwareError(f"pose '{pose_name}' not set — run: python -m bookworm save-pose {pose_name}")

    def _capture_sharp(self):
        v = self.cfg["vision"]
        frame = None
        for attempt in range(v["max_capture_retries"] + 1):
            frame = self.robot.capture()
            s = sharpness(frame)
            if s >= v["min_sharpness"]:
                break
            log.info("blurry capture (%.1f), retry %d", s, attempt + 1)
            time.sleep(0.3)
        return frame

    def capture_spread(self) -> Spread:
        pages, shots = {}, []
        bcfg, ppm = self.cfg["book"], self.mcfg["px_per_mm"]
        for shot in self.mcfg["shots"]:
            self._goto(shot["pose"])
            frame = self._capture_sharp()
            geom = self.robot.camera_geometry() if self.book else None
            quad = size = None
            if geom is not None:
                quad_mm = page_quad(shot["region"], bcfg["page_width_mm"], bcfg["page_height_mm"])
                quad = book_quad_pixels(self.book, geom, quad_mm)
                size = ((quad_mm[1, 0] - quad_mm[0, 0]) * ppm, bcfg["page_height_mm"] * ppm)
            pages.update(process_shot(frame, shot["region"], self.cfg["vision"], quad, size))
            shots.append(Shot(shot["region"], frame, geom))
        return Spread(pages, shots, self._page_number(pages))

    def _page_number(self, pages) -> int | None:
        frac = self.cfg["vision"]["page_number_region"]
        if "left" in pages and (n := self.ocr.page_number(pages["left"].binary, frac)) is not None:
            return n
        if "right" in pages and (n := self.ocr.page_number(pages["right"].binary, frac)) is not None:
            return n - 1
        return None

    # ---- flip ---------------------------------------------------------------
    def _corner(self, spread: Spread):
        if not (self.cfg["flip"]["use_vision_corner"] and self.book):
            return None
        for shot in spread.shots:
            if shot.region in ("spread", "right") and shot.geom is not None:
                c = locate_right_corner(shot.frame, shot.geom, self.book, self.cfg["book"],
                                        self.cfg["vision"]["corner_search_mm"])
                log.info("vision corner: %s", c)
                return c
        return None

    def flip_and_verify(self, before: Spread) -> tuple[FlipOutcome, Spread | None]:
        book = self.book or BookFrame.identity()
        if self.book is None and self.cfg["robot"]["backend"] == "viam":
            raise HardwareError("book frame not calibrated — run: python -m bookworm calibrate-book")
        after = None
        for attempt in range(self.cfg["verify"]["max_flip_retries"] + 1):
            run_flip(self.robot, book, self.cfg, self._corner(after or before), self.flips_done)
            if hasattr(self.robot, "simulate_page_turn"):
                self.robot.simulate_page_turn()
            if getattr(getattr(self.robot, "camera", None), "exhausted", False):
                return FlipOutcome.OK, None              # photo playback finished
            after = self.capture_spread()
            side = "left" if "left" in before.pages else "right"
            outcome = classify_flip(before.pages[side].binary, after.pages[side].binary,
                                    before.page_no, after.page_no, self.cfg["verify"]["change_threshold"])
            log.info("flip attempt %d: %s (pages %s -> %s)", attempt + 1, outcome.value,
                     before.page_no, after.page_no)
            if outcome != FlipOutcome.NO_FLIP:
                self.flips_done += 1
                return outcome, after
        return FlipOutcome.NO_FLIP, after

    # ---- main loop ----------------------------------------------------------
    def _save_debug(self, i, spread, results):
        if not self.run_dir:
            return
        for k, shot in enumerate(spread.shots):
            cv2.imwrite(str(self.run_dir / f"spread{i:03d}_shot{k}_{shot.region}.jpg"), shot.frame)
        lines = [f"page_no={spread.page_no}"]
        lines += [f"=== {s} (conf {r.mean_conf:.0f}, {r.word_count} words) ===\n{r.text}" for s, r in results.items()]
        (self.run_dir / f"spread{i:03d}.txt").write_text("\n\n".join(lines) + "\n")

    def run(self, max_spreads: int | None = None) -> RunSummary:
        summary = RunSummary()
        if self.cfg["poses"].get("home"):
            self.robot.move_to(self.cfg["poses"]["home"])
        spread = self.capture_spread()
        blanks = 0
        while True:
            results = {side: self.ocr.read(page.binary) for side, page in spread.pages.items()}
            self._save_debug(summary.spreads, spread, results)
            self.handler.on_spread(summary.spreads, spread, results)
            summary.spreads += 1

            min_words = self.cfg["ocr"]["min_words_for_text"]
            blanks = 0 if any(r.word_count >= min_words for r in results.values()) else blanks + 1
            if blanks >= self.cfg["run"]["blank_spreads_to_stop"]:
                summary.stopped_because = "end of book"
                break
            if max_spreads is not None and summary.spreads >= max_spreads:
                summary.stopped_because = "max spreads reached"
                break

            outcome, nxt = self.flip_and_verify(spread)
            if nxt is None:
                summary.stopped_because = "no more images"
                break
            summary.flips[outcome.value] = summary.flips.get(outcome.value, 0) + 1
            if outcome == FlipOutcome.NO_FLIP:
                self.handler.on_event("flip_failed")
                summary.stopped_because = "flip failed"
                break
            if outcome in (FlipOutcome.MULTI_FLIP, FlipOutcome.BACKWARD):
                self.handler.on_event(outcome.value)
            spread = nxt

        if self.cfg["poses"].get("home"):
            self.robot.move_to(self.cfg["poses"]["home"])
        self.handler.finish(summary)
        log.info("summary: %s", summary)
        return summary
