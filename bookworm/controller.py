"""Main read loop:  capture -> OCR -> speak -> flip -> verify -> repeat."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from .hardware import Actuators, run_sequence
from .vision import FlipOutcome, classify_flip, process_spread, sharpness

log = logging.getLogger("bookworm")


@dataclass
class Spread:
    frame: object
    pages: dict           # side -> binarised page image
    page_no: int | None   # left-page number (None if not found)


@dataclass
class RunSummary:
    spreads_read: int = 0
    flips: dict = field(default_factory=dict)   # FlipOutcome.value -> count
    stopped_because: str = ""


class BookReader:
    def __init__(self, camera, actuators: Actuators, ocr, speaker, cfg: dict):
        self.camera, self.act, self.ocr, self.speaker, self.cfg = camera, actuators, ocr, speaker, cfg
        self.run_dir = None
        if cfg["run"].get("save_dir"):
            self.run_dir = Path(cfg["run"]["save_dir"]) / time.strftime("%Y%m%d-%H%M%S")
            self.run_dir.mkdir(parents=True, exist_ok=True)

    # ---- vision -------------------------------------------------------------
    def capture_sharp(self):
        v = self.cfg["vision"]
        frame = None
        for attempt in range(v["max_capture_retries"] + 1):
            frame = self.camera.capture()
            s = sharpness(frame)
            if s >= v["min_sharpness"]:
                return frame
            log.info("blurry capture (%.1f), retry %d", s, attempt + 1)
            time.sleep(0.3)
        return frame  # best effort

    def analyse(self, frame) -> Spread:
        out = process_spread(frame, self.cfg["vision"])
        frac = self.cfg["vision"]["page_number_region"]
        page_no = self.ocr.page_number(out["left"], frac)
        if page_no is None:
            right = self.ocr.page_number(out["right"], frac)
            page_no = right - 1 if right is not None else None
        return Spread(out["spread"], {"left": out["left"], "right": out["right"]}, page_no)

    # ---- reading ------------------------------------------------------------
    def read_spread(self, spread: Spread, index: int) -> bool:
        """OCR and speak both pages. Returns False if the spread was blank."""
        o = self.cfg["ocr"]
        any_text = False
        texts = []
        for side in self.cfg["run"]["read_order"]:
            res = self.ocr.read(spread.pages[side])
            texts.append(f"=== {side} (conf {res.mean_conf:.0f}, {res.word_count} words) ===\n{res.text}")
            if res.word_count < o["min_words_for_text"]:
                continue  # blank page or illustration
            any_text = True
            if res.mean_conf < o["min_page_conf"]:
                self.speaker.say(f"I can't read the {side} page clearly.")
            else:
                self.speaker.say(res.text)
        self._save(index, spread, "\n\n".join(texts))
        return any_text

    def _save(self, index, spread, text):
        if not self.run_dir:
            return
        cv2.imwrite(str(self.run_dir / f"spread_{index:03d}.jpg"), spread.frame)
        (self.run_dir / f"spread_{index:03d}.txt").write_text(f"page_no={spread.page_no}\n{text}\n")

    # ---- flipping -----------------------------------------------------------
    def flip_and_verify(self, before: Spread) -> tuple[FlipOutcome, Spread | None]:
        retries = self.cfg["verify"]["max_flip_retries"]
        after = None
        for attempt in range(retries + 1):
            run_sequence(self.act, self.cfg["flip_sequence"])
            if getattr(self.camera, "exhausted", False):
                return FlipOutcome.OK, None
            after = self.analyse(self.capture_sharp())
            outcome = classify_flip(before.frame, after.frame, before.page_no, after.page_no,
                                    self.cfg["verify"]["change_threshold"])
            log.info("flip attempt %d: %s (pages %s -> %s)", attempt + 1, outcome.value,
                     before.page_no, after.page_no)
            if outcome != FlipOutcome.NO_FLIP:
                return outcome, after
        return FlipOutcome.NO_FLIP, after

    # ---- main loop ----------------------------------------------------------
    def run(self, max_spreads: int | None = None) -> RunSummary:
        summary = RunSummary()
        self.act.home()
        spread = self.analyse(self.capture_sharp())
        blanks = 0
        while True:
            if self.read_spread(spread, summary.spreads_read):
                blanks = 0
            else:
                blanks += 1
                if blanks >= self.cfg["run"]["blank_spreads_to_stop"]:
                    summary.stopped_because = "end of book (blank spreads)"
                    break
            summary.spreads_read += 1
            if max_spreads is not None and summary.spreads_read >= max_spreads:
                summary.stopped_because = "max spreads reached"
                break

            outcome, nxt = self.flip_and_verify(spread)
            if nxt is None:
                summary.stopped_because = "no more images"
                break
            summary.flips[outcome.value] = summary.flips.get(outcome.value, 0) + 1
            if outcome == FlipOutcome.NO_FLIP:
                self.speaker.say("I couldn't turn the page. Please help me.")
                summary.stopped_because = "flip failed"
                break
            if outcome == FlipOutcome.MULTI_FLIP:
                self.speaker.say("I may have turned more than one page.")
            elif outcome == FlipOutcome.BACKWARD:
                self.speaker.say("The page went backwards.")
            spread = nxt

        self.speaker.say("The end." if "end of book" in summary.stopped_because else "Stopping.")
        self.act.home()
        log.info("summary: %s", summary)
        return summary
