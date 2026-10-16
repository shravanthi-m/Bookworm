"""Core page loop shared by both modes:  capture -> OCR -> handler -> flip -> verify -> repeat.

A mode (read aloud / scan) is a handler object with:
    on_spread(index, spread, ocr_results)   called once per spread
    on_event(message)                        flip problems, end of book
    finish(summary)                          after the loop
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from .hardware import BookFrame, HardwareError, Robot, page_quad, run_flip
from .vision import ocr_image
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
    results: dict | None = None   # OCR per side, when already done during capture (best-of-N)


def clean_words(result) -> int:
    """How much of a page would be read aloud: words in clean sentences."""
    from .modes.read_aloud import excerpt

    return len(excerpt(result.text, None).split())


def best_of_frames(frames, region, vcfg, ocr, quad=None, size=None):
    """OCR every frame and keep, per side, the page whose text came out cleanest.

    Video frames differ slightly (noise, focus breathing); on the phone feed the amount of
    cleanly-read text varied 2-3x between consecutive frames of the same page.
    Returns ({side: Page}, {side: OcrResult}).
    """
    best: dict = {}
    for frame in frames:
        for side, page in process_shot(frame, region, vcfg, quad, size).items():
            res = ocr.read(ocr_image(ocr, page))
            score = (clean_words(res), res.word_count)
            if side not in best or score > best[side][0]:
                best[side] = (score, page, res)
    return {s: b[1] for s, b in best.items()}, {s: b[2] for s, b in best.items()}


@dataclass
class RunSummary:
    spreads: int = 0
    flips: dict = field(default_factory=dict)   # FlipOutcome.value -> count
    stopped_because: str = ""
    seconds: float = 0.0
    timing: dict = field(default_factory=dict)  # stage -> {"n", "mean_s", "total_s"}

    def timing_table(self) -> str:
        rows = [f"{'stage':<12}{'n':>4}{'mean s':>9}{'total s':>9}"]
        for stage, t in sorted(self.timing.items(), key=lambda kv: -kv[1]["total_s"]):
            rows.append(f"{stage:<12}{t['n']:>4}{t['mean_s']:>9.2f}{t['total_s']:>9.1f}")
        rows.append(f"{'wall clock':<12}{'':>4}{'':>9}{self.seconds:>9.1f}")
        return "\n".join(rows)


class BookSession:
    def __init__(self, robot: Robot, ocr, handler, cfg: dict, mode: str):
        self.robot, self.ocr, self.handler, self.cfg = robot, ocr, handler, cfg
        self.mcfg = cfg["modes"][mode]
        self.book = BookFrame.from_config(cfg["book"])
        self.flips_done = 0
        self.times: dict[str, list[float]] = defaultdict(list)
        self.pool = ThreadPoolExecutor(max_workers=4)   # tesseract runs as subprocesses -> real parallelism
        self.run_dir = None
        if cfg["run"].get("save_dir"):
            self.run_dir = Path(cfg["run"]["save_dir"]) / f"{time.strftime('%Y%m%d-%H%M%S')}-{mode}"
            self.run_dir.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _timed(self, stage: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.times[stage].append(time.perf_counter() - t0)

    # ---- capture ------------------------------------------------------------
    def _goto(self, pose_name: str) -> None:
        pose = self.cfg["poses"].get(pose_name)
        if pose:
            with self._timed("move"):
                self.robot.move_to(pose)
            with self._timed("settle"):
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
        pages, shots, results = {}, [], {}
        n_frames = max(1, int(self.cfg["ocr"].get("frames", 1)))
        bcfg, ppm = self.cfg["book"], self.mcfg["px_per_mm"]
        for shot in self.mcfg["shots"]:
            self._goto(shot["pose"])
            with self._timed("capture"):
                frame = self._capture_sharp()
                geom = self.robot.camera_geometry() if self.book else None
            t0 = time.perf_counter()
            quad = size = None
            if geom is not None:
                quad_mm = page_quad(shot["region"], bcfg["page_width_mm"], bcfg["page_height_mm"])
                quad = book_quad_pixels(self.book, geom, quad_mm)
                size = ((quad_mm[1, 0] - quad_mm[0, 0]) * ppm, bcfg["page_height_mm"] * ppm)
            if n_frames > 1:
                frames = [frame]
                for _ in range(n_frames - 1):
                    time.sleep(self.cfg["ocr"].get("frame_gap_s", 0.3))
                    frames.append(self.robot.capture())
                p, r = best_of_frames(frames, shot["region"], self.cfg["vision"], self.ocr, quad, size)
                pages.update(p)
                results.update(r)
            else:
                pages.update(process_shot(frame, shot["region"], self.cfg["vision"], quad, size))
            self.times["preprocess"].append(time.perf_counter() - t0)
            shots.append(Shot(shot["region"], frame, geom))
        with self._timed("page_no"):
            page_no = self._page_number(pages)
        return Spread(pages, shots, page_no, results or None)

    def _page_number(self, pages) -> int | None:
        frac = self.cfg["vision"]["page_number_region"]
        futs = {s: self.pool.submit(self.ocr.page_number, ocr_image(self.ocr, pages[s]), frac)
                for s in ("left", "right") if s in pages}
        nums = {s: f.result() for s, f in futs.items()}
        if nums.get("left") is not None:
            return nums["left"]
        if nums.get("right") is not None:
            return nums["right"] - 1
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
            corner = self._corner(after or before)
            with self._timed("flip"):
                run_flip(self.robot, book, self.cfg, corner, self.flips_done)
            if hasattr(self.robot, "simulate_page_turn"):
                self.robot.simulate_page_turn()
            if getattr(getattr(self.robot, "camera", None), "exhausted", False):
                return FlipOutcome.OK, None              # photo playback finished
            after = self.capture_spread()
            side = "left" if "left" in before.pages else "right"
            outcome = classify_flip(before.pages[side].ocr, after.pages[side].ocr,
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
        t_start = time.perf_counter()
        try:
            summary = self._run(max_spreads)
        finally:
            self.pool.shutdown(wait=False)
        summary.seconds = time.perf_counter() - t_start
        summary.timing = {k: {"n": len(v), "mean_s": sum(v) / len(v), "total_s": sum(v)}
                          for k, v in self.times.items() if v}
        log.info("summary: %s", summary)
        print(summary.timing_table())
        return summary

    def _run(self, max_spreads: int | None) -> RunSummary:
        summary = RunSummary()
        if self.cfg["poses"].get("home"):
            self.robot.move_to(self.cfg["poses"]["home"])
        spread = self.capture_spread()
        blanks = 0
        while True:
            with self._timed("ocr"):
                if spread.results:                # already read while picking the best frame
                    results = spread.results
                else:
                    futs = {side: self.pool.submit(self.ocr.read, ocr_image(self.ocr, page))
                            for side, page in spread.pages.items()}
                    results = {side: f.result() for side, f in futs.items()}
            self._save_debug(summary.spreads, spread, results)
            with self._timed("handler"):
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
        with self._timed("finish"):
            self.handler.finish(summary)        # read mode: waits for speech to end
        return summary
