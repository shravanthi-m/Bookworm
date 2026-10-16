"""Watch the page during a flip and raise STOP before it folds. Standalone: it only
watches a camera; it never moves the arm. (Wiring it into the flip motion comes later.)

No training. The flip follows the same path over the same book every time, so every
frame of a good flip looks like *some* frame of other good flips. Each frame is reduced
to a heavily blurred, contrast-normalised "shape signature" — the text washes out, the
page's silhouette, curl and shadows remain — and compared with frames recorded from
flips a human labelled GOOD or BAD:

    anomalous if it's far from every good frame       (fold, missed page, double page...)
    anomalous if it's closer to a bad frame than to any good one
    STOP after `stop_after` anomalous frames in a row  (one odd frame is just noise)

References live in <reference_dir>/<good|bad>/<recording>/NNNN.png (small grayscale).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

SIG_SIZE = (96, 72)          # signature resolution: page shape survives, letters don't
STORE_SIZE = (192, 144)      # stored reference frame size (small, repo-friendly)


def signature(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    small = cv2.resize(gray, SIG_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)
    small = cv2.GaussianBlur(small, (0, 0), 1.5)
    return ((small - small.mean()) / (small.std() + 1e-6)).ravel()   # robust to overall brightness


def _nearest(sig: np.ndarray, refs: np.ndarray | None) -> float | None:
    """Mean-abs distance to the closest reference (0 = identical shape, ~0.8+ = unrelated)."""
    if refs is None or not len(refs):
        return None
    return float(np.abs(refs - sig).mean(axis=1).min())


@dataclass
class FrameScore:
    dev_good: float | None
    dev_bad: float | None
    anomalous: bool
    stop: bool
    reason: str


class FlipMonitor:
    def __init__(self, reference_dir: str | Path, max_deviation: float = 0.2, stop_after: int = 3):
        self.ref_dir = Path(reference_dir)
        self.max_deviation = max_deviation
        self.stop_after = stop_after
        self.refs = {label: self._load(label) for label in ("good", "bad")}
        self._streak = 0

    def _load(self, label) -> np.ndarray | None:
        files = sorted((self.ref_dir / label).glob("*/*.png"))
        if not files:
            return None
        return np.stack([signature(cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)) for f in files])

    def counts(self) -> dict:
        return {k: 0 if v is None else len(v) for k, v in self.refs.items()}

    def reset(self) -> None:
        self._streak = 0

    def score(self, frame: np.ndarray) -> FrameScore:
        sig = signature(frame)
        dev_good, dev_bad = _nearest(sig, self.refs["good"]), _nearest(sig, self.refs["bad"])
        reason = ""
        if dev_bad is not None and (dev_good is None or dev_bad < dev_good):
            reason = f"looks like a BAD flip (bad {dev_bad:.2f} < good {dev_good if dev_good is None else round(dev_good, 2)})"
        elif dev_good is not None and dev_good > self.max_deviation:
            reason = f"unlike any good flip ({dev_good:.2f} > {self.max_deviation})"
        anomalous = bool(reason)
        self._streak = self._streak + 1 if anomalous else 0
        return FrameScore(dev_good, dev_bad, anomalous, self._streak >= self.stop_after, reason)


class ReferenceRecorder:
    """Saves frames of one flip as a GOOD or BAD reference recording."""

    def __init__(self, reference_dir: str | Path, label: str):
        assert label in ("good", "bad")
        base = Path(reference_dir) / label / time.strftime("%Y%m%d-%H%M%S")
        self.dir, k = base, 1
        while self.dir.exists():                         # two recordings in the same second
            k += 1
            self.dir = base.with_name(f"{base.name}-{k}")
        self.dir.mkdir(parents=True)
        self.n = 0

    def add(self, frame: np.ndarray) -> None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        cv2.imwrite(str(self.dir / f"{self.n:04d}.png"), cv2.resize(gray, STORE_SIZE, interpolation=cv2.INTER_AREA))
        self.n += 1
