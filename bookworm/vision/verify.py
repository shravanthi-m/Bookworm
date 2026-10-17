"""Did the flip work? Compare before/after images and page numbers."""
from __future__ import annotations

from enum import Enum

import cv2
import numpy as np


class FlipOutcome(Enum):
    OK = "ok"                # exactly one leaf turned
    NO_FLIP = "no_flip"      # spread looks the same -> retry
    MULTI_FLIP = "multi"     # page numbers jumped by more than 2 -> skipped pages
    BACKWARD = "backward"    # page numbers went down
    UNKNOWN = "unknown"      # image changed but no page numbers to confirm


def image_difference(a: np.ndarray, b: np.ndarray, size=(160, 110)) -> float:
    """Mean absolute difference of small blurred grayscale thumbnails, in [0, 1]."""
    def thumb(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        return cv2.GaussianBlur(cv2.resize(g, size), (5, 5), 0).astype(np.float32) / 255.0

    return float(np.abs(thumb(a) - thumb(b)).mean())


def classify_flip(before_img, after_img, before_page: int | None, after_page: int | None,
                  change_threshold: float = 0.08) -> FlipOutcome:
    if before_page is not None and after_page is not None:
        delta = after_page - before_page
        if delta == 2:
            return FlipOutcome.OK
        if delta == 0:
            return FlipOutcome.NO_FLIP
        if delta < 0:
            return FlipOutcome.BACKWARD
        if delta > 2:
            return FlipOutcome.MULTI_FLIP
        # delta == 1: probably an OCR misread on one side; fall through to image check
    if image_difference(before_img, after_img) < change_threshold:
        return FlipOutcome.NO_FLIP
    return FlipOutcome.UNKNOWN


def page_signature(page_img: np.ndarray, width: int = 320) -> np.ndarray:
    """Text-layout fingerprint of a straightened page: ink marks at ~word scale. Two shots of
    the same page match closely; a different page (new words, new line breaks) does not."""
    g = cv2.cvtColor(page_img, cv2.COLOR_BGR2GRAY) if page_img.ndim == 3 else page_img
    g = cv2.resize(g, (width, max(1, int(g.shape[0] * width / g.shape[1]))), interpolation=cv2.INTER_AREA)
    ink = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10)
    return cv2.GaussianBlur(ink, (0, 0), 1.5).astype(np.float32)


def same_page_score(a: np.ndarray, b: np.ndarray, shift: int = 10) -> float:
    """Correlation (0..1) of two page signatures, allowing a small shift (camera jitter,
    slightly different crop). ~1 = same page; a turned page scores much lower."""
    b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)   # crops vary in size
    h, w = a.shape
    templ = b[shift:h - shift, shift:w - shift]
    if templ.size == 0 or templ.std() < 1e-3 or a.std() < 1e-3:
        return 0.0
    return float(cv2.matchTemplate(a, templ, cv2.TM_CCOEFF_NORMED).max())


SAME_PAGE = 0.75     # data/ocr_eval: same spread median 0.93 (failed flip 0.83-0.98), different spreads <= 0.54.
                     # Leans to "it turned": a false "didn't turn" would retry and skip a page.


def judge_flip(before: dict, after: dict, same_page: float = SAME_PAGE) -> tuple[FlipOutcome, str]:
    """Did a flip turn exactly one leaf? ``before``/``after``: {"sig": {side: page_signature},
    "page_no": int | None}. The image decides "nothing turned" (robust to OCR misreads);
    page numbers, when both are readable, catch two pages turned at once."""
    scores = [same_page_score(before["sig"][s], after["sig"][s]) for s in ("left", "right")
              if s in before["sig"] and s in after["sig"]]
    score = min(scores) if scores else 0.0
    if score >= same_page:
        return FlipOutcome.NO_FLIP, f"spread looks the same (match {score:.2f})"
    b, a = before.get("page_no"), after.get("page_no")
    if b is not None and a is not None:
        if a - b >= 4:
            return FlipOutcome.MULTI_FLIP, f"pages jumped {b} -> {a}"
        if a - b < 0:
            return FlipOutcome.BACKWARD, f"pages went back {b} -> {a}"
        return FlipOutcome.OK, f"pages {b} -> {a} (match {score:.2f})"
    return FlipOutcome.OK, f"spread changed (match {score:.2f}); page numbers not readable"
