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
