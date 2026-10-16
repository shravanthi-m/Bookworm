import shutil

import cv2
import numpy as np
import pytest

from bookworm.vision import FlipOutcome, classify_flip, image_difference, sharpness
from bookworm.vision.ocr import clean_text, parse_page_number
from bookworm.vision.preprocess import find_gutter, mask_color, mask_rects, process_spread


def make_spread(left_words="hello world", right_words="second page", page=12, w=1600, h=1100):
    """Synthetic open book: white pages, dark gutter at x=w/2, text and page numbers."""
    img = np.full((h, w, 3), 245, np.uint8)
    cv2.rectangle(img, (w // 2 - 6, 0), (w // 2 + 6, h), (60, 60, 60), -1)
    for i in range(6):
        cv2.putText(img, f"{left_words} line {i}", (80, 150 + i * 70), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
        cv2.putText(img, f"{right_words} line {i}", (w // 2 + 80, 150 + i * 70), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
    cv2.putText(img, str(page), (w // 4, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
    cv2.putText(img, str(page + 1), (3 * w // 4, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
    return img


def test_gutter_found_near_center():
    img = make_spread()
    assert abs(find_gutter(img) - 800) < 15


def test_sharpness_drops_when_blurred():
    img = make_spread()
    assert sharpness(cv2.GaussianBlur(img, (31, 31), 0)) < sharpness(img) / 5


def test_mask_rects_whites_out_region():
    img = np.zeros((100, 100), np.uint8)
    out = mask_rects(img, [[0, 0, 0.5, 0.5]])
    assert out[:50, :50].min() == 255 and out[60:, 60:].max() == 0


def test_mask_color_removes_green_clamp():
    img = make_spread()
    cv2.rectangle(img, (0, 400), (90, 600), (0, 200, 0), -1)  # green clamp finger
    out = mask_color(img, [[40, 80, 60], [85, 255, 255]])
    assert out[500, 45].tolist() == [255, 255, 255]


def test_process_spread_outputs_binary_pages():
    out = process_spread(make_spread(), {"book_corners": None, "warp_size": [1600, 1100],
                                         "clamp_masks": [], "clamp_color_hsv": None})
    assert set(np.unique(out["left"])) <= {0, 255}
    assert out["left"].shape[0] == 1100


def test_image_difference():
    a, b = make_spread(page=10), make_spread("other text", "entirely new", page=12)
    assert image_difference(a, a) < 0.01
    assert image_difference(a, b) > image_difference(a, a)


@pytest.mark.parametrize("before,after,expected", [
    (10, 12, FlipOutcome.OK), (10, 10, FlipOutcome.NO_FLIP),
    (10, 16, FlipOutcome.MULTI_FLIP), (10, 8, FlipOutcome.BACKWARD),
])
def test_classify_by_page_number(before, after, expected):
    img = make_spread()
    assert classify_flip(img, img, before, after) == expected


def test_classify_without_page_numbers_uses_image():
    a = make_spread()
    assert classify_flip(a, a, None, None) == FlipOutcome.NO_FLIP
    b = np.full_like(a, 30)
    assert classify_flip(a, b, None, None) == FlipOutcome.UNKNOWN


def test_clean_text():
    assert clean_text("The quick bro-\nwn fox\njumps.\n\n42\n\nNext para.") == "The quick brown fox jumps.\n\nNext para."
    assert parse_page_number("  - 17 -\n") == 17
    assert parse_page_number("no digits") is None


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
def test_tesseract_end_to_end():
    from bookworm.vision import TesseractOcr

    out = process_spread(make_spread(), {"book_corners": None, "warp_size": [1600, 1100],
                                         "clamp_masks": [], "clamp_color_hsv": None})
    ocr = TesseractOcr()
    assert "hello" in ocr.read(out["left"]).text.lower()
    assert ocr.page_number(out["left"], 0.08) == 12
