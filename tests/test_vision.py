import shutil

import cv2
import numpy as np
import pytest

from bookworm.hardware import BookFrame, CameraGeometry, page_quad
from bookworm.vision import (FlipOutcome, book_quad_pixels, classify_flip, find_paper_corners,
                             image_difference, locate_right_corner, pixel_to_plane, process_shot,
                             sharpness, world_to_pixel)
from bookworm.vision.ocr import clean_text, parse_page_number
from bookworm.vision.preprocess import find_gutter, mask_color, mask_rects

VCFG = {"book_corners": None, "fallback_warp_size": [1600, 1100], "clamp_masks": [], "clamp_color_hsv": None}


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


def overhead_camera(height=400.0, f=1000.0, size=(1600, 1200)):
    """Camera looking straight down from (0, 0, height); image x = world x, image y = -world y."""
    K = np.array([[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = np.diag([1.0, -1.0, -1.0])   # cam z (forward) points world -z (down)
    T[:3, 3] = [0, 0, height]
    return CameraGeometry(K, None, T)


# ---- preprocessing ------------------------------------------------------------
def test_gutter_found_near_center():
    assert abs(find_gutter(make_spread()) - 800) < 15


def test_sharpness_drops_when_blurred():
    img = make_spread()
    assert sharpness(cv2.GaussianBlur(img, (31, 31), 0)) < sharpness(img) / 5


def test_mask_rects_whites_out_region():
    out = mask_rects(np.zeros((100, 100), np.uint8), [[0, 0, 0.5, 0.5]])
    assert out[:50, :50].min() == 255 and out[60:, 60:].max() == 0


def test_mask_color_removes_green_holder():
    img = make_spread()
    cv2.rectangle(img, (0, 400), (90, 600), (0, 200, 0), -1)
    assert mask_color(img, [[40, 80, 60], [85, 255, 255]])[500, 45].tolist() == [255, 255, 255]


def test_process_shot_spread_and_single_page():
    pages = process_shot(make_spread(), "spread", VCFG)
    assert set(pages) == {"left", "right"}
    assert set(np.unique(pages["left"].binary)) <= {0, 255}
    assert pages["left"].color.ndim == 3
    # fallback (no geometry) for a single-page region still splits the spread
    assert set(process_shot(make_spread(), "right", VCFG)) == {"right"}


def test_process_shot_with_projected_quad():
    img = make_spread()
    quad = np.array([[800, 0], [1599, 0], [1599, 1099], [800, 1099]], float)
    pages = process_shot(img, "right", VCFG, quad, (700, 1000))
    assert pages["right"].color.shape[:2] == (1000, 700)


# ---- geometry -----------------------------------------------------------------
def test_pixel_plane_round_trip():
    g = overhead_camera()
    world = np.array([[50.0, -30.0, 0.0], [-120.0, 80.0, 0.0]])
    px = world_to_pixel(world, g.K, g.T_world_cam)
    assert np.allclose(px[0], [800 + 125, 600 + 75])     # f * x / z
    assert np.allclose(pixel_to_plane(px, g.K, g.T_world_cam, 0.0), world, atol=1e-6)


def test_book_frame_axes():
    book = BookFrame([100, 0, 5], [100, 200, 5], [250, 3, 8])   # spine along world y
    assert np.allclose(book.to_world(140, 0, 0), [240, 0, 8])
    assert np.allclose(book.from_world(book.to_world(-30, 50, 2)), [-30, 50, 2])


def test_locate_right_corner_from_image():
    g = overhead_camera(height=450)
    book = BookFrame([0, 0, 0], [0, 1, 0], [1, 0, 0])
    bcfg = {"page_width_mm": 140, "page_height_mm": 210}
    img = np.full((1200, 1600, 3), 40, np.uint8)                 # dark table
    true_quad = page_quad("spread", 140, 210) + [3, -4]          # book shifted 3 mm, -4 mm
    px = book_quad_pixels(book, g, true_quad)
    cv2.fillPoly(img, [px.astype(np.int32)], (240, 240, 240))
    assert find_paper_corners(img) is not None
    c = locate_right_corner(img, g, book, bcfg, search_mm=25)
    assert c is not None and abs(c[0] - 143) < 2 and abs(c[1] + 4) < 2


# ---- verification / text -------------------------------------------------------
def test_image_difference():
    a, b = make_spread(page=10), make_spread("other text", "entirely new", page=12)
    assert image_difference(a, a) < 0.01 < image_difference(a, b)


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
    assert classify_flip(a, np.full_like(a, 30), None, None) == FlipOutcome.UNKNOWN


def test_clean_text():
    assert clean_text("The quick bro-\nwn fox\njumps.\n\n42\n\nNext para.") == "The quick brown fox jumps.\n\nNext para."
    assert parse_page_number("  - 17 -\n") == 17
    assert parse_page_number("no digits") is None


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
def test_tesseract_end_to_end():
    from bookworm.vision import TesseractOcr

    pages = process_shot(make_spread(), "spread", VCFG)
    ocr = TesseractOcr()
    assert "hello" in ocr.read(pages["left"].binary).text.lower()
    assert ocr.page_number(pages["left"].binary, 0.08) == 12


def test_word_accuracy():
    from bookworm.vision.bench import word_accuracy

    assert word_accuracy("the cat sat", "the cat sat") == 1.0
    assert word_accuracy("the cat sat down", "the bat sat down") == 0.75
    assert word_accuracy("", "") == 1.0


def test_bench_on_photo_folder(tmp_path):
    from bookworm.config import load_config
    from bookworm.vision.bench import VisionBench

    class FakeOcr:
        def read(self, img):
            from bookworm.vision import OcrResult
            return OcrResult("hello world line", 90.0, ["hello", "world", "line"])

        def page_number(self, img, frac):
            return None

    (tmp_path / "a.left.txt").write_text("hello world line")
    bench = VisionBench(load_config(), FakeOcr(), tmp_path / "out")
    bench.process("a.png", make_spread(), truth_dir=tmp_path)
    bench.process("b.png", make_spread("x", "y"), truth_dir=tmp_path)
    summary = bench.report()
    assert summary["images"] == 2 and summary["mean_word_accuracy"] == 1.0
    assert (tmp_path / "out" / "a.panel.jpg").exists()
    assert "flip_vs_previous" in bench.results[1]


def test_flatten_illumination_evens_out_shadow():
    from bookworm.vision.preprocess import flatten_illumination

    gray = cv2.cvtColor(make_spread(), cv2.COLOR_BGR2GRAY).astype(np.float32)
    shaded = (gray * np.linspace(0.45, 1.0, gray.shape[1])[None, :]).astype(np.uint8)   # dark on the left
    flat = flatten_illumination(shaded)
    paper_l, paper_r = flat[600:650, 100:300].mean(), flat[600:650, 1300:1500].mean()
    assert abs(paper_l - paper_r) < 15 < abs(shaded[600:650, 100:300].mean() - shaded[600:650, 1300:1500].mean())


def test_tool_mask_finds_static_pencil():
    from bookworm.vision.tool_mask import build_tool_mask

    rng = np.random.default_rng(0)
    frames = []
    for _ in range(5):
        f = rng.integers(0, 255, (360, 640, 3), dtype=np.uint8)              # a different scene each time
        cv2.rectangle(f, (600, 150), (639, 359), (40, 160, 220), -1)        # pencil from the right edge
        frames.append(f)
    mask = build_tool_mask(frames, dilate_px=0)
    assert mask[250, 620] == 255 and mask[50, 50] == 0
    assert 0.02 < (mask > 0).mean() < 0.06


def test_tool_mask_applied_in_process_shot(tmp_path):
    from bookworm.vision.preprocess import apply_tool_mask

    img = make_spread()
    cv2.rectangle(img, (1500, 300), (1599, 700), (0, 0, 0), -1)             # black "pencil"
    m = np.zeros(img.shape[:2], np.uint8)
    m[290:710, 1490:] = 255
    cv2.imwrite(str(tmp_path / "m.png"), m)
    out = apply_tool_mask(img, str(tmp_path / "m.png"))
    assert out[500, 1550].mean() > 150                                       # painted over with paper
