"""Session + modes with a fake camera/OCR and the mock robot — no arm, Viam, or Tesseract needed."""
import json

import numpy as np

from bookworm.config import load_config
from bookworm.hardware import BookFrame, MockRobot, run_flip, tip_to_flange
from bookworm.modes import ReadAloud, Scanner
from bookworm.session import BookSession
from bookworm.speech import PrintSpeaker
from bookworm.vision import OcrResult


class FakeCamera:
    """Each spread is (page_no, words); a flip advances unless `stuck`."""

    def __init__(self, spreads, stuck=False):
        self.spreads, self.i, self.stuck = spreads, 0, stuck

    def capture(self):
        rng = np.random.default_rng(min(self.i, len(self.spreads) - 1))  # distinct, sharp per spread
        return rng.integers(0, 255, (1100, 1600, 3), dtype=np.uint8)

    def advance(self):
        if not self.stuck:
            self.i += 1

    @property
    def exhausted(self):
        return self.i >= len(self.spreads)


class FakeOcr:
    def __init__(self, cam):
        self.cam = cam

    def _cur(self):
        return self.cam.spreads[min(self.cam.i, len(self.cam.spreads) - 1)]

    def read(self, img):
        words = self._cur()[1]
        return OcrResult(" ".join(words), 90.0, words)

    def page_number(self, img, frac):
        return self._cur()[0]


WORDS = "once upon a time there was".split()


def make_session(spreads, mode="read", stuck=False, handler=None, tmp_path=None):
    cfg = load_config()
    cfg["robot"]["backend"] = "mock"
    cfg["run"]["save_dir"] = None
    cfg["vision"]["settle_s"] = 0
    cfg["ocr"]["frames"] = 1
    cfg["flip"]["sequence"] = [s for s in cfg["flip"]["sequence"] if "wait" not in s]
    cam = FakeCamera(spreads, stuck)
    robot = MockRobot(cam)
    if handler is None:
        handler = ReadAloud(PrintSpeaker(), cfg) if mode == "read" else Scanner(tmp_path, cfg)
    return BookSession(robot, FakeOcr(cam), handler, cfg, mode), robot, handler


def test_read_mode_reads_every_spread():
    session, _, handler = make_session([(2, WORDS), (4, WORDS), (6, WORDS)])
    summary = session.run()
    assert summary.spreads == 3 and summary.flips == {"ok": 2}
    assert handler.speaker.spoken.count(" ".join(WORDS)) == 6


def test_stops_at_blank_pages():
    session, _, handler = make_session([(2, WORDS), (4, []), (6, []), (8, WORDS)])
    assert session.run().stopped_because == "end of book"
    assert handler.speaker.spoken[-1] == "The end."


def test_stuck_page_asks_for_help_after_retries():
    session, robot, handler = make_session([(2, WORDS), (4, WORDS)], stuck=True)
    assert session.run().stopped_because == "flip failed"
    assert any("help" in s for s in handler.speaker.spoken)
    turns = [e for e in robot.log if e[0] == "page_turned"]
    assert len(turns) == 1 + session.cfg["verify"]["max_flip_retries"]


def test_multi_flip_is_announced():
    session, _, handler = make_session([(2, WORDS), (8, WORDS)])
    session.run()
    assert any("more than one page" in s for s in handler.speaker.spoken)


def test_scan_mode_writes_pages_text_pdf(tmp_path):
    session, _, _ = make_session([(2, WORDS), (4, WORDS)], mode="scan", tmp_path=tmp_path)
    session.run()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert [p["printed_page"] for p in manifest["pages"]] == [2, 3, 4, 5]
    assert (tmp_path / "pages" / "0004.png").exists()
    assert (tmp_path / "book.pdf").stat().st_size > 0
    assert "once upon" in (tmp_path / "book.txt").read_text()


def test_flip_waypoints_follow_book_frame_and_corner():
    cfg = load_config()
    cfg["flip"]["sequence"] = [{"move": {"x": -20, "y": 20, "z": 30, "from": "corner"}},
                               {"move": {"x": 0, "y": 60, "z": 120, "from": "spine"}},
                               {"gripper": {"action": "grab"}}]
    robot = MockRobot()
    book = BookFrame([500, 0, 10], [500, 300, 10], [700, 0, 10])   # spine along world y at x=500
    run_flip(robot, book, cfg, corner_xy=(142, 1), flip_index=0)
    L = cfg["robot"]["tool_length_mm"]
    assert robot.log[0] == ("move", {"x": 622.0, "y": 21.0, "z": 40.0 + L})
    assert robot.log[1] == ("move", {"x": 500.0, "y": 60.0, "z": 130.0 + L})
    assert robot.log[2] == ("gripper", "grab")


def test_tip_to_flange_points_tool_down():
    rcfg = {"tool_length_mm": 190, "tip_orientation": {"o_x": 0, "o_y": 0, "o_z": -1, "theta": 0}}
    assert tip_to_flange([1, 2, 3], rcfg) == {"x": 1, "y": 2, "z": 193, "o_x": 0, "o_y": 0, "o_z": -1, "theta": 0}


def test_background_speaker_overlaps_with_work():
    import time

    from bookworm.speech import BackgroundSpeaker

    spoken = []
    sp = BackgroundSpeaker(lambda t: (time.sleep(0.2), spoken.append(t)))
    t0 = time.perf_counter()
    for w in ("one", "two", "three"):
        sp.say(w)
    assert time.perf_counter() - t0 < 0.1          # say() doesn't block -> arm can flip meanwhile
    sp.wait()
    assert spoken == ["one", "two", "three"] and time.perf_counter() - t0 >= 0.6


def test_summary_has_timing():
    session, _, _ = make_session([(2, WORDS), (4, WORDS)])
    summary = session.run()
    assert {"ocr", "flip", "capture"} <= set(summary.timing)
    assert "wall clock" in summary.timing_table()


def test_excerpt_keeps_whole_sentences():
    from bookworm.modes import excerpt

    text = "The cat sat. The dog ran far away. Then it rained all day long."
    assert excerpt(text, 5) == "The cat sat."
    assert excerpt(text, 10) == "The cat sat. The dog ran far away."
    assert excerpt(text, None) == text
    assert excerpt("one two three four five six", 3) == "one two three..."


def test_read_and_save_scan_together(tmp_path):
    from bookworm.modes import Combined

    session, _, _ = make_session([(2, WORDS), (4, WORDS)], handler=None)
    speaker = PrintSpeaker()
    session.handler = Combined(ReadAloud(speaker, session.cfg), Scanner(tmp_path, session.cfg))
    session.run()
    assert len(speaker.spoken) >= 4
    assert (tmp_path / "book.pdf").exists() and len(list((tmp_path / "pages").glob("*.png"))) == 4


def test_searchable_pdf_has_text_layer(tmp_path):
    import shutil

    import cv2
    import pytest
    from pypdf import PdfReader

    if shutil.which("tesseract") is None:
        pytest.skip("tesseract not installed")
    from bookworm.archive import build_pdf

    (tmp_path / "pages").mkdir()
    img = np.full((1100, 800, 3), 245, np.uint8)
    for i, line in enumerate(["The quick brown fox", "jumps over the lazy dog", "Spinoza and Descartes"]):
        cv2.putText(img, line, (40, 150 + 90 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
    cv2.circle(img, (400, 800), 150, (40, 120, 200), -1)                     # an "illustration"
    cv2.imwrite(str(tmp_path / "pages" / "0001.png"), img)
    pdf = build_pdf(tmp_path, page_width_mm=140)
    reader = PdfReader(pdf)
    assert len(reader.pages) == 1
    text = reader.pages[0].extract_text().lower()
    assert "brown fox" in text and "spinoza" in text
    w_in = float(reader.pages[0].mediabox.width) / 72
    assert abs(w_in - 140 / 25.4) < 0.1                                     # real page size


def test_excerpt_skips_garbled_ocr_and_headers():
    from bookworm.modes import excerpt

    ocr = ("200 : JOSTEIN GAARDER\n\n"
           "So nature is not a dream after all. "
           "Wag dot pat web tha bark eet Leber gee en de od he fe. "
           "For example, only man has a mind. "
           "Are we automatons as well?")
    out = excerpt(ocr, 30)
    assert out == "So nature is not a dream after all. For example, only man has a mind. Are we automatons as well?"
    assert excerpt("Xq zzv brrp. Qwt plk mnnb.", 30) == ""          # nothing clean -> stay quiet


def test_best_of_frames_keeps_cleanest_page_per_side():
    from bookworm.session import best_of_frames

    class OcrBySeed:                    # frame brightness decides how clean the "OCR" is
        wants_color = True              # gets the raw colour page, like the Apple engine

        def read(self, img):
            clean = img.mean() > 100
            text = "The cat sat on the mat. The dog ran far away." if clean else "Thx cxt sxt qn tvb mzt."
            return OcrResult(text, 90.0, text.split())

    dark, bright = np.full((200, 400, 3), 40, np.uint8), np.full((200, 400, 3), 200, np.uint8)
    vcfg = {"book_corners": None, "clamp_masks": [], "clamp_color_hsv": None}
    pages, results = best_of_frames([dark, bright, dark], "spread", vcfg, OcrBySeed())
    assert set(results) == {"left", "right"} and all("cat sat" in r.text for r in results.values())
    assert pages["left"].color.mean() > 100
