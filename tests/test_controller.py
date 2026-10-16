"""Controller logic with fake camera/OCR/hardware — no rig, Viam, or Tesseract needed."""
import numpy as np

from bookworm.config import load_config
from bookworm.controller import BookReader
from bookworm.hardware import MockActuators, run_sequence
from bookworm.speech import PrintSpeaker
from bookworm.vision import OcrResult


class FakeCamera:
    """Each spread is (page_no, words); a flip advances unless `stuck`."""

    def __init__(self, spreads, stuck=False):
        self.spreads, self.i, self.stuck = spreads, 0, stuck

    def capture(self):
        # Distinct sharp image per spread so image-diff and sharpness checks behave.
        rng = np.random.default_rng(min(self.i, len(self.spreads) - 1))
        return rng.integers(0, 255, (1100, 1600, 3), dtype=np.uint8)

    def advance(self):
        if not self.stuck:
            self.i += 1

    @property
    def exhausted(self):
        return self.i >= len(self.spreads)

    def close(self):
        pass


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


def make_reader(spreads, stuck=False):
    cfg = load_config()
    cfg["run"]["save_dir"] = None
    cfg["flip_sequence"] = [s for s in cfg["flip_sequence"] if "wait" not in s]
    cam = FakeCamera(spreads, stuck)
    act = MockActuators(on_flip=cam.advance)
    speaker = PrintSpeaker()
    return BookReader(cam, act, FakeOcr(cam), speaker, cfg), act, speaker


WORDS = "once upon a time there was".split()


def test_reads_every_spread_then_stops():
    reader, act, speaker = make_reader([(2, WORDS), (4, WORDS), (6, WORDS)])
    summary = reader.run()
    assert summary.spreads_read == 3
    assert summary.flips.get("ok") == 2
    assert speaker.spoken.count("once upon a time there was") == 6  # 2 pages x 3 spreads


def test_stops_at_blank_pages():
    reader, _, speaker = make_reader([(2, WORDS), (4, []), (6, []), (8, WORDS)])
    summary = reader.run()
    assert "end of book" in summary.stopped_because
    assert speaker.spoken[-1] == "The end."


def test_stuck_page_asks_for_help_after_retries():
    reader, act, speaker = make_reader([(2, WORDS), (4, WORDS)], stuck=True)
    summary = reader.run()
    assert summary.stopped_because == "flip failed"
    assert any("help" in s for s in speaker.spoken)
    sweeps = [e for e in act.log if e == ("arm", 175.0)]
    assert len(sweeps) == 1 + reader.cfg["verify"]["max_flip_retries"]


def test_multi_flip_is_announced():
    reader, _, speaker = make_reader([(2, WORDS), (8, WORDS)])
    reader.run()
    assert any("more than one page" in s for s in speaker.spoken)


def test_run_sequence_dispatch():
    act = MockActuators()
    run_sequence(act, [{"clamp": {"side": "left", "engaged": False}}, {"lift": {"down": True}},
                       {"arm": {"angle": 90}}])
    assert act.log == [("clamp", "left", False), ("lift", True), ("arm", 90.0)]
