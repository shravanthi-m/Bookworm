"""Fold detector: synthetic overhead frames of a flip; no arm, no camera."""
import cv2
import numpy as np

from bookworm.vision.flip_monitor import FlipMonitor, ReferenceRecorder


def frame(t: float, kind: str = "good", seed: int = 0) -> np.ndarray:
    """Overhead view at flip progress t in [0, 1]. Text differs per seed (= different page)."""
    img = np.full((480, 640, 3), 90, np.uint8)
    cv2.rectangle(img, (60, 80), (580, 420), (235, 235, 235), -1)
    rng = np.random.default_rng(seed)
    for y in range(110, 400, 14):
        for x0, x1 in ((80, 300), (340, 560)):
            cv2.line(img, (x0, y), (int(rng.integers(x0 + 120, x1)), y), (60, 60, 60), 2)
    cv2.circle(img, (int(560 - 300 * t), 300), 12, (30, 120, 220), -1)              # the pencil moving left
    if kind == "good":                                                              # curl grows smoothly
        overlay = img.copy()
        cv2.ellipse(overlay, (540, 400), (int(30 + 80 * t), int(20 + 50 * t)), 0, 180, 360, (170, 170, 170), -1)
        img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
    elif kind == "fold" and t > 0.4:                                                # a creased flap appears
        cv2.fillPoly(img, [np.array([[580, 420], [430, 420], [580, 230]])], (40, 40, 40))
    return img


def record(ref_dir, kind, seed, label=None):
    rec = ReferenceRecorder(ref_dir, label or ("good" if kind == "good" else "bad"))
    for t in np.linspace(0, 1, 12):
        rec.add(frame(t, kind, seed))


def run(monitor, kind, seed):
    monitor.reset()
    return [monitor.score(frame(t, kind, seed)) for t in np.linspace(0, 1, 12)]


def test_good_flip_on_a_new_page_never_stops(tmp_path):
    for s in (1, 2, 3):
        record(tmp_path, "good", s)
    scores = run(FlipMonitor(tmp_path), "good", 99)
    assert not any(s.stop for s in scores)


def test_fold_triggers_stop_without_bad_examples(tmp_path):
    record(tmp_path, "good", 1)
    scores = run(FlipMonitor(tmp_path), "fold", 99)
    first = next(i for i, s in enumerate(scores) if s.stop)
    assert 5 <= first <= 9                                  # fold starts at t>0.4 (frame 5) + 3-frame streak
    assert not any(s.anomalous for s in scores[:5])         # no false alarm before the fold


def test_bad_examples_catch_it_even_with_loose_threshold(tmp_path):
    record(tmp_path, "good", 1)
    record(tmp_path, "fold", 2)
    m = FlipMonitor(tmp_path, max_deviation=10)             # deviation alone would never trigger
    assert any(s.stop and "BAD" in s.reason for s in run(m, "fold", 99))


def test_no_references_means_no_alarm(tmp_path):
    m = FlipMonitor(tmp_path)
    assert m.counts() == {"good": 0, "bad": 0}
    assert not any(s.stop for s in run(m, "fold", 1))
