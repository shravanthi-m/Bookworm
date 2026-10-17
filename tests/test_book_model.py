"""The book model: depth heights + the table's z + a correction fitted to the page touches."""
import numpy as np
import pytest

from bookworm.gui import book_model as bmod


def _res(h=30.0):
    xs, ys = np.arange(300, 500, 2.0), np.arange(-150, 150, 2.0)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    book = (X > 320) & (X < 480) & (np.abs(Y) < 130)
    z = np.where(book, h + 0.02 * (X - 400), 0.0)
    return {"x": xs, "y": ys, "z": z, "book": book.astype(np.uint8)}, X, Y


def test_no_page_touch_keeps_the_depth_map():
    res, X, Y = _res()
    m = bmod.build(res, 16.0, [{"key": "table", "kind": "table", "x": 310, "y": 140, "z": 16.0}])
    assert m["correction"]["kind"] == "none" and m["touches"][0]["residual"] is None
    assert bmod.z_at(m, 400, 0) == pytest.approx(46.0, abs=0.1)


def test_one_page_touch_shifts_the_book_but_not_the_table():
    res, X, Y = _res()
    m = bmod.build(res, 16.0, [{"key": "cl", "kind": "page_check", "x": 360, "y": 0, "z": 16 + 30 - 0.8 + 3.0}])
    assert m["correction"]["kind"] == "constant" and m["correction"]["a"] == pytest.approx(3.0, abs=0.1)
    assert bmod.z_at(m, 360, 0) == pytest.approx(16 + 30 - 0.8 + 3.0, abs=0.1)
    assert bmod.z_at(m, 310, 140) == pytest.approx(16.0, abs=0.1)                 # the table did not move


def test_spread_touches_fit_a_plane():
    res, X, Y = _res()
    true = lambda x, y: 16 + 30 + 0.02 * (x - 400) + 1.0 + 0.01 * (x - 400) - 0.02 * y   # noqa: E731  map off by a tilt
    t = [{"key": str(i), "kind": "demo", "x": x, "y": y, "z": true(x, y)}
         for i, (x, y) in enumerate([(340, -100), (460, -100), (460, 100), (340, 100), (400, 0)])]
    m = bmod.build(res, 16.0, t)
    c = m["correction"]
    assert c["kind"] == "plane" and c["rms_after"] < 0.05 < c["rms_before"]
    assert bmod.z_at(m, 420, 50) == pytest.approx(true(420, 50), abs=0.1)


def test_collinear_touches_fall_back_to_a_constant():
    res, X, Y = _res()
    t = [{"key": str(i), "kind": "demo", "x": 360, "y": y, "z": 46 - 0.8 + 2} for i, y in enumerate((-100, 0, 100))]
    assert bmod.build(res, 16.0, t)["correction"]["kind"] == "constant"


def test_save_and_load(tmp_path):
    res, X, Y = _res()
    m = bmod.build(res, 16.0, [{"key": "cl", "kind": "page_check", "x": 360, "y": 0, "z": 46.0}],
                   points={"cl": [360, 0], "cr": [440, 0]}, meta={"tool_length_mm": 155})
    assert set(m["pages"]) == {"left", "right"} and m["pages"]["left"]["tilt_deg"] == pytest.approx(1.15, abs=0.1)
    back = bmod.load(bmod.save(m, tmp_path / "b"))
    assert back["tool_length_mm"] == 155 and bmod.z_at(back, 400, 0) == pytest.approx(bmod.z_at(m, 400, 0))
