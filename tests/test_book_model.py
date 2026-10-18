"""The book model: driven by the page touches; the depth map is only a reference (the outline and a 'depth is off by' number)."""
import numpy as np
import pytest

from bookworm.gui import book_model as bmod


def _res(h=30.0):
    xs, ys = np.arange(300, 500, 2.0), np.arange(-150, 150, 2.0)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    book = (X > 320) & (X < 480) & (np.abs(Y) < 130)
    z = np.where(book, h + 0.02 * (X - 400), 0.0)
    return {"x": xs, "y": ys, "z": z, "book": book.astype(np.uint8)}, X, Y


def test_no_page_touch_falls_back_to_the_depth_map():
    res, X, Y = _res()
    m = bmod.build(res, 16.0, [{"key": "table", "kind": "table", "x": 310, "y": 140, "z": 16.0}])
    assert m["driven"] == "depth" and m["level"]["kind"] == "none"
    assert bmod.z_at(m, 400, 0) == pytest.approx(46.0, abs=0.1)


def test_one_page_touch_sets_one_level_and_the_depth_map_does_not_shape_it():
    res, X, Y = _res()
    m = bmod.build(res, 16.0, [{"key": "bl", "kind": "page_check", "x": 360, "y": 0, "z": 35.0}])
    assert m["driven"] == "touch" and m["level"]["kind"] == "constant" and m["level"]["a"] == pytest.approx(35.0)
    assert bmod.z_at(m, 360, 0) == pytest.approx(35.0, abs=0.01)
    assert bmod.z_at(m, 470, 100) == pytest.approx(35.0, abs=0.01)                 # flat over the book, whatever depth says
    assert bmod.z_at(m, 310, 140) == pytest.approx(16.0, abs=0.01)                 # outside the outline: the table
    t = m["touches"][0]
    assert t["depth_z"] == pytest.approx(16 + 30 - 0.8, abs=0.1) and t["depth_off"] == pytest.approx(35 - (16 + 30 - 0.8), abs=0.1)


def test_spread_touches_fit_a_plane():
    res, X, Y = _res()
    true = lambda x, y: 40 + 0.03 * (x - 400) - 0.02 * y                          # noqa: E731
    t = [{"key": str(i), "kind": "demo", "x": x, "y": y, "z": true(x, y)}
         for i, (x, y) in enumerate([(340, -100), (460, -100), (460, 100), (340, 100), (400, 0)])]
    m = bmod.build(res, 16.0, t)
    c = m["level"]
    assert c["kind"] == "plane" and c["rms_after"] < 0.05 < c["spread"]
    assert bmod.z_at(m, 420, 50) == pytest.approx(true(420, 50), abs=0.1)


def test_collinear_touches_fall_back_to_one_level():
    res, X, Y = _res()
    t = [{"key": str(i), "kind": "demo", "x": 360, "y": y, "z": 36.0 + i} for i, y in enumerate((-100, 0, 100))]
    m = bmod.build(res, 16.0, t)
    assert m["level"]["kind"] == "constant" and m["level"]["a"] == pytest.approx(37.0)


def test_save_and_load(tmp_path):
    res, X, Y = _res()
    m = bmod.build(res, 16.0, [{"key": "cl", "kind": "page_check", "x": 360, "y": 0, "z": 46.0}],
                   points={"cl": [360, 0], "cr": [440, 0]}, meta={"tool_length_mm": 155})
    assert set(m["pages"]) == {"left", "right"} and m["pages"]["left"]["tilt_deg"] == pytest.approx(0.0, abs=0.01)
    back = bmod.load(bmod.save(m, tmp_path / "b"))
    assert back["tool_length_mm"] == 155 and back["driven"] == "touch"
    assert bmod.z_at(back, 400, 0) == pytest.approx(bmod.z_at(m, 400, 0))
