import numpy as np
import pytest

from bookworm.grasp.probe import flange_for_tip, tip_of, tool_axis

DOWN = {"x": 400.0, "y": -50.0, "z": 300.0, "o_x": 0.0, "o_y": 0.0, "o_z": -1.0, "theta": -5.6}


def test_tip_is_tool_length_along_the_tool_axis():
    assert np.allclose(tip_of(DOWN, 155), [400, -50, 145])
    tilted = dict(DOWN, o_x=0.1, o_z=-0.995)
    t = tip_of(tilted, 155)
    assert np.isclose(np.linalg.norm(t - [400, -50, 300]), 155) and t[0] > 400        # a tilted tool reaches sideways


def test_flange_for_tip_round_trips_and_keeps_the_orientation():
    tilted = dict(DOWN, o_x=0.05, o_z=-0.9987)
    f = flange_for_tip([420.0, -30.0, 40.0], tilted, 155)
    assert np.allclose(tip_of(f, 155), [420, -30, 40]) and f["theta"] == tilted["theta"] and f["o_x"] == tilted["o_x"]


def test_tool_axis_is_normalised():
    assert np.isclose(np.linalg.norm(tool_axis(dict(DOWN, o_z=-2.0))), 1.0)
