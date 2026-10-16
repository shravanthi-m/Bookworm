"""Flip planning, stack heights, safety limits, safe stop and pause-for-help. No arm needed."""
import pytest

from bookworm.config import load_config
from bookworm.hardware import BookFrame, HardwareError, MockRobot, plan_flip, retreat
from bookworm.hardware.flip import surface_z
from bookworm.vision import FlipOutcome
from test_session import WORDS, make_session

# Book 128 mm wide, left stack 3 mm lower than the right at calibration.
BOOK = BookFrame([100, 0, 50], [100, 190, 50], [228, 0, 50], left_corner=[-28, 0, 47])


def cfg():
    c = load_config()
    c["flip"]["sequence"] = [s for s in c["flip"]["sequence"] if "wait" not in s]
    return c


def test_calibrated_width_and_left_offset():
    assert BOOK.right_width == pytest.approx(128)
    assert BOOK.left_dz == pytest.approx(-3)
    assert BookFrame.identity().right_width is None


def test_stacks_move_in_opposite_directions():
    b = cfg()["book"]
    assert surface_z(BOOK, b, "right", 10) == pytest.approx(-10 * b["z_drop_per_flip_mm"])
    assert surface_z(BOOK, b, "left", 10) == pytest.approx(-3 + 10 * b["z_drop_per_flip_mm"])


def test_default_corner_comes_from_calibration_not_config():
    c = cfg()
    c["book"]["page_width_mm"] = 999          # would be way off if it were used
    first = plan_flip(BOOK, c)[0]
    assert first[2][0] == pytest.approx(128 - 20)   # hover 20 mm inside the touched corner


def test_left_waypoints_track_the_left_stack():
    c = cfg()
    early = [s for s in plan_flip(BOOK, c, flip_index=0) if s[0] == "move" and s[3] == "left"]
    late = [s for s in plan_flip(BOOK, c, flip_index=100) if s[0] == "move" and s[3] == "left"]
    assert early, "flip sequence should lay and flatten the leaf on the left stack"
    rise = 100 * c["book"]["z_drop_per_flip_mm"]
    for e, l in zip(early, late):
        assert l[2][2] - e[2][2] == pytest.approx(rise)


def test_pressing_too_hard_is_refused_before_any_move():
    c = cfg()
    c["flip"]["sequence"].insert(1, {"move": {"x": 0, "y": 20, "z": -10, "from": "corner"}})
    with pytest.raises(HardwareError, match="presses"):
        plan_flip(BOOK, c)


def test_waypoint_off_the_book_is_refused():
    c = cfg()
    with pytest.raises(HardwareError, match="outside the book"):
        plan_flip(BOOK, c, corner_xy=(400.0, 0.0))     # a wild vision corner


def test_retreat_lifts_the_pencil_then_goes_home():
    c = cfg()
    c["poses"]["home"] = {"x": 1, "y": 2, "z": 300, "o_x": 0, "o_y": 0, "o_z": -1, "theta": 0}
    robot = MockRobot()
    robot.move_to({"x": 10, "y": 20, "z": 5, "o_x": 0, "o_y": 0, "o_z": -1, "theta": 0})
    retreat(robot, c)
    moves = [e[1] for e in robot.log if e[0] == "move"]
    assert moves[-2]["z"] == pytest.approx(5 + c["robot"]["safe_height_mm"])
    assert moves[-1]["z"] == 300


def test_ctrl_c_mid_run_retreats():
    session, robot, _ = make_session([(2, WORDS), (4, WORDS)])

    def boom(*a, **k):
        raise KeyboardInterrupt
    session.flip_and_verify = boom
    with pytest.raises(KeyboardInterrupt):
        session.run()
    assert robot.log[-1][0] == "move"     # the retreat lift happened


def test_failed_flip_pauses_for_help_then_continues():
    session, robot, handler = make_session([(2, WORDS), (4, WORDS), (6, WORDS)], stuck=True)
    cam = robot.camera

    def person_turns_page(prompt):
        cam.i += 1
        return ""
    session.ask_for_help = person_turns_page
    cam.stuck = False
    session.flip_and_verify = lambda before: (FlipOutcome.NO_FLIP, session.capture_spread())
    summary = session.run(max_spreads=2)
    assert summary.flips.get("by_hand") == 1 and summary.spreads == 2
    assert any("help" in s for s in handler.speaker.spoken)


def test_failed_flip_stops_when_person_types_q():
    session, _, _ = make_session([(2, WORDS), (4, WORDS)], stuck=True)
    session.ask_for_help = lambda prompt: "q"
    assert session.run().stopped_because == "flip failed"
