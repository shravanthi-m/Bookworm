"""Midhat's recorded flip (scripts/flip_path.py) and the control page running it over its own
robot connection — with a fake arm, so no robot is needed."""
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import flip_path  # noqa: E402


class FakeArm:
    def __init__(self, off_at=None):
        self.moves, self.stops, self.now, self.off_at = [], 0, [0.0] * 6, off_at

    def move_to_joint_positions(self, jp):
        self.moves.append(list(jp.values))
        self.now = [v + (2.0 if len(self.moves) == self.off_at else 0.0) for v in jp.values]

    def get_joint_positions(self):
        return SimpleNamespace(values=self.now)

    def stop(self):
        self.stops += 1


class FakeConn:
    """Stands in for ViamConnection: the fake arm's calls return plain values, not coroutines."""

    def __init__(self, arm):
        self.arm, self.dials = arm, 0

    def call(self, value, timeout=None):
        return value

    def component(self, cls, name):
        return self.arm


def test_v2_plan_goes_1_to_11_then_parks():
    pts, path, between = flip_path.load("v2")
    plan = flip_path.make_plan(path, between)
    assert plan[:11] == [str(n) for n in range(1, 12)]
    assert plan[11:] == ["out_of_the_way", "wait 0"]
    assert all(p in pts for p in plan[:-1])


def test_v1_plan_unchanged():
    pts, path, between = flip_path.load("v1")
    plan = flip_path.make_plan(path, between)
    assert plan == ["0", "0.5", "start", "deep", "lift", "3b", "press_down", "push_back", "final",
                    "6b", "0", "out_of_the_way", "wait 0"]


def test_play_moves_through_every_point_in_order():
    pts, path, between = flip_path.load("v2")
    plan = flip_path.make_plan(path, between)
    arm, steps = FakeArm(), []
    code, _ = flip_path.play(FakeConn(arm), arm, plan, pts, on_step=steps.append, log=lambda m: None)
    assert code == 0
    assert steps == plan[:-1]
    assert arm.moves == [pts[p] for p in plan[:-1]]


def test_play_stops_between_moves():
    pts, path, between = flip_path.load("v2")
    arm = FakeArm()
    code, _ = flip_path.play(FakeConn(arm), arm, flip_path.make_plan(path, between), pts,
                             stopped=lambda: len(arm.moves) >= 3, log=lambda m: None)
    assert code == 130 and len(arm.moves) == 3


def test_play_halts_on_joint_error():
    pts, path, between = flip_path.load("v2")
    arm = FakeArm(off_at=4)
    code, msg = flip_path.play(FakeConn(arm), arm, flip_path.make_plan(path, between), pts, log=lambda m: None)
    assert code == 2 and "STOPPED at 4" in msg
    assert len(arm.moves) == 4 and arm.stops == 1


def test_control_page_flips_over_its_own_connection():
    from bookworm.ui.server import Controller

    arm = FakeArm()
    ctrl = Controller.__new__(Controller)            # skip camera/OCR/speech setup
    ctrl.cfg = {"robot": {"arm": "arm"}, "ui": {"flip_path": "v2", "verify_flip": False}}   # as in default.yaml
    ctrl.camera = SimpleNamespace(conn=FakeConn(arm))
    ctrl.state = {"flip_step": None, "flipping": False, "status": "Ready"}
    ctrl._stop = threading.Event()

    def no_subprocess(*_):
        raise AssertionError("flip must not start a separate script (it would dial the robot again)")
    ctrl._command = no_subprocess

    ctrl._flip()
    assert len(arm.moves) == 12                      # points 1-11 + park
    assert ctrl.state["flip_step"] == "v2:out_of_the_way"
    assert ctrl.state["status"] == "Flipped" and ctrl.state["flipping"] is False
