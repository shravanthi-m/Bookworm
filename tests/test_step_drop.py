import numpy as np
import pytest

from bookworm.grasp.step_drop import StepDrop


def _feed(sd, zs, cmds=None, loads=None):
    for i, z in enumerate(zs):
        r = sd.check(z, None if cmds is None else cmds[i], None if loads is None else loads[i])
        if r is not None:
            return i, r
    return None, None


def test_free_air_steps_never_stall():
    sd = StepDrop()
    assert _feed(sd, [50, 48, 46, 44.1, 42.0, 40.2]) == (None, None)


def test_a_step_that_does_not_go_down_is_a_stall_and_reports_where_to_go_back_to():
    sd = StepDrop()
    i, r = _feed(sd, [50, 48, 46, 44, 43.9, 43.9])
    assert i == 4 and r.tip_z == 43.9 and r.previous_tip_z == 44 and r.dropped_mm == pytest.approx(0.1)
    assert "stalled" in r.text() or "position" in r.text()
    assert r.step == 3                                                            # the 4th drop (0-based 3)


def test_a_half_landed_step_counts_a_fully_landed_one_does_not():
    assert _feed(StepDrop(), [50, 48, 46.9])[0] is None                           # 1.1 mm of 2: arm noise, keep going
    assert _feed(StepDrop(), [50, 48, 47.5])[0] == 2                              # 0.5 mm of 2: ran into something


def test_switched_off_never_reports():
    sd = StepDrop(enabled=False)
    assert _feed(sd, [50, 48, 48, 48, 48]) == (None, None) and sd.history == [50, 48, 48, 48, 48]


def test_the_first_reading_is_only_a_baseline():
    assert StepDrop().check(50) is None


def test_bad_settings_are_refused():
    with pytest.raises(ValueError):
        StepDrop(step_mm=0)
    with pytest.raises(ValueError):
        StepDrop(step_mm=2, min_drop_mm=3)
    with pytest.raises(ValueError):
        StepDrop(mode="guess")


# ---- regression mode --------------------------------------------------------------------------------------
CMD = [50 - 2 * i for i in range(12)]


def _real(landed_until, n=12, rng=None):
    """Measured tip z: follows the command exactly (+ tiny noise) until the surface, then sits on it."""
    rng = rng or np.random.default_rng(0)
    return [max(c, landed_until) + rng.normal(0, 0.03) for c in CMD[:n]]


def test_regression_catches_a_partly_landed_step_the_fixed_rule_misses():
    z = _real(38.4)                       # surface 0.4 mm above the step at cmd 38: that step lands 1.6 of 2 mm
    fixed = _feed(StepDrop(mode="fixed"), z, CMD)[0]
    reg = _feed(StepDrop(mode="regression"), z, CMD)[0]
    assert reg == 6                       # the step itself: it landed higher than the trend says
    assert fixed == 7                     # the fixed rule only sees the NEXT step, which cannot move at all


def test_regression_stays_quiet_in_free_air_with_noise():
    z = _real(-100, rng=np.random.default_rng(3))
    assert _feed(StepDrop(mode="regression"), z, CMD) == (None, None)


def test_regression_flags_a_joint_load_that_leaves_its_trend_before_any_stall():
    rng = np.random.default_rng(1)
    loads = [[0.0, 1.9 - 0.004 * c + rng.normal(0, 0.004), 0.9, 0.05, 0.3, 0.0] for c in CMD[:10]]
    loads[7][2] += 6.0                                                          # joint 3 starts to push back at step 7 (Nm)
    z = _real(-100, n=10)                                                       # but the tip still followed
    i, r = _feed(StepDrop(mode="regression"), z, CMD, loads)
    assert i == 7 and r.reason == "load"
    assert _feed(StepDrop(mode="fixed"), z, CMD, loads) == (None, None)         # the fixed rule does not look at load


def test_load_can_be_left_out_and_the_trace_records_both_rules_whichever_is_active():
    z = _real(38.4)
    sd = StepDrop(mode="fixed", use_load=False)
    _feed(sd, z, CMD)
    assert all("flags" in t and "pos_z" in t for t in sd.trace)
    assert any(t["flags"]["pos"] for t in sd.trace)                             # the chart can show what regression would have said
    assert sd.predict(38) is not None and StepDrop().predict(38) is None


def test_regression_gives_no_false_alarms_over_many_noisy_free_air_descents():
    false_alarms = 0
    for seed in range(400):
        rng = np.random.default_rng(seed)
        cmds = [60 - 2 * i for i in range(20)]
        z = [c + rng.normal(0, 0.05) for c in cmds]                              # 0.05 mm of read-back noise
        loads = [[0.0, 1.9 - 0.004 * c + rng.normal(0, 0.004), 0.9 - 0.002 * c + rng.normal(0, 0.004), 0.05, 0.3, 0.0] for c in cmds]
        if _feed(StepDrop(mode="regression"), z, cmds, loads)[0] is not None:
            false_alarms += 1
    assert false_alarms == 0, false_alarms


# ---- the real arm (armfarm22, 2026-10-17): the tip hit at step 14, the read-back never showed it ---------------
def _real_run():
    import json
    from pathlib import Path
    d = json.loads((Path(__file__).parent / "data" / "zlink_real_20261017_table.json").read_text())
    tr = d["traces"]["table"]
    return [t["meas"] for t in tr], [t["cmd"] for t in tr], [t["load"] for t in tr]


def test_real_run_the_position_rules_miss_the_contact():
    z, cmd, loads = _real_run()
    assert _feed(StepDrop(mode="fixed"), z, cmd, loads) == (None, None)
    assert _feed(StepDrop(mode="regression", use_load=False), z, cmd, loads) == (None, None)


def test_real_run_the_load_rule_stops_at_the_step_that_hit():
    z, cmd, loads = _real_run()
    i, r = _feed(StepDrop(mode="regression"), z, cmd, loads)
    assert i == 14 and r.reason == "load" and abs(r.previous_tip_z - 17.5) < 0.1    # back to step 13, just above the contact


def test_real_run_free_air_steps_raise_no_alarm():
    z, cmd, loads = _real_run()
    assert _feed(StepDrop(mode="regression"), z[:14], cmd[:14], loads[:14]) == (None, None)


def test_a_contact_before_there_is_a_trend_is_still_caught():
    """The page can be higher than expected: contact in the first steps, before min_points, must not join the trend."""
    sd = StepDrop(mode="regression", load_joints=[1, 2], load_floor=0.3, load_k=8)
    loads = [[0, -21.0 - 0.05 * i, -14.0 + 0.07 * i, 0, -1.86, 0] for i in range(3)] + [[0, -15.2, -10.6, 0, -1.86, 0]]
    hits = [sd.check(40.0 - 2 * i, 40.0 - 2 * i, ld) for i, ld in enumerate(loads)]
    assert hits[:3] == [None, None, None] and hits[3] is not None and hits[3].reason == "load"
