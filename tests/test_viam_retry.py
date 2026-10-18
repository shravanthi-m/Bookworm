import pytest

from bookworm.hardware.workshop import WorkshopConnection as ViamConnection, is_transient


class Expired(Exception):
    pass


def _conn(outcomes):
    """A ViamConnection whose call() plays back the given outcomes (exceptions or values)."""
    c = ViamConnection.__new__(ViamConnection)
    seen = []

    def call(coro, timeout=60.0):
        seen.append(coro)
        o = outcomes.pop(0)
        if isinstance(o, BaseException):
            raise o
        return o

    c.call = call
    return c, seen


def test_session_expired_is_retried_with_a_fresh_call():
    c, seen = _conn([Expired("(<Status.INVALID_ARGUMENT: 3>, 'SESSION_EXPIRED', None)"), "moved"])
    said = []
    made = []
    assert c.call_retry(lambda: made.append(1) or "coro", say=said.append, what="arm move") == "moved"
    assert len(made) == 2 and len(seen) == 2 and "retrying" in said[0]


def test_real_errors_are_not_retried():
    c, seen = _conn([ValueError("all IK solutions failed constraints")])
    with pytest.raises(ValueError):
        c.call_retry(lambda: "coro")
    assert len(seen) == 1


def test_gives_up_after_the_last_attempt():
    c, seen = _conn([Expired("SESSION_EXPIRED")] * 3)
    with pytest.raises(Expired):
        c.call_retry(lambda: "coro", attempts=3)
    assert len(seen) == 3


def test_transient_markers():
    assert is_transient(Exception("context canceled")) and not is_transient(Exception("planner refused"))
