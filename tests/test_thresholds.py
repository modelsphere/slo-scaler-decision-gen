"""Both sides of every threshold, in BOTH directions (R2, I3 lesson).

The wrong implementation (ceiling-on-both) is what these tests must
reject. If a test here is green on a Direction.FLOOR violation below
threshold AND on Direction.FLOOR safety above, the same-kind rule on
CEILING would also be green — and that's the point: the two
directions must not agree.

Equality is neutral (C14): not a violation, not a comfort.
"""

from decision_gen.thresholds import (
    Direction, Verdict, classify, comfortable, violated,
)


# ---------- ceiling ----------

def test_ceiling_violated_above():
    assert violated(Direction.CEILING, 21.0, 20.0)


def test_ceiling_safe_below():
    assert not violated(Direction.CEILING, 19.0, 20.0)


def test_ceiling_comfortable_well_below():
    assert comfortable(Direction.CEILING, 9.9, 20.0, 0.5)


def test_ceiling_grey_between_threshold_and_headroom():
    assert not violated(Direction.CEILING, 15.0, 20.0)
    assert not comfortable(Direction.CEILING, 15.0, 20.0, 0.5)
    assert classify(Direction.CEILING, 15.0, 20.0, 0.5) is Verdict.GREY


def test_ceiling_equal_to_threshold_is_neutral():
    assert not violated(Direction.CEILING, 20.0, 20.0)
    assert not comfortable(Direction.CEILING, 20.0, 20.0, 0.5)
    assert classify(Direction.CEILING, 20.0, 20.0, 0.5) is Verdict.GREY


# ---------- floor (the I3 side) ----------

def test_floor_violated_below():
    assert violated(Direction.FLOOR, 29.0, 30.0)


def test_floor_safe_above():
    assert not violated(Direction.FLOOR, 31.0, 30.0)


def test_floor_comfortable_well_above():
    assert comfortable(Direction.FLOOR, 61.0, 30.0, 0.5)


def test_floor_grey_between_headroom_and_threshold():
    assert not violated(Direction.FLOOR, 45.0, 30.0)
    assert not comfortable(Direction.FLOOR, 45.0, 30.0, 0.5)
    assert classify(Direction.FLOOR, 45.0, 30.0, 0.5) is Verdict.GREY


def test_floor_equal_to_threshold_is_neutral():
    assert not violated(Direction.FLOOR, 30.0, 30.0)
    assert not comfortable(Direction.FLOOR, 30.0, 30.0, 0.5)
    assert classify(Direction.FLOOR, 30.0, 30.0, 0.5) is Verdict.GREY


# ---------- classify at headroom edges ----------

def test_classify_ceiling_at_headroom_edge():
    # thr × 0.5 exactly: strict < means edge is NOT comfortable.
    assert classify(Direction.CEILING, 10.0, 20.0, 0.5) is Verdict.GREY
    assert classify(Direction.CEILING, 9.99, 20.0, 0.5) is Verdict.COMFORTABLE


def test_classify_floor_at_headroom_edge():
    # thr ÷ 0.5 exactly: strict > means edge is NOT comfortable.
    assert classify(Direction.FLOOR, 60.0, 30.0, 0.5) is Verdict.GREY
    assert classify(Direction.FLOOR, 60.01, 30.0, 0.5) is Verdict.COMFORTABLE


# ---------- classify must be asymmetric across directions ----------

def test_wrong_direction_would_flip_results():
    """I3 regression: a value that's violated under FLOOR must NOT be
    violated under CEILING, and vice-versa. If an impl treats both
    directions the same, exactly one branch of this test would fail
    — which is the bug we shipped."""
    assert classify(Direction.FLOOR,   25.0, 30.0, 0.5) is Verdict.VIOLATED
    assert classify(Direction.CEILING, 25.0, 30.0, 0.5) is Verdict.GREY  # not violated: 25 < 30
    assert classify(Direction.CEILING, 35.0, 30.0, 0.5) is Verdict.VIOLATED
    assert classify(Direction.FLOOR,   35.0, 30.0, 0.5) is Verdict.GREY  # not violated: 35 > 30
