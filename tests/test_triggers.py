"""R1a / R1b / R1c sizing-rule tests.

Every rule has both sides of its gate: rejection ≥ 5% AND < 5%; some
verdict violated AND all-comfortable-and-none-violated; shed gate open
AND closed. Also the C1/C5 zero-base corner on both up paths.
"""

import pytest

from decision_gen.thresholds import Verdict
from decision_gen.triggers import (
    Gates, Proposal, evaluate, fire_alarm, quiet_shed, steady_growth,
)


GATES_CLOSED = Gates(up_cooldown_open=False, shed_ready=False)
GATES_OPEN = Gates(up_cooldown_open=True, shed_ready=True)


def _viol(**kv):
    """Build a verdict dict: _viol(**{'ttft.p80': Verdict.VIOLATED})."""
    return kv


def _comfy(n=2):
    return {f"m{i}": Verdict.COMFORTABLE for i in range(n)}


# ---------- R1a fire_alarm ----------

def test_r1a_fires_at_threshold():
    p = fire_alarm(0.05, current=10)
    assert p is not None
    assert p.rule == "r1a-fire"


def test_r1a_silent_below_threshold():
    assert fire_alarm(0.049, current=10) is None
    assert fire_alarm(0.0, current=10) is None
    assert fire_alarm(None, current=10) is None


def test_r1a_sizes_to_deficit():
    # r=0.0526 ≈ 1/19. deficit = 0.0526/(1-0.0526) ≈ 0.0555.
    # mult = 1 + 0.0555 × 2.0 = 1.111; ceil(10 × 1.111) = 12.
    p = fire_alarm(0.0526, current=10)
    assert p.replicas == 12


def test_r1a_cap_limits_huge_rejection():
    # Huge r → mult capped at 1.5.
    p = fire_alarm(0.9, current=10)
    assert p.replicas == 15


def test_r1a_never_below_current_plus_one():
    # Tiny but above-threshold rejection; ceil(current × mult) may not
    # exceed current. Floor is current+1.
    p = fire_alarm(0.05, current=100)
    assert p.replicas >= 101


def test_r1a_zero_base_scales_to_one(C1=None):
    # C1 / C5: current=0 must not pin.
    p = fire_alarm(0.1, current=0)
    assert p.replicas >= 1


def test_r1a_ignores_gate_state():
    """Ungated by construction: fire_alarm takes no gates argument.
    This test pins that evaluate() checks it first even with gates closed."""
    p = evaluate({}, rejection_rate=0.1, current=10, gates=GATES_CLOSED)
    assert p is not None and p.rule == "r1a-fire"


# ---------- R1b steady_growth ----------

def test_r1b_fires_on_any_violation():
    v = _viol(**{"ttft.p80": Verdict.VIOLATED, "otps.p80": Verdict.COMFORTABLE})
    p = steady_growth(v, current=10)
    assert p is not None and p.rule == "r1b-steady"
    assert p.replicas == 10 + max(1, int(10 * 0.15 + 0.999))


def test_r1b_silent_without_violation():
    v = _viol(**{"ttft.p80": Verdict.GREY, "otps.p80": Verdict.COMFORTABLE})
    assert steady_growth(v, current=10) is None


def test_r1b_min_step_covers_zero_base():
    v = _viol(**{"ttft.p80": Verdict.VIOLATED})
    p = steady_growth(v, current=0)
    assert p is not None
    assert p.replicas >= 1


def test_r1b_blocked_when_gate_closed():
    v = _viol(**{"ttft.p80": Verdict.VIOLATED})
    p = evaluate(v, rejection_rate=0.0, current=10, gates=GATES_CLOSED)
    # Up gate closed → r1b unreachable. Shed gate also closed → None.
    assert p is None


def test_r1b_fires_when_gate_open():
    v = _viol(**{"ttft.p80": Verdict.VIOLATED})
    p = evaluate(v, rejection_rate=0.0, current=10, gates=GATES_OPEN)
    assert p is not None and p.rule == "r1b-steady"


# ---------- R1c quiet_shed ----------

def test_r1c_fires_on_all_comfortable():
    p = quiet_shed(_comfy(), rejection_rate=0.0005, current=10)
    assert p is not None and p.rule == "r1c-shed"
    assert p.replicas == 10 - max(1, int(10 * 0.15 + 0.999))


def test_r1c_refused_on_grey():
    v = _viol(**{"ttft.p80": Verdict.GREY, "otps.p80": Verdict.COMFORTABLE})
    assert quiet_shed(v, rejection_rate=0.0, current=10) is None


def test_r1c_refused_on_violation():
    v = _viol(**{"ttft.p80": Verdict.VIOLATED, "otps.p80": Verdict.COMFORTABLE})
    assert quiet_shed(v, rejection_rate=0.0, current=10) is None


def test_r1c_refused_on_rejection_at_floor():
    assert quiet_shed(_comfy(), rejection_rate=0.001, current=10) is None
    assert quiet_shed(_comfy(), rejection_rate=0.0009, current=10) is not None


def test_r1c_refused_without_rejection_reading():
    assert quiet_shed(_comfy(), rejection_rate=None, current=10) is None


def test_r1c_blocked_when_gate_closed():
    p = evaluate(_comfy(), rejection_rate=0.0005, current=10, gates=GATES_CLOSED)
    assert p is None


def test_r1c_fires_when_gate_open():
    p = evaluate(_comfy(), rejection_rate=0.0005, current=10, gates=GATES_OPEN)
    assert p is not None and p.rule == "r1c-shed"


# ---------- priority ----------

def test_r1a_wins_over_r1b():
    v = _viol(**{"ttft.p80": Verdict.VIOLATED})
    p = evaluate(v, rejection_rate=0.1, current=10, gates=GATES_OPEN)
    assert p.rule == "r1a-fire"


def test_hold_returns_none():
    p = evaluate(_comfy(), rejection_rate=0.0, current=10, gates=GATES_CLOSED)
    assert p is None
