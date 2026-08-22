"""Table-driven tests for decision_gen.estimator."""

import pytest

from decision_gen import estimator
from decision_gen.estimator import (
    REJECTION_OK_FLOOR,
    REJECTION_THRESHOLD,
    SCALE_DOWN_COOLDOWN_S,
    SCALE_UP_COOLDOWN_S,
    decide,
    is_no_signal,
    slo_comfortably_met,
    slo_violated,
)


def _cr(min_v=1, max_v=4, ttft=None, otps=None):
    return {
        "serviceId": "svc",
        "priority": 5,
        "minimumDeployment": {"type": "replica", "value": min_v},
        "maximumDeployment": {"type": "replica", "value": max_v},
        "ttft": {"default": ttft if ttft is not None else {
            "metrics": [{"type": "p80", "threshold": 20.0}]}},
        "otps": {"default": otps if otps is not None else {
            "metrics": [{"type": "p80", "threshold": 30.0}]}},
    }


def _sig(ttft=None, otps=None, rej=0.0):
    return {
        "ttft": ttft if ttft is not None else {"p80": 10.0},
        "otps": otps if otps is not None else {"p80": 15.0},
        "rejection_rate": rej,
    }


NOW = 1_000_000
FAR_PAST = NOW - SCALE_DOWN_COOLDOWN_S - 60  # down-cooldown fully elapsed
UP_RECENT = NOW - SCALE_UP_COOLDOWN_S - 60   # up-cooldown elapsed, down still gated
RECENT = NOW - 10                            # both cooldowns still active


# ---------- is_no_signal / helpers ----------

@pytest.mark.parametrize("v,expected", [
    (None, True), (float("nan"), True), (0.0, False), (1.0, False), (0, False), (5, False),
])
def test_is_no_signal(v, expected):
    assert is_no_signal(v) is expected


def test_slo_violated_basic():
    spec = {"metrics": [{"type": "p80", "threshold": 20.0}]}
    assert slo_violated(spec, {"p80": 25.0}) is True
    assert slo_violated(spec, {"p80": 15.0}) is False
    assert slo_violated(spec, {"p80": None}) is False        # no signal → not violation
    assert slo_violated(spec, {"p80": float("nan")}) is False
    assert slo_violated(spec, {}) is False                   # missing key → no signal


def test_slo_violated_any_of_many():
    spec = {"metrics": [
        {"type": "avg", "threshold": 50.0},
        {"type": "p95", "threshold": 200.0},
    ]}
    assert slo_violated(spec, {"avg": 10.0, "p95": 250.0}) is True
    assert slo_violated(spec, {"avg": 60.0, "p95": 100.0}) is True
    assert slo_violated(spec, {"avg": 10.0, "p95": 100.0}) is False


def test_slo_comfortably_met_basic():
    spec = {"metrics": [{"type": "p80", "threshold": 20.0}]}
    assert slo_comfortably_met(spec, {"p80": 9.9}) is True     # < 20*0.5=10
    assert slo_comfortably_met(spec, {"p80": 10.0}) is False   # not strict-less
    assert slo_comfortably_met(spec, {"p80": 15.0}) is False
    assert slo_comfortably_met(spec, {"p80": None}) is False
    assert slo_comfortably_met(spec, {}) is False


def test_slo_comfortably_met_empty_metrics():
    """Empty metrics list is trivially comfortable."""
    assert slo_comfortably_met({"metrics": []}, {}) is True
    assert slo_comfortably_met({}, {}) is True
    assert slo_comfortably_met(None, {}) is True


def test_slo_violated_floor():
    """OTPS is a throughput floor — violated when observed < threshold."""
    spec = {"metrics": [{"type": "p80", "threshold": 30.0}]}
    assert slo_violated(spec, {"p80": 25.0}, direction="floor") is True
    assert slo_violated(spec, {"p80": 35.0}, direction="floor") is False
    assert slo_violated(spec, {"p80": None}, direction="floor") is False
    assert slo_violated(spec, {"p80": float("nan")}, direction="floor") is False
    assert slo_violated(spec, {}, direction="floor") is False


def test_slo_comfortably_met_floor():
    """Floor comfort requires observed > threshold / SLO_HEADROOM = 60."""
    spec = {"metrics": [{"type": "p80", "threshold": 30.0}]}
    assert slo_comfortably_met(spec, {"p80": 61.0}, direction="floor") is True
    assert slo_comfortably_met(spec, {"p80": 60.0}, direction="floor") is False   # not strict-greater
    assert slo_comfortably_met(spec, {"p80": 45.0}, direction="floor") is False
    assert slo_comfortably_met(spec, {"p80": None}, direction="floor") is False
    assert slo_comfortably_met(spec, {}, direction="floor") is False


# ---------- decide: rejection spike ----------

def test_rejection_spike_at_threshold_caps_multiplier():
    """At rej=0.20 the mult formula gives 1+2*(0.2/0.8)=1.5 — exactly at the cap."""
    cr = _cr(min_v=1, max_v=10)
    sig = _sig(rej=0.20)
    desired, reason = decide(cr, sig, current_replicas=4, last_change_at=NOW, now=NOW)
    assert desired == 6  # ceil(4 * 1.5)
    assert "rejection" in reason


def test_rejection_spike_below_cap_uses_deficit_formula():
    """rej=0.125 → mult = 1+2*(0.125/0.875) ≈ 1.2857, below the 1.5 cap."""
    cr = _cr(min_v=1, max_v=10)
    sig = _sig(rej=0.125)
    desired, _ = decide(cr, sig, current_replicas=4, last_change_at=NOW, now=NOW)
    assert desired == 6  # ceil(4 * 1.2857) = ceil(5.143)


def test_rejection_spike_small_leak_rounds_to_at_least_plus_one():
    """A 5% spike against a small pool still grows by at least 1 replica,
    even though the deficit formula would ceil to less than current."""
    cr = _cr(min_v=1, max_v=10)
    sig = _sig(rej=REJECTION_THRESHOLD)  # exactly 5%
    desired, _ = decide(cr, sig, current_replicas=4, last_change_at=NOW, now=NOW)
    assert desired == 5  # 4 + 1, not 4


def test_rejection_spike_short_circuits_nan_ttft():
    """Rule 1 fires even when the SLO signals are NaN — spike overrides."""
    cr = _cr()
    sig = _sig(ttft={"p80": float("nan")}, otps={"p80": float("nan")}, rej=0.5)
    desired, reason = decide(cr, sig, current_replicas=2, last_change_at=NOW, now=NOW)
    assert desired == 3  # ceil(2 * 1.5) — capped multiplier
    assert "rejection" in reason


def test_rejection_spike_from_zero_scales_to_one():
    """ceil(0 * mult) = 0 mustn't pin us at zero — scale up to at least 1."""
    cr = _cr(min_v=0, max_v=10)
    sig = _sig(rej=0.5)
    desired, _ = decide(cr, sig, current_replicas=0, last_change_at=NOW, now=NOW)
    assert desired == 1


def test_rejection_below_threshold_does_not_trigger_rule1():
    """4% is hot but sub-threshold — rule 1 must NOT fire; only cooldown
    + SLO state governs this tick. otps sits at threshold exactly, so no
    violation fires either."""
    cr = _cr()
    sig = _sig(rej=0.04, otps={"p80": 35.0})
    desired, _ = decide(cr, sig, current_replicas=2, last_change_at=UP_RECENT, now=NOW)
    assert desired == 2


# ---------- decide: SLO violation ----------

def test_ttft_violated_scales_up_by_step():
    cr = _cr()
    sig = _sig(ttft={"p80": 25.0}, rej=0.0)
    desired, reason = decide(cr, sig, current_replicas=2, last_change_at=UP_RECENT, now=NOW)
    assert desired == 3
    assert "slo" in reason


def test_otps_violated_scales_up_by_step():
    """OTPS floor: below threshold is a violation; above is fine."""
    cr = _cr()
    sig = _sig(otps={"p80": 25.0}, rej=0.0)  # below 30-floor → violated
    desired, _ = decide(cr, sig, current_replicas=2, last_change_at=UP_RECENT, now=NOW)
    assert desired == 3


def test_otps_below_floor_but_ttft_fine_scales_up():
    """Even with ttft fine, otps dipping below its floor forces scale-up."""
    cr = _cr()
    sig = _sig(ttft={"p80": 25.0}, otps={"p80": 25.0}, rej=0.0)
    desired, _ = decide(cr, sig, current_replicas=2, last_change_at=UP_RECENT, now=NOW)
    assert desired == 3


def test_otps_comfortable_but_ttft_violated_scales_up():
    """otps above floor but ttft above ceiling still scales up — counters
    the old wrong-direction misread that called otps a violation."""
    cr = _cr()
    sig = _sig(ttft={"p80": 100.0}, otps={"p80": 45.0}, rej=0.0)
    desired, _ = decide(cr, sig, current_replicas=2, last_change_at=UP_RECENT, now=NOW)
    assert desired == 3


def test_both_violated_still_single_step():
    """Both violating doesn't mean double-step — one additive step per tick."""
    cr = _cr()
    sig = _sig(ttft={"p80": 100.0}, otps={"p80": 25.0}, rej=0.0)  # ttft above ceil, otps below floor
    desired, _ = decide(cr, sig, current_replicas=2, last_change_at=UP_RECENT, now=NOW)
    assert desired == 3


def test_step_scales_with_pool_size():
    """ceil(20 * 0.15) = 3 — step grows with current, not stuck at +1."""
    cr = _cr(min_v=1, max_v=40)
    sig = _sig(ttft={"p80": 100.0}, rej=0.0)
    desired, _ = decide(cr, sig, current_replicas=20, last_change_at=UP_RECENT, now=NOW)
    assert desired == 23


def test_violation_during_up_cooldown_holds():
    """Even with an SLO violation, if we changed within the last
    SCALE_UP_COOLDOWN_S we hold — jitter protection on +step path."""
    cr = _cr()
    sig = _sig(ttft={"p80": 25.0}, rej=0.0)
    desired, reason = decide(cr, sig, current_replicas=2, last_change_at=RECENT, now=NOW)
    assert desired == 2
    assert "cooldown" in reason


# ---------- decide: cooldown-gated scale-down ----------

def test_cooldown_not_elapsed_holds():
    cr = _cr()
    sig = _sig(rej=0.0)  # everything fine
    desired, reason = decide(cr, sig, current_replicas=3, last_change_at=NOW - 100, now=NOW)
    assert desired == 3
    assert "cooldown" in reason


def test_cooldown_elapsed_comfortable_quiet_scales_down():
    cr = _cr()
    sig = _sig(ttft={"p80": 5.0}, otps={"p80": 61.0}, rej=REJECTION_OK_FLOOR / 2)
    desired, reason = decide(cr, sig, current_replicas=3, last_change_at=FAR_PAST, now=NOW,
                             comfortable_since=FAR_PAST)
    assert desired == 2
    assert "down" in reason


def test_scale_down_step_scales_with_pool_size():
    """ceil(25 * 0.15) = 4 — shed grows with pool, mirrors rule 2's step."""
    cr = _cr(min_v=1, max_v=40)
    sig = _sig(ttft={"p80": 5.0}, otps={"p80": 61.0}, rej=0.0)
    desired, reason = decide(cr, sig, current_replicas=25, last_change_at=FAR_PAST, now=NOW,
                             comfortable_since=FAR_PAST)
    assert desired == 21
    assert "step=4" in reason


def test_scale_down_step_clamps_to_min():
    """Step overshooting min clamps, not bypasses: current=12, min=10,
    step=2 → desired=10 exactly. current=13, min=10, step=2 → desired=11."""
    cr = _cr(min_v=10, max_v=40)
    sig = _sig(ttft={"p80": 5.0}, otps={"p80": 61.0}, rej=0.0)
    desired, _ = decide(cr, sig, current_replicas=11, last_change_at=FAR_PAST, now=NOW,
                        comfortable_since=FAR_PAST)
    assert desired == 10


def test_cooldown_elapsed_but_rejection_above_floor_holds():
    cr = _cr()
    sig = _sig(ttft={"p80": 5.0}, otps={"p80": 61.0}, rej=REJECTION_OK_FLOOR * 2)
    desired, reason = decide(cr, sig, current_replicas=3, last_change_at=FAR_PAST, now=NOW)
    assert desired == 3
    assert "rejection-above-floor" in reason


def test_cooldown_elapsed_but_not_comfortable_holds():
    cr = _cr()
    sig = _sig(ttft={"p80": 15.0}, otps={"p80": 61.0}, rej=0.0)  # ttft 15 > 20*0.5=10
    desired, _ = decide(cr, sig, current_replicas=3, last_change_at=FAR_PAST, now=NOW)
    assert desired == 3


def test_cooldown_elapsed_but_otps_not_comfortable_holds():
    """Otps just above floor but below comfort boundary (30/0.5=60)
    blocks shed."""
    cr = _cr()
    sig = _sig(ttft={"p80": 5.0}, otps={"p80": 50.0}, rej=0.0)
    desired, _ = decide(cr, sig, current_replicas=3, last_change_at=FAR_PAST, now=NOW,
                        comfortable_since=FAR_PAST)
    assert desired == 3


def test_rule3_requires_sustained_comfort_window():
    """Comfortable ticks only count toward shed once SCALE_DOWN_COMFORT_S
    of continuous comfort has accumulated — one quiet minute after a hot
    hour is not enough."""
    cr = _cr()
    sig = _sig(ttft={"p80": 5.0}, otps={"p80": 61.0}, rej=0.0)
    # Comfortable for only 2 min — not yet 30.
    desired, reason = decide(cr, sig, current_replicas=3, last_change_at=FAR_PAST, now=NOW,
                             comfortable_since=NOW - 120)
    assert desired == 3
    assert "comfort-window" in reason
    # 30 min of comfort — shed allowed.
    desired, reason = decide(cr, sig, current_replicas=3, last_change_at=FAR_PAST, now=NOW,
                             comfortable_since=NOW - 1800)
    assert desired == 2
    assert "down" in reason


def test_rule3_no_comfort_history_blocks_shed():
    """None comfortable_since means 'we have never observed this service
    comfortably' — its shed gate is effectively infinite."""
    cr = _cr()
    sig = _sig(ttft={"p80": 5.0}, otps={"p80": 61.0}, rej=0.0)
    desired, reason = decide(cr, sig, current_replicas=3, last_change_at=FAR_PAST, now=NOW,
                             comfortable_since=None)
    assert desired == 3
    assert "comfort-window" in reason


# ---------- decide: NaN / missing signals ----------

def test_nan_everywhere_with_cooldown_elapsed_holds():
    """NaN input is never a violation AND never a comfortable-meet."""
    cr = _cr()
    sig = _sig(ttft={"p80": float("nan")}, otps={"p80": float("nan")}, rej=None)
    desired, _ = decide(cr, sig, current_replicas=3, last_change_at=FAR_PAST, now=NOW)
    assert desired == 3


def test_partial_nan_on_slo_uses_other_signal():
    """One SLO NaN doesn't block the other signal from driving decisions."""
    cr = _cr()
    sig = _sig(ttft={"p80": float("nan")}, otps={"p80": 25.0}, rej=0.0)  # below floor → violated
    desired, _ = decide(cr, sig, current_replicas=2, last_change_at=UP_RECENT, now=NOW)
    assert desired == 3  # OTPS violation drives


# ---------- decide: clamps ----------

def test_clamp_to_max_on_scale_up():
    cr = _cr(min_v=1, max_v=3)
    sig = _sig(rej=0.5)
    desired, reason = decide(cr, sig, current_replicas=3, last_change_at=NOW, now=NOW)
    assert desired == 3
    assert "clamp" in reason


def test_clamp_to_min_on_scale_down():
    cr = _cr(min_v=2, max_v=4)
    # otps sits comfortably above its 30/0.5=60 comfort boundary
    sig = _sig(ttft={"p80": 5.0}, otps={"p80": 61.0}, rej=0.0)
    desired, _ = decide(cr, sig, current_replicas=2, last_change_at=FAR_PAST, now=NOW)
    assert desired == 2   # raw step-1 down tries to go to 2-1=1, clamped to min=2


def test_malformed_cr_min_greater_than_max_clamps():
    """Defensive: min>max treated as max=min."""
    cr = {
        "minimumDeployment": {"value": 5},
        "maximumDeployment": {"value": 2},
        "ttft": {"default": {"metrics": [{"type": "p80", "threshold": 20}]}},
        "otps": {"default": {"metrics": [{"type": "p80", "threshold": 30}]}},
    }
    sig = _sig()
    desired, _ = decide(cr, sig, current_replicas=1, last_change_at=FAR_PAST, now=NOW)
    assert desired == 5  # max treated as min=5


# ---------- scale-from-zero ----------

def test_scale_from_zero_on_violation():
    cr = _cr(min_v=0, max_v=4)
    sig = _sig(ttft={"p80": 100.0}, rej=0.0)
    desired, _ = decide(cr, sig, current_replicas=0, last_change_at=UP_RECENT, now=NOW)
    assert desired == 1  # 0 + max(1, ceil(0*0.1)) = 0 + 1
