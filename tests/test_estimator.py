"""Table-driven tests for decision_gen.estimator."""

import pytest

from decision_gen import estimator
from decision_gen.estimator import (
    REJECTION_OK_FLOOR,
    REJECTION_THRESHOLD,
    SCALE_DOWN_COOLDOWN_S,
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
FAR_PAST = NOW - SCALE_DOWN_COOLDOWN_S - 60  # cooldown fully elapsed
RECENT = NOW - 10                            # still in cooldown


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


# ---------- decide: rejection spike ----------

def test_rejection_spike_multiplicative():
    cr = _cr(min_v=1, max_v=10)
    sig = _sig(rej=0.02)  # > REJECTION_THRESHOLD
    desired, reason = decide(cr, sig, current_replicas=4, last_change_at=NOW, now=NOW)
    assert desired == 6  # ceil(4*1.5)
    assert "rejection" in reason


def test_rejection_spike_short_circuits_nan_ttft():
    """Rule 1 fires even when the SLO signals are NaN — spike overrides."""
    cr = _cr()
    sig = _sig(ttft={"p80": float("nan")}, otps={"p80": float("nan")}, rej=0.5)
    desired, reason = decide(cr, sig, current_replicas=2, last_change_at=NOW, now=NOW)
    assert desired == 3  # ceil(2*1.5)
    assert "rejection" in reason


def test_rejection_spike_from_zero_scales_to_one():
    """ceil(0 * 1.5) = 0 mustn't pin us at zero — we must scale up to at least 1."""
    cr = _cr(min_v=0, max_v=10)
    sig = _sig(rej=0.05)
    desired, _ = decide(cr, sig, current_replicas=0, last_change_at=NOW, now=NOW)
    assert desired == 1


def test_rejection_at_threshold_does_not_trigger():
    cr = _cr()
    sig = _sig(rej=REJECTION_THRESHOLD)  # not strictly greater
    # falls through to violation/comfortable paths
    desired, reason = decide(cr, sig, current_replicas=2, last_change_at=RECENT, now=NOW)
    # cooldown still active → hold at 2 (violation path also False)
    assert desired == 2


# ---------- decide: SLO violation ----------

def test_ttft_violated_scales_up_by_step():
    cr = _cr()
    sig = _sig(ttft={"p80": 25.0}, rej=0.0)
    desired, reason = decide(cr, sig, current_replicas=2, last_change_at=RECENT, now=NOW)
    assert desired == 3
    assert "slo" in reason


def test_otps_violated_scales_up_by_step():
    cr = _cr()
    sig = _sig(otps={"p80": 35.0}, rej=0.0)
    desired, _ = decide(cr, sig, current_replicas=2, last_change_at=RECENT, now=NOW)
    assert desired == 3


def test_both_violated_still_plus_one():
    """Both violating doesn't mean +2 — one additive step per tick."""
    cr = _cr()
    sig = _sig(ttft={"p80": 100.0}, otps={"p80": 100.0}, rej=0.0)
    desired, _ = decide(cr, sig, current_replicas=2, last_change_at=RECENT, now=NOW)
    assert desired == 3


# ---------- decide: cooldown-gated scale-down ----------

def test_cooldown_not_elapsed_holds():
    cr = _cr()
    sig = _sig(rej=0.0)  # everything fine
    desired, reason = decide(cr, sig, current_replicas=3, last_change_at=NOW - 100, now=NOW)
    assert desired == 3
    assert "cooldown" in reason


def test_cooldown_elapsed_comfortable_quiet_scales_down():
    cr = _cr()
    sig = _sig(ttft={"p80": 5.0}, otps={"p80": 10.0}, rej=REJECTION_OK_FLOOR / 2)
    desired, reason = decide(cr, sig, current_replicas=3, last_change_at=FAR_PAST, now=NOW)
    assert desired == 2
    assert "down" in reason


def test_cooldown_elapsed_but_rejection_above_floor_holds():
    cr = _cr()
    sig = _sig(ttft={"p80": 5.0}, otps={"p80": 10.0}, rej=REJECTION_OK_FLOOR * 2)
    desired, reason = decide(cr, sig, current_replicas=3, last_change_at=FAR_PAST, now=NOW)
    assert desired == 3
    assert "rejection-above-floor" in reason


def test_cooldown_elapsed_but_not_comfortable_holds():
    cr = _cr()
    sig = _sig(ttft={"p80": 15.0}, otps={"p80": 10.0}, rej=0.0)  # 15 > 20*0.5=10
    desired, _ = decide(cr, sig, current_replicas=3, last_change_at=FAR_PAST, now=NOW)
    assert desired == 3


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
    sig = _sig(ttft={"p80": float("nan")}, otps={"p80": 100.0}, rej=0.0)
    desired, _ = decide(cr, sig, current_replicas=2, last_change_at=RECENT, now=NOW)
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
    sig = _sig(rej=0.0)
    desired, _ = decide(cr, sig, current_replicas=2, last_change_at=FAR_PAST, now=NOW)
    assert desired == 2


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
    desired, _ = decide(cr, sig, current_replicas=0, last_change_at=RECENT, now=NOW)
    assert desired == 1  # 0 + 1
