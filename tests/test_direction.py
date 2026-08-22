"""R5 reconciliation — the I4 regression test lives here.

v1 shipped without the IN_DRAIN guard; a `40→34` shed commit bounced
back to 40 the next tick because the estimator saw `physical=40` and
committed=34. These tests pin BOTH halves of the symmetric rule.
"""

from decision_gen.direction import Phase, phase_of, reconcile


# ---------- phase derivation ----------

def test_phase_ramp_up():
    assert phase_of(committed=5, physical=3) is Phase.RAMP_UP


def test_phase_in_drain():
    assert phase_of(committed=3, physical=5) is Phase.IN_DRAIN


def test_phase_settled():
    assert phase_of(committed=4, physical=4) is Phase.SETTLED


# ---------- I4 regression: shed must stay shed through drain ----------

def test_i4_shed_proposal_during_drain_snaps_to_committed():
    """committed=34 (just shed from 40), physical still 40 → IN_DRAIN.
    Estimator at current=40 proposes 40 (no rule fires, looks like
    scale-up from committed). Must snap to committed=34."""
    out, reason = reconcile(proposed=40, committed=34, phase=Phase.IN_DRAIN)
    assert out == 34
    assert reason.startswith("freeze-in-drain")


def test_i4_mirror_ramp_up_growth_snap():
    """committed=40 (just grew from 34), physical still 34 → RAMP_UP.
    A propose-below (e.g. quiet_shed off a quiet tick before the new
    capacity lands) must snap back to 40."""
    out, reason = reconcile(proposed=34, committed=40, phase=Phase.RAMP_UP)
    assert out == 40
    assert reason.startswith("freeze-ramp-up")


# ---------- authorized directions pass through ----------

def test_in_drain_further_shed_is_allowed():
    out, reason = reconcile(proposed=30, committed=34, phase=Phase.IN_DRAIN)
    assert out == 30 and reason == ""


def test_ramp_up_further_growth_is_allowed():
    out, reason = reconcile(proposed=45, committed=40, phase=Phase.RAMP_UP)
    assert out == 45 and reason == ""


def test_settled_passes_through_up_and_down():
    assert reconcile(45, 40, Phase.SETTLED) == (45, "")
    assert reconcile(35, 40, Phase.SETTLED) == (35, "")


def test_equal_proposal_always_passes():
    for p in (Phase.RAMP_UP, Phase.IN_DRAIN, Phase.SETTLED):
        assert reconcile(40, 40, p) == (40, "")
