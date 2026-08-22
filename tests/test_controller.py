"""Controller integration with fake tier-3 collaborators.

Per arch §8: deleted CR frees capacity the same tick (C7), pool read
failure holds the previous snapshot (§5), one uneventful tick produces
a legible per-service Transition log row.
"""

import logging

import pytest

from decision_gen.controller import Controller
from decision_gen.k8s_state import Placement
from decision_gen.signals import Reading


PLACEMENT = Placement(
    namespace="ns", name="svc", kind="deployment",
    pool="p", gpus_per_replica=8,
    workload_name="svc", spec_replicas=2,
)


def _boot(fake_k8s, fake_signals, placement=PLACEMENT, physical=2):
    fake_k8s.placements[("ns", "svc")] = placement
    fake_k8s.capacity = {"p": 64}
    fake_signals.set_replicas_ready(physical)
    fake_signals.set_ttft("ns", "svc", "p80", 5.0)
    fake_signals.set_otps("ns", "svc", "p80", 100.0)
    fake_signals.set_rejection(0.0)


def _mk_ctl(fake_slo, fake_k8s, fake_signals, clock, cr_spec):
    fake_slo.set("ns", "svc", cr_spec)
    return Controller(
        slo=fake_slo, k8s=fake_k8s, signals=fake_signals,
        tick_seconds=60, clock=clock,
    )


def _decisions(ctl):
    return {d["serviceId"]: d["replicas"]["active"]
            for d in ctl.snapshot()["decisions"]}


def test_one_tick_produces_decision_row(
    fake_slo, fake_k8s, fake_signals, clock, cr_spec,
):
    _boot(fake_k8s, fake_signals)
    ctl = _mk_ctl(fake_slo, fake_k8s, fake_signals, clock, cr_spec)
    ctl.tick()
    assert _decisions(ctl) == {"svc": 2}     # boot seed = spec_replicas


def test_unresolvable_placement_skips_service(
    fake_slo, fake_k8s, fake_signals, clock, cr_spec,
):
    # No placement registered → skip entirely, no decision row.
    fake_signals.set_ttft("ns", "svc", "p80", 5.0)
    ctl = _mk_ctl(fake_slo, fake_k8s, fake_signals, clock, cr_spec)
    ctl.tick()
    assert _decisions(ctl) == {}


def test_deleted_cr_frees_capacity_same_tick(
    fake_slo, fake_k8s, fake_signals, clock, cr_spec,
):
    """C7: delete a CR; ledger entry must free BEFORE gap, so the very
    next gap computation sees the GPUs back."""
    _boot(fake_k8s, fake_signals)
    ctl = _mk_ctl(fake_slo, fake_k8s, fake_signals, clock, cr_spec)
    ctl.tick()
    assert _decisions(ctl) == {"svc": 2}

    fake_slo.delete("ns", "svc")
    ctl.tick()
    # No CRs → empty payload; view deleted; ledger freed.
    assert _decisions(ctl) == {}
    assert ("ns", "svc") not in ctl._views


def test_pool_capacity_failure_holds_previous_snapshot(
    fake_slo, fake_k8s, fake_signals, clock, cr_spec,
):
    _boot(fake_k8s, fake_signals)
    ctl = _mk_ctl(fake_slo, fake_k8s, fake_signals, clock, cr_spec)
    ctl.tick()
    assert _decisions(ctl) == {"svc": 2}

    class Boom:
        def pool_capacity(self):
            raise RuntimeError("k8s down")
    ctl.k8s = Boom()
    # _loop swallows; called directly via tick — confirm raise propagates
    # so the loop's try/except is what swallows, keeping the snapshot.
    with pytest.raises(RuntimeError):
        ctl.tick()
    # Snapshot unchanged from previous tick.
    assert _decisions(ctl) == {"svc": 2}


def test_one_tick_transition_log_is_legible(
    fake_slo, fake_k8s, fake_signals, clock, cr_spec, caplog,
):
    """Acceptance check #7: one uneventful tick's log shows placement,
    raw signals with labels, verdicts, stage names, hold reason."""
    _boot(fake_k8s, fake_signals)
    ctl = _mk_ctl(fake_slo, fake_k8s, fake_signals, clock, cr_spec)
    with caplog.at_level(logging.DEBUG, logger="decision_gen.controller"):
        ctl.tick()
    rows = [r.getMessage() for r in caplog.records
            if r.name == "decision_gen.controller" and "est=" in r.getMessage()]
    assert rows, "no per-service transition log"
    row = rows[0]
    for needle in (
        "pool=p", "gpr=8", "kind=deployment",
        "[ttft.p80]=5", "[otps.p80]=100",
        "est=", "want=",
        "phase=", "hold-",
    ):
        assert needle in row, f"missing {needle!r} in log row: {row}"
