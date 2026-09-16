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


def test_not_ready_until_first_tick(
    fake_slo, fake_k8s, fake_signals, clock, cr_spec,
):
    """Server 503s until controller.ready() flips; it flips after the
    first tick regardless of success. Bug the log at 04:49 exposed:
    /decisions served a boot-empty payload as if it were data."""
    _boot(fake_k8s, fake_signals)
    ctl = _mk_ctl(fake_slo, fake_k8s, fake_signals, clock, cr_spec)
    assert ctl.ready() is False
    ctl.tick()
    assert ctl.ready() is True


def test_ready_flips_even_on_tick_failure(
    fake_slo, fake_k8s, fake_signals, clock, cr_spec,
):
    """Posture matches the loop: a failed first tick means 'serve the
    (still-empty) previous snapshot', not 'stay unready forever'."""
    _boot(fake_k8s, fake_signals)
    ctl = _mk_ctl(fake_slo, fake_k8s, fake_signals, clock, cr_spec)

    class Boom:
        def pool_capacity(self):
            raise RuntimeError("k8s down")
    ctl.k8s = Boom()
    with pytest.raises(RuntimeError):
        ctl.tick()
    assert ctl.ready() is True


def test_loop_first_tick_waits_for_slo_sync(fake_slo, fake_k8s, fake_signals, clock, cr_spec):
    """Race regression: `_loop` must not tick off an empty pre-watch cache."""
    import threading, time as pytime

    fake_slo.mark_unsynced()
    _boot(fake_k8s, fake_signals)
    ctl = _mk_ctl(fake_slo, fake_k8s, fake_signals, clock, cr_spec)
    ctl.tick_seconds = 0.05

    t = threading.Thread(target=ctl._loop, daemon=True)
    t.start()
    pytime.sleep(0.2)
    assert _decisions(ctl) == {}

    fake_slo.mark_synced()
    deadline = pytime.time() + 2
    while pytime.time() < deadline and _decisions(ctl) == {}:
        pytime.sleep(0.02)
    ctl.stop()
    t.join(timeout=2)
    assert _decisions(ctl) == {"svc": 2}


def test_loop_first_tick_proceeds_when_already_synced(fake_slo, fake_k8s, fake_signals, clock, cr_spec):
    """Baseline: with synced=True from the start, the first tick doesn't block."""
    import threading, time as pytime
    _boot(fake_k8s, fake_signals)
    ctl = _mk_ctl(fake_slo, fake_k8s, fake_signals, clock, cr_spec)
    ctl.tick_seconds = 0.05

    t = threading.Thread(target=ctl._loop, daemon=True)
    t.start()
    deadline = pytime.time() + 2
    while pytime.time() < deadline and _decisions(ctl) == {}:
        pytime.sleep(0.02)
    ctl.stop()
    t.join(timeout=2)
    assert _decisions(ctl) == {"svc": 2}


def test_skip_keeps_service_on_wire_at_last_committed(
    fake_slo, fake_k8s, fake_signals, clock, cr_spec,
):
    """physical=0 mid-stream: service must NOT vanish from /decisions
    (the executor would read absence differently from "same value").
    It also must NOT commit a new value."""
    _boot(fake_k8s, fake_signals, physical=3)
    ctl = _mk_ctl(fake_slo, fake_k8s, fake_signals, clock, cr_spec)
    ctl.tick()
    assert _decisions(ctl) == {"svc": 2}    # PLACEMENT.spec_replicas=2

    # Drop physical to 0 — exporter sees no backends.
    fake_signals.set_replicas_ready(0)
    clock.advance(60)
    ctl.tick()
    assert _decisions(ctl) == {"svc": 2}    # stays, unchanged


def test_skip_on_first_tick_still_serves_seeded_value(
    fake_slo, fake_k8s, fake_signals, clock, cr_spec,
):
    """Brand-new service whose first reading is physical=0: the seed
    comes from spec.replicas, not prom — so the wire advertises the
    service at its seeded committed right away, then tick N+1 (once
    physical is available) proceeds from that baseline without a
    visible flap."""
    _boot(fake_k8s, fake_signals, physical=0)
    ctl = _mk_ctl(fake_slo, fake_k8s, fake_signals, clock, cr_spec)
    ctl.tick()
    assert _decisions(ctl) == {"svc": 2}    # seeded from spec_replicas=2
    # Next tick, physical arrives — committed stays pinned by the seed.
    fake_signals.set_replicas_ready(3)
    clock.advance(60)
    ctl.tick()
    assert _decisions(ctl) == {"svc": 2}    # unchanged; seed held


def test_warns_when_pool_cannot_fit_all_mins(
    fake_slo, fake_k8s, fake_signals, clock, cr_spec, caplog,
):
    """Σ CR min × gpr = 16 but capacity = 8: must log a single LOUD
    warning so the under-provisioning is visible. Mins still served."""
    import dataclasses
    cr2 = dict(cr_spec, minimumDeployment={"type": "replica", "value": 1})
    fake_slo.set("ns", "svc", cr_spec)
    fake_slo.set("ns", "svc2", cr2)
    fake_k8s.placements[("ns", "svc")] = PLACEMENT
    fake_k8s.placements[("ns", "svc2")] = dataclasses.replace(PLACEMENT, name="svc2")
    fake_k8s.capacity = {"p": 8}
    fake_signals.set_replicas_ready(1)
    fake_signals.set_ttft("ns", "svc", "p80", 5.0)
    fake_signals.set_ttft("ns", "svc2", "p80", 5.0)
    fake_signals.set_otps("ns", "svc", "p80", 100.0)
    fake_signals.set_otps("ns", "svc2", "p80", 100.0)
    fake_signals.set_rejection(0.0)
    ctl = Controller(slo=fake_slo, k8s=fake_k8s, signals=fake_signals,
                     tick_seconds=60, clock=clock)
    with caplog.at_level(logging.WARNING, logger="decision_gen.controller"):
        ctl.tick()
    warned = [r.getMessage() for r in caplog.records if "UNDER-PROVISIONED" in r.getMessage()]
    assert warned, "expected a loud warning when mins exceed pool capacity"
    msg = warned[0]
    assert "16 GPUs" in msg and "capacity = 8" in msg and "short 8" in msg


def test_missing_max_keeps_service_off_wire(
    fake_slo, fake_k8s, fake_signals, clock, caplog,
):
    """CR without maximumDeployment → user opted out. Service is skipped
    (logged once per tick at WARNING) and never appears in /decisions."""
    cr_no_max = {
        "serviceId": "svc",
        "ttft": {"default": {"metrics": [{"type": "p80", "threshold": 20.0}]}},
        "otps": {"default": {"metrics": [{"type": "p80", "threshold": 30.0}]}},
    }
    fake_slo.set("ns", "svc", cr_no_max)
    fake_k8s.placements[("ns", "svc")] = PLACEMENT
    fake_k8s.capacity = {"p": 64}
    fake_signals.set_replicas_ready(2)
    ctl = Controller(slo=fake_slo, k8s=fake_k8s, signals=fake_signals,
                     tick_seconds=60, clock=clock)
    with caplog.at_level(logging.WARNING):
        ctl.tick()
    assert _decisions(ctl) == {}
    warned = [r.getMessage() for r in caplog.records
              if "missing maximumDeployment" in r.getMessage()]
    assert warned, "expected loud warning on missing max"


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
