"""Tests for decision_gen.controller._tick.

Drives Controller._tick directly with fakes for slo/k8s/prom. No
sleeping on the tick loop — we inject the tick body and call it.
"""

import time as time_module

import pytest

from decision_gen import controller as ctrl_mod
from decision_gen.controller import Controller
from decision_gen.k8s_state import Placement
from decision_gen import estimator as est_mod


POOL = "NVIDIA-H100-80GB-HBM3"


def _placement(pool=POOL, gpr=8, spec_replicas=None):
    """Default spec_replicas to a current-mirroring value (2) so that
    boot-seed == typical prom current in most tests. Override per-test
    when you specifically want to exercise the seed path."""
    return Placement(
        namespace="", name="", kind="deployment",
        pool=pool, gpus_per_replica=gpr,
        spec_replicas=spec_replicas if spec_replicas is not None else 2,
    )


def _cr(min_v=1, max_v=4, priority=5, ttft_metrics=None, otps_metrics=None):
    return {
        "serviceId": "svc",
        "priority": priority,
        "minimumDeployment": {"type": "replica", "value": min_v},
        "maximumDeployment": {"type": "replica", "value": max_v},
        "ttft": {"default": {"metrics": ttft_metrics or [{"type": "p80", "threshold": 20.0}]}},
        "otps": {"default": {"metrics": otps_metrics or [{"type": "p80", "threshold": 30.0}]}},
    }


def _healthy_prom(prom, current=2, ttft_s=10.0, otps=15.0, rej=0.0):
    prom.current_replicas_fn = lambda ns, svc: current
    prom.ttft_fn = lambda ns, svc, kind: ttft_s
    prom.otps_fn = lambda ns, svc, kind: otps
    prom.rejection_rate_fn = lambda ns, svc: rej


# ---------- basic wiring ----------

def test_empty_cr_map_yields_empty_decisions(fake_slo_store, fake_prom, fake_k8s):
    c = Controller(slo=fake_slo_store, k8s=fake_k8s, prom=fake_prom)
    c._tick()
    assert c.snapshot()["decisions"] == []


def test_healthy_service_appears_in_decisions(fake_slo_store, fake_prom, fake_k8s):
    fake_slo_store.set("kimi", "kimi-k25", _cr(min_v=2, max_v=4))
    fake_k8s.placements[("kimi", "kimi-k25")] = _placement(gpr=8)
    fake_k8s.capacity[POOL] = 32
    _healthy_prom(fake_prom, current=2)

    c = Controller(slo=fake_slo_store, k8s=fake_k8s, prom=fake_prom)
    c._tick()

    snap = c.snapshot()
    assert snap["apiVersion"] == "llmscaling.inference.x-k8s.io/v1alpha1"
    rows = {d["serviceId"]: d for d in snap["decisions"]}
    assert "kimi-k25" in rows
    assert rows["kimi-k25"]["namespace"] == "kimi"
    assert rows["kimi-k25"]["replicas"]["active"] >= 2


# ---------- failure postures ----------

def test_unmanageable_service_excluded(fake_slo_store, fake_prom, fake_k8s):
    fake_slo_store.set("kimi", "kimi-k25", _cr())
    fake_k8s.placements[("kimi", "kimi-k25")] = None   # unmanageable
    fake_k8s.capacity[POOL] = 32
    _healthy_prom(fake_prom, current=2)

    c = Controller(slo=fake_slo_store, k8s=fake_k8s, prom=fake_prom)
    c._tick()
    assert c.snapshot()["decisions"] == []


def test_nan_ttft_holds_at_current(fake_slo_store, fake_prom, fake_k8s):
    fake_slo_store.set("kimi", "kimi-k25", _cr(min_v=2, max_v=4))
    fake_k8s.placements[("kimi", "kimi-k25")] = _placement()
    fake_k8s.capacity[POOL] = 32
    _healthy_prom(fake_prom, current=2)
    fake_prom.ttft_fn = lambda ns, svc, kind: float("nan")
    fake_prom.rejection_rate_fn = lambda ns, svc: 0.5   # ← spike, but nan ttft → hold

    c = Controller(slo=fake_slo_store, k8s=fake_k8s, prom=fake_prom)
    c._tick()
    rows = {d["serviceId"]: d for d in c.snapshot()["decisions"]}
    assert rows["kimi-k25"]["replicas"]["active"] == 2   # hold at current


def test_nan_on_one_service_doesnt_affect_another(fake_slo_store, fake_prom, fake_k8s):
    fake_slo_store.set("kimi", "kimi-k25", _cr(min_v=1, max_v=4))
    fake_slo_store.set("modelforge", "fb", _cr(min_v=1, max_v=4, priority=0))
    fake_k8s.placements[("kimi", "kimi-k25")] = _placement(spec_replicas=1)
    fake_k8s.placements[("modelforge", "fb")] = _placement(spec_replicas=1)
    fake_k8s.capacity[POOL] = 32

    def _ttft_per_svc(ns, svc, kind):
        return float("nan") if svc == "kimi-k25" else 100.0   # ← modelforge sees violation
    fake_prom.current_replicas_fn = lambda ns, svc: 1
    fake_prom.ttft_fn = _ttft_per_svc
    fake_prom.otps_fn = lambda ns, svc, kind: 10.0
    fake_prom.rejection_rate_fn = lambda ns, svc: 0.0

    c = Controller(slo=fake_slo_store, k8s=fake_k8s, prom=fake_prom)
    c._tick()
    rows = {d["serviceId"]: d for d in c.snapshot()["decisions"]}
    assert rows["kimi-k25"]["replicas"]["active"] == 1   # hold
    assert rows["fb"]["replicas"]["active"] == 1         # hold: up-cooldown still gated

    # Give fb enough idle time for the up-cooldown; second tick scales it.
    c._last_change_at[("modelforge", "fb")] = time_module.time() - est_mod.SCALE_UP_COOLDOWN_S
    c._tick()
    rows = {d["serviceId"]: d for d in c.snapshot()["decisions"]}
    assert rows["fb"]["replicas"]["active"] == 2         # violation now scales


def test_missing_current_replicas_falls_back_to_last_served(
    fake_slo_store, fake_prom, fake_k8s,
):
    fake_slo_store.set("kimi", "kimi-k25", _cr(min_v=1, max_v=4))
    fake_k8s.placements[("kimi", "kimi-k25")] = _placement()
    fake_k8s.capacity[POOL] = 32
    _healthy_prom(fake_prom, current=2)

    c = Controller(slo=fake_slo_store, k8s=fake_k8s, prom=fake_prom)
    c._tick()   # establishes last_served = 2
    rows = {d["serviceId"]: d for d in c.snapshot()["decisions"]}
    assert rows["kimi-k25"]["replicas"]["active"] == 2

    # next tick: replicas_ready absent → fall back to last_served, not to
    # CR min or zero
    fake_prom.current_replicas_fn = lambda ns, svc: None
    c._tick()
    rows = {d["serviceId"]: d for d in c.snapshot()["decisions"]}
    assert rows["kimi-k25"]["replicas"]["active"] == 2


def test_missing_current_replicas_seeds_from_workload_spec(
    fake_slo_store, fake_prom, fake_k8s,
):
    """Boot with Prom absent: ledger boots from spec.replicas, not skipped."""
    fake_slo_store.set("kimi", "kimi-k25", _cr(min_v=1, max_v=4))
    fake_k8s.placements[("kimi", "kimi-k25")] = _placement(spec_replicas=3)
    fake_k8s.capacity[POOL] = 32
    _healthy_prom(fake_prom, current=2, ttft_s=5.0)
    fake_prom.current_replicas_fn = lambda ns, svc: None   # Prom is dark

    c = Controller(slo=fake_slo_store, k8s=fake_k8s, prom=fake_prom)
    c._tick()
    rows = {d["serviceId"]: d for d in c.snapshot()["decisions"]}
    # Boot: book spec=3 (estimator input falls back to ledger, healthy → hold or cooldown)
    assert rows["kimi-k25"]["replicas"]["active"] >= 1
    # The ledger records *a* value (not skip) — service is managed.
    assert ("kimi", "kimi-k25") in c._last_served


# ---------- boot seed (spec.replicas → ledger) ----------

def test_boot_seed_within_bounds_uses_spec(
    fake_slo_store, fake_prom, fake_k8s,
):
    """spec.replicas in [min, max] → last_served boots to that value."""
    fake_slo_store.set("kimi", "kimi-k25", _cr(min_v=1, max_v=4))
    fake_k8s.placements[("kimi", "kimi-k25")] = _placement(spec_replicas=3)
    fake_k8s.capacity[POOL] = 32
    _healthy_prom(fake_prom, current=3, ttft_s=5.0)

    c = Controller(slo=fake_slo_store, k8s=fake_k8s, prom=fake_prom)
    c._tick()
    assert c._last_served[("kimi", "kimi-k25")] == 3


def test_boot_seed_above_max_clamps_down(
    fake_slo_store, fake_prom, fake_k8s,
):
    """Someone manually scaled up beyond CR max → boot clamps us back."""
    fake_slo_store.set("kimi", "kimi-k25", _cr(min_v=1, max_v=2))
    fake_k8s.placements[("kimi", "kimi-k25")] = _placement(spec_replicas=5)
    fake_k8s.capacity[POOL] = 64
    _healthy_prom(fake_prom, current=5, ttft_s=5.0)

    c = Controller(slo=fake_slo_store, k8s=fake_k8s, prom=fake_prom)
    c._tick()
    rows = {d["serviceId"]: d for d in c.snapshot()["decisions"]}
    assert rows["kimi-k25"]["replicas"]["active"] == 2    # clamped to max


def test_boot_seed_below_min_clamps_up(
    fake_slo_store, fake_prom, fake_k8s,
):
    """spec.replicas < CR min → boot lifts to min (this is also the
    'service is not running' case — a stopped service gets restarted
    as soon as we boot and CR floor > 0)."""
    fake_slo_store.set("kimi", "kimi-k25", _cr(min_v=2, max_v=4))
    fake_k8s.placements[("kimi", "kimi-k25")] = _placement(spec_replicas=0)
    fake_k8s.capacity[POOL] = 64
    _healthy_prom(fake_prom, current=0, ttft_s=5.0)

    c = Controller(slo=fake_slo_store, k8s=fake_k8s, prom=fake_prom)
    c._tick()
    rows = {d["serviceId"]: d for d in c.snapshot()["decisions"]}
    assert rows["kimi-k25"]["replicas"]["active"] == 2    # clamped to min


def test_boot_seed_min_zero_respects_shutdown(
    fake_slo_store, fake_prom, fake_k8s,
):
    """spec.replicas=0 AND CR min=0 → we respect the shutdown, serve 0."""
    fake_slo_store.set("kimi", "kimi-k25", _cr(min_v=0, max_v=4))
    fake_k8s.placements[("kimi", "kimi-k25")] = _placement(spec_replicas=0)
    fake_k8s.capacity[POOL] = 32
    _healthy_prom(fake_prom, current=0, ttft_s=5.0)

    c = Controller(slo=fake_slo_store, k8s=fake_k8s, prom=fake_prom)
    c._tick()
    rows = {d["serviceId"]: d for d in c.snapshot()["decisions"]}
    assert rows["kimi-k25"]["replicas"]["active"] == 0


# ---------- boot freeze: seed stamps cooldown ----------

def test_boot_freeze_healthy_service_holds(
    fake_slo_store, fake_prom, fake_k8s,
):
    """Regression: a long-running healthy service must NOT scale down on
    the very first managed tick. Seeding stamps last_change_at=now, so
    rule 3's cooldown gate stays shut for a full window.

    (Before this fix, last_change_at defaulted to epoch 0 → cooldown
    read as 'long since changed' → immediate shed off unproven data.)"""
    fake_slo_store.set("kimi", "kimi-k25", _cr(min_v=1, max_v=4))
    fake_k8s.placements[("kimi", "kimi-k25")] = _placement(spec_replicas=3)
    fake_k8s.capacity[POOL] = 64
    # Unambiguously comfortable: ttft 5 < 20*0.5=10; otps 61 > 30/0.5=60.
    _healthy_prom(fake_prom, current=3, ttft_s=5.0, otps=61.0, rej=0.0)

    c = Controller(slo=fake_slo_store, k8s=fake_k8s, prom=fake_prom)
    for _ in range(3):   # several ticks, same story
        c._tick()
        rows = {d["serviceId"]: d for d in c.snapshot()["decisions"]}
        assert rows["kimi-k25"]["replicas"]["active"] == 3   # frozen
    # Cooldown clock is ticking from seed time.
    assert ("kimi", "kimi-k25") in c._last_change_at


def test_boot_freeze_expires_after_cooldown_window(
    fake_slo_store, fake_prom, fake_k8s,
):
    """Once the seeded cooldown elapses AND signals still look
    comfortable, scale-down resumes — the freeze is not permanent."""
    fake_slo_store.set("kimi", "kimi-k25", _cr(min_v=1, max_v=4))
    fake_k8s.placements[("kimi", "kimi-k25")] = _placement(spec_replicas=3)
    fake_k8s.capacity[POOL] = 64
    _healthy_prom(fake_prom, current=3, ttft_s=5.0, otps=61.0, rej=0.0)

    c = Controller(slo=fake_slo_store, k8s=fake_k8s, prom=fake_prom)
    c._tick()
    assert c.snapshot()["decisions"][0]["replicas"]["active"] == 3

    # Simulate the cooldown window elapsing since the seed, AND seed the
    # comfort tracker so rule 3's 30-min gate is satisfied.
    c._last_change_at[("kimi", "kimi-k25")] = 0
    c._comfortable_since[("kimi", "kimi-k25")] = 0
    c._tick()
    rows = {d["serviceId"]: d for d in c.snapshot()["decisions"]}
    assert rows["kimi-k25"]["replicas"]["active"] == 2   # now allowed


def test_violation_during_boot_shed_freeze_holds_until_up_cooldown(
    fake_slo_store, fake_prom, fake_k8s,
):
    """Boot freeze gates rule 2 for SCALE_UP_COOLDOWN_S after the seed —
    SLO-following-up only kicks in after that window. Rule 1 (rejection
    spike) is always fire-alarm and not gated by this freeze."""
    fake_slo_store.set("kimi", "kimi-k25", _cr(min_v=1, max_v=6))
    fake_k8s.placements[("kimi", "kimi-k25")] = _placement(spec_replicas=2)
    fake_k8s.capacity[POOL] = 64
    fake_prom.current_replicas_fn = lambda ns, svc: 2
    fake_prom.ttft_fn = lambda ns, svc, kind: 50.0   # violated (thr 20)
    fake_prom.otps_fn = lambda ns, svc, kind: 10.0
    fake_prom.rejection_rate_fn = lambda ns, svc: 0.0

    c = Controller(slo=fake_slo_store, k8s=fake_k8s, prom=fake_prom)
    c._tick()
    rows = {d["serviceId"]: d for d in c.snapshot()["decisions"]}
    assert rows["kimi-k25"]["replicas"]["active"] == 2   # hold: up-cooldown gated

    # Let the up-cooldown elapse — then the violation scales.
    c._last_change_at[("kimi", "kimi-k25")] = (
        time_module.time() - est_mod.SCALE_UP_COOLDOWN_S
    )
    c._tick()
    rows = {d["serviceId"]: d for d in c.snapshot()["decisions"]}
    assert rows["kimi-k25"]["replicas"]["active"] == 3


# ---------- direction rule (shed freeze) ----------

def test_direction_rampup_slo_violation_freezes_at_booked(
    fake_slo_store, fake_prom, fake_k8s,
):
    """Core invariant: pending scale-up (prom < booked) must NOT let
    the estimator's violated-SLO rule shed the in-flight commitment."""
    fake_slo_store.set("kimi", "kimi-k25", _cr(min_v=1, max_v=12))
    fake_k8s.placements[("kimi", "kimi-k25")] = _placement(spec_replicas=10)
    fake_k8s.capacity[POOL] = 96
    fake_prom.current_replicas_fn = lambda ns, svc: 8
    fake_prom.ttft_fn = lambda ns, svc, kind: 50.0   # violated
    fake_prom.otps_fn = lambda ns, svc, kind: 10.0
    fake_prom.rejection_rate_fn = lambda ns, svc: 0.0

    c = Controller(slo=fake_slo_store, k8s=fake_k8s, prom=fake_prom)
    c._tick()
    # Tick 1: boot seeds ledger=10; violation is up-cooldown gated → hold.
    rows = {d["serviceId"]: d for d in c.snapshot()["decisions"]}
    assert rows["kimi-k25"]["replicas"]["active"] == 10

    # Tick 2: up-cooldown elapsed → estimator proposes +1 off prom=8 →
    # desired=9 < booked=10 → direction rule freezes at 10.
    c._last_change_at[("kimi", "kimi-k25")] = (
        time_module.time() - est_mod.SCALE_UP_COOLDOWN_S
    )
    c._tick()
    rows = {d["serviceId"]: d for d in c.snapshot()["decisions"]}
    # Estimator wanted 8+1=9 < booked=10, but direction-vs-ledger froze it.
    # Without the freeze we'd commit 9 and revoke the pending pod.
    assert rows["kimi-k25"]["replicas"]["active"] == 10


def test_direction_rampup_rejection_spike_still_scales_up(
    fake_slo_store, fake_prom, fake_k8s,
):
    """Rejection spike magnitude on prom=8 can exceed booked=10 — must
    not be capped by the freeze."""
    fake_slo_store.set("kimi", "kimi-k25", _cr(min_v=1, max_v=20))
    fake_k8s.placements[("kimi", "kimi-k25")] = _placement(spec_replicas=10)
    fake_k8s.capacity[POOL] = 160
    fake_prom.current_replicas_fn = lambda ns, svc: 8
    fake_prom.ttft_fn = lambda ns, svc, kind: 10.0
    fake_prom.otps_fn = lambda ns, svc, kind: 15.0
    fake_prom.rejection_rate_fn = lambda ns, svc: 0.05   # spike

    c = Controller(slo=fake_slo_store, k8s=fake_k8s, prom=fake_prom)
    c._tick()
    rows = {d["serviceId"]: d for d in c.snapshot()["decisions"]}
    # rej=0.05 is at threshold, mult = 1+2·(0.05/0.95) ≈ 1.105.
    # desired = max(ceil(8×1.105), 8+1) = 9 < booked=10 → direction rule
    # lifts to 10 (the freeze: rule-1 spikes below booked don't shed).
    assert rows["kimi-k25"]["replicas"]["active"] == 10


def test_direction_scale_down_via_rule3_still_works(
    fake_slo_store, fake_prom, fake_k8s,
):
    """The ONLY path below booked is estimator rule 3 (cooldown +
    comfortable + quiet). Anything else freezes."""
    fake_slo_store.set("kimi", "kimi-k25", _cr(min_v=1, max_v=4))
    fake_k8s.placements[("kimi", "kimi-k25")] = _placement(spec_replicas=2)
    fake_k8s.capacity[POOL] = 32
    _healthy_prom(fake_prom, current=2, ttft_s=5.0, otps=61.0, rej=0.0)

    c = Controller(slo=fake_slo_store, k8s=fake_k8s, prom=fake_prom)
    c._tick()
    rows = {d["serviceId"]: d for d in c.snapshot()["decisions"]}
    assert rows["kimi-k25"]["replicas"]["active"] == 2   # frozen during cooldown

    # Force cooldown elapsed + comfort + quiet + 30-min comfort window →
    # rule 3 fires → shed.
    c._last_change_at[("kimi", "kimi-k25")] = 0
    c._comfortable_since[("kimi", "kimi-k25")] = 0
    c._tick()
    rows = {d["serviceId"]: d for d in c.snapshot()["decisions"]}
    assert rows["kimi-k25"]["replicas"]["active"] == 1


def test_direction_cooldown_expired_but_uncomfortable_freezes(
    fake_slo_store, fake_prom, fake_k8s,
):
    """Cooldown expired alone is not enough — must also be comfortable
    + rejection quiet, else freeze at booked."""
    fake_slo_store.set("kimi", "kimi-k25", _cr(min_v=1, max_v=4))
    fake_k8s.placements[("kimi", "kimi-k25")] = _placement(spec_replicas=2)
    fake_k8s.capacity[POOL] = 32
    fake_prom.current_replicas_fn = lambda ns, svc: 2
    fake_prom.ttft_fn = lambda ns, svc, kind: 5.0
    fake_prom.otps_fn = lambda ns, svc, kind: 25.0   # not comfortable (25 > 15)
    fake_prom.rejection_rate_fn = lambda ns, svc: 0.0

    c = Controller(slo=fake_slo_store, k8s=fake_k8s, prom=fake_prom)
    c._last_change_at[("kimi", "kimi-k25")] = 0   # force cooldown elapsed
    c._tick()
    rows = {d["serviceId"]: d for d in c.snapshot()["decisions"]}
    # Estimator: cooldown open but not comfortable → desired=2, hold.
    # Direction: 2 == 2, no shed anyway. But even if the estimator had
    # somehow returned 1, rule 4 would freeze it.
    assert rows["kimi-k25"]["replicas"]["active"] == 2


def test_direction_max_shrink_bypasses_freeze(
    fake_slo_store, fake_prom, fake_k8s,
):
    """CR max edited down below booked: we MUST compress in this tick,
    regardless of cooldown state. Holding above a newly-lowered cap
    is wronger than breaking the freeze."""
    fake_slo_store.set("kimi", "kimi-k25", _cr(min_v=1, max_v=4))
    fake_k8s.placements[("kimi", "kimi-k25")] = _placement(spec_replicas=10)
    fake_k8s.capacity[POOL] = 96
    _healthy_prom(fake_prom, current=10, ttft_s=5.0)

    c = Controller(slo=fake_slo_store, k8s=fake_k8s, prom=fake_prom)
    c._tick()
    # Boot seed: booked=clamp(10, 1, 4)=4 already max-clamped. Not our case here.
    rows = {d["serviceId"]: d for d in c.snapshot()["decisions"]}
    assert rows["kimi-k25"]["replicas"]["active"] == 4

    # Now spec.replicas is 4 (executor applied), but operator's CR
    # max gets tightened: 4 → 2.
    fake_slo_store.set("kimi", "kimi-k25", _cr(min_v=1, max_v=2))
    # Make ttft uncomfortable so estimator can't rule-3 its way down
    # (definitely exercising the max-shrink path).
    fake_prom.ttft_fn = lambda ns, svc, kind: 18.0
    fake_prom.current_replicas_fn = lambda ns, svc: 4
    c._tick()
    rows = {d["serviceId"]: d for d in c.snapshot()["decisions"]}
    assert rows["kimi-k25"]["replicas"]["active"] == 2   # compressed into new cap


def test_direction_nan_signals_freeze_capped_by_max(
    fake_slo_store, fake_prom, fake_k8s,
):
    """NaN hold path also compresses into a newly-shrunk CR max,
    not just the estimator-decide path."""
    fake_slo_store.set("kimi", "kimi-k25", _cr(min_v=1, max_v=4))
    fake_k8s.placements[("kimi", "kimi-k25")] = _placement(spec_replicas=4)
    fake_k8s.capacity[POOL] = 64
    _healthy_prom(fake_prom, current=4, ttft_s=5.0)

    c = Controller(slo=fake_slo_store, k8s=fake_k8s, prom=fake_prom)
    c._tick()
    assert c.snapshot()["decisions"][0]["replicas"]["active"] == 4

    # Shrink max to 2; feed NaN signals (goes through the hold path).
    fake_slo_store.set("kimi", "kimi-k25", _cr(min_v=1, max_v=2))
    fake_prom.ttft_fn = lambda ns, svc, kind: float("nan")
    c._tick()
    rows = {d["serviceId"]: d for d in c.snapshot()["decisions"]}
    assert rows["kimi-k25"]["replicas"]["active"] == 2


def test_preemption_shed_commits_without_freeze(
    fake_slo_store, fake_prom, fake_k8s,
):
    """Planner phase-3 sheds a low-priority donor to feed a high-priority
    grower. The donor's shed must not be frozen by the direction rule —
    the planner's word is final downstream of the estimator."""
    fake_slo_store.set("kimi", "hi", _cr(min_v=1, max_v=10, priority=10))
    fake_slo_store.set("kimi", "lo", _cr(min_v=1, max_v=10, priority=0))
    fake_k8s.placements[("kimi", "hi")] = _placement(spec_replicas=1)
    fake_k8s.placements[("kimi", "lo")] = _placement(spec_replicas=4)
    fake_k8s.capacity[POOL] = 40   # 5 replicas × 8; lo is using 32 of it

    # Current matches booked => no ramp-up freeze interference.
    fake_prom.current_replicas_fn = lambda ns, svc: 1 if svc == "hi" else 4
    fake_prom.ttft_fn = lambda ns, svc, kind: 50.0 if svc == "hi" else 5.0
    fake_prom.otps_fn = lambda ns, svc, kind: 10.0
    fake_prom.rejection_rate_fn = lambda ns, svc: 0.0

    c = Controller(slo=fake_slo_store, k8s=fake_k8s, prom=fake_prom)
    c._tick()
    rows = {d["serviceId"]: d for d in c.snapshot()["decisions"]}
    # hi: ttft violated but up-cooldown gated → want = booked = 1 (no preemption).
    # lo: comfortable but cooldown gated → hold 4.
    assert rows["hi"]["replicas"]["active"] == 1
    assert rows["lo"]["replicas"]["active"] == 4

    # Let hi's up-cooldown elapse → estimator now scales hi up → planner
    # preempts lo to fund the bump.
    c._last_change_at[("kimi", "hi")] = time_module.time() - est_mod.SCALE_UP_COOLDOWN_S
    c._tick()
    rows = {d["serviceId"]: d for d in c.snapshot()["decisions"]}
    # hi: est +1 → want 2, committed 2 (planner preempts lo).
    # lo: est comfortable-but-cooldown → want = booked = 4. Planner sheds 4→3.
    assert rows["hi"]["replicas"]["active"] == 2
    assert rows["lo"]["replicas"]["active"] == 3
    # Cooldown stamps: both changed → both stamped.
    assert ("kimi", "hi") in c._last_change_at
    assert ("kimi", "lo") in c._last_change_at


# ---------- estimator integration ----------

def test_violation_scales_up_via_controller(fake_slo_store, fake_prom, fake_k8s):
    fake_slo_store.set("kimi", "kimi-k25", _cr(min_v=1, max_v=4))
    fake_k8s.placements[("kimi", "kimi-k25")] = _placement()
    fake_k8s.capacity[POOL] = 32
    fake_prom.current_replicas_fn = lambda ns, svc: 2
    fake_prom.ttft_fn = lambda ns, svc, kind: 50.0   # > 20
    fake_prom.otps_fn = lambda ns, svc, kind: 10.0
    fake_prom.rejection_rate_fn = lambda ns, svc: 0.0

    c = Controller(slo=fake_slo_store, k8s=fake_k8s, prom=fake_prom)
    c._tick()
    # Tick 1: boot stamp; violation gated by up-cooldown → hold at booked.
    rows = {d["serviceId"]: d for d in c.snapshot()["decisions"]}
    assert rows["kimi-k25"]["replicas"]["active"] == 2

    # Tick 2: up-cooldown elapsed → violation scales +1.
    c._last_change_at[("kimi", "kimi-k25")] = (
        time_module.time() - est_mod.SCALE_UP_COOLDOWN_S
    )
    c._tick()
    rows = {d["serviceId"]: d for d in c.snapshot()["decisions"]}
    assert rows["kimi-k25"]["replicas"]["active"] == 3


def test_rejection_spike_scales_up_multiplicatively(fake_slo_store, fake_prom, fake_k8s):
    fake_slo_store.set("kimi", "kimi-k25", _cr(min_v=1, max_v=10))
    fake_k8s.placements[("kimi", "kimi-k25")] = _placement()
    fake_k8s.capacity[POOL] = 64
    fake_prom.current_replicas_fn = lambda ns, svc: 2
    fake_prom.ttft_fn = lambda ns, svc, kind: 10.0
    fake_prom.otps_fn = lambda ns, svc, kind: 15.0
    fake_prom.rejection_rate_fn = lambda ns, svc: 0.05   # > 0.01

    c = Controller(slo=fake_slo_store, k8s=fake_k8s, prom=fake_prom)
    c._tick()
    rows = {d["serviceId"]: d for d in c.snapshot()["decisions"]}
    # rej=0.05 is at threshold, mult ≈ 1.105. ceil(2×1.105)=3, and the
    # +1 floor would also give 3 — either way we grow by 1.
    assert rows["kimi-k25"]["replicas"]["active"] == 3


# ---------- cooldown bookkeeping ----------

def test_cooldown_updated_only_when_planner_changes_value(
    fake_slo_store, fake_prom, fake_k8s, monkeypatch,
):
    """Stamps happen on ledger-seed and on planner moves — not on hold.
    Tick 1 (planner stalled): seed stamp only. Tick 2 (capacity given):
    stamp advances with the move."""
    fake_slo_store.set("kimi", "kimi-k25", _cr(min_v=1, max_v=10))
    fake_k8s.placements[("kimi", "kimi-k25")] = _placement(spec_replicas=1)
    fake_k8s.capacity[POOL] = 0   # ← zero spare; can't actually grow
    fake_prom.current_replicas_fn = lambda ns, svc: 1
    fake_prom.ttft_fn = lambda ns, svc, kind: 50.0
    fake_prom.otps_fn = lambda ns, svc, kind: 10.0
    fake_prom.rejection_rate_fn = lambda ns, svc: 0.0

    c = Controller(slo=fake_slo_store, k8s=fake_k8s, prom=fake_prom)
    c._tick()
    # Tick 1 seeded the ledger → stamp (boot freeze). Planner stalled
    # (capacity 0, alloc == booked) — so the stamp is from the seed,
    # not from a planner move.
    assert ("kimi", "kimi-k25") in c._last_change_at
    seed_stamp = c._last_change_at[("kimi", "kimi-k25")]

    # Give capacity AND let the up-cooldown elapse; planner moves 1 → 2
    # → stamp advances.
    fake_k8s.capacity[POOL] = 32
    c._last_change_at[("kimi", "kimi-k25")] = (
        time_module.time() - est_mod.SCALE_UP_COOLDOWN_S
    )
    time_module.sleep(0.01)
    c._tick()
    assert c._last_change_at[("kimi", "kimi-k25")] > seed_stamp


# ---------- kinds_needed ----------

def test_kinds_needed_only_queries_declared_metrics(
    fake_slo_store, fake_prom, fake_k8s,
):
    """If CR only specifies p80, we don't query p50/p95/p99/avg."""
    fake_slo_store.set("kimi", "kimi-k25", _cr(
        ttft_metrics=[{"type": "p80", "threshold": 20.0}],
        otps_metrics=[{"type": "p95", "threshold": 30.0}],
    ))
    fake_k8s.placements[("kimi", "kimi-k25")] = _placement(spec_replicas=1)
    fake_k8s.capacity[POOL] = 32
    _healthy_prom(fake_prom, current=1)

    c = Controller(slo=fake_slo_store, k8s=fake_k8s, prom=fake_prom)
    c._tick()
    ttft_kinds = [c[3] for c in fake_prom.calls if c[0] == "ttft"]
    otps_kinds = [c[3] for c in fake_prom.calls if c[0] == "otps"]
    assert ttft_kinds == ["p80"]
    assert otps_kinds == ["p95"]


# ---------- snapshot shape ----------

def test_snapshot_wire_format(fake_slo_store, fake_prom, fake_k8s):
    fake_slo_store.set("kimi", "kimi-k25", _cr(min_v=2, max_v=2))
    fake_k8s.placements[("kimi", "kimi-k25")] = _placement()
    fake_k8s.capacity[POOL] = 32
    _healthy_prom(fake_prom, current=2)

    c = Controller(slo=fake_slo_store, k8s=fake_k8s, prom=fake_prom)
    c._tick()
    snap = c.snapshot()
    assert snap["apiVersion"] == "llmscaling.inference.x-k8s.io/v1alpha1"
    assert isinstance(snap["decisions"], list)
    d = snap["decisions"][0]
    assert set(d.keys()) == {"namespace", "serviceId", "replicas"}
    assert set(d["replicas"].keys()) == {"active"}
    assert "warm" not in d["replicas"]


# ---------- 3-tick scenario ----------

def test_three_tick_lifecycle(fake_slo_store, fake_prom, fake_k8s):
    """Scale-up → hold → cooldown-elapse → scale-down."""
    fake_slo_store.set("kimi", "kimi-k25", _cr(min_v=1, max_v=4))
    fake_k8s.placements[("kimi", "kimi-k25")] = _placement(spec_replicas=1)
    fake_k8s.capacity[POOL] = 32
    fake_prom.current_replicas_fn = lambda ns, svc: 1
    fake_prom.otps_fn = lambda ns, svc, kind: 61.0
    fake_prom.rejection_rate_fn = lambda ns, svc: 0.0

    c = Controller(slo=fake_slo_store, k8s=fake_k8s, prom=fake_prom)

    # Tick 1: boot seeds ledger=1; violation is up-cooldown gated → hold.
    fake_prom.ttft_fn = lambda ns, svc, kind: 50.0
    c._tick()
    rows = {d["serviceId"]: d for d in c.snapshot()["decisions"]}
    assert rows["kimi-k25"]["replicas"]["active"] == 1

    # Tick 2: up-cooldown elapsed → violation scales +1.
    c._last_change_at[("kimi", "kimi-k25")] = (
        time_module.time() - est_mod.SCALE_UP_COOLDOWN_S
    )
    c._tick()
    rows = {d["serviceId"]: d for d in c.snapshot()["decisions"]}
    assert rows["kimi-k25"]["replicas"]["active"] == 2

    # Tick 3: everything fine, but up/down cooldowns active from tick 2's change.
    fake_prom.ttft_fn = lambda ns, svc, kind: 5.0
    fake_prom.current_replicas_fn = lambda ns, svc: 2
    c._tick()
    rows = {d["serviceId"]: d for d in c.snapshot()["decisions"]}
    assert rows["kimi-k25"]["replicas"]["active"] == 2   # hold, cooldown

    # Tick 4: pretend time has moved past down-cooldown (implies up too),
    # and that the comfort window has also elapsed.
    c._last_change_at[("kimi", "kimi-k25")] = 0   # force elapsed
    c._comfortable_since[("kimi", "kimi-k25")] = 0
    c._tick()
    rows = {d["serviceId"]: d for d in c.snapshot()["decisions"]}
    assert rows["kimi-k25"]["replicas"]["active"] == 1   # scale down
