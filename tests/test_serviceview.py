"""ServiceView state machine tests.

Per arch §8: comfort streak accumulates and resets on each of the
four named events; boot seeding clamps spec.replicas into [min,max];
commit() always stamps clock and zeroes comfort; max-shrink compresses
this tick bypassing everything; missing SLO signal holds + resets
comfort.
"""

from decision_gen.k8s_state import Placement
from decision_gen.serviceview import (
    COMFORT_SUSTAIN_S, DOWN_COOLDOWN_S, LOG_STALE_S, UP_COOLDOWN_S,
    ServiceView,
)
from decision_gen.signals import Reading
from decision_gen.thresholds import Verdict


PLACEMENT = Placement(
    namespace="ns", name="svc", kind="deployment",
    pool="p", gpus_per_replica=8,
    workload_name="svc", spec_replicas=3,
)

CR = {
    "minimumDeployment": {"type": "replica", "value": 1},
    "maximumDeployment": {"type": "replica", "value": 8},
    "ttft": {"default": {"metrics": [{"type": "p80", "threshold": 20.0}]}},
    "otps": {"default": {"metrics": [{"type": "p80", "threshold": 30.0}]}},
}


def ok(v): return Reading(v, "ok")
def miss(): return Reading(None, "missing_series")
def notraffic(): return Reading(None, "no_traffic")


def comfy_readings():
    """Deep on the safe side of both SLOs. Counts are above the R1a/R1b
    evidence floors (5 rejections / 20 requests) so tests can focus on
    comfort-vs-hold, not sample-size."""
    return {
        "ttft": {"p80": ok(5.0)},        # ceiling at 20; comfy <10
        "otps": {"p80": ok(100.0)},      # floor at 30; comfy >60
        "rejection": ok(0.0002),
        "rejection_count": ok(0.0),
        "request_count": ok(50.0),
    }


def violated_readings():
    return {
        "ttft": {"p80": ok(25.0)},       # > 20 → violated (ceiling)
        "otps": {"p80": ok(100.0)},
        "rejection": ok(0.001),
        "rejection_count": ok(0.0),
        "request_count": ok(50.0),       # over the R1B_MIN_REQUESTS floor
    }


def _step(view, readings, physical=3, now=0.0):
    return view.step(readings, PLACEMENT, CR, physical, now)


# ---------- boot (R7) ----------

def test_boot_seeds_from_spec_replicas():
    v = ServiceView("ns", "svc")
    _step(v, comfy_readings(), now=0.0)
    assert v.committed == 3


def test_boot_clamps_into_min_max():
    p = Placement("ns", "svc", "deployment", "p", 8, "svc", spec_replicas=99)
    v = ServiceView("ns", "svc")
    v.step(comfy_readings(), p, CR, physical=1, now=0.0)
    assert v.committed == CR["maximumDeployment"]["value"]


def test_missing_max_means_skip_service():
    """No maximumDeployment in CR → user opted out of autoscale. Service
    is skipped every tick (with a warning). No seeding, no commit, no
    clock stamps. State stays frozen until max is set."""
    from decision_gen.serviceview import _bounds
    cr_no_max = {
        "serviceId": "svc",
        "namespace": "ns",
        "minimumDeployment": {"type": "replica", "value": 1},
        "ttft": {"default": {"metrics": [{"type": "p80", "threshold": 20.0}]}},
        "otps": {"default": {"metrics": [{"type": "p80", "threshold": 30.0}]}},
    }
    assert _bounds(cr_no_max) is None

    # End-to-end: spec.replicas=99 would have been the boot seed under
    # the old clamp-to-max path. Now the service is simply skipped.
    p = Placement("ns", "svc", "deployment", "p", 8, "svc", spec_replicas=99)
    v = ServiceView("ns", "svc")
    t = v.step(comfy_readings(), p, cr_no_max, physical=5, now=0.0)
    assert t.skip is True
    assert "no-max-in-cr" in t.hold_reason
    assert v.committed is None              # never seeded
    assert v.last_change_at == 0.0          # no clock stamp


def test_boot_stamps_clock_and_zeroes_comfort():
    v = ServiceView("ns", "svc")
    v.step(comfy_readings(), PLACEMENT, CR, physical=3, now=100.0)
    assert v.last_change_at == 100.0
    # This very tick starts the comfort streak — nothing sheds on boot.
    assert v.comfort_s == 0.0


# ---------- comfort streak (R4) ----------

def test_comfort_streak_accumulates():
    v = ServiceView("ns", "svc")
    v.step(comfy_readings(), PLACEMENT, CR, 3, now=0.0)
    assert v.comfort_s == 0.0
    v.step(comfy_readings(), PLACEMENT, CR, 3, now=60.0)
    assert v.comfort_s == 60.0
    assert v.step(comfy_readings(), PLACEMENT, CR, 3, now=120.0).comfort_s == 120


def test_comfort_resets_on_violation():
    v = ServiceView("ns", "svc")
    v.step(comfy_readings(), PLACEMENT, CR, 3, now=0.0)
    assert v.comfort_s is not None
    v.step(violated_readings(), PLACEMENT, CR, 3, now=60.0)
    assert v.comfort_s is None


def test_comfort_pauses_on_missing_signal():
    """No evidence, no judgement: the missing tick's gap is not counted,
    the comfort before it is kept."""
    v = ServiceView("ns", "svc")
    v.step(comfy_readings(), PLACEMENT, CR, 3, now=0.0)
    v.step(comfy_readings(), PLACEMENT, CR, 3, now=60.0)
    bad = comfy_readings()
    bad["ttft"]["p80"] = miss()
    t = v.step(bad, PLACEMENT, CR, 3, now=120.0)
    assert t.comfort_s == 60                 # frozen, not reset
    t = v.step(comfy_readings(), PLACEMENT, CR, 3, now=180.0)
    assert t.comfort_s == 120                # 0→60 and 120→180


def test_comfort_resets_on_violation_beside_missing_signal():
    """A missing signal does not shield real discomfort elsewhere."""
    v = ServiceView("ns", "svc")
    v.step(comfy_readings(), PLACEMENT, CR, 3, now=0.0)
    bad = violated_readings()
    bad["otps"]["p80"] = miss()
    v.step(bad, PLACEMENT, CR, 3, now=60.0)
    assert v.comfort_s is None


def test_missing_signal_does_not_start_a_streak():
    v = ServiceView("ns", "svc")
    v.step(violated_readings(), PLACEMENT, CR, 3, now=0.0)
    bad = comfy_readings()
    bad["ttft"]["p80"] = miss()
    v.step(bad, PLACEMENT, CR, 3, now=60.0)
    assert v.comfort_s is None


def test_no_traffic_counts_as_comfort_for_shed():
    """R1c: `no_traffic` verdicts are COMFORTABLE (zero demand = deepest
    comfort); comfort streak does NOT reset; shed proceeds once the
    down-cooldown and sustainment both pass. `missing_series` holds and
    pauses the streak (test_comfort_pauses_on_missing_signal above)."""
    v = ServiceView("ns", "svc")
    # Seed at t=0 with normal comfy readings.
    v.step(comfy_readings(), PLACEMENT, CR, 3, now=0.0)
    assert v.comfort_s == 0.0

    # Service goes idle — NaN on both SLO channels.
    idle = comfy_readings()
    idle["ttft"]["p80"] = notraffic()
    idle["otps"]["p80"] = notraffic()
    t = v.step(idle, PLACEMENT, CR, 3, now=60.0)
    # Comfort streak must NOT have reset at t=60 — it kept running.
    assert v.comfort_s == 60.0
    # Verdicts should be COMFORTABLE (not part of missing_slo), so no
    # missing-signal hold fired.
    assert not t.hold_reason.startswith("hold-missing-signal"), t.hold_reason


def test_no_traffic_sheds_after_cooldowns():
    """End-to-end idle-shed: seed at T, stay idle for DOWN_COOLDOWN_S,
    comfort streak (which now covers idle) also clears COMFORT_SUSTAIN_S.
    Shed must fire."""
    v = ServiceView("ns", "svc")
    v.step(comfy_readings(), PLACEMENT, CR, 3, now=0.0)
    idle = comfy_readings()
    idle["ttft"]["p80"] = notraffic()
    idle["otps"]["p80"] = notraffic()
    # Wait past both cooldowns; the streak runs the whole way.
    t = v.step(idle, PLACEMENT, CR, 3, now=float(DOWN_COOLDOWN_S + COMFORT_SUSTAIN_S + 60))
    assert t.proposal is not None and t.proposal.rule == "r1c-shed", t.hold_reason


def test_comfort_resets_on_commit():
    v = ServiceView("ns", "svc")
    v.step(comfy_readings(), PLACEMENT, CR, 3, now=0.0)
    assert v.comfort_s is not None
    v.commit(4, now=120.0, cause="planner")
    assert v.comfort_s is None


# ---------- commit semantics ----------

def test_commit_idempotent_on_same_value():
    v = ServiceView("ns", "svc")
    v.step(comfy_readings(), PLACEMENT, CR, 3, now=0.0)
    assert v.committed == 3
    assert v.last_change_at == 0.0
    changed = v.commit(3, now=500.0, cause="planner")
    assert changed is False
    assert v.last_change_at == 0.0        # unchanged value → no stamp


def test_commit_different_value_stamps():
    v = ServiceView("ns", "svc")
    v.step(comfy_readings(), PLACEMENT, CR, 3, now=0.0)
    changed = v.commit(5, now=99.0, cause="preempt")
    assert changed is True
    assert v.committed == 5
    assert v.last_change_at == 99.0


# ---------- missing-signal hold (§5 row 1) ----------

def test_missing_slo_holds_at_committed_and_names_signal():
    v = ServiceView("ns", "svc")
    v.step(comfy_readings(), PLACEMENT, CR, 3, now=0.0)
    bad = comfy_readings()
    bad["otps"]["p80"] = miss()
    t = v.step(bad, PLACEMENT, CR, 3, now=DOWN_COOLDOWN_S + 60.0)
    assert t.want == v.committed
    assert "hold-missing-signal" in t.hold_reason
    assert "otps.p80:missing_series" in t.hold_reason


# ---------- physical ∈ {None, 0} → skip this tick ----------

def test_skip_on_physical_none():
    """replicas_ready missing → no-stats tick. Don't seed, don't commit,
    don't fire anything even on a perfect fire signal."""
    v = ServiceView("ns", "svc")
    v.step(comfy_readings(), PLACEMENT, CR, physical=3, now=0.0)
    assert v.committed == 3
    hot = comfy_readings()
    hot["rejection"] = ok(1.0)
    hot["rejection_count"] = ok(10.0)
    t = v.step(hot, PLACEMENT, CR, physical=None, now=60.0)
    assert t.skip is True
    assert t.proposal is None
    assert "no-physical-data" in t.hold_reason
    # No clock stamp, no comfort reset, since_change hasn't moved from 0 base.
    assert v.last_change_at == 0.0
    assert v.comfort_s == 0.0              # preserved, not reset


def test_skip_on_physical_zero():
    """replicas_ready = 0 means no backends — all signals from it are
    dead-air data. Same skip posture as None."""
    v = ServiceView("ns", "svc")
    v.step(comfy_readings(), PLACEMENT, CR, physical=3, now=0.0)
    now = 120.0
    hot = comfy_readings()
    hot["rejection"] = ok(1.0)
    hot["rejection_count"] = ok(10.0)
    t = v.step(hot, PLACEMENT, CR, physical=0, now=now)
    assert t.skip is True
    assert t.proposal is None
    assert v.committed == 3
    assert v.last_change_at == 0.0         # not stamped at t=120
    assert v.comfort_s == 0.0              # preserved from t=0


def test_skip_still_seeds_new_service():
    """Fresh service, physical=0 on first sight: still boot-seed from
    spec.replicas (which doesn't depend on prom), so /decisions can show
    the service at its seeded committed. Skip means "don't decide", not
    "deny the service exists on the wire"."""
    v = ServiceView("ns", "svc")
    t = v.step(comfy_readings(), PLACEMENT, CR, physical=0, now=50.0)
    assert t.skip is True
    assert v.committed == 3                 # seeded from spec.replicas
    assert v.last_change_at == 50.0         # clock stamped at seed time


def test_skip_does_not_reset_comfort_streak():
    """Comfort was accumulating before the dead-air tick; it must resume
    seamlessly when real data returns — no reset, no restart."""
    v = ServiceView("ns", "svc")
    v.step(comfy_readings(), PLACEMENT, CR, physical=3, now=0.0)
    v.step(comfy_readings(), PLACEMENT, CR, physical=3, now=60.0)
    assert v.comfort_s == 60.0
    # Dead-air tick mid-streak.
    v.step(comfy_readings(), PLACEMENT, CR, physical=0, now=120.0)
    assert v.comfort_s == 60.0             # unchanged, NOT None
    # Next real tick: streak is 180s, not 60s.
    t = v.step(comfy_readings(), PLACEMENT, CR, physical=3, now=180.0)
    assert t.skip is False
    assert t.comfort_s == 180


# ---------- scale-to-zero: physical=0 with committed=0 evaluates ----------

def _zero_cr():
    cr = dict(CR)
    cr["minimumDeployment"] = {"type": "replica", "value": 0}
    return cr


def _zero_placement(spec_replicas=0):
    return Placement("ns", "svc", "deployment", "p", 8, "svc",
                     spec_replicas=spec_replicas)


def _idle_zero_readings():
    """No backends, no traffic: SLO histograms read no_traffic (deep
    comfort, not missing), rejection quiet."""
    return {
        "ttft": {"p80": notraffic()},
        "otps": {"p80": notraffic()},
        "rejection": ok(0.0002),
        "rejection_count": ok(0.0),
        "request_count": ok(0.0),
    }


def test_shed_reaches_zero_with_min_zero():
    """With minimumDeployment.value=0 the comfort shed walks the last
    step 1→0 instead of flooring at 1."""
    v = ServiceView("ns", "svc")
    p = _zero_placement(spec_replicas=1)
    v.step(_idle_zero_readings(), p, _zero_cr(), physical=1, now=0.0)
    assert v.committed == 1
    t = v.step(_idle_zero_readings(), p, _zero_cr(), physical=1,
               now=DOWN_COOLDOWN_S + COMFORT_SUSTAIN_S + 1)
    assert t.skip is False
    assert t.proposal is not None and t.proposal.rule == "r1c-shed"
    assert t.est == 0
    assert t.want == 0


# ---------- CR max shrink (C4) ----------

def test_max_shrink_compresses_this_tick():
    v = ServiceView("ns", "svc")
    v.step(comfy_readings(), PLACEMENT, CR, 3, now=0.0)
    assert v.committed == 3
    tighter = dict(CR)
    tighter["maximumDeployment"] = {"type": "replica", "value": 2}
    v.step(comfy_readings(), PLACEMENT, tighter, 3, now=60.0)
    assert v.committed == 2


def test_max_shrink_works_with_missing_signal():
    """Compression bypasses the missing-signal hold: CR authority wins."""
    v = ServiceView("ns", "svc")
    v.step(comfy_readings(), PLACEMENT, CR, 3, now=0.0)
    tighter = dict(CR)
    tighter["maximumDeployment"] = {"type": "replica", "value": 2}
    bad = comfy_readings()
    bad["ttft"]["p80"] = miss()
    v.step(bad, PLACEMENT, tighter, 3, now=60.0)
    assert v.committed == 2


# ---------- CR bounds hold on skipped ticks (R9) ----------

def _bounded(mn, mx):
    cr = dict(CR)
    cr["minimumDeployment"] = {"type": "replica", "value": mn}
    cr["maximumDeployment"] = {"type": "replica", "value": mx}
    return cr


def test_max_shrink_applies_when_physical_missing():
    """Suspend (min=max=0) while replicas_ready is absent: the skip
    must not serve the pre-suspend count."""
    v = ServiceView("ns", "svc")
    v.step(comfy_readings(), PLACEMENT, CR, 3, now=0.0)
    for physical in (None, 0):
        v.step(comfy_readings(), PLACEMENT, _bounded(0, 0), physical, now=60.0)
        assert v.committed == 0


def test_min_raise_applies_when_physical_zero():
    """Resume from suspend: the workload sits at zero ready replicas,
    so every tick skips — the raised min must still land."""
    v = ServiceView("ns", "svc")
    v.step(comfy_readings(), PLACEMENT, CR, 3, now=0.0)
    v.step(comfy_readings(), PLACEMENT, _bounded(0, 0), 3, now=60.0)
    assert v.committed == 0
    t = v.step(comfy_readings(), PLACEMENT, _bounded(1, 8), 0, now=120.0)
    assert t.skip is True
    assert v.committed == 1


def test_enforce_bounds_clamps_both_ways():
    v = ServiceView("ns", "svc")
    v.committed = 5
    v.enforce_bounds(_bounded(1, 3), now=0.0)
    assert v.committed == 3
    v.enforce_bounds(_bounded(4, 8), now=60.0)
    assert v.committed == 4
    assert v.last_change_at == 60.0


def test_within_bounds():
    from decision_gen import pd
    from decision_gen.k8s_state import PDSpec
    from decision_gen.serviceview import within_bounds
    assert within_bounds(3, _bounded(1, 8))
    assert not within_bounds(0, _bounded(1, 8))
    assert not within_bounds(9, _bounded(1, 8))
    assert not within_bounds(3, {"minimumDeployment": {"value": 1}})
    pd_cr = {"minimumDeployment": {"type": "pdengine", "value": 6},
             "maximumDeployment": {"type": "pdengine", "value": 8}}
    spec = PDSpec(slices=1, prefill=4, decode=2,
                  role_group_gpus={"prefill": 8, "decode": 8})
    assert within_bounds(pd.PdShape(4, 2, 6), pd_cr, spec)
    assert not within_bounds(pd.PdShape(1, 1, 2), pd_cr, spec)
    assert not within_bounds(pd.PdShape(4, 2, 6), pd_cr, None)
    # Currency mismatch is never in bounds.
    assert not within_bounds(6, pd_cr, spec)
    assert not within_bounds(pd.PdShape(1, 1, 2), _bounded(1, 8))


# ---------- end-to-end one-tick stages ----------

def test_r1a_fires_immediately_after_boot_with_rejection():
    """Boot just seeded (last_change=now), gates closed, but r1a is ungated."""
    v = ServiceView("ns", "svc")
    hot = comfy_readings()
    hot["rejection"] = ok(0.1)
    hot["rejection_count"] = ok(10.0)
    # current physical=3, mult = 1 + 0.1/0.9 × 2 = 1.222; ceil(3×1.222)=4.
    t = v.step(hot, PLACEMENT, CR, physical=3, now=0.0)
    assert t.proposal is not None and t.proposal.rule == "r1a-fire"
    assert t.est == 4
    assert t.want == 4


def test_r1b_blocked_by_up_cooldown():
    v = ServiceView("ns", "svc")
    v.step(comfy_readings(), PLACEMENT, CR, 3, now=0.0)
    t = v.step(violated_readings(), PLACEMENT, CR, 3, now=60.0)
    assert t.hold_reason.startswith("hold-cooldown-up")
    assert t.want == v.committed


def test_r1b_fires_after_up_cooldown():
    v = ServiceView("ns", "svc")
    v.step(comfy_readings(), PLACEMENT, CR, 3, now=0.0)
    t = v.step(violated_readings(), PLACEMENT, CR, 3,
               now=UP_COOLDOWN_S + 1)
    assert t.proposal is not None and t.proposal.rule == "r1b-steady"
    assert t.want == t.proposal.replicas


def test_r1c_requires_both_legs_of_shed_ready():
    """Down cooldown alone is not enough — comfort must sustain too.
    Boot at T. Stay comfortable until T + down-cooldown + tiny → still
    no shed, because comfort streak only started at T."""
    v = ServiceView("ns", "svc")
    v.step(comfy_readings(), PLACEMENT, CR, 3, now=0.0)
    # Advance well under comfort_sustain from boot. shed_ready should be False.
    t = v.step(comfy_readings(), PLACEMENT, CR, 3,
               now=DOWN_COOLDOWN_S - 60)
    assert t.proposal is None
    # Advance past down cooldown but keep comfort streak short.
    t = v.step(comfy_readings(), PLACEMENT, CR, 3,
               now=DOWN_COOLDOWN_S + COMFORT_SUSTAIN_S - 1)
    # Comfort started at now=0 → comfort_s = DOWN_COOLDOWN_S + COMFORT_SUSTAIN_S - 1
    # which is > COMFORT_SUSTAIN_S, so shed_ready should be True here.
    assert t.proposal is not None and t.proposal.rule == "r1c-shed"


def test_i4_shed_stays_shed_through_drain():
    """The acceptance-check #3 unit form. After shed 40→34 commits,
    physical stays 40 the next tick; estimator sees current=40 and
    wants to re-propose 40; direction.reconcile must snap to 34."""
    big_cr = dict(CR, maximumDeployment={"type": "replica", "value": 100})
    p = Placement("ns", "svc", "deployment", "p", 1, "svc", spec_replicas=40)
    v = ServiceView("ns", "svc")
    # Seed at 40 to simulate a stable, big service.
    v.step(comfy_readings(), p, big_cr, physical=40, now=0.0)
    assert v.committed == 40

    # Wait out cooldowns while staying comfortable so shed_ready opens.
    t_shed = None
    for tick in range(1, 90):                     # 90 minutes of quiet ticks
        now = tick * 60.0
        t = v.step(comfy_readings(), p, big_cr, physical=40, now=now)
        if t.proposal is not None and t.proposal.rule == "r1c-shed":
            t_shed = t
            shed_at = now
            break
    assert t_shed is not None, "r1c never fired in 90 min of perfect comfort"
    assert t_shed.want == t_shed.proposal.replicas == 40 - max(1, int(40 * 0.15 + 0.999))
    v.commit(t_shed.want, now=shed_at, cause="planner")
    assert v.committed == t_shed.want    # 34

    # Next tick: executor is still draining. Physical still 40, committed 34.
    t_next = v.step(comfy_readings(), p, big_cr, physical=40, now=shed_at + 60)
    assert t_next.phase.value == "in-drain"
    # No rule fires (gates closed) → est=committed=34; want stays 34.
    assert t_next.want == 34

    # Stronger: even a fresh rejection spike during drain must not
    # "grow back" into the surplus. r1a fires at current=40 with
    # mult>1, but direction.reconcile snaps to committed.
    hot = comfy_readings()
    hot["rejection"] = ok(0.1)
    hot["rejection_count"] = ok(10.0)   # over the R1A_MIN_REJECTIONS floor
    t_spike = v.step(hot, p, big_cr, physical=40, now=shed_at + 120)
    assert t_spike.phase.value == "in-drain"
    assert t_spike.proposal is not None and t_spike.proposal.rule == "r1a-fire"
    assert t_spike.proposal.replicas > 34       # r1a computed at current=40
    assert t_spike.est > 34                     # passed the clamp (max=100)
    assert t_spike.want == 34                   # freeze-in-drain snapped it
    assert t_spike.hold_reason.startswith("freeze-in-drain")


# ---------- request-log freshness gate (issue #14) ----------
#
# A dead log pipeline makes every bodylog-derived signal go flat, which
# reads exactly like an idle service, while the proxy-polled 429 counter
# keeps counting. The global staleness reading is an input signal: stale
# or missing → no evidence → hold, comfort paused.

def _with_log(readings, log):
    readings = dict(readings)
    readings["log"] = log
    return readings


def _idle():
    r = comfy_readings()
    r["ttft"]["p80"] = notraffic()
    r["otps"]["p80"] = notraffic()
    r["rejection"] = ok(0.0)
    r["request_count"] = ok(0.0)
    return r


def test_stale_log_holds_and_pauses_comfort():
    v = ServiceView("ns", "svc")
    v.step(_with_log(comfy_readings(), ok(5.0)), PLACEMENT, CR, 3, now=0.0)
    v.step(_with_log(comfy_readings(), ok(5.0)), PLACEMENT, CR, 3, now=60.0)
    t = v.step(_with_log(_idle(), ok(120.0)), PLACEMENT, CR, 3, now=120.0)
    assert t.hold_reason == "hold-missing-signal log:stale=120s"
    assert t.comfort_s == 60


def test_quiet_gap_does_not_restart_comfort():
    """10 min comfortable, a 2 min cluster-wide gap, 10 min comfortable:
    20 min of judged comfort, so the streak meets COMFORT_SUSTAIN_S."""
    v = ServiceView("ns", "svc")
    now = 0.0
    for _ in range(11):                      # t=0..600: 600s of comfort
        t = v.step(_with_log(comfy_readings(), ok(5.0)), PLACEMENT, CR, 3, now=now)
        now += 60.0
    for _ in range(2):                       # t=660, 720: stale, paused
        t = v.step(_with_log(_idle(), ok(LOG_STALE_S + 30.0)), PLACEMENT, CR, 3, now=now)
        assert t.comfort_s == 600
        now += 60.0
    for _ in range(10):                      # t=780..1320: 600s more
        t = v.step(_with_log(comfy_readings(), ok(5.0)), PLACEMENT, CR, 3, now=now)
        now += 60.0
    assert t.comfort_s == COMFORT_SUSTAIN_S


def test_log_staleness_at_the_limit_is_fresh():
    v = ServiceView("ns", "svc")
    t = v.step(_with_log(comfy_readings(), ok(float(LOG_STALE_S))),
               PLACEMENT, CR, 3, now=0.0)
    assert "log:" not in t.hold_reason
    assert v.comfort_s == 0.0


def test_missing_log_reading_holds():
    v = ServiceView("ns", "svc")
    t = v.step(_with_log(comfy_readings(), miss()), PLACEMENT, CR, 3, now=0.0)
    assert t.hold_reason == "hold-missing-signal log:missing_series"


def test_stale_log_never_sheds_an_idle_service():
    """The issue's first half: the quantiles go no_traffic once the 5m
    window empties, and without the gate comfort accrues and sheds."""
    v = ServiceView("ns", "svc")
    v.step(_with_log(comfy_readings(), ok(5.0)), PLACEMENT, CR, 3, now=0.0)
    for minute in range(1, 69):
        t = v.step(_with_log(_idle(), ok(60.0 * minute)),
                   PLACEMENT, CR, 3, now=60.0 * minute)
        assert t.want == 3, (minute, t.hold_reason)
    # Pipeline back: the 68 stale minutes are not comfort. Only t=0→60
    # (the minute-1 reading was still under LOG_STALE_S) and the first
    # fresh minute count.
    t = v.step(_with_log(_idle(), ok(5.0)), PLACEMENT, CR, 3, now=60.0 * 69)
    assert t.want == 3
    assert t.comfort_s == 120


def test_stale_log_blocks_the_clamped_rejection_fire_alarm():
    """The issue's second half: the request-rate denominator is clamped
    to 0.001, so 429s alone read as rej≈9000 and R1a fires."""
    burst = _idle()
    burst["rejection"] = ok(8970.0)
    burst["rejection_count"] = ok(1000.0)
    v = ServiceView("ns", "svc")
    v.step(_with_log(comfy_readings(), ok(5.0)), PLACEMENT, CR, 3, now=0.0)
    t = v.step(_with_log(burst, ok(3600.0)), PLACEMENT, CR, 3, now=60.0)
    assert t.proposal is None
    assert t.want == 3
