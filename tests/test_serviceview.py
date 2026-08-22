"""ServiceView state machine tests.

Per arch §8: comfort streak accumulates and resets on each of the
four named events; boot seeding clamps spec.replicas into [min,max];
commit() always stamps clock and zeroes comfort; max-shrink compresses
this tick bypassing everything; missing SLO signal holds + resets
comfort.
"""

from decision_gen.k8s_state import Placement
from decision_gen.serviceview import (
    COMFORT_SUSTAIN_S, DOWN_COOLDOWN_S, UP_COOLDOWN_S,
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
    """Deep on the safe side of both SLOs."""
    return {
        "ttft": {"p80": ok(5.0)},        # ceiling at 20; comfy <10
        "otps": {"p80": ok(100.0)},      # floor at 30; comfy >60
        "rejection": ok(0.0002),
    }


def violated_readings():
    return {
        "ttft": {"p80": ok(25.0)},       # > 20 → violated (ceiling)
        "otps": {"p80": ok(100.0)},
        "rejection": ok(0.001),
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
    v.step(comfy_readings(), p, CR, None, 0.0)
    assert v.committed == CR["maximumDeployment"]["value"]


def test_boot_stamps_clock_and_zeroes_comfort():
    v = ServiceView("ns", "svc")
    v.step(comfy_readings(), PLACEMENT, CR, physical=3, now=100.0)
    assert v.last_change_at == 100.0
    # This very tick starts the comfort streak — nothing sheds on boot.
    assert v.comfortable_since == 100.0


# ---------- comfort streak (R4) ----------

def test_comfort_streak_accumulates():
    v = ServiceView("ns", "svc")
    v.step(comfy_readings(), PLACEMENT, CR, 3, now=0.0)
    assert v.comfortable_since == 0.0
    v.step(comfy_readings(), PLACEMENT, CR, 3, now=60.0)
    assert v.comfortable_since == 0.0        # still anchored at start
    assert v.step(comfy_readings(), PLACEMENT, CR, 3, now=120.0).comfort_s == 120


def test_comfort_resets_on_violation():
    v = ServiceView("ns", "svc")
    v.step(comfy_readings(), PLACEMENT, CR, 3, now=0.0)
    assert v.comfortable_since is not None
    v.step(violated_readings(), PLACEMENT, CR, 3, now=60.0)
    assert v.comfortable_since is None


def test_comfort_resets_on_missing_signal():
    v = ServiceView("ns", "svc")
    v.step(comfy_readings(), PLACEMENT, CR, 3, now=0.0)
    bad = comfy_readings()
    bad["ttft"]["p80"] = miss()
    v.step(bad, PLACEMENT, CR, 3, now=60.0)
    assert v.comfortable_since is None


def test_comfort_resets_on_no_traffic_nan():
    """NaN ≠ no evidence is a real state; comfort cannot sustain off it."""
    v = ServiceView("ns", "svc")
    v.step(comfy_readings(), PLACEMENT, CR, 3, now=0.0)
    nan = comfy_readings()
    nan["otps"]["p80"] = notraffic()
    v.step(nan, PLACEMENT, CR, 3, now=60.0)
    assert v.comfortable_since is None


def test_comfort_resets_on_commit():
    v = ServiceView("ns", "svc")
    v.step(comfy_readings(), PLACEMENT, CR, 3, now=0.0)
    assert v.comfortable_since is not None
    v.commit(4, now=120.0, cause="planner")
    assert v.comfortable_since is None


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


def test_replicas_ready_missing_falls_back_to_committed():
    """§5 row 2: replication signal missing; estimator magnitude = committed.
    Service otherwise continues — rule 1 still fires on rejection spike."""
    v = ServiceView("ns", "svc")
    v.step(comfy_readings(), PLACEMENT, CR, physical=3, now=0.0)
    hot = comfy_readings()
    hot["rejection"] = ok(0.1)             # r1a fires
    t = v.step(hot, PLACEMENT, CR, physical=None, now=60.0)
    # current=committed=3 (fallback), mult > 1 → proposal > 3
    assert t.proposal is not None and t.proposal.replicas > 3


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


# ---------- end-to-end one-tick stages ----------

def test_r1a_fires_immediately_after_boot_with_rejection():
    """Boot just seeded (last_change=now), gates closed, but r1a is ungated."""
    v = ServiceView("ns", "svc")
    hot = comfy_readings()
    hot["rejection"] = ok(0.1)
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
    t_spike = v.step(hot, p, big_cr, physical=40, now=shed_at + 120)
    assert t_spike.phase.value == "in-drain"
    assert t_spike.proposal is not None and t_spike.proposal.rule == "r1a-fire"
    assert t_spike.proposal.replicas > 34       # r1a computed at current=40
    assert t_spike.est > 34                     # passed the clamp (max=100)
    assert t_spike.want == 34                   # freeze-in-drain snapped it
    assert t_spike.hold_reason.startswith("freeze-in-drain")
