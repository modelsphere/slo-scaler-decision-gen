"""PD (DisaggregatedSet, pdengine) scaling coverage.

Split between pure shape math (`pd.correct_to_bounds` /
`role_margins` / `pick_shed_role`), the trigger axis
(`triggers.evaluate_pd`), per-role direction freezes, and the
serviceview PD branch which wires the whole thing together.

A `pdengine` CR bounds *engines* (fleet-total prefill + decode),
whereas a normal CR bounds replica/bundle counts. The autoscaler
never moves `spec.slices` — that stays a deploy-time property.
"""

import pytest

from decision_gen import direction, pd, triggers
from decision_gen import serviceview as sv
from decision_gen.k8s_state import PDSpec, Placement
from decision_gen.signals import Reading
from decision_gen.thresholds import Verdict

from conftest import FakeK8sState, FakeSignals, FakeSLOStore, FakeClock
from decision_gen.controller import Controller


# ---------------------------------------------------------------------
# pure shape math

def test_correct_to_bounds_in_range_is_identity():
    s = pd.correct_to_bounds(2, 1, 2, 6)
    # E=3 is inside [2,6] → identity.
    assert (s.prefill, s.decode, s.engines) == (2, 1, 3)


def test_correct_to_bounds_grow_to_min():
    s = pd.correct_to_bounds(6, 3, 12, 24)
    # E=9 < 12 → r=12/9=4/3, ceil(6*4/3)=8, ceil(3*4/3)=4; E'=12 ✓
    assert s.engines == 12
    assert (s.prefill, s.decode) == (8, 4)


def test_correct_to_bounds_shrink_to_max():
    s = pd.correct_to_bounds(6, 3, 6, 24)
    # E=9 in range — no-op.
    assert (s.prefill, s.decode) == (6, 3)

    s = pd.correct_to_bounds(6, 3, 2, 6)
    # E=9 > 6 → r=6/9=2/3, floor(4, 2); E'=6 ✓
    assert (s.prefill, s.decode, s.engines) == (4, 2, 6)


def test_correct_to_bounds_grow_with_residual_walk():
    s = pd.correct_to_bounds(2, 1, 8, 8)
    # E=3 < 8 → r=8/3 ≈ 2.67, ceil(2*2.67)=6, ceil(1*2.67)=3; E'=9 overshoot.
    # Residual walk removes the overshoot from decode (largest ceil over).
    assert s.engines == 8
    assert (s.prefill, s.decode) in ((6, 2), (5, 3))
    assert s.prefill >= 2 and s.decode >= 1          # one-engine floor


def test_correct_to_bounds_odd_bound_against_equal_ratio_walks_off_pair():
    s = pd.correct_to_bounds(1, 1, 3, 3)
    # Pair ratio 1:1 cannot represent E=3 exactly; walk one engine off
    # the pair (ties to decode per policy) and land at 1p2d.
    assert s.engines == 3
    assert (s.prefill, s.decode) in ((2, 1), (1, 2))


def test_correct_to_bounds_scale_to_zero():
    s = pd.correct_to_bounds(2, 1, 0, 0)
    assert (s.prefill, s.decode, s.engines) == (0, 0, 0)


def test_correct_to_bounds_zero_shape_refuses():
    assert pd.correct_to_bounds(0, 0, 6, 24) is None


def test_per_slice_window_closes():
    assert pd.per_slice_window(6, 24, 2) == (3, 12)
    assert pd.per_slice_window(7, 7, 2) == (None, None)   # inexpressible
    assert pd.per_slice_window(3, 3, 2) == (None, None)   # inexpressible
    assert pd.per_slice_window(4, 8, 2) == (2, 4)


# ---------------------------------------------------------------------
# margins + shed pick

def test_role_margins_uses_min_of_kinds():
    margins = pd.role_margins(
        {"p80": (10.0, "ok"), "p99": (12.0, "ok")},
        {"p80": (50.0, "ok")},
        {"p80": 20.0, "p99": 20.0},
        {"p80": 100.0},
        headroom=0.5,
    )
    # ttft: p80 → (20*0.5)/10 = 1.0, p99 → (20*0.5)/12 ≈ 0.83 → min 0.83
    # otps: (50*0.5)/100 = 0.25
    assert margins.prefill == pytest.approx((20 * 0.5) / 12)
    assert margins.decode == pytest.approx((50 * 0.5) / 100)


def test_role_margins_no_traffic_is_infinitely_comfortable():
    margins = pd.role_margins(
        {"p80": (None, "no_traffic")},
        {"p80": (50.0, "ok")},
        {"p80": 20.0}, {"p80": 100.0}, 0.5,
    )
    assert margins.prefill == pytest.approx(float("inf"))


def test_pick_shed_role_prefers_bigger_margin():
    m = pd.RoleMargins(prefill=2.0, decode=1.2)
    assert pd.pick_shed_role(6, 3, m) == "prefill"

    m = pd.RoleMargins(prefill=1.2, decode=2.0)
    assert pd.pick_shed_role(6, 3, m) == "decode"


def test_pick_shed_role_floors_at_one_each():
    m = pd.RoleMargins(prefill=5.0, decode=1.0)
    # only prefill can shed (decode already at floor)
    assert pd.pick_shed_role(2, 1, m) == "prefill"

    # both at floor — refuse
    assert pd.pick_shed_role(1, 1, m) is None


def test_pick_shed_role_tie_breaks_by_size_then_decode():
    m = pd.RoleMargins(prefill=1.5, decode=1.5)
    # equal margin: bigger role sheds
    assert pd.pick_shed_role(8, 4, m) == "prefill"
    assert pd.pick_shed_role(4, 8, m) == "decode"
    # equal size: final tie to decode
    assert pd.pick_shed_role(4, 4, m) == "decode"


# ---------------------------------------------------------------------
# triggers on engines

G = triggers.Gates


def test_evaluate_pd_fire_scales_both_roles():
    s = triggers.evaluate_pd(
        {}, pd.RoleMargins(None, None), 0.2, pd.PdShape(2, 1, 3),
        G(up_cooldown_open=False, shed_ready=False),
        rejection_count=10, request_count=None,
    )
    assert s.rule == "r1a-fire"
    assert s.replicas.prefill >= 3 and s.replicas.decode >= 2


def test_evaluate_pd_ttft_violated_grows_prefill_only():
    v = {"ttft.p80": Verdict.VIOLATED, "otps.p80": Verdict.COMFORTABLE}
    s = triggers.evaluate_pd(
        v, pd.RoleMargins(None, None), None, pd.PdShape(2, 1, 3),
        G(up_cooldown_open=True, shed_ready=False),
        rejection_count=None, request_count=100,
    )
    assert s.rule == "r1b-steady"
    assert s.replicas.prefill > 2 and s.replicas.decode == 1


def test_evaluate_pd_otps_violated_grows_decode_only():
    v = {"ttft.p80": Verdict.COMFORTABLE, "otps.p80": Verdict.VIOLATED}
    s = triggers.evaluate_pd(
        v, pd.RoleMargins(None, None), None, pd.PdShape(2, 1, 3),
        G(up_cooldown_open=True, shed_ready=False),
        rejection_count=None, request_count=100,
    )
    assert s.rule == "r1b-steady"
    assert s.replicas.prefill == 2 and s.replicas.decode > 1


def test_evaluate_pd_shed_uses_bigger_margin_role():
    verdicts = {"ttft.p80": Verdict.COMFORTABLE, "otps.p80": Verdict.VIOLATED}
    # all-COMFORTABLE required for shed; replace otps to also-comfy.
    verdicts["otps.p80"] = Verdict.COMFORTABLE
    m = pd.RoleMargins(prefill=3.0, decode=1.1)
    s = triggers.evaluate_pd(
        verdicts, m, 0.0, pd.PdShape(6, 3, 9),
        G(up_cooldown_open=False, shed_ready=True),
        rejection_count=None, request_count=None,
    )
    assert s.rule == "r1c-shed"
    assert s.replicas.prefill < 6 and s.replicas.decode == 3
    assert "prefill" in s.reason


# ---------------------------------------------------------------------
# direction on engines

def test_direction_pd_phases_are_per_role():
    phases = direction.phase_of_pd((2, 1), (3, 2))
    assert phases["prefill"] is direction.Phase.IN_DRAIN
    assert phases["decode"] is direction.Phase.IN_DRAIN

    phases = direction.phase_of_pd((3, 2), (2, 1))
    assert phases["prefill"] is direction.Phase.RAMP_UP
    assert phases["decode"] is direction.Phase.RAMP_UP


def test_direction_reconcile_pd_freezes_aggressive_side_only():
    # prefill is IN_DRAIN (physical > committed — shedding), decode is
    # RAMP_UP (physical < committed — growing). A proposal that grows
    # prefill (against the drain) and sheds decode (against the ramp)
    # must snap both back to committed.
    committed = pd.PdShape(2, 2, 4)
    physical = (3, 1)
    phases = direction.phase_of_pd(
        (committed.prefill, committed.decode), physical,
    )
    proposed = pd.PdShape(4, 1, 5)
    want, reason = direction.reconcile_pd(proposed, committed, phases)
    assert want.prefill == 2 and want.decode == 2
    assert "freeze-in-drain" in reason
    assert "freeze-ramp-up" in reason


# ---------------------------------------------------------------------
# serviceview PD branch

PD_PLACEMENT = Placement(
    namespace="ns", name="svc", kind="disaggregatedset",
    pool="p", gpus_per_replica=24, workload_name="svc",
    spec_replicas=1, members=(("lws", "svc-prefill"), ("lws", "svc-decode")),
    pd=PDSpec(slices=1, prefill=2, decode=1,
              role_group_gpus={"prefill": 8, "decode": 8}),
)


def _pd_readings():
    return {
        "ttft": {"p80": Reading(5.0, "ok")},
        "otps": {"p80": Reading(100.0, "ok")},
        "rejection": Reading(0.0, "ok"),
        "rejection_count": Reading(0, "ok"),
        "request_count": Reading(0, "ok"),
        "replicas_ready_prefill": Reading(2, "ok"),
        "replicas_ready_decode": Reading(1, "ok"),
    }


def _pd_cr():
    return {
        "serviceId": "svc",
        "priority": 5,
        "minimumDeployment": {"type": "pdengine", "value": 6},
        "maximumDeployment": {"type": "pdengine", "value": 24},
        "ttft": {"default": {"metrics": [{"type": "p80", "threshold": 20.0}]}},
        "otps": {"default": {"metrics": [{"type": "p80", "threshold": 30.0}]}},
    }


def test_pd_step_seeds_from_live_spec():
    """Boot: committed comes from the live DS spec (2p1d), not prom.
    Bounds-fix sees E=3 inside the per-slice window [6,24] — wait,
    that's below the floor: min=6 engines per slice. So the boot
    *also* walks the shape up to (4, 2) to satisfy the CR floor."""
    v = sv.ServiceView("ns", "svc")
    t = v.step(_pd_readings(), PD_PLACEMENT, _pd_cr(), 3, now=0.0)
    assert t.skip is False
    assert v.committed == pd.PdShape(4, 2, 6)


def test_pd_step_bounds_fix_moves_committed():
    """External drift: live shape is (4, 2) — six engines — but the
    CR only allows up to four. Bounds-fix shrinks at the live ratio
    (2:1), floors at ≥1 per role: r=4/6=0.67 → (floor 2.67, floor
    1.33) = (2, 1), then the residual walk adds the spare engine
    back to whichever role is closest to its ideal (3, 1) — decode
    was already exactly on its ideal, so prefill takes it."""
    cr = _pd_cr()
    cr["maximumDeployment"] = {"type": "pdengine", "value": 4}
    cr["minimumDeployment"] = {"type": "pdengine", "value": 4}
    over = Placement(
        namespace="ns", name="svc", kind="disaggregatedset",
        pool="p", gpus_per_replica=48, workload_name="svc",
        spec_replicas=1, members=(("lws", "svc-prefill"), ("lws", "svc-decode")),
        pd=PDSpec(slices=1, prefill=4, decode=2,
                  role_group_gpus={"prefill": 8, "decode": 8}),
    )
    readings = _pd_readings()
    readings["replicas_ready_prefill"] = Reading(4, "ok")
    readings["replicas_ready_decode"] = Reading(2, "ok")
    v = sv.ServiceView("ns", "svc")
    t = v.step(readings, over, cr, 3, now=0.0)
    assert t.skip is False
    assert t.pd_shape == pd.PdShape(3, 1, 4)


def test_pd_step_bounds_too_tight_compresses_to_1p1d():
    """Live shape 4p2d (six engines), CR window [2,2] — below the
    ≥1-per-role floor. We do NOT drop the service from the wire;
    that would orphan an already-live shape. Compress to the
    smallest liveable shape (1, 1) and log an error so the CR
    author can fix the bounds/shape disagreement."""
    cr = _pd_cr()
    cr["maximumDeployment"] = {"type": "pdengine", "value": 2}
    cr["minimumDeployment"] = {"type": "pdengine", "value": 2}
    over = Placement(
        namespace="ns", name="svc", kind="disaggregatedset",
        pool="p", gpus_per_replica=48, workload_name="svc",
        spec_replicas=1, members=(("lws", "svc-prefill"), ("lws", "svc-decode")),
        pd=PDSpec(slices=1, prefill=4, decode=2,
                  role_group_gpus={"prefill": 8, "decode": 8}),
    )
    readings = _pd_readings()
    readings["replicas_ready_prefill"] = Reading(4, "ok")
    readings["replicas_ready_decode"] = Reading(2, "ok")
    v = sv.ServiceView("ns", "svc")
    t = v.step(readings, over, cr, 3, now=0.0)
    assert t.skip is False
    assert t.pd_shape == pd.PdShape(1, 1, 2)


def test_pd_step_non_ds_refuses():
    v = sv.ServiceView("ns", "svc")
    placement = Placement(
        namespace="ns", name="svc", kind="deployment",
        pool="p", gpus_per_replica=8, workload_name="svc",
        spec_replicas=2,
    )
    t = v.step(_pd_readings(), placement, _pd_cr(), 2, now=0.0)
    assert t.skip is True
    assert "pdengine-non-ds" in t.hold_reason


def test_pd_step_multi_slice_refuses():
    placement = Placement(
        namespace="ns", name="svc", kind="disaggregatedset",
        pool="p", gpus_per_replica=48, workload_name="svc",
        spec_replicas=2,
        pd=PDSpec(slices=2, prefill=2, decode=1,
                  role_group_gpus={"prefill": 8, "decode": 8}),
    )
    v = sv.ServiceView("ns", "svc")
    t = v.step(_pd_readings(), placement, _pd_cr(), 3, now=0.0)
    assert t.skip is True
    assert "pdengine-slices" in t.hold_reason


def test_pd_step_ttft_violation_grows_prefill():
    """ttft violated past the up-cooldown → grow prefill only.
    Boot already walked the shape from (2,1) to (4,2) to reach min=6
    engines; the violation fires on the *next* tick so the
    bounds-commit's cooldown has aged past UP_COOLDOWN_S."""
    v = sv.ServiceView("ns", "svc")
    readings = _pd_readings()
    v.step(readings, PD_PLACEMENT, _pd_cr(), 3, now=0.0)
    assert v.committed == pd.PdShape(4, 2, 6)      # bounds-fix to min

    readings = _pd_readings()
    readings["ttft"]["p80"] = Reading(50.0, "ok")
    readings["request_count"] = Reading(100, "ok")
    t = v.step(readings, PD_PLACEMENT, _pd_cr(), 3,
               now=sv.UP_COOLDOWN_S + 1)
    assert t.skip is False
    assert t.pd_shape.prefill > 4
    assert t.pd_shape.decode == 2


# ---------------------------------------------------------------------
# controller end-to-end

def test_controller_pd_goes_through_planner_and_wire_shape_block(
        fake_slo, fake_k8s, fake_signals, clock,
):
    """A pdengine service joins the planner in per-role engine units
    and the wire carries the shape.pd block on top of
    replicas.active = slices."""
    engine_cr = {
        "serviceId": "svc",
        "priority": 5,
        "minimumDeployment": {"type": "pdengine", "value": 3},
        "maximumDeployment": {"type": "pdengine", "value": 24},
        "ttft": {"default": {"metrics": [{"type": "p80", "threshold": 20.0}]}},
        "otps": {"default": {"metrics": [{"type": "p80", "threshold": 30.0}]}},
    }
    fake_slo.set("ns", "svc", engine_cr)
    fake_k8s.placements[("ns", "svc")] = PD_PLACEMENT
    fake_k8s.capacity = {"p": 64}
    fake_signals.set_replicas_ready(3)
    fake_signals.set_replicas_ready_role("prefill", 2)
    fake_signals.set_replicas_ready_role("decode", 1)
    fake_signals.set_ttft("ns", "svc", "p80", 5.0)
    fake_signals.set_otps("ns", "svc", "p80", 100.0)
    fake_signals.set_rejection(0.0)

    ctl = Controller(slo=fake_slo, k8s=fake_k8s, signals=fake_signals,
                     tick_seconds=60, clock=clock)
    ctl.tick()

    rows = {d["serviceId"]: d for d in ctl.snapshot()["decisions"]}
    assert "svc" in rows
    row = rows["svc"]
    assert row["replicas"]["active"] == 1      # slices never moved
    assert "shape" in row and "pd" in row["shape"]
    pd_block = row["shape"]["pd"]
    assert pd_block["slices"] == 1
    # CR min=3 = the live (2p,1d) shape's engine total. The group
    # floor is met exactly, so neither role lifts.
    assert pd_block["prefill"]["replicas"] == 2
    assert pd_block["decode"]["replicas"] == 1
    assert pd_block["prefill"]["liveReplicas"] == 2
    assert pd_block["decode"]["liveReplicas"] == 1


def test_controller_pd_non_pd_rows_unchanged(fake_slo, fake_k8s,
                                             fake_signals, clock):
    """Regression: a replica-typed service still renders v1 wire and
    doesn't grow a `shape` block."""
    fake_slo.set("ns", "svc", {
        "serviceId": "svc", "priority": 5,
        "minimumDeployment": {"type": "replica", "value": 1},
        "maximumDeployment": {"type": "replica", "value": 8},
        "ttft": {"default": {"metrics": [{"type": "p80", "threshold": 20.0}]}},
        "otps": {"default": {"metrics": [{"type": "p80", "threshold": 30.0}]}},
    })
    fake_k8s.placements[("ns", "svc")] = Placement(
        namespace="ns", name="svc", kind="deployment",
        pool="p", gpus_per_replica=8, workload_name="svc",
        spec_replicas=2,
    )
    fake_k8s.capacity = {"p": 64}
    fake_signals.set_replicas_ready(2)
    fake_signals.set_ttft("ns", "svc", "p80", 5.0)
    fake_signals.set_otps("ns", "svc", "p80", 100.0)
    fake_signals.set_rejection(0.0)

    ctl = Controller(slo=fake_slo, k8s=fake_k8s, signals=fake_signals,
                     tick_seconds=60, clock=clock)
    ctl.tick()

    row = ctl.snapshot()["decisions"][0]
    assert row == {"namespace": "ns", "serviceId": "svc",
                   "replicas": {"active": 2}}
    assert "shape" not in row


# ---------------------------------------------------------------------
# regression: review findings on PR #11 round 1

def _replica_cr(mn=1, mx=8):
    return {
        "serviceId": "svc", "priority": 5,
        "minimumDeployment": {"type": "replica", "value": mn},
        "maximumDeployment": {"type": "replica", "value": mx},
        "ttft": {"default": {"metrics": [{"type": "p80", "threshold": 20.0}]}},
        "otps": {"default": {"metrics": [{"type": "p80", "threshold": 30.0}]}},
    }


_DEPLOY_PLACEMENT = Placement(
    namespace="ns", name="svc", kind="deployment",
    pool="p", gpus_per_replica=8, workload_name="svc", spec_replicas=2,
)


def test_cr_type_flip_replica_to_pdengine_reseeds():
    """A replica-typed view whose CR flips to pdengine must reseed from
    the live DS spec, not crash comparing an int against a PdShape."""
    v = sv.ServiceView("ns", "svc")
    t = v.step(_pd_readings(), _DEPLOY_PLACEMENT, _replica_cr(), 2, now=0.0)
    assert v.committed == 2
    t = v.step(_pd_readings(), PD_PLACEMENT, _pd_cr(), 3, now=60.0)
    assert t.skip is False
    assert isinstance(v.committed, pd.PdShape)


def test_cr_type_flip_pdengine_to_replica_reseeds():
    """The reverse flip — pdengine view becomes replica-typed — must
    also reseed rather than raise a TypeError on int > PdShape."""
    v = sv.ServiceView("ns", "svc")
    v.step(_pd_readings(), PD_PLACEMENT, _pd_cr(), 3, now=0.0)
    assert isinstance(v.committed, pd.PdShape)
    t = v.step(_pd_readings(), _DEPLOY_PLACEMENT, _replica_cr(), 2, now=60.0)
    assert t.skip is False
    assert v.committed == 2


def test_pd_step_seeds_from_ds_spec_when_physical_missing():
    """A pdengine service whose replicas_ready never comes up must
    still appear on the wire at its seeded shape (spec), matching the
    replica path's boot posture."""
    empty = {
        "rejection": Reading(None, "missing_series"),
        "replicas_ready_prefill": Reading(None, "missing_series"),
        "replicas_ready_decode": Reading(None, "missing_series"),
    }
    v = sv.ServiceView("ns", "svc")
    t = v.step(empty, PD_PLACEMENT, _pd_cr(), None, now=0.0)
    assert v.committed == pd.PdShape(2, 1, 3)
    assert t.pd_shape == pd.PdShape(2, 1, 3)


def test_pd_step_holds_when_role_physical_missing():
    """Without per-role replicas_ready evidence the direction freeze
    cannot see ramps/drains (I4); the PD branch must hold rather than
    fall back to the committed value."""
    readings = _pd_readings()
    readings["replicas_ready_prefill"] = Reading(None, "missing_series")
    v = sv.ServiceView("ns", "svc")
    v.step(_pd_readings(), PD_PLACEMENT, _pd_cr(), 3, now=0.0)
    t = v.step(readings, PD_PLACEMENT, _pd_cr(), 3,
               now=sv.UP_COOLDOWN_S + 1)
    assert t.skip is False
    assert t.hold_reason == "hold-pd-no-role-physical"
    assert t.pd_shape == v.committed


def test_pd_step_est_clamped_at_min():
    """A shed below the CR floor must not reach the wire; the replica
    path clamps est to [min, max], the PD path must behave the same.
    Shedding still fires from >= min shapes — only the crossing is
    pinned."""
    v = sv.ServiceView("ns", "svc")
    placement = Placement(
        namespace="ns", name="svc", kind="disaggregatedset",
        pool="p", gpus_per_replica=32, workload_name="svc",
        spec_replicas=1, members=(("lws", "svc-prefill"), ("lws", "svc-decode")),
        pd=PDSpec(slices=1, prefill=2, decode=2,
                  role_group_gpus={"prefill": 8, "decode": 8}),
    )
    cr = _pd_cr()
    cr["minimumDeployment"] = {"type": "pdengine", "value": 3}
    cr["maximumDeployment"] = {"type": "pdengine", "value": 24}
    # Comfortably deep so R1c fires after the comfort/down-cooldown
    # window; min=3 lets exactly one engine leave before the clamp.
    cr["ttft"]["default"]["metrics"][0]["threshold"] = 100.0
    cr["otps"]["default"]["metrics"][0]["threshold"] = 200.0
    readings = _pd_readings()
    readings["ttft"]["p80"] = Reading(1.0, "ok")
    readings["otps"]["p80"] = Reading(500.0, "ok")
    readings["replicas_ready_prefill"] = Reading(2, "ok")
    readings["replicas_ready_decode"] = Reading(2, "ok")
    v.step(readings, placement, cr, 4, now=0.0)
    big = sv.DOWN_COOLDOWN_S + sv.COMFORT_SUSTAIN_S + 100
    t = v.step(readings, placement, cr, 4, now=big)
    assert t.pd_shape == pd.PdShape(1, 2, 3)   # first shed lands at min
    # Next comfort-aged tick must NOT dip below 3.
    v.commit(t.pd_shape, big, cause="test")    # simulate controller commit
    # Re-age; the shape is now sitting at the floor.
    readings["replicas_ready_prefill"] = Reading(1, "ok")
    t2 = v.step(readings, placement, cr, 3,
                now=big + sv.DOWN_COOLDOWN_S + 1)
    assert t2.pd_shape is None or t2.pd_shape.engines >= 3


def test_pd_step_max_clamp_is_a_hold_not_a_rebalance():
    """Growth past the CR ceiling trims the *delta* back so the
    sibling role is untouched. A fully-crossing proposal (delta ==
    entire growth) still holds — no side sheds engines it didn't
    ask for."""
    v = sv.ServiceView("ns", "svc")
    cr = _pd_cr()
    cr["minimumDeployment"] = {"type": "pdengine", "value": 6}
    cr["maximumDeployment"] = {"type": "pdengine", "value": 6}
    tight = Placement(
        namespace="ns", name="svc", kind="disaggregatedset",
        pool="p", gpus_per_replica=48, workload_name="svc",
        spec_replicas=1, members=(("lws", "svc-prefill"), ("lws", "svc-decode")),
        pd=PDSpec(slices=1, prefill=1, decode=5,
                  role_group_gpus={"prefill": 8, "decode": 8}),
    )
    readings = _pd_readings()
    readings["replicas_ready_prefill"] = Reading(1, "ok")
    readings["replicas_ready_decode"] = Reading(5, "ok")
    v.step(readings, tight, cr, 6, now=0.0)
    readings["ttft"]["p80"] = Reading(50.0, "ok")     # violated
    readings["request_count"] = Reading(100, "ok")
    t = v.step(readings, tight, cr, 6, now=sv.UP_COOLDOWN_S + 1)
    # Trim of (+1, 0) against cap 6 lands at committed — decode
    # keeps its 5, prefill keeps its 1.
    assert t.pd_shape == v.committed
    assert "hold-pd-max-clamp" in t.hold_reason


def test_pd_step_max_clamp_trims_delta_to_bound():
    """6p1d at cap 8, violated by ttft: the +step proposal lands
    at the cap by trimming the *step* (here `step=2`), never by
    shedding the sibling. Reviewer round-2 claim C."""
    v = sv.ServiceView("ns", "svc")
    cr = _pd_cr()
    cr["minimumDeployment"] = {"type": "pdengine", "value": 2}
    cr["maximumDeployment"] = {"type": "pdengine", "value": 8}
    wide = Placement(
        namespace="ns", name="svc", kind="disaggregatedset",
        pool="p", gpus_per_replica=56, workload_name="svc",
        spec_replicas=1, members=(("lws", "svc-prefill"), ("lws", "svc-decode")),
        pd=PDSpec(slices=1, prefill=6, decode=1,
                  role_group_gpus={"prefill": 8, "decode": 8}),
    )
    readings = _pd_readings()
    readings["replicas_ready_prefill"] = Reading(6, "ok")
    readings["replicas_ready_decode"] = Reading(1, "ok")
    v.step(readings, wide, cr, 7, now=0.0)
    # Subsequent tick: violation, cooldown open. step = ceil(7 ×
    # 0.15) = 2 → proposal (8, 1), trimmed to (7, 1) at the cap.
    readings["ttft"]["p80"] = Reading(50.0, "ok")
    readings["request_count"] = Reading(500, "ok")
    t = v.step(readings, wide, cr, 7, now=sv.UP_COOLDOWN_S + 1)
    assert t.pd_shape == pd.PdShape(7, 1, 8)
    # Partial trim = a real change at the bound: clamped-*, not hold-*.
    assert "clamped-pd-max" in t.hold_reason
    assert "hold-pd" not in t.hold_reason


def test_pd_step_heterogeneous_role_group_gpus():
    """Prefill TP4 (4 GPUs) / decode TP8 (8 GPUs) is a common PD
    shape. The controller must charge each role its own rate and
    never raise on the non-uniform mix."""
    v = sv.ServiceView("ns", "svc")
    cr = _pd_cr()
    mixed = Placement(
        namespace="ns", name="svc", kind="disaggregatedset",
        pool="p", gpus_per_replica=16, workload_name="svc",
        spec_replicas=1, members=(("lws", "svc-prefill"), ("lws", "svc-decode")),
        pd=PDSpec(slices=1, prefill=2, decode=1,
                  role_group_gpus={"prefill": 4, "decode": 8}),
    )
    t = v.step(_pd_readings(), mixed, cr, 3, now=0.0)
    assert t.skip is False
    assert v.committed == pd.PdShape(4, 2, 6)     # bounds-fix to min=6
    # GPU cost must be 4*4 + 2*8 = 32, not 6 * a single rate.
    cost = pd.shape_gpu_cost(v.committed, mixed.pd.role_group_gpus)
    assert cost == 32


def test_pd_commit_log_line_never_crashes(caplog):
    """The audit line uses %s + _fmt_committed for both replica ints
    and PdShapes; %d formatting would raise TypeError and the line
    would be lost."""
    import logging
    v = sv.ServiceView("ns", "svc")
    v.committed = pd.PdShape(2, 1, 3)
    with caplog.at_level(logging.INFO):
        v.commit(pd.PdShape(4, 2, 6), 0.0, cause="test")
    rows = [r.getMessage() for r in caplog.records if "commit" in r.getMessage()]
    assert rows and "4p2d" in rows[-1]


def test_controller_pd_replica_wins_contested_pool_gpus(
        fake_slo, fake_k8s, fake_signals, clock):
    """A replica service that wants capacity in a saturated pool wins
    it; a pdengine service that wants growth the same tick is denied
    (`hold-pd-no-capacity`). pdengine preemption against replica
    services is intentionally out of scope for this PR — see the
    CHANGELOG."""
    pd_cr = {
        "serviceId": "svc", "priority": 9,   # higher than svc2
        "minimumDeployment": {"type": "pdengine", "value": 2},
        "maximumDeployment": {"type": "pdengine", "value": 24},
        "ttft": {"default": {"metrics": [{"type": "p80", "threshold": 20.0}]}},
        "otps": {"default": {"metrics": [{"type": "p80", "threshold": 30.0}]}},
    }
    rep_cr = {
        "serviceId": "svc2", "priority": 1,
        "minimumDeployment": {"type": "replica", "value": 1},
        "maximumDeployment": {"type": "replica", "value": 3},
        "ttft": {"default": {"metrics": [{"type": "p80", "threshold": 20.0}]}},
        "otps": {"default": {"metrics": [{"type": "p80", "threshold": 30.0}]}},
    }
    fake_slo.set("ns", "svc", pd_cr)
    fake_slo.set("ns", "svc2", rep_cr)
    fake_k8s.placements[("ns", "svc")] = Placement(
        namespace="ns", name="svc", kind="disaggregatedset",
        pool="p", gpus_per_replica=24, workload_name="svc",
        spec_replicas=1, members=(("lws", "svc-prefill"), ("lws", "svc-decode")),
        pd=PDSpec(slices=1, prefill=2, decode=1,
                  role_group_gpus={"prefill": 8, "decode": 8}),
    )
    fake_k8s.placements[("ns", "svc2")] = Placement(
        namespace="ns", name="svc2", kind="deployment",
        pool="p", gpus_per_replica=8, workload_name="svc2",
        spec_replicas=1,
    )
    # 32 GPUs: PD holds 24, replica holds 8. Free = 0.
    fake_k8s.capacity = {"p": 32}
    fake_signals.set_replicas_ready(3, "ns", "svc")
    fake_signals.set_replicas_ready(1, "ns", "svc2")
    fake_signals.set_replicas_ready_role("prefill", 2)
    fake_signals.set_replicas_ready_role("decode", 1)
    fake_signals.ttft_fn[("ns", "svc", "p80")] = 5.0
    fake_signals.otps_fn[("ns", "svc", "p80")] = 100.0
    fake_signals.ttft_fn[("ns", "svc2", "p80")] = 5.0
    fake_signals.otps_fn[("ns", "svc2", "p80")] = 100.0
    fake_signals.rejection_val = 0.0

    ctl = Controller(slo=fake_slo, k8s=fake_k8s, signals=fake_signals,
                     tick_seconds=60, clock=clock)
    ctl.tick()
    assert ctl._views[("ns", "svc")].committed == pd.PdShape(2, 1, 3)
    assert ctl._views[("ns", "svc2")].committed == 1

    # Tick 2: svc's own estimator wants one more engine (ttft
    # violated, up-cooldown open) but the pool is saturated — the
    # growth is denied by the pool ledger, not the CR window.
    fake_signals.ttft_fn[("ns", "svc", "p80")] = 50.0
    fake_signals.request_count_val = 500       # R1b evidence floor
    clock.advance(sv.UP_COOLDOWN_S * 2)
    ctl.tick()
    assert ctl._views[("ns", "svc")].committed == pd.PdShape(2, 1, 3)

    # Tick 3: free up 8 GPUs (pool grows) and let the same want land.
    fake_k8s.capacity = {"p": 40}
    clock.advance(sv.UP_COOLDOWN_S * 2)
    ctl.tick()
    assert ctl._views[("ns", "svc")].committed == pd.PdShape(3, 1, 4)


def test_controller_pd_placement_unresolvable_keeps_serving_from_last_spec(
        fake_slo, fake_k8s, fake_signals, clock):
    """Placement read fails on tick N+1 for a pdengine service whose
    view is already seeded — the wire must keep rendering the row
    from the PDSpec it last resolved, not crash or drop to nulls."""
    engine_cr = {
        "serviceId": "svc", "priority": 5,
        "minimumDeployment": {"type": "pdengine", "value": 2},
        "maximumDeployment": {"type": "pdengine", "value": 24},
        "ttft": {"default": {"metrics": [{"type": "p80", "threshold": 20.0}]}},
        "otps": {"default": {"metrics": [{"type": "p80", "threshold": 30.0}]}},
    }
    fake_slo.set("ns", "svc", engine_cr)
    fake_k8s.placements[("ns", "svc")] = PD_PLACEMENT
    fake_k8s.capacity = {"p": 64}
    fake_signals.set_replicas_ready(3, "ns", "svc")
    fake_signals.set_replicas_ready_role("prefill", 2)
    fake_signals.set_replicas_ready_role("decode", 1)
    fake_signals.set_ttft("ns", "svc", "p80", 5.0)
    fake_signals.set_otps("ns", "svc", "p80", 100.0)
    fake_signals.set_rejection(0.0)

    ctl = Controller(slo=fake_slo, k8s=fake_k8s, signals=fake_signals,
                     tick_seconds=60, clock=clock)
    ctl.tick()
    assert ctl.snapshot()["decisions"][0]["shape"]["pd"]["prefill"]["replicas"] == 2

    # Placement goes away; the tick must still serve, off the spec the
    # controller cached on a good tick.
    del fake_k8s.placements[("ns", "svc")]
    clock.advance(60)
    ctl.tick()      # must not raise
    row = [d for d in ctl.snapshot()["decisions"] if d["serviceId"] == "svc"]
    assert row, "pdengine row must survive placement loss"
    assert row[0]["shape"]["pd"]["prefill"]["liveReplicas"] is not None


def test_controller_pd_max_removed_and_readded_full_cycle(
        fake_slo, fake_k8s, fake_signals, clock):
    """maximumDeployment removed (opt-out) on a pdengine service that
    was managed drops the row; re-adding `maximumDeployment` re-lists
    it from a *fresh* boot seed — the controller clears its cached
    pd spec and re-seeds from the live DS spec."""
    engine_cr = {
        "serviceId": "svc", "priority": 5,
        "minimumDeployment": {"type": "pdengine", "value": 2},
        "maximumDeployment": {"type": "pdengine", "value": 24},
        "ttft": {"default": {"metrics": [{"type": "p80", "threshold": 20.0}]}},
        "otps": {"default": {"metrics": [{"type": "p80", "threshold": 30.0}]}},
    }
    fake_slo.set("ns", "svc", engine_cr)
    fake_k8s.placements[("ns", "svc")] = PD_PLACEMENT
    fake_k8s.capacity = {"p": 64}
    fake_signals.set_replicas_ready(3, "ns", "svc")
    fake_signals.set_replicas_ready_role("prefill", 2)
    fake_signals.set_replicas_ready_role("decode", 1)
    fake_signals.set_ttft("ns", "svc", "p80", 5.0)
    fake_signals.set_otps("ns", "svc", "p80", 100.0)
    fake_signals.set_rejection(0.0)

    ctl = Controller(slo=fake_slo, k8s=fake_k8s, signals=fake_signals,
                     tick_seconds=60, clock=clock)
    ctl.tick()
    row = [d for d in ctl.snapshot()["decisions"] if d["serviceId"] == "svc"]
    assert row, "row listed with shape.pd on a healthy tick"

    broken = dict(engine_cr)
    broken.pop("maximumDeployment")
    fake_slo.set("ns", "svc", broken)
    clock.advance(60)
    ctl.tick()
    rows = ctl.snapshot()["decisions"]
    assert rows == [] or all(d["serviceId"] != "svc" for d in rows)
    assert ("ns", "svc") not in ctl._views
    assert ("ns", "svc") not in ctl._pd_specs

    # Re-add the max: the service comes back from a fresh boot seed
    # (its committed is the live DS shape again, clamped by the new
    # CR window) and shape.pd reappears.
    fake_slo.set("ns", "svc", engine_cr)
    clock.advance(60)
    ctl.tick()
    row = [d for d in ctl.snapshot()["decisions"] if d["serviceId"] == "svc"]
    assert row
    pd_blk = row[0]["shape"]["pd"]
    assert pd_blk["prefill"]["liveReplicas"] is not None
    assert pd_blk["decode"]["liveReplicas"] is not None


def test_controller_replica_max_removed_and_readded_full_cycle(
        fake_slo, fake_k8s, fake_signals, clock):
    """Opt-out on a replica service drops the row on the next tick;
    re-adding `maximumDeployment` re-lists it from a fresh boot seed."""
    rep_cr = {
        "serviceId": "svc", "priority": 5,
        "minimumDeployment": {"type": "replica", "value": 1},
        "maximumDeployment": {"type": "replica", "value": 8},
        "ttft": {"default": {"metrics": [{"type": "p80", "threshold": 20.0}]}},
        "otps": {"default": {"metrics": [{"type": "p80", "threshold": 30.0}]}},
    }
    fake_slo.set("ns", "svc", rep_cr)
    fake_k8s.placements[("ns", "svc")] = Placement(
        namespace="ns", name="svc", kind="deployment",
        pool="p", gpus_per_replica=8, workload_name="svc",
        spec_replicas=2,
    )
    fake_k8s.capacity = {"p": 64}
    fake_signals.set_replicas_ready(2, "ns", "svc")
    fake_signals.set_ttft("ns", "svc", "p80", 5.0)
    fake_signals.set_otps("ns", "svc", "p80", 100.0)
    fake_signals.set_rejection(0.0)

    ctl = Controller(slo=fake_slo, k8s=fake_k8s, signals=fake_signals,
                     tick_seconds=60, clock=clock)
    ctl.tick()
    assert ctl.snapshot()["decisions"] == [
        {"namespace": "ns", "serviceId": "svc", "replicas": {"active": 2}}
    ]

    broken = dict(rep_cr)
    broken.pop("maximumDeployment")
    fake_slo.set("ns", "svc", broken)
    clock.advance(60)
    ctl.tick()
    assert ctl.snapshot()["decisions"] == []
    assert ("ns", "svc") not in ctl._views

    # Re-add: fresh boot seed, committed = clamped spec.replicas (2).
    fake_slo.set("ns", "svc", rep_cr)
    clock.advance(60)
    ctl.tick()
    assert ctl.snapshot()["decisions"] == [
        {"namespace": "ns", "serviceId": "svc", "replicas": {"active": 2}}
    ]