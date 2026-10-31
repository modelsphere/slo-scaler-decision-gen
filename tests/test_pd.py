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

def test_controller_pd_omitted_from_gap_math(fake_slo, fake_k8s,
                                            fake_signals, clock):
    """A pdengine service bypasses the bundle planner: it commits its
    own pd_shape, and the wire carries the shape.pd block on top of
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
