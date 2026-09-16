"""JobSLO (queue-driven) tick behavior: verdict algebra, step rules,
comfort streak, holds, planner participation.

Both sides of every threshold — see the feedback-tests-domain-semantics
memory note. Thresholds under test:
  VIOLATED if depth/replica > maxDepth
  COMFORTABLE if depth/replica < maxDepth × SLO_HEADROOM (0.5)
  GREY between
"""

import pytest

from decision_gen.controller import Controller
from decision_gen.k8s_state import Placement
from decision_gen.serviceview import (
    COMFORT_SUSTAIN_S,
    DOWN_COOLDOWN_S,
    UP_COOLDOWN_S,
    ServiceView,
)
from decision_gen.signals import Reading
from decision_gen.thresholds import Verdict
from decision_gen.triggers import evaluate_job, Gates


# ---------- helpers ----------

def _placement(pool="NVIDIA-H100-80GB-HBM3", gpr=1, spec_replicas=2):
    return Placement(
        namespace="ns", name="svc", kind="deployment",
        pool=pool, gpus_per_replica=gpr,
        workload_name="svc", spec_replicas=spec_replicas,
    )


def _signals_with(fake_signals, ns, svc, depth, replicas=2):
    fake_signals.set_replicas_ready(replicas)
    fake_signals.set_queue_depth(ns, svc, depth)


def _controller_with_job(fake_slo, fake_k8s, fake_signals, clock, cr_spec):
    """Wire a Controller whose job-store carries `cr_spec` for ns/svc."""
    from conftest import FakeSLOStore
    fake_slo_job = FakeSLOStore(specs={("ns", "svc"): cr_spec},
                                plural="jobslorequirements")
    return Controller(
        slo=fake_slo, k8s=fake_k8s, signals=fake_signals,
        clock=clock, slo_job=fake_slo_job, tick_seconds=60,
    )


# ---------- verdict algebra (both sides) ----------

class TestJobVerdicts:
    """_job_verdicts: classify depth/replica vs maxDepth."""

    def _classify(self, cr_spec, depth, physical=2):
        view = ServiceView("ns", "svc")
        readings = {"queue_depth": Reading(depth, "ok")}
        verdicts, missing = ServiceView._job_verdicts(readings, cr_spec, physical)
        return verdicts["queue"]["depth"], missing

    def test_violated_when_above_maxdepth(self, job_cr_spec):
        # per_replica = 12/2 = 6 > 5
        v, missing = self._classify(job_cr_spec, depth=12, physical=2)
        assert v is Verdict.VIOLATED
        assert missing == []

    def test_violated_strict_above_maxdepth_at_boundary(self, job_cr_spec):
        # depth=11 → per=5.5 > 5 → violated (strict: exactly 5 is NOT comfort)
        v, _ = self._classify(job_cr_spec, depth=11, physical=2)
        assert v is Verdict.VIOLATED

    def test_grey_at_exactly_maxdepth(self, job_cr_spec):
        # depth=10 → per=5 = threshold → GREY (strict comparisons)
        v, _ = self._classify(job_cr_spec, depth=10, physical=2)
        assert v is Verdict.GREY

    def test_grey_between_headroom_and_maxdepth(self, job_cr_spec):
        # per=3 is in (2.5, 5) → GREY. Blocks shed; doesn't fire up.
        v, _ = self._classify(job_cr_spec, depth=6, physical=2)
        assert v is Verdict.GREY

    def test_comfortable_below_headroom(self, job_cr_spec):
        # per=1 < 2.5 → COMFORTABLE.
        v, _ = self._classify(job_cr_spec, depth=2, physical=2)
        assert v is Verdict.COMFORTABLE

    def test_comfortable_at_zero_depth(self, job_cr_spec):
        # The canonical `or vector(0)` shape: reading is 0, state ok → comfy.
        v, _ = self._classify(job_cr_spec, depth=0, physical=2)
        assert v is Verdict.COMFORTABLE

    def test_no_traffic_reading_counts_as_comfortable(self, job_cr_spec):
        # NaN histogram read — deepest possible comfort.
        view = ServiceView("ns", "svc")
        readings = {"queue_depth": Reading(None, "no_traffic")}
        verdicts, missing = ServiceView._job_verdicts(readings, job_cr_spec, 2)
        assert verdicts["queue"]["depth"] is Verdict.COMFORTABLE
        assert missing == []

    def test_missing_series_produces_hold(self, job_cr_spec):
        # Prom 0 rows / error / bad shape → hold, reset comfort.
        readings = {"queue_depth": Reading(None, "missing_series")}
        verdicts, missing = ServiceView._job_verdicts(readings, job_cr_spec, 2)
        assert verdicts["queue"]["depth"] is None
        assert missing == ["queue.depth:missing_series"]

    def test_unconfigured_maxdepth_produces_hold(self):
        # maxDepth=0/None would make comparison meaningless → hold.
        cr = {"serviceId": "svc", "queue": {"maxDepth": 0, "promql": "x"}}
        readings = {"queue_depth": Reading(0, "ok")}
        verdicts, missing = ServiceView._job_verdicts(readings, cr, 2)
        assert verdicts["queue"]["depth"] is None
        assert "queue.depth:unconfigured" in missing

    def test_no_queue_depth_reading_is_unconfigured(self, job_cr_spec):
        verdicts, missing = ServiceView._job_verdicts({}, job_cr_spec, 2)
        assert verdicts["queue"]["depth"] is None
        assert "queue.depth:unconfigured" in missing


# ---------- triggers (pure) ----------

class TestEvaluateJob:
    def test_fires_step_up_when_violated_and_cooldown_open(self):
        gates = Gates(up_cooldown_open=True, shed_ready=False)
        p = evaluate_job({"queue.depth": Verdict.VIOLATED}, current=4, gates=gates)
        assert p is not None
        assert p.rule == "rjob-step-up"
        assert p.replicas == 4 + max(1, int(4 * 0.15 + 0.999))  # ceil
        assert p.replicas == 5

    def test_step_up_blocked_when_cooldown_closed(self):
        gates = Gates(up_cooldown_open=False, shed_ready=False)
        p = evaluate_job({"queue.depth": Verdict.VIOLATED}, current=4, gates=gates)
        assert p is None

    def test_step_up_blocked_on_grey(self):
        gates = Gates(up_cooldown_open=True, shed_ready=False)
        p = evaluate_job({"queue.depth": Verdict.GREY}, current=4, gates=gates)
        assert p is None

    def test_step_up_single_replica_floor(self):
        # current=1, frac=0.15 → ceil(0.15)=1; max(1,...)=1
        gates = Gates(up_cooldown_open=True, shed_ready=False)
        p = evaluate_job({"queue.depth": Verdict.VIOLATED}, current=1, gates=gates)
        assert p.replicas == 2

    def test_shed_fires_when_comfortable_and_shed_ready(self):
        gates = Gates(up_cooldown_open=False, shed_ready=True)
        p = evaluate_job({"queue.depth": Verdict.COMFORTABLE}, current=4, gates=gates)
        assert p is not None
        assert p.rule == "rjob-shed"
        assert p.replicas == 3

    def test_shed_blocked_when_shed_not_ready(self):
        gates = Gates(up_cooldown_open=False, shed_ready=False)
        p = evaluate_job({"queue.depth": Verdict.COMFORTABLE}, current=4, gates=gates)
        assert p is None

    def test_shed_blocked_on_grey(self):
        gates = Gates(up_cooldown_open=False, shed_ready=True)
        p = evaluate_job({"queue.depth": Verdict.GREY}, current=4, gates=gates)
        assert p is None

    def test_no_verdicts_is_hold(self):
        gates = Gates(up_cooldown_open=True, shed_ready=True)
        assert evaluate_job({}, current=4, gates=gates) is None


# ---------- ServiceView end-to-end ----------

class TestJobView:
    def _tick(self, view, cr_spec, readings, physical, now):
        return view.step(readings, _placement(), cr_spec, physical, now,
                         kind="job")

    def test_seed_from_spec_replicas_and_clamp(self, job_cr_spec):
        view = ServiceView("ns", "svc")
        readings = {"queue_depth": Reading(0, "ok")}
        t = self._tick(view, job_cr_spec, readings, physical=2, now=100)
        assert view.committed == 2              # seeded from spec_replicas
        assert t.skip is False
        # Freshly-seeded clock + COMFORTABLE verdict ⇒ hold on down-cooldown.
        assert t.hold_reason.startswith("hold-cooldown-down")

    def test_fire_step_up_when_violated_and_cooldown_open(self, job_cr_spec):
        view = ServiceView("ns", "svc")
        # Pre-seed via a past commit so the up-cooldown clock is old.
        view.commit(2, now=0, cause="boot-seed")
        # depth=12 → per=6 > 5 → violated; gate open (0 was long ago)
        readings = {"queue_depth": Reading(12, "ok")}
        t = self._tick(view, job_cr_spec, readings, physical=2,
                       now=UP_COOLDOWN_S + 1)
        assert t.proposal is not None
        assert t.proposal.rule == "rjob-step-up"
        assert t.want == 3                       # 2 + ceil(2 × 0.15) = 2 + 1
        assert t.hold_reason == ""

    def test_step_up_clamped_at_cr_max(self, job_cr_spec):
        cr = dict(job_cr_spec)
        cr["maximumDeployment"] = {"type": "replica", "value": 2}
        view = ServiceView("ns", "svc")
        view.commit(2, now=0, cause="boot-seed")
        readings = {"queue_depth": Reading(50, "ok")}
        t = self._tick(view, cr, readings, physical=2, now=UP_COOLDOWN_S + 1)
        assert t.want == 2                       # clamped at max

    def test_hold_when_cooldown_closed(self, job_cr_spec):
        view = ServiceView("ns", "svc")
        view.commit(2, now=0, cause="boot-seed")
        readings = {"queue_depth": Reading(12, "ok")}
        t = self._tick(view, job_cr_spec, readings, physical=2, now=10)
        assert t.proposal is None
        assert t.want == 2
        assert t.hold_reason.startswith("hold-cooldown-up")

    def test_hold_when_signal_missing(self, job_cr_spec):
        view = ServiceView("ns", "svc")
        view.commit(2, now=0, cause="boot-seed")
        readings = {"queue_depth": Reading(None, "missing_series")}
        t = self._tick(view, job_cr_spec, readings, physical=2, now=UP_COOLDOWN_S + 1)
        assert t.proposal is None
        assert t.want == 2
        assert t.hold_reason.startswith("hold-missing-signal")
        assert "queue.depth:missing_series" in t.hold_reason

    def test_comfortable_accumulates_comfort_streak(self, job_cr_spec):
        view = ServiceView("ns", "svc")
        view.commit(2, now=0, cause="boot-seed")
        readings = {"queue_depth": Reading(0, "ok")}
        # Tick 1: comfortable arrives
        t1 = self._tick(view, job_cr_spec, readings, physical=2, now=100)
        assert view.comfortable_since == 100
        # Tick 2: still comfortable → streak accumulates
        t2 = self._tick(view, job_cr_spec, readings, physical=2, now=200)
        assert view.comfortable_since == 100

    def test_grey_resets_comfort_and_blocks_shed(self, job_cr_spec):
        """GREY verdicts reset the comfort streak (same as LLM) AND block
        the shed path even when shed_ready=True. Both sides of R3."""
        view = ServiceView("ns", "svc")
        view.commit(3, now=0, cause="boot-seed")
        # depth=9, physical=3 → per=3, in (2.5, 5) → GREY.
        readings = {"queue_depth": Reading(9, "ok")}
        t = self._tick(
            view, job_cr_spec, readings, physical=3,
            now=DOWN_COOLDOWN_S + COMFORT_SUSTAIN_S + 1000,
        )
        assert view.comfortable_since is None
        assert t.proposal is None
        assert t.want == 3

    def test_physical_zero_skips_service_tick(self, job_cr_spec):
        """physical=0 = no-stats tick: skipped, state frozen, not fed to planner."""
        view = ServiceView("ns", "svc")
        view.commit(2, now=0, cause="boot-seed")
        readings = {"queue_depth": Reading(50, "ok")}
        t = self._tick(view, job_cr_spec, readings, physical=0, now=100)
        assert t.skip is True
        assert "hold-no-physical-data" in t.hold_reason
        assert view.committed == 2  # unchanged

    def test_physical_none_skips_service_tick(self, job_cr_spec):
        """physical=None (prom replicas_ready missing) → same skip path."""
        view = ServiceView("ns", "svc")
        view.commit(2, now=0, cause="boot-seed")
        readings = {"queue_depth": Reading(50, "ok")}
        t = self._tick(view, job_cr_spec, readings, physical=None, now=100)
        assert t.skip is True
        assert "hold-no-physical-data" in t.hold_reason
        assert view.committed == 2

    def test_shed_fires_after_full_comfort_streak(self, job_cr_spec):
        view = ServiceView("ns", "svc")
        view.commit(3, now=0, cause="boot-seed")
        # depth=0 → COMFORTABLE. Streak must run for COMFORT_SUSTAIN_S and
        # last_change_at must be ≥ DOWN_COOLDOWN_S.
        readings = {"queue_depth": Reading(0, "ok")}
        # Tick 1: accumulate comfort at t=DOWN_COOLDOWN_S (down cooldown done)
        _ = self._tick(view, job_cr_spec, readings, physical=3, now=DOWN_COOLDOWN_S)
        # Tick 2: after full COMFORT_SUSTAIN_S
        t = self._tick(
            view, job_cr_spec, readings, physical=3,
            now=DOWN_COOLDOWN_S + COMFORT_SUSTAIN_S,
        )
        assert t.proposal is not None
        assert t.proposal.rule == "rjob-shed"
        assert t.want == 2

    def test_idle_after_violated_resets_comfort(self, job_cr_spec):
        view = ServiceView("ns", "svc")
        view.commit(3, now=0, cause="boot-seed")
        # Comfortable tick
        _ = self._tick(view, job_cr_spec, {"queue_depth": Reading(0, "ok")},
                       physical=3, now=100)
        assert view.comfortable_since == 100
        # Violated tick — comfort resets
        _ = self._tick(view, job_cr_spec, {"queue_depth": Reading(50, "ok")},
                       physical=3, now=200)
        assert view.comfortable_since is None

    def test_cr_max_shrink_compresses_immediately(self, job_cr_spec):
        view = ServiceView("ns", "svc")
        view.commit(3, now=0, cause="boot-seed")
        cr = dict(job_cr_spec)
        cr["maximumDeployment"] = {"type": "replica", "value": 2}
        readings = {"queue_depth": Reading(50, "ok")}   # violated, but irrelevant
        t = self._tick(view, cr, readings, physical=3, now=10)
        assert view.committed == 2
        assert t.want == 2

    def test_missing_max_skips_service(self, job_cr_spec):
        cr = dict(job_cr_spec)
        cr.pop("maximumDeployment")
        view = ServiceView("ns", "svc")
        t = self._tick(view, cr, {"queue_depth": Reading(0, "ok")},
                       physical=2, now=100)
        assert t.skip is True
        assert t.hold_reason.startswith("hold-no-max-in-cr")


# ---------- Controller integration ----------

class TestControllerJobIntegration:
    def _setup_k8s(self, fake_k8s, ns, svc, replicas=2, gpr=1):
        pool, cap = "NVIDIA-H100-80GB-HBM3", 8
        fake_k8s.placements[(ns, svc)] = _placement(pool=pool, gpr=gpr,
                                                     spec_replicas=replicas)
        fake_k8s.capacity = {pool: cap}

    def test_job_service_flows_through_tick(
            self, fake_slo, fake_k8s, fake_signals, clock, job_cr_spec):
        """End-to-end: job CR → view created → want computed → snapshot has it."""
        ns, svc = "ns", "svc"
        self._setup_k8s(fake_k8s, ns, svc, replicas=2, gpr=1)
        fake_signals.set_replicas_ready(2)
        fake_signals.set_queue_depth(ns, svc, 0)   # idle → COMFORTABLE
        c = _controller_with_job(fake_slo, fake_k8s, fake_signals, clock, job_cr_spec)
        c.tick()
        snap = c.snapshot()
        rows = {(d["namespace"], d["serviceId"]): d["replicas"]["active"]
                for d in snap["decisions"]}
        assert (ns, svc) in rows
        assert rows[(ns, svc)] == 2              # seeded from spec_replicas

    def test_job_service_upscales_when_violated_and_cooldown_open(
            self, fake_slo, fake_k8s, fake_signals, clock, job_cr_spec):
        ns, svc = "ns", "svc"
        self._setup_k8s(fake_k8s, ns, svc, replicas=2, gpr=1)
        fake_signals.set_replicas_ready(2)
        fake_signals.set_queue_depth(ns, svc, 50)  # deep overshoot
        c = _controller_with_job(fake_slo, fake_k8s, fake_signals, clock, job_cr_spec)
        c.tick()                                   # seed tick: commits 2
        clock.advance(UP_COOLDOWN_S + 1)
        c.tick()                                   # gate open now; fires
        snap = c.snapshot()
        rows = {(d["namespace"], d["serviceId"]): d["replicas"]["active"]
                for d in snap["decisions"]}
        assert rows[(ns, svc)] == 3              # 2 + 1

    def test_job_service_does_not_exceed_cr_max(
            self, fake_slo, fake_k8s, fake_signals, clock, job_cr_spec):
        ns, svc = "ns", "svc"
        self._setup_k8s(fake_k8s, ns, svc, replicas=2, gpr=1)
        cr = dict(job_cr_spec)
        cr["maximumDeployment"] = {"type": "replica", "value": 2}
        fake_signals.set_replicas_ready(2)
        fake_signals.set_queue_depth(ns, svc, 50)
        c = _controller_with_job(fake_slo, fake_k8s, fake_signals, clock, cr)
        c.tick()
        clock.advance(UP_COOLDOWN_S + 1)
        c.tick()
        rows = {(d["namespace"], d["serviceId"]): d["replicas"]["active"]
                for d in c.snapshot()["decisions"]}
        assert rows[(ns, svc)] == 2

    def test_job_signal_missing_holds_at_committed(
            self, fake_slo, fake_k8s, fake_signals, clock, job_cr_spec):
        ns, svc = "ns", "svc"
        self._setup_k8s(fake_k8s, ns, svc, replicas=2, gpr=1)
        fake_signals.set_replicas_ready(2)
        fake_signals.set_queue_depth(ns, svc, 0)   # start healthy
        c = _controller_with_job(fake_slo, fake_k8s, fake_signals, clock, job_cr_spec)
        c.tick()
        clock.advance(UP_COOLDOWN_S + 1)
        # Now signal goes missing — must hold, not shed nor fire
        fake_signals.set_queue_depth(ns, svc, None)   # → missing_series
        c.tick()
        rows = {(d["namespace"], d["serviceId"]): d["replicas"]["active"]
                for d in c.snapshot()["decisions"]}
        assert rows[(ns, svc)] == 2

    def test_job_and_llm_coexist_in_same_tick(
            self, fake_slo, fake_k8s, fake_signals, clock,
            cr_spec, job_cr_spec):
        """One llm service + one job service share a tick and planner gap."""
        ns = "ns"
        pool = "NVIDIA-H100-80GB-HBM3"
        # llm service
        fake_slo.set(ns, "llm-svc", cr_spec)
        fake_k8s.placements[(ns, "llm-svc")] = _placement(pool=pool, gpr=1,
                                                           spec_replicas=2)
        # job service
        from conftest import FakeSLOStore
        fake_slo_job = FakeSLOStore(specs={(ns, "svc"): job_cr_spec},
                                    plural="jobslorequirements")
        fake_k8s.placements[(ns, "svc")] = _placement(pool=pool, gpr=1,
                                                       spec_replicas=2)
        fake_k8s.capacity = {pool: 8}
        fake_signals.set_replicas_ready(2)
        fake_signals.set_queue_depth(ns, "svc", 0)      # job: idle
        # llm signals default to missing_series → hold
        c = Controller(slo=fake_slo, k8s=fake_k8s, signals=fake_signals,
                       clock=clock, slo_job=fake_slo_job, tick_seconds=60)
        c.tick()
        rows = {(d["namespace"], d["serviceId"]): d["replicas"]["active"]
                for d in c.snapshot()["decisions"]}
        assert rows[(ns, "llm-svc")] == 2
        assert rows[(ns, "svc")] == 2

    def test_job_uses_job_replicas_ready_not_llm_signal(
            self, fake_slo, fake_k8s, fake_signals, clock, job_cr_spec):
        """Jobs read physical via kube-state-metrics (job_replicas_ready),
        not the LLM bodylog replicas_ready. Pin the regression: if the
        controller ever re-checks which signal to call based on spec
        shape or reversed the branch, this test fails."""
        ns, svc = "ns", "svc"
        self._setup_k8s(fake_k8s, ns, svc, replicas=2, gpr=1)
        fake_signals.set_replicas_ready(2)        # shared setter; both fakes read it
        fake_signals.set_queue_depth(ns, svc, 0)
        c = _controller_with_job(fake_slo, fake_k8s, fake_signals, clock, job_cr_spec)
        c.tick()
        # Filter calls to the job service only (llm services would
        # legitimately call replicas_ready; we only care about svc here).
        job_calls = [c_ for c_ in fake_signals.calls
                     if c_[0] == "job_replicas_ready" and c_[1] == ns and c_[2] == svc]
        llm_calls = [c_ for c_ in fake_signals.calls
                     if c_[0] == "replicas_ready" and c_[1] == ns and c_[2] == svc]
        assert job_calls, "expected job_replicas_ready to be called for the job service"
        assert not llm_calls, "job service must NOT call LLM replicas_ready"

    def test_job_cr_deleted_frees_ledger_entry(
            self, fake_slo, fake_k8s, fake_signals, clock, job_cr_spec):
        ns, svc = "ns", "svc"
        self._setup_k8s(fake_k8s, ns, svc, replicas=2, gpr=1)
        fake_signals.set_replicas_ready(2)
        fake_signals.set_queue_depth(ns, svc, 0)
        from conftest import FakeSLOStore
        fake_slo_job = FakeSLOStore(specs={(ns, svc): job_cr_spec},
                                    plural="jobslorequirements")
        c = Controller(slo=fake_slo, k8s=fake_k8s, signals=fake_signals,
                       clock=clock, slo_job=fake_slo_job, tick_seconds=60)
        c.tick()
        assert (ns, svc) in c._views
        fake_slo_job.delete(ns, svc)
        clock.advance(1)
        c.tick()
        assert (ns, svc) not in c._views
