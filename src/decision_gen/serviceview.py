"""Per-service state machine — sole owner of §4 state.

The re-architecture's main addition: v1 left these fields scattered
across controller dicts (`_last_change_at`, `_last_served`,
`_comfortable_since`), which is why the tick function was a tangle.

Owns, per (ns, service):
  - committed           int   — direction authority; capacity booking
  - last_change_at      epoch — single clock, two thresholds (D1)
  - comfortable_since   epoch | None — unbroken comfort streak (R4)

Single mutation path: `commit()`. Every commit stamps the clock and
zeroes comfort on delta — the R6 re-arm rule is un-skippable.

`step()` produces a `Transition`: one per-service log record that
carries each stage value (est → want). §8's "reconstruct the tick
from logs" requirement is delivered as a data structure so the log
format can't drift from the decision structure.
"""

import logging
import os
from collections import namedtuple

from decision_gen import direction, triggers
from decision_gen.thresholds import METRIC_DIRECTION, Verdict, classify

log = logging.getLogger(__name__)


def _env_int(name, default):
    return int(os.environ.get(name, default))


def _env_float(name, default):
    return float(os.environ.get(name, default))


UP_COOLDOWN_S     = _env_int("SCALE_UP_COOLDOWN_S",     900)   # 15 min
DOWN_COOLDOWN_S   = _env_int("SCALE_DOWN_COOLDOWN_S",  1800)   # 30 min
COMFORT_SUSTAIN_S = _env_int("SCALE_DOWN_COMFORT_S",   1200)   # 20 min
SLO_HEADROOM      = _env_float("SLO_HEADROOM",          0.5)


Transition = namedtuple("Transition", [
    "ns", "service",
    # placement
    "pool", "gpr", "kind", "spec_replicas",
    # raw evidence
    "readings",            # llm: {'ttft': {kind: Reading}, 'otps': {...}, 'rejection': Reading}
                           # job: {'queue_depth': Reading}
    "verdicts",            # llm: {'ttft': {kind: Verdict|None}, 'otps': {...}}
                           # job: {'queue': {'depth': Verdict|None}}
    "physical",            # prom replicas_ready, int | None
    # gates + streaks at decision time
    "gates",               # triggers.Gates
    "since_change_s",      # int — seconds since last commit
    "comfort_s",           # int | None — seconds of unbroken comfort
    # stages in order: est (post-clamp) → want (post-direction)
    "phase",               # direction.Phase
    "proposal",            # triggers.Proposal | None
    "est",
    "want",
    "hold_reason",         # '' on propose, else e.g. 'hold-cooldown-up 1200s<1800s'
    "skip",                # bool — controller must not touch this service this tick
])


def _bounds(cr_spec):
    """(min, max, priority), or None if maximumDeployment is absent.

    Missing max = user opted out of autoscaling. Caller decides what to
    do with that (skip vs unbounded) and logs any diagnostic — this
    helper stays pure. Malformed min>max clamps to max=min and warns,
    since that's a repairable shape problem, not an opt-out signal."""
    def _i(blk, default):
        try:
            return int((blk or {}).get("value", default))
        except (TypeError, ValueError):
            return default
    if cr_spec.get("maximumDeployment") is None:
        return None
    mn = _i(cr_spec.get("minimumDeployment"), 1)
    mx = _i(cr_spec.get("maximumDeployment"), mn)
    if mn > mx:
        log.warning("CR min>max (%s>%s); clamp max=min", mn, mx)
        mx = mn
    pri = int(cr_spec.get("priority") or 0)
    return mn, mx, pri


def _threshold(cr_spec, signal, kind):
    block = (cr_spec.get(signal) or {}).get("default") or {}
    for m in block.get("metrics") or []:
        if m.get("type") == kind:
            return m.get("threshold")
    return None


class ServiceView:
    __slots__ = ("ns", "service", "committed", "last_change_at", "comfortable_since")

    def __init__(self, ns, service):
        self.ns = ns
        self.service = service
        self.committed = None          # None = unseeded
        self.last_change_at = 0.0
        self.comfortable_since = None

    # ---------- sole mutation path ----------

    def commit(self, replicas, now, cause):
        """Move `committed` and stamp the clock. Idempotent on same value.
        Delta commits zero comfort (R4 reset event 1). Returns True iff
        the committed value changed."""
        if self.committed == replicas:
            return False
        log.info("%s/%s: commit %s→%d (%s)",
                 self.ns, self.service, self.committed, replicas, cause)
        self.committed = replicas
        self.last_change_at = now
        self.comfortable_since = None
        return True

    # ---------- boot (R7) ----------

    def _ensure_seeded(self, cr_spec, placement, now):
        """First sight: seed committed from spec.replicas clamped to
        [min, max]. Never from Prometheus (prom lags intent at boot).
        Seeding IS a change event for the clock."""
        if self.committed is not None:
            return
        mn, mx, _ = _bounds(cr_spec)
        self.commit(max(mn, min(mx, placement.spec_replicas)), now, cause="boot-seed")

    # ---------- main ----------

    def step(self, readings, placement, cr_spec, physical, now, kind="llm"):
        """One tick for this service. Returns a Transition. Does NOT
        commit; the controller commits after the planner arbitrates
        across services.

        readings: llm — {'ttft': {kind: Reading}, 'otps': {...},
                         'rejection': Reading, ...}
                  job — {'queue_depth': Reading}
        physical: int | None — replicas_ready; None or 0 = no-stats, skip.
        CR missing maximumDeployment → skip: user opted out of autoscale.
        kind: "llm" | "job" — which verdict algebra to apply. The
              controller knows this from which watcher produced the CR;
              pass it down rather than sniffing the spec shape here.
        """
        bounds = _bounds(cr_spec)
        if bounds is None:
            return Transition(
                ns=self.ns, service=self.service,
                pool=placement.pool, gpr=placement.gpus_per_replica,
                kind=placement.kind, spec_replicas=placement.spec_replicas,
                readings=readings, verdicts={"ttft": {}, "otps": {}},
                physical=physical,
                gates=triggers.Gates(up_cooldown_open=False, shed_ready=False),
                since_change_s=int(now - self.last_change_at),
                comfort_s=(None if self.comfortable_since is None
                           else int(now - self.comfortable_since)),
                phase=direction.Phase.SETTLED,
                proposal=None,
                est=self.committed if self.committed is not None else 0,
                want=self.committed if self.committed is not None else 0,
                hold_reason="hold-no-max-in-cr (user opted out of autoscale)",
                skip=True,
            )
        # Seed FIRST. spec.replicas is authoritative boot state; physical
        # is only needed for the *decision* phase below. A service whose
        # physical signal is missing must still appear on the wire with
        # its seeded committed, otherwise /decisions reads as "unmanaged"
        # for any service whose replicas_ready exporter is absent —
        # which includes whole classes of workloads that never had the
        # LLM exporter (jobs being the first example).
        self._ensure_seeded(cr_spec, placement, now)
        if physical is None or physical == 0:
            empty_verdicts = {"queue": {}} if kind == "job" else {"ttft": {}, "otps": {}}
            return Transition(
                ns=self.ns, service=self.service,
                pool=placement.pool, gpr=placement.gpus_per_replica,
                kind=placement.kind, spec_replicas=placement.spec_replicas,
                readings=readings, verdicts=empty_verdicts,
                physical=physical,
                gates=triggers.Gates(up_cooldown_open=False, shed_ready=False),
                since_change_s=int(now - self.last_change_at),
                comfort_s=(None if self.comfortable_since is None
                           else int(now - self.comfortable_since)),
                phase=direction.Phase.SETTLED,
                proposal=None,
                est=self.committed if self.committed is not None else 0,
                want=self.committed if self.committed is not None else 0,
                hold_reason=f"hold-no-physical-data physical={physical}",
                skip=True,
            )
        mn, mx, _pri = bounds

        # C4: CR max shrunk below committed → compress immediately.
        # Bypasses cooldowns, freezes, missing-signal holds; CR is
        # authority. Goes through commit() so clock stamps and comfort
        # zeroes.
        if self.committed > mx:
            self.commit(mx, now, cause="cr-max-shrink")

        is_job = (kind == "job")

        # ---- verdicts ----
        # `no_traffic` (histogram_quantile NaN = zero requests in window,
        # or an `or vector(0)` job PromQL returning literal 0) contributes
        # COMFORTABLE, not missing: zero demand IS the deepest possible
        # comfort. `missing_series` (query 0-rows / bad shape / transport
        # error) continues to mean "no evidence; hold".
        if is_job:
            verdicts, missing_slo = self._job_verdicts(
                readings, cr_spec, physical,
            )
            rej = None             # jobs have no rejection signal
            rej_value = None
        else:
            verdicts, missing_slo = self._llm_verdicts(readings, cr_spec)
            rej = readings.get("rejection")
            rej_value = rej.value if rej is not None and rej.state == "ok" else None

        # ---- comfort streak (R4) ----
        # Reset events: commit (inside commit()), any MISSING-series signal,
        # any verdict leaving COMFORTABLE, any violation. `no_traffic` is
        # NOT a reset — it feeds COMFORTABLE above, so an idle service's
        # comfort streak accumulates and it sheds after the down-cooldown.
        flat = [v for sig in verdicts.values() for v in sig.values()]
        any_missing = missing_slo or (rej is not None and rej.state != "ok")
        all_comfy = (not any_missing
                     and bool(flat)
                     and all(v is Verdict.COMFORTABLE for v in flat))
        if all_comfy:
            if self.comfortable_since is None:
                self.comfortable_since = now
        else:
            self.comfortable_since = None

        # ---- gates (D1: one clock, two thresholds) ----
        since_change = now - self.last_change_at
        comfort_s = None if self.comfortable_since is None else now - self.comfortable_since
        gates = triggers.Gates(
            up_cooldown_open=since_change >= UP_COOLDOWN_S,
            shed_ready=(since_change >= DOWN_COOLDOWN_S
                        and comfort_s is not None
                        and comfort_s >= COMFORT_SUSTAIN_S),
        )

        # ---- magnitude (R5 ledger vs prom; §5 fallback) ----
        current = physical if physical is not None else self.committed
        ref = current
        phase = direction.phase_of(self.committed, ref)

        # ---- SLO-missing hold (§5 row 1) ----
        if missing_slo:
            return self._transition(
                readings, verdicts, placement, physical, now,
                gates, phase, proposal=None,
                est=self.committed, want=self.committed,
                hold_reason=f"hold-missing-signal {','.join(missing_slo)}",
            )

        # ---- triggers (pure) ----
        flat_verdicts = {
            f"{sig}.{kind}": v
            for sig, by_kind in verdicts.items()
            for kind, v in by_kind.items()
        }
        if is_job:
            proposal = triggers.evaluate_job(flat_verdicts, current, gates)
        else:
            # Evidence floors for R1a/R1b come in as sample counts.
            rej_count_reading = readings.get("rejection_count")
            req_count_reading = readings.get("request_count")
            rej_count = (rej_count_reading.value
                         if rej_count_reading is not None and rej_count_reading.state == "ok"
                         else None)
            req_count = (req_count_reading.value
                         if req_count_reading is not None and req_count_reading.state == "ok"
                         else None)
            proposal = triggers.evaluate(
                flat_verdicts, rej_value, current, gates,
                rejection_count=rej_count, request_count=req_count,
            )

        # ---- est stage: post-clamp (R9) ----
        if proposal is None:
            est = self.committed
            hold_reason = _hold_reason(
                flat_verdicts, gates, since_change, comfort_s,
            )
        else:
            est = max(mn, min(mx, proposal.replicas))
            hold_reason = ""

        # ---- want stage: post-direction (R5 symmetric, I4) ----
        want, d_reason = direction.reconcile(est, self.committed, phase)
        if d_reason:
            hold_reason = d_reason

        return self._transition(
            readings, verdicts, placement, physical, now,
            gates, phase, proposal,
            est=est, want=want, hold_reason=hold_reason,
        )

    # ---------- internals ----------

    @staticmethod
    def _llm_verdicts(readings, cr_spec):
        """Classify TTFT/OTPS readings against CR thresholds.

        Returns (verdicts, missing_slo). Verdicts is
        {"ttft": {kind: Verdict|None}, "otps": {kind: Verdict|None}}.
        """
        verdicts = {"ttft": {}, "otps": {}}
        missing_slo = []
        for signal in ("ttft", "otps"):
            for kind, r in (readings.get(signal) or {}).items():
                if r.state == "ok":
                    thr = _threshold(cr_spec, signal, kind)
                    verdicts[signal][kind] = classify(
                        METRIC_DIRECTION[signal], r.value, thr, SLO_HEADROOM,
                    )
                elif r.state == "no_traffic":
                    verdicts[signal][kind] = Verdict.COMFORTABLE
                else:   # missing_series / unknown
                    missing_slo.append(f"{signal}.{kind}:{r.state}")
                    verdicts[signal][kind] = None
        return verdicts, missing_slo

    @staticmethod
    def _job_verdicts(readings, cr_spec, physical):
        """Classify queue depth (total ÷ live replicas) against maxDepth.

        The CR's `queue.promql` returns the FLEET-TOTAL depth; `maxDepth`
        is per-replica, so divide here. physical>0 is guaranteed by the
        caller's no-stats guard; max() is belt-and-braces against races.

        Returns (verdicts, missing_slo). Verdicts is
        {"queue": {"depth": Verdict|None}} — the same shape the LLM path
        emits so downstream (comfort streak, flat_verdicts, hold_reason)
        behaves identically for both kinds.
        """
        verdicts = {"queue": {}}
        missing_slo = []
        r = readings.get("queue_depth")
        max_depth = int((cr_spec.get("queue") or {}).get("maxDepth") or 0)
        if r is None or max_depth <= 0:
            missing_slo.append("queue.depth:unconfigured")
            verdicts["queue"]["depth"] = None
        elif r.state == "ok":
            per_replica = r.value / max(physical, 1)
            verdicts["queue"]["depth"] = classify(
                METRIC_DIRECTION["queue"], per_replica, max_depth, SLO_HEADROOM,
            )
        elif r.state == "no_traffic":
            # Queue empty (hist NaN) — deepest possible comfort. An
            # `or vector(0)` PromQL reads as ok(0) above and lands
            # COMFORTABLE through classify; the NaN form arrives here.
            verdicts["queue"]["depth"] = Verdict.COMFORTABLE
        else:
            missing_slo.append(f"queue.depth:{r.state}")
            verdicts["queue"]["depth"] = None
        return verdicts, missing_slo

    def _transition(self, readings, verdicts, placement, physical, now,
                    gates, phase, proposal, est, want, hold_reason):
        since_change = now - self.last_change_at
        comfort_s = None if self.comfortable_since is None else now - self.comfortable_since
        return Transition(
            ns=self.ns, service=self.service,
            pool=placement.pool, gpr=placement.gpus_per_replica,
            kind=placement.kind, spec_replicas=placement.spec_replicas,
            readings=readings, verdicts=verdicts,
            physical=physical,
            gates=gates,
            since_change_s=int(since_change),
            comfort_s=None if comfort_s is None else int(comfort_s),
            phase=phase, proposal=proposal,
            est=est, want=want, hold_reason=hold_reason,
            skip=False,
        )


def _hold_reason(flat_verdicts, gates, since_change_s, comfort_s):
    """§8.5: hold reasons name the blocking gate; comfort blocks name
    which leg failed."""
    any_violated = any(v is Verdict.VIOLATED for v in flat_verdicts.values())
    all_comfy = bool(flat_verdicts) and all(
        v is Verdict.COMFORTABLE for v in flat_verdicts.values()
    )
    if any_violated and not gates.up_cooldown_open:
        return f"hold-cooldown-up {int(since_change_s)}s<{UP_COOLDOWN_S}s"
    if all_comfy and not gates.shed_ready:
        if since_change_s < DOWN_COOLDOWN_S:
            return f"hold-cooldown-down {int(since_change_s)}s<{DOWN_COOLDOWN_S}s"
        return f"hold-comfort {int(comfort_s or 0)}s<{COMFORT_SUSTAIN_S}s"
    return "hold-no-rule"
