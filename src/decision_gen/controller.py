"""Tick driver. The only writer of the served snapshot.

A tick runs strictly top-down (arch §3.3, boundary rule 5):

    est → want → alloc → committed

No stage reads a later stage's value. The names above appear verbatim
in `Transition` records so §8's "reconstruct the tick from logs" is
a property of the data, not of a formatter that could drift.

The controller contains NO rule logic — every decision is delegated
to `ServiceView` (per-service) or `planner` (cross-service). An `if`
about thresholds, cooldowns, or direction in this file is a boundary
bug by definition.

Failure posture (arch §5):
  - slo snapshot:        free ledger entries for deleted CRs before
                         gap computation (C7).
  - pool capacity read:  exception → abort tick; keep previous snapshot.
  - placement unreadable for S: skip S that tick (no decision row).
  - SLO signal missing:  serviceview holds S at committed (per S, not
                         global).
  - replicas_ready missing: serviceview falls back to committed for
                         magnitude.
  - prom globally down:  every signal returns missing_series; every
                         service holds; previous snapshot stays live.
"""

import logging
import threading
import time

from decision_gen import planner, snapshot as wire
from decision_gen.serviceview import (
    ServiceView, _bounds, _bounds_kind,
)
from decision_gen.thresholds import Verdict

log = logging.getLogger(__name__)


def _kinds_needed(cr_spec, signal):
    """Unique list of declared metric kinds for one signal."""
    block = (cr_spec.get(signal) or {}).get("default") or {}
    kinds = []
    for m in block.get("metrics") or []:
        t = m.get("type")
        if t and t not in kinds:
            kinds.append(t)
    return kinds


class Controller:
    def __init__(self, slo, k8s, signals, tick_seconds=60, clock=time.time,
                 slo_job=None):
        self.slo = slo
        self.slo_job = slo_job  # JobSLO watcher; None in tests that only exercise LLM
        self.k8s = k8s
        self.signals = signals
        self.tick_seconds = tick_seconds
        self.clock = clock      # injectable for tests

        self._lock = threading.Lock()
        self._snapshot = wire.to_wire({})
        self._views = {}        # (ns, svc) → ServiceView
        self._stop = threading.Event()
        self._has_ticked = threading.Event()    # set after first tick attempt
        self._thread = None

    # ---------- lifecycle ----------

    def start(self):
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="controller",
        )
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def snapshot(self):
        with self._lock:
            return {
                "apiVersion": self._snapshot["apiVersion"],
                "decisions": list(self._snapshot["decisions"]),
            }

    def ready(self):
        """True only after the controller has attempted at least one tick.

        Between process start and the first tick there's nothing meaningful
        to serve — the snapshot is the empty placeholder. Serving that as a
        real answer is the bug 04:49 <log line> exposed: an early consumer
        sees decisions=[] and concludes no services are managed."""
        return self._has_ticked.is_set()

    # ---------- loop ----------

    def _loop(self):
        # First tick must not run off an empty slo cache. Wait (bounded) for
        # the watch to report contact with the API server; on timeout tick
        # anyway — same fail-open posture as "serve previous snapshot" errors.
        t = time.monotonic()
        if self.slo.wait_synced(timeout=10):
            log.info("controller: slo synced in %.3fs; first tick",
                     time.monotonic() - t)
        else:
            log.warning("controller: slo not synced after 10s; first tick anyway")
        # Job store: same posture but tolerate-404 — jobslo CRD may not
        # be installed on every cluster. A missing CRD leaves the watcher
        # in retry-with-backoff; wait_synced times out and we tick with
        # an empty job set.
        if self.slo_job is not None:
            if self.slo_job.wait_synced(timeout=5):
                log.info("controller: job-slo synced; first tick")
            else:
                log.warning(
                    "controller: job-slo not synced after 5s "
                    "(jobslo CRD may be absent); continuing with zero job CRs",
                )
        while not self._stop.is_set():
            started = time.time()
            try:
                self.tick()
            except Exception:
                log.exception("tick failed; serving previous snapshot")
            elapsed = time.time() - started
            self._stop.wait(max(0.0, self.tick_seconds - elapsed))

    # ---------- tick ----------

    def tick(self):
        t0 = time.monotonic()
        now = self.clock()
        # Set on attempt as well as success: "serving previous snapshot"
        # posture is identical whether the previous was real data or the
        # boot placeholder, so readiness shouldn't distinguish them either.
        self._has_ticked.set()
        crs = self.slo.snapshot()
        log.info("tick start: slo.snapshot %+0.3fs (%d CRs)",
                 time.monotonic() - t0, len(crs))

        # 1. Free ledger entries for deleted CRs BEFORE gap computation.
        #    Job SLOs arrive via a separate watcher; a key is live iff
        #    it appears in either snapshot.
        crs_job = self.slo_job.snapshot() if self.slo_job is not None else {}
        for key in list(self._views):
            if key in crs or key in crs_job:
                continue
            del self._views[key]
            log.info("%s/%s: CR deleted; freed ledger entry", *key)

        # Merge both kinds into one iteration. Placement, bounds,
        # planner input, and the transition format are shared; only the
        # readings dict shape differs (view.step() dispatches on the
        # CR spec's shape, not on a kind tag).
        merged = [(k, s, "llm") for k, s in sorted(crs.items())]
        merged += [(k, s, "job") for k, s in sorted(crs_job.items())]

        if not merged:
            with self._lock:
                self._snapshot = wire.to_wire({})
            return

        # 2. Pool capacity. Failure aborts the whole tick.
        t = time.monotonic()
        capacity = self.k8s.pool_capacity()
        log.info("pool capacity: %s (%+0.3fs into tick)",
                 capacity, time.monotonic() - t0)

        # 3. Per-service step → est / want.
        transitions = []
        booked = {}        # {(ns,svc): committed} — planner's baseline
        wants = {}         # {(ns,svc): want}
        placements = {}    # {(ns,svc): (pool, gpr)}
        bounds = {}        # {(ns,svc): {'min','max','priority'}}
        views = {}
        pd_placements = {} # {(ns,svc): Placement} — pdengine only
        pd_bookings = {}   # {(ns,svc): PdShape} — pdengine only

        for (ns, svc), spec, cr_kind in merged:
            key = (ns, svc)
            log.info("─── %s/%s (%s) %s", ns, svc, cr_kind,
                     "─" * max(0, 56 - len(ns) - len(svc) - len(cr_kind)))
            # Missing maximumDeployment is an opt-out signal from the CR
            # author: skip the service entirely, don't even fetch signals.
            if _bounds(spec) is None:
                log.warning(
                    "%s/%s: CR missing maximumDeployment — unmanaged "
                    "(not autoscaled); skipping every tick until set",
                    ns, svc,
                )
                continue

            placement = self.k8s.resolve_placement(ns, svc)
            if placement is None:
                log.warning("%s/%s: placement unresolvable; skipping tick", ns, svc)
                continue

            if cr_kind == "job":
                physical = self.signals.job_replicas_ready(ns, svc)
                physical_val = physical.value if physical.state == "ok" else None
                promql = (spec.get("queue") or {}).get("promql")
                if not promql:
                    log.warning("%s/%s: job CR missing queue.promql; skipping", ns, svc)
                    continue
                readings = {
                    "queue_depth": self.signals.queue_depth(ns, svc, promql),
                }
            else:
                physical = self.signals.replicas_ready(ns, svc)
                physical_val = physical.value if physical.state == "ok" else None
                readings = {
                    "ttft": {k: self.signals.ttft(ns, svc, k)
                             for k in _kinds_needed(spec, "ttft")},
                    "otps": {k: self.signals.otps(ns, svc, k)
                             for k in _kinds_needed(spec, "otps")},
                    "rejection": self.signals.rejection_rate(ns, svc),
                    "rejection_count": self.signals.rejection_count_2m(ns, svc),
                    "request_count": self.signals.request_count_5m(ns, svc),
                }
                if _bounds_kind(spec) == "pdengine":
                    for role in ("prefill", "decode"):
                        readings[f"replicas_ready_{role}"] = (
                            self.signals.replicas_ready_role(ns, svc, role)
                        )

            view = self._views.get(key)
            if view is None:
                view = ServiceView(ns, svc)
                self._views[key] = view
            # pdengine services need placement.pd available even on
            # transient skip ticks so the wire can keep serving the
            # last committed shape; it only changes via the success
            # path but the read happens before `continue` below.
            # Gate on the CR kind — a replica-typed CR on a DS
            # workload must NOT populate pd_placements (to_wire would
            # then try to render an int as a PdShape).
            if (_bounds_kind(spec) == "pdengine"
                    and placement.pd is not None):
                pd_placements[key] = placement

            t = view.step(readings, placement, spec, physical_val, now,
                          kind=cr_kind)
            transitions.append(t)
            pd_kind = _bounds_kind(spec) == "pdengine"
            if t.skip:
                # Transient skip → serve the last committed shape on the
                # wire, same posture as the replica path. Structural
                # skips (non-ds, slices != 1, inexpressible bounds)
                # drop the view entirely so /decisions goes quiet until
                # the configuration heals.
                if pd_kind and t.pd_shape is None:
                    self._drop_pd_view(key)
                continue

            if pd_kind:
                # pdengine services bypass the cross-service planner:
                # engines are exchanged inside the sandwich, bundles
                # are not. The want is already bounds-valid — commit
                # goes through the single mutation path. Book the
                # committed GPU spend against the pool ledger *after*
                # the replica loop below, so the planner never sees a
                # PdShape in its inputs.
                view.commit(t.pd_shape, now, cause="pd-engine")
                views[key] = view
                pd_bookings[key] = view.committed
                continue

            mn, mx, pri = _bounds(spec)
            views[key] = view
            wants[key] = t.want
            booked[key] = view.committed
            placements[key] = (placement.pool, placement.gpus_per_replica)
            bounds[key] = {"min": mn, "max": mx, "priority": pri}

        # 3b. Warn loudly when Σ min × per-unit > capacity for any pool —
        # mins are CR authority and we will honor them regardless, but
        # the scheduler cannot place what doesn't fit; pods will go
        # Pending. Replica services pay `min × gpr-per-replica`. PD
        # services pay committed engines × per-group GPUs — engines are
        # the unit on the CR and bundle_gpus is what one slice costs.
        min_needed = {}
        for key, b in bounds.items():
            pool, gpr = placements[key]
            min_needed[pool] = min_needed.get(pool, 0) + b["min"] * gpr
        for key, n in pd_bookings.items():
            pool, pd_spec = (pd_placements[key].pool,
                             pd_placements[key].pd)
            cost = (n.prefill * pd_spec.role_group_gpus["prefill"]
                    + n.decode * pd_spec.role_group_gpus["decode"])
            min_needed[pool] = min_needed.get(pool, 0) + cost
        for pool, need in min_needed.items():
            have = capacity.get(pool, 0)
            if need > have:
                log.warning(
                    "pool %s UNDER-PROVISIONED: Σ CR min × gpr = %d GPUs "
                    "but pool capacity = %d (short %d). Honoring mins "
                    "anyway; kube-scheduler will mark excess pods Pending.",
                    pool, need, have, need - have,
                )

        # 4. Ledger gap: Σ ready GPUs − Σ committed × per-unit cost.
        # Replica bookings go through the planner's bookkeeping; pd
        # bookings land here directly.
        gap = dict(capacity)
        for key, n in booked.items():
            pool, gpr = placements[key]
            gap[pool] = gap.get(pool, 0) - n * gpr
        for key, n in pd_bookings.items():
            pool, pd_spec = (pd_placements[key].pool,
                             pd_placements[key].pd)
            cost = (n.prefill * pd_spec.role_group_gpus["prefill"]
                    + n.decode * pd_spec.role_group_gpus["decode"])
            gap[pool] = gap.get(pool, 0) - cost
        log.info("pool gap after booking: %s", gap)

        # 5. Planner arbiters capacity. Output is final (R8).
        alloc = planner.resolve(
            wants=wants, current=booked,
            placement=placements, cr=bounds, gap=gap,
        )

        # 6. Commit through the only mutation path; emit log rows;
        #    swap the served snapshot atomically.
        for key, n in sorted(alloc.items()):
            views[key].commit(n, now, cause="planner")

        for t in transitions:
            log.info("tick %s/%s %s", t.ns, t.service, _fmt_transition(t))

        # Serve every tracked view, including ones skipped this tick —
        # a temporarily-no-physical service must not vanish from the wire
        # (the executor would read absence differently from "same value").
        # pdengine views that skipped this tick have already been
        # dropped from self._views by _drop_pd_view above.
        committed = {
            key: v.committed
            for key, v in self._views.items()
            if v.committed is not None      # never-seeded: nothing to serve
        }
        with self._lock:
            self._snapshot = wire.to_wire(committed, pd_placements)
        log.info(
            "tick: %s",
            "; ".join(f"{k[0]}/{k[1]}={_fmt_committed(n)}"
                      for k, n in sorted(committed.items())),
        )

    def _drop_pd_view(self, key):
        """Drop a pdengine view after a skip/hold. The next tick re-seeds
        the view from the live DS spec, so the wire never serves a
        bound-breaking or stale shape."""
        view = self._views.pop(key, None)
        if view is not None and view.committed is not None:
            log.info("%s/%s: pdengine skip; dropping committed %s",
                     key[0], key[1], view.committed)


def _fmt_committed(n):
    """int for replica services; PdShape for pdengine. Log-friendly
    one-liner so the `tick: ns/svc=...` line doesn't blow up on the
    object repr."""
    if hasattr(n, "prefill") and hasattr(n, "decode"):
        return f"{n.prefill}p{n.decode}d"
    return str(n)


def _fmt_transition(t):
    """§8.3 / §8.4: stage transitions by name, verdicts with comparisons
    and threshold, hold reasons by gate. One function, one format — the
    log cannot drift from the Transition data structure."""
    parts = [
        f"pool={t.pool} gpr={t.gpr} kind={t.kind} spec_replicas={t.spec_replicas}",
    ]
    for signal in ("ttft", "otps"):
        for kind, r in (t.readings.get(signal) or {}).items():
            tag = f"{signal}.{kind}"
            if r.state != "ok":
                parts.append(f"[{tag}]={r.state}")
                continue
            v = t.verdicts.get(signal, {}).get(kind)
            sym = {Verdict.VIOLATED: "✗", Verdict.COMFORTABLE: "✓"}.get(v, "~")
            parts.append(f"[{tag}]={r.value:g}{sym}")
    # Jobs carry depth under the flat "queue_depth" key instead of a
    # per-kind nested dict. Print it explicitly with its verdict.
    qd = t.readings.get("queue_depth")
    if qd is not None:
        v = t.verdicts.get("queue", {}).get("depth")
        sym = {Verdict.VIOLATED: "✗", Verdict.COMFORTABLE: "✓"}.get(v, "~")
        if qd.state == "ok":
            parts.append(f"[queue.depth]={qd.value:g}{sym}")
        else:
            parts.append(f"[queue.depth]={qd.state}")
    rej = t.readings.get("rejection")
    if rej is not None:
        parts.append(f"[rej]={rej.value:g}" if rej.state == "ok" else f"[rej]={rej.state}")
    if t.physical is None:
        parts.append("physical=?")
    else:
        parts.append(f"physical={t.physical}")
    parts.append(f"since={t.since_change_s}s")
    parts.append(f"comfort={t.comfort_s}s" if t.comfort_s is not None else "comfort=–")
    parts.append(f"gates(up={t.gates.up_cooldown_open} shed={t.gates.shed_ready})")
    parts.append(f"phase={t.phase.value}")
    if t.proposal is not None:
        parts.append(f"proposal={t.proposal.replicas}({t.proposal.rule} {t.proposal.reason})")
    parts.append(f"est={t.est}")
    parts.append(f"want={t.want}")
    if t.hold_reason:
        parts.append(f"({t.hold_reason})")
    return " ".join(parts)
