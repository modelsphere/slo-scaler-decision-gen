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
  - CR bounds (R9):      never served outside. Every live view is
                         clamped before anything can skip or abort, and
                         publish omits any row still outside.
  - pool capacity read:  exception → abort tick; republish the ledger
                         (previous values, clamped to current bounds).
  - placement unreadable for S: skip S that tick; S keeps its last
                         committed row, clamped to current bounds.
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

from decision_gen import pd, planner, snapshot as wire
from decision_gen.serviceview import (
    ServiceView, _bounds, _bounds_kind, _fmt_committed, within_bounds,
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
        self._pd_specs = {}     # (ns, svc) → PDSpec — last resolved
                                # for a pdengine CR; survives ticks
                                # whose placement read fails.
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
            self._pd_specs.pop(key, None)
            log.info("%s/%s: CR deleted; freed ledger entry", *key)

        # R9: pull every live view into its CR's current bounds before
        #    anything below can skip it or abort the tick. The publish
        #    in `finally` then serves those values even when the tick
        #    fails, instead of a snapshot the CRs no longer allow.
        #    Job CRs win a key collision, as in the step order below.
        specs = {**crs, **crs_job}
        for key, view in self._views.items():
            view.enforce_bounds(specs[key], now)

        try:
            self._decide(now, crs, crs_job, t0)
        finally:
            self._publish(specs)

    def _decide(self, now, crs, crs_job, t0):
        """Steps 2-6 of the tick: per-service step, planner, commit."""
        # Merge both kinds into one iteration. Placement, bounds,
        # planner input, and the transition format are shared; only the
        # readings dict shape differs (view.step() dispatches on the
        # CR spec's shape, not on a kind tag).
        merged = [(k, s, "llm") for k, s in sorted(crs.items())]
        merged += [(k, s, "job") for k, s in sorted(crs_job.items())]

        if not merged:
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
        pd_wants = {}      # {(ns,svc): (PdShape, Placement)} — growth
                           # candidates; the placement supplies pool +
                           # role_group_gpus for the leftover math

        # Request-log freshness: one global reading, fed to every LLM
        # service as an input signal (job services don't read the log).
        log_stale = self.signals.log_staleness() if crs else None

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
                # Opt-out CR on a service we were managing: drop the
                # ledger row, and clean up any cached pdengine spec so
                # the wire doesn't keep serving a stale shape.
                self._views.pop(key, None)
                self._pd_specs.pop(key, None)
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
                    "log": log_stale,
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
            # Persist the pd spec for wire rendering whenever the
            # placement resolves — a bodylog lull or a later failed
            # resolve must not put nulls (or worse, crash) in shape.pd.
            pd_kind = _bounds_kind(spec) == "pdengine"
            if pd_kind and placement.pd is not None:
                self._pd_specs[key] = placement.pd

            t = view.step(readings, placement, spec, physical_val, now,
                          kind=cr_kind)
            transitions.append(t)
            if t.skip:
                # Transient skip → serve the last committed shape on the
                # wire, same posture as the replica path. Structural
                # skips (non-ds, slices != 1, inexpressible bounds)
                # drop the view entirely so /decisions goes quiet until
                # the configuration heals.
                if pd_kind and t.pd_shape is None:
                    self._drop_pd_view(key)
                continue

            mn, mx, pri = _bounds(spec)
            views[key] = view

            if pd_kind:
                # pdengine services bypass the cross-service planner:
                # the replica path owns priority preemption today;
                # pdengine growth rides on whatever the replicas
                # leave behind (see step 6 below). Sheds always go
                # through. The want is already bounds- and
                # direction-valid by `_step_pd`; pool-ledger accounting
                # references this service's *committed* shape, not its
                # want.
                pd_wants[key] = (t.pd_shape, placement)
                continue

            wants[key] = t.want
            booked[key] = view.committed
            placements[key] = (placement.pool, placement.gpus_per_replica)
            bounds[key] = {"min": mn, "max": mx, "priority": pri}

        # 3b. Warn loudly when Σ min × per-unit > capacity for any pool —
        # mins are CR authority and we will honor them regardless, but
        # the scheduler cannot place what doesn't fit; pods will go
        # Pending. Replica services book `min × gpr-per-replica`;
        # pdengine services book their *committed* shape — the planner
        # doesn't grow them, so the floor is what they're holding.
        min_needed = {}
        for key, b in bounds.items():
            pool, gpr = placements[key]
            min_needed[pool] = min_needed.get(pool, 0) + b["min"] * gpr
        for pool, need in min_needed.items():
            have = capacity.get(pool, 0)
            if need > have:
                log.warning(
                    "pool %s UNDER-PROVISIONED: Σ per-service floor "
                    "× unit cost = %d GPUs but pool capacity = %d "
                    "(short %d). Honoring mins anyway; kube-scheduler "
                    "will mark excess pods Pending.",
                    pool, need, have, need - have,
                )

        # 4. Ledger gap: Σ ready GPUs − Σ committed × per-unit cost.
        # Replica bookings go through the planner; pd bookings land here.
        gap = dict(capacity)
        for key, n in booked.items():
            pool, gpr = placements[key]
            gap[pool] = gap.get(pool, 0) - n * gpr
        for key, (_want, placed) in pd_wants.items():
            view = views[key]
            gap[placed.pool] = gap.get(placed.pool, 0) - pd.shape_gpu_cost(
                view.committed, placed.pd.role_group_gpus)
        log.info("pool gap after booking: %s", gap)

        # 5. Planner arbiters capacity. Replica/job only — pdengine
        # wants don't enter. Output is final (R8).
        alloc = planner.resolve(
            wants=wants, current=booked,
            placement=placements, cr=bounds, gap=gap,
        )

        # 6. Commit through the only mutation path. Replicas first;
        # pdengine growth is granted out of the pool's post-alloc
        # leftover, never contested against a replica want. Sheds
        # (want.engines < committed.engines) are always allowed —
        # the leftover check only guards growth.
        leftover = dict(gap)
        for key, n in alloc.items():
            pool, gpr = placements[key]
            booked_n = booked.get(key, 0)
            growth = n - booked_n
            if growth > 0:
                leftover[pool] = leftover.get(pool, 0) - growth * gpr
        # Transitions were built before the ledger ran; annotate the
        # outcome so `tick ns/svc ... want=...` always agrees with the
        # committed value that actually lands.
        transitions_by_key = {(t.ns, t.service): t for t in transitions}
        for key, (want, placed) in pd_wants.items():
            view = views[key]
            t = transitions_by_key.get(key)
            delta = pd.PdShape(
                max(0, want.prefill - view.committed.prefill),
                max(0, want.decode - view.committed.decode),
                0,
            )
            need = pd.shape_gpu_cost(delta, placed.pd.role_group_gpus)
            free = leftover.get(placed.pool, 0)
            if need <= free:
                view.commit(want, now, cause="pd-engine")
                leftover[placed.pool] = free - need
            else:
                hold = (f"hold-pd-no-capacity need={need} "
                        f"free={free}")
                log.info("%s/%s: pd-engine %s; keeping %s",
                         key[0], key[1], hold,
                         _fmt_committed(view.committed))
                if t is not None:
                    transitions[transitions.index(t)] = t._replace(
                        want=view.committed, pd_shape=view.committed,
                        hold_reason=hold)
        for key, n in sorted(alloc.items()):
            views[key].commit(n, now, cause="planner")

        for t in transitions:
            log.info("tick %s/%s %s", t.ns, t.service, _fmt_transition(t))

    def _publish(self, specs):
        """Rebuild the served snapshot from the ledger. Runs at the end
        of every tick, including one that raised."""
        # Serve every tracked view, including ones skipped this tick —
        # a temporarily-no-physical service must not vanish from the wire
        # (the executor would read absence differently from "same value").
        # pdengine views that skipped this tick have already been
        # dropped from self._views by _drop_pd_view above. A pdengine
        # row whose DS is unreadable renders against the PDSpec the
        # controller last resolved, not nulls.
        committed = {}
        for key, v in self._views.items():
            if v.committed is None:         # never-seeded: nothing to serve
                continue
            # R9 backstop: a row outside the CR's bounds is omitted,
            # never served. The paths above keep committed in bounds;
            # this is where a missed one becomes visible.
            if not within_bounds(v.committed, specs.get(key),
                                 self._pd_specs.get(key)):
                log.error("%s/%s: committed %s not within the CR's "
                          "bounds (or CR has none); omitting from /decisions",
                          key[0], key[1], _fmt_committed(v.committed))
                continue
            committed[key] = v.committed
        with self._lock:
            self._snapshot = wire.to_wire(committed, self._pd_specs)
        log.info(
            "tick: %s",
            "; ".join(f"{k[0]}/{k[1]}={_fmt_committed(n)}"
                      for k, n in sorted(committed.items())),
        )

    def _drop_pd_view(self, key):
        """Drop a pdengine view after a structural-skip transition.

        Structural skips carry `pd_shape=None` (non-ds, slices != 1,
        inexpressible bounds); the next tick re-seeds the view from the
        live DS spec, so the wire never serves a bound-breaking or
        stale shape. Transient skips (`physical` down) leave the view
        alone."""
        view = self._views.pop(key, None)
        self._pd_specs.pop(key, None)
        if view is not None and view.committed is not None:
            log.info("%s/%s: pdengine structural skip; dropping committed %s",
                     key[0], key[1], _fmt_committed(view.committed))


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
