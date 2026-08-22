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
from decision_gen.serviceview import ServiceView, _bounds
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
    def __init__(self, slo, k8s, signals, tick_seconds=60, clock=time.time):
        self.slo = slo
        self.k8s = k8s
        self.signals = signals
        self.tick_seconds = tick_seconds
        self.clock = clock      # injectable for tests

        self._lock = threading.Lock()
        self._snapshot = wire.to_wire({})
        self._views = {}        # (ns, svc) → ServiceView
        self._stop = threading.Event()
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

    # ---------- loop ----------

    def _loop(self):
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
        now = self.clock()
        crs = self.slo.snapshot()

        # 1. Free ledger entries for deleted CRs BEFORE gap computation.
        for key in list(self._views):
            if key not in crs:
                del self._views[key]
                log.info("%s/%s: CR deleted; freed ledger entry", *key)

        if not crs:
            with self._lock:
                self._snapshot = wire.to_wire({})
            return

        # 2. Pool capacity. Failure aborts the whole tick.
        capacity = self.k8s.pool_capacity()
        log.debug("pool capacity: %s", capacity)

        # 3. Per-service step → est / want.
        transitions = []
        booked = {}        # {(ns,svc): committed} — planner's baseline
        wants = {}         # {(ns,svc): want}
        placements = {}    # {(ns,svc): (pool, gpr)}
        bounds = {}        # {(ns,svc): {'min','max','priority'}}
        views = {}

        for (ns, svc), spec in sorted(crs.items()):
            key = (ns, svc)
            placement = self.k8s.resolve_placement(ns, svc)
            if placement is None:
                log.warning("%s/%s: placement unresolvable; skipping tick", ns, svc)
                continue

            physical = self.signals.replicas_ready(ns, svc)
            physical_val = physical.value if physical.state == "ok" else None

            readings = {
                "ttft": {k: self.signals.ttft(ns, svc, k)
                         for k in _kinds_needed(spec, "ttft")},
                "otps": {k: self.signals.otps(ns, svc, k)
                         for k in _kinds_needed(spec, "otps")},
                "rejection": self.signals.rejection_rate(ns, svc),
            }

            view = self._views.get(key)
            if view is None:
                view = ServiceView(ns, svc)
                self._views[key] = view

            t = view.step(readings, placement, spec, physical_val, now)
            transitions.append(t)

            mn, mx, pri = _bounds(spec)
            views[key] = view
            wants[key] = t.want
            booked[key] = view.committed
            placements[key] = (placement.pool, placement.gpus_per_replica)
            bounds[key] = {"min": mn, "max": mx, "priority": pri}

        # 4. Ledger gap: Σ ready GPUs − Σ committed × gpr.
        gap = dict(capacity)
        for key, n in booked.items():
            pool, gpr = placements[key]
            gap[pool] = gap.get(pool, 0) - n * gpr
        log.debug("pool gap after booking: %s", gap)

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
            log.debug("tick %s/%s %s", t.ns, t.service, _fmt_transition(t))

        committed = {key: v.committed for key, v in views.items()}
        with self._lock:
            self._snapshot = wire.to_wire(committed)
        log.info(
            "tick: %s",
            "; ".join(f"{k[0]}/{k[1]}={n}" for k, n in sorted(committed.items())),
        )


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
