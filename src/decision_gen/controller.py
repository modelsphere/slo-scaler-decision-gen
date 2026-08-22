"""60s ticker. Per tick: metrics → estimator → planner → snapshot swap.

The only writer to the decision snapshot. HTTP handlers read via
`controller.snapshot()` from another thread; reads take a short lock.

Failure semantics (arch §5.1):
  - Prometheus error on a *required* signal for service S → hold S at
    last-served value this tick. Other services unaffected.
  - Prometheus missing current_replicas for S → the estimator input
    falls back to the ledger value for S; the service still runs.
  - K8s resolve_placement failure for S → skip S this tick.
  - K8s pool_capacity failure → propagate to tick-level except; last
    snapshot stays live.

Capacity model: ledger accounting. Per pool,
  gap[pool] = Σ node allocatable  −  Σ last_served[svc] × gpr[svc]
over services in that pool. We never read pod state — pending pods
can't double-book (the decision itself books them), and foreign
consumers are assumed not to exist (this is a ModelForge fleet).

Partial-metric failure (our rule): if any required signal is None or
NaN, hold S at its `current` (don't decide on half the signals). This
biases toward stability.
"""

import logging
import threading
import time

from decision_gen import estimator, planner

log = logging.getLogger(__name__)

API_VERSION = "llmscaling.inference.x-k8s.io/v1alpha1"


def _kinds_needed(spec, category):
    """Return the list of signal kinds a CR needs us to query.

    category: 'ttft' or 'otps'. Reads only .default.metrics[] types
    (ranges are v2).
    """
    block = (spec.get(category) or {}).get("default") or {}
    metrics = block.get("metrics") or []
    kinds = []
    for m in metrics:
        t = m.get("type")
        if t and t not in kinds:
            kinds.append(t)
    return kinds


class Controller:
    def __init__(self, slo, k8s, prom, tick_seconds=60):
        self.slo = slo
        self.k8s = k8s
        self.prom = prom
        self.tick_seconds = tick_seconds

        self._lock = threading.Lock()
        self._snapshot = {"apiVersion": API_VERSION, "decisions": []}
        self._last_change_at = {}      # svc_key → epoch
        self._last_served = {}         # svc_key → int (replicas we last put on the wire)
        self._comfortable_since = {}   # svc_key → epoch of start of unbroken comfort
        self._stop = threading.Event()
        self._thread = None

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
                self._tick()
            except Exception:
                log.exception("tick failed")
            elapsed = time.time() - started
            self._stop.wait(max(0.0, self.tick_seconds - elapsed))

    # ---------- tick ----------

    def _tick(self):
        now = time.time()
        crs = self.slo.snapshot()

        # Drop ledger entries for services whose CR went away, so their
        # committed GPUs free up in the next gap computation.
        for key in list(self._last_served.keys()):
            if key not in crs:
                del self._last_served[key]
                self._last_change_at.pop(key, None)
                self._comfortable_since.pop(key, None)
                log.info("%s/%s: CR deleted; freed ledger entry", *key)

        if not crs:
            log.debug("no CRs; skipping tick")
            with self._lock:
                self._snapshot = {"apiVersion": API_VERSION, "decisions": []}
            return

        # Ledger capacity: gross allocatable per pool. Gap is this
        # minus what we've committed — booked per service below.
        capacity = self.k8s.pool_capacity()

        wants = {}
        booked = {}      # planner's "current" — may differ from Prom's
        placements = {}
        cr_bounds = {}
        log_rows = []

        for (ns, svc), spec in sorted(crs.items()):
            key = (ns, svc)

            placement = self.k8s.resolve_placement(ns, svc)
            if placement is None:
                log_rows.append(f"{ns}/{svc}:unmanageable")
                continue

            # Fetch only the signal kinds the CR declares.
            ttft_kinds = _kinds_needed(spec, "ttft")
            otps_kinds = _kinds_needed(spec, "otps")
            signals = {
                "ttft": {k: self.prom.ttft(ns, svc, k) for k in ttft_kinds},
                "otps": {k: self.prom.otps(ns, svc, k) for k in otps_kinds},
                "rejection_rate": self.prom.rejection_rate(ns, svc),
            }

            # Physical serving replicas — drives the estimator's signal /
            # rejection feedback.
            prom_current = self.prom.current_replicas(ns, svc)

            # Ledger current: what we've last put on the wire for this
            # service. On the very first managed tick for a service
            # there is no last_served entry; seed from the workload's
            # `spec.replicas`, clamped to the CR's [min, max]. We never
            # seed from Prometheus — spec is the operator's declared
            # intent; Prom would just be lagging it at boot.
            booked_current = self._last_served.get(key)
            if booked_current is None:
                bounds = _cr_bounds(spec)
                seeded = placement.spec_replicas
                clamped = max(bounds["min"], min(bounds["max"], seeded))
                if clamped != seeded:
                    log.info(
                        "%s/%s: seeded spec.replicas=%d clamped to [%d,%d] → %d",
                        ns, svc, seeded, bounds["min"], bounds["max"], clamped,
                    )
                else:
                    log.info(
                        "%s/%s: seeding ledger from workload spec.replicas=%d",
                        ns, svc, seeded,
                    )
                booked_current = clamped
                # Seeding IS a change event for the cooldown clock. If we
                # leave last_change_at at epoch 0, the estimator reads it
                # as "long since changed" and may scale down on the very
                # first tick off unproven Prom data. We must prove comfort
                # for a full cooldown window before any shed.
                self._last_change_at[key] = now
            if prom_current is None:
                log.warning(
                    "%s/%s: current_replicas missing; estimator input "
                    "falls back to ledger (%d)",
                    ns, svc, booked_current,
                )
                prom_current = booked_current

            bounds = _cr_bounds(spec)

            # Partial failure: any required signal absent / NaN → hold.
            flat_signals = (
                list(signals["ttft"].values())
                + list(signals["otps"].values())
            )
            partial_missing = any(estimator.is_no_signal(v) for v in flat_signals)

            # Sustained-comfort tracker: rule 3 needs 30 min of continuous
            # comfortable signals before it'll shed. Reset on missing data,
            # advancing on fully-comfortable ticks. Comfortable is the same
            # test rule 3 uses internally (no rejection, both SLO blocks
            # comfortable with SLO_HEADROOM margin).
            ttft_block = (spec.get("ttft") or {}).get("default")
            otps_block = (spec.get("otps") or {}).get("default")
            if partial_missing:
                comfortable = False
            else:
                rej_val = signals.get("rejection_rate")
                comfortable = (
                    (estimator.is_no_signal(rej_val)
                     or rej_val < estimator.REJECTION_OK_FLOOR)
                    and estimator.slo_comfortably_met(ttft_block, signals["ttft"], "ceiling")
                    and estimator.slo_comfortably_met(otps_block, signals["otps"], "floor")
                )
            if comfortable:
                self._comfortable_since.setdefault(key, now)
            else:
                self._comfortable_since.pop(key, None)

            if partial_missing:
                want = _sanitize_hold(booked_current, bounds)
                wants[key] = want
                booked[key] = booked_current
                placements[key] = (placement.pool, placement.gpus_per_replica)
                cr_bounds[key] = bounds
                log_rows.append(
                    f"{ns}/{svc}:hold-no-signal booked={booked_current}",
                )
                missing = [k for k, v in list(signals["ttft"].items()) + list(signals["otps"].items())
                           if estimator.is_no_signal(v)]
                log.debug(
                    "%s/%s: hold (missing %s) booked=%d prom=%s",
                    ns, svc,
                    ",".join(missing) if missing else "?",
                    booked_current,
                    prom_current if prom_current is not None else "?",
                )
                continue

            est, reason = estimator.decide(
                cr_spec=spec,
                signals=signals,
                current_replicas=prom_current,
                last_change_at=self._last_change_at.get(key, 0),
                now=now,
                comfortable_since=self._comfortable_since.get(key),
            )

            # Direction rule: magnitude from prom (estimator), direction
            # from ledger. If the estimator proposes a shed (`desired <
            # booked`) without rule 3 having authorized it (reason ==
            # "scale-down"), lift the want to the ledger value. Planner
            # output below `booked` is untouched — preemption sheds are
            # legitimate by construction and the cooldown timer already
            # re-anchors when they land.
            if est < booked_current and not reason.startswith("scale-down"):
                want = _sanitize_hold(booked_current, bounds)
                reason = f"freeze-shed {reason}"
            else:
                want = est
            wants[key] = want
            booked[key] = booked_current
            placements[key] = (placement.pool, placement.gpus_per_replica)
            cr_bounds[key] = bounds
            log.debug(
                "%s/%s: est=%d → want=%d (%s)", ns, svc, est, want, reason,
            )
            log_rows.append(
                f"{ns}/{svc}:est={est} want={want} ({reason})",
            )

        # Ledger gap: total − Σ booked (priced at this tick's gpr).
        gap = dict(capacity)
        for key, n in booked.items():
            pool, gpr = placements[key]
            gap[pool] = gap.get(pool, 0) - n * gpr

        alloc = planner.resolve(
            wants=wants, current=booked,
            placement=placements, cr=cr_bounds, gap=gap,
        )

        for k, v in sorted(alloc.items()):
            log.debug(
                "%s/%s: want=%d → alloc=%d", k[0], k[1], wants.get(k, 0), v,
            )

        # Cooldown bookkeeping: stamp only when the planner moves a
        # service relative to what it walked in with this tick. Seeding
        # the ledger from Prometheus and holding-steady both leave the
        # value unchanged → no stamp.
        for k, v in alloc.items():
            if booked.get(k) != v:
                self._last_change_at[k] = now
                self._comfortable_since.pop(k, None)
            self._last_served[k] = v

        payload = {
            "apiVersion": API_VERSION,
            "decisions": [
                {"namespace": ns, "serviceId": svc, "replicas": {"active": n}}
                for (ns, svc), n in sorted(alloc.items())
            ],
        }
        with self._lock:
            self._snapshot = payload
        log.info("tick: %s", "; ".join(log_rows))


def _sanitize_hold(booked_current, bounds):
    """Hold value for the ledger. Never emit above the current CR max —
    a CR that shrank beneath us wins over the freeze; never below min —
    the boot seed already guarantees booked ≥ min so the freeze just
    keeps it."""
    return max(bounds["min"], min(bounds["max"], booked_current))


def _cr_bounds(spec):
    mn = (spec.get("minimumDeployment") or {}).get("value", 1)
    mx = (spec.get("maximumDeployment") or {}).get("value", 1)
    try:
        mn = int(mn)
    except (TypeError, ValueError):
        mn = 1
    try:
        mx = int(mx)
    except (TypeError, ValueError):
        mx = mn
    if mn > mx:
        mx = mn
    return {
        "min": mn,
        "max": mx,
        "priority": int((spec.get("priority") or 0)),
    }
