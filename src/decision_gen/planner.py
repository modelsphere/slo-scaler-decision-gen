"""Pure global allocator. No I/O.

Inputs (all plain dicts / ints):
  wants:      {svc_key: desired_replicas}
  current:    {svc_key: live_replicas}
  placement:  {svc_key: (pool, gpus_per_replica)}
  cr:         {svc_key: {'min': int, 'max': int, 'priority': int}}
  gap:        {pool: free_gpus}

Output:
  {svc_key: allocated_replicas}

Per-pool isolation — services in different pools never interact.
Three phases: floors → priority round-robin → preemption.

Semantics:

  Phase 1 (floors): start from the *baseline* we believe the world is
  already paying for. That's `max(min[s], min(want[s], current[s]))` —
  at least the CR floor, at most what the estimator asked for, and
  never exceeding what's already running (so we never charge ourselves
  for GPUs we don't actually need to allocate this tick).

  Phase 2 (priority): top-down by tier, round-robin within a tier by
  largest deficit first. Each bump of one replica consumes `gpr[s]`
  from pool spare. Skips (rather than brakes the tier) services whose
  per-replica cost exceeds remaining spare — cheaper services in the
  same tier may still fit.

  Phase 3 (preemption): for each still-unsatisfied service in priority
  order, shed lower-priority donors toward their min and grant the
  freed GPUs to the unsatisfied service.

Invariants:
  - alloc[s] ∈ [min[s], max[s]]
  - Σ over svc in pool (alloc[s] − current[s])_+ × gpus_per_replica[s]
      ≤ gap[pool]
  - phase-3 never shrinks a donor below its min
  - cooldown bypass: preemption-driven downscaling is the planner's own
    decision and does not block on estimator cooldown
"""

import logging
import math

log = logging.getLogger(__name__)


def resolve(wants, current, placement, cr, gap):
    """Return {svc_key: allocated_replicas}. Pure."""
    pools = {}
    for k, (pool, _gpr) in placement.items():
        pools.setdefault(pool, []).append(k)

    alloc = {}
    for pool, members in pools.items():
        _allocate_in_pool(
            pool, members,
            wants=wants, current=current, placement=placement,
            cr=cr, gap=gap, alloc_out=alloc,
        )
    return alloc


def _allocate_in_pool(pool, members, *, wants, current, placement, cr, gap, alloc_out):
    gpr = {k: placement[k][1] for k in members}
    want_clamped = {
        k: max(cr[k]["min"], min(cr[k]["max"], wants[k]))
        for k in members
    }

    # Phase 1 — baseline.
    for k in members:
        alloc_out[k] = max(cr[k]["min"], min(want_clamped[k], current[k]))
    spare = gap.get(pool, 0)
    # Account for any service whose baseline exceeds its current (can
    # only happen when min > current — estimator-driven floor bump).
    # In that case we must consume spare now to pay for the floor.
    for k in members:
        spare -= max(0, alloc_out[k] - current[k]) * gpr[k]

    # Phase 2 — priority pass, highest tier first.
    by_tier = {}
    for k in members:
        by_tier.setdefault(cr[k]["priority"], []).append(k)

    for tier_prio in sorted(by_tier.keys(), reverse=True):
        tier = by_tier[tier_prio]
        while True:
            progressed = False
            # D6 — stable tie-break. Deficit desc first; service key asc
            # next so equal-deficit services don't flap tick to tick.
            order = sorted(
                tier,
                key=lambda k: (
                    -(want_clamped[k] - alloc_out[k]) * gpr[k],
                    k,
                ),
            )
            for k in order:
                if alloc_out[k] >= want_clamped[k]:
                    continue
                if gpr[k] > spare:
                    continue
                alloc_out[k] += 1
                spare -= gpr[k]
                progressed = True
            if not progressed:
                break

    # Phase 3 — preemption.
    unsatisfied = [k for k in members if alloc_out[k] < want_clamped[k]]
    for k in sorted(unsatisfied, key=lambda x: cr[x]["priority"], reverse=True):
        while alloc_out[k] < want_clamped[k]:
            donors = [t for t in members
                      if cr[t]["priority"] < cr[k]["priority"]
                      and alloc_out[t] > cr[t]["min"]]
            if not donors:
                break
            donor_tier = min(cr[t]["priority"] for t in donors)
            tier_donors = [t for t in donors if cr[t]["priority"] == donor_tier]

            need_gpus = (want_clamped[k] - alloc_out[k]) * gpr[k]
            for t in tier_donors:
                take = min(
                    math.ceil(need_gpus / len(tier_donors) / gpr[t]),
                    alloc_out[t] - cr[t]["min"],
                )
                if take <= 0:
                    continue
                freed = take * gpr[t]
                alloc_out[t] -= take
                bumps = min(want_clamped[k] - alloc_out[k], freed // gpr[k])
                alloc_out[k] += bumps
                spare += freed - bumps * gpr[k]
