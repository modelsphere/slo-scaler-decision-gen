"""R5 ledger/physical reconciliation — the I4 fix.

The smallest, most load-bearing module in the codebase. v1 shipped
with a one-directional guard inside `_tick`: it caught shed-during-
ramp-up but not grow-back-during-drain, so a `40→34` commit followed
by a drain still at physical=40 caused the estimator to re-propose 40
("looks like scale-up from prom=40 vs committed=34") and the ledger
bounced. The shed never survived one tick.

Phase is derived (D2 — chosen model): compare `committed` against
physical evidence each tick, don't track a side flag.

The rule is SYMMETRIC: a proposal against the phase's authorized
direction snaps to committed. No asymmetry, no exceptions.

  - RAMP_UP   (physical < committed): executor is still adding
              replicas. A proposal below committed would shed in-flight
              capacity we haven't even received yet. Freeze up.
  - IN_DRAIN  (physical > committed): executor is draining after a
              shed. A proposal above committed would "grow back" into
              the not-yet-drained surplus. Freeze down.
  - SETTLED   (physical == committed): pass through.
"""

from enum import Enum


class Phase(Enum):
    RAMP_UP   = "ramp-up"
    IN_DRAIN  = "in-drain"
    SETTLED   = "settled"


def phase_of(committed, physical):
    if physical < committed:
        return Phase.RAMP_UP
    if physical > committed:
        return Phase.IN_DRAIN
    return Phase.SETTLED


def phase_of_pd(committed, physical):
    """Per-role phases for a PD service. Returns
    {"prefill": Phase, "decode": Phase}.

    `committed` / `physical` are 2-tuples (prefill, decode). Each role
    runs the same RAMP_UP / IN_DRAIN / SETTLED algebra independently —
    shedding decode while prefill is still ramping must not mask the
    shed, and vice-versa.
    """
    return {
        "prefill": phase_of(committed[0], physical[0]),
        "decode":  phase_of(committed[1], physical[1]),
    }


def reconcile(proposed, committed, phase):
    """Return (replicas, reason).

    `reason` is the empty string on pass-through, or one of the named
    freezes. Both freezes carry the same shape so callers / tests can
    pattern-match the prefix `freeze-`.
    """
    if phase is Phase.IN_DRAIN and proposed > committed:
        return committed, f"freeze-in-drain proposed={proposed} committed={committed}"
    if phase is Phase.RAMP_UP and proposed < committed:
        return committed, f"freeze-ramp-up proposed={proposed} committed={committed}"
    return proposed, ""


def reconcile_pd(proposed, committed, phases):
    """Per-role `reconcile`. `proposed` / `committed` are PdShapes,
    `phases` the dict from `phase_of_pd`.

    Each role runs its own symmetric freeze independently; a role that
    is SETTLED passes through untouched even if its sibling is in
    RAMP_UP or IN_DRAIN. Reasons are concatenated with `|` so log
    parsers keep the `freeze-` prefix contract.
    """
    p_new, p_reason = reconcile(
        proposed.prefill, committed.prefill, phases["prefill"])
    d_new, d_reason = reconcile(
        proposed.decode,  committed.decode,  phases["decode"])
    reasons = [r for r in (p_reason, d_reason) if r]
    return proposed.__class__(p_new, d_new, p_new + d_new), "|".join(reasons)
