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
