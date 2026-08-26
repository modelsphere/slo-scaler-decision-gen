"""R1a / R1b / R1c sizing rules, applied top-down, first match wins.

Pure. No state, no clock, no env — every input arrives as an argument;
every decision is a function of them. What this module does NOT know
(by contract, arch §3.2):

  - No `last_change_at`, `comfortable_since`, `now`. Cooldown and
    comfort arrive as `Gates` booleans; clock count is invisible here.
  - No clamping. `[min, max]` is CR authority (R9), applied by
    serviceview at the est stage. Proposals may exceed max.
  - No ledger. `current` is magnitude evidence (R5); the
    replicas_ready-missing fallback is chosen *before* the call.
  - Hold is `None`, not `proposed == current`. "No rule fires" must
    not be confused with "propose-stay" — the module doesn't know the
    committed value.

P4: replicas are absolute ints everywhere. Deltas have an implicit
base whose ambiguity is the I4 bug class; we don't ship them.
"""

import math
import os
from collections import namedtuple

from decision_gen.thresholds import Verdict


def _env_float(name, default):
    return float(os.environ.get(name, default))


REJECTION_THRESHOLD   = _env_float("REJECTION_THRESHOLD",      0.05)
REJECTION_OK_FLOOR    = _env_float("REJECTION_OK_FLOOR",       0.001)

R1A_GAIN              = _env_float("SCALE_UP_MULTIPLIER_GAIN", 2.0)
R1A_CAP               = _env_float("SCALE_UP_MULTIPLIER_CAP",  1.5)
R1B_STEP_FRAC         = _env_float("SCALE_UP_STEP_FRAC",       0.15)
R1C_STEP_FRAC         = _env_float("SCALE_DOWN_STEP_FRAC",     0.15)

# Evidence floors: a high rate off tiny N is noise, not signal.
R1A_MIN_REJECTIONS    = _env_float("R1A_MIN_REJECTIONS_2M",    5.0)   # over 2m
R1B_MIN_REQUESTS      = _env_float("R1B_MIN_REQUESTS_5M",     20.0)   # over 5m


# Cooldown/comfort answers, pre-computed by serviceview. R1a reads
# neither field — fire alarm is ungated — so neither may shut it off.
Gates = namedtuple("Gates", ["up_cooldown_open", "shed_ready"])

# Absolute target (P4). rule: 'r1a-fire' | 'r1b-steady' | 'r1c-shed'.
# reason is human-readable, with numbers, for §8 logs.
Proposal = namedtuple("Proposal", ["replicas", "rule", "reason"])


def fire_alarm(rejection_rate, current, rejection_count=None):
    """R1a: rejection ≥ 5% AND ≥5 rejections in window → multiplicative.

    Multiplier is 1 + (r/(1−r)) × GAIN, capped at R1A_CAP. Deficit
    ratio r/(1−r) is the rejected-load fraction of accepted traffic;
    overshooting by ~2× makes recovery faster than drain rate.

    Evidence floor: `rejection_count` is the absolute number of 429s in
    the 2m window. r = rate is meaningless off N=1; 1 rejected request
    out of 1 total is 100% but not a capacity emergency. Floor of 5
    rejects per 2m filters singleton noise. Caller passes None when the
    reading is missing_series, which also skips — no evidence, no fire.

    Never gated. Never a shed: floor is current+1 (C1 zero-base corner
    gives `current+1` when `ceil(current × mult) ≤ current`).
    """
    if rejection_rate is None or rejection_rate < REJECTION_THRESHOLD:
        return None
    if rejection_count is None or rejection_count < R1A_MIN_REJECTIONS:
        return None
    # r=1.0 fired ZeroDivisionError at 15:00:42 — the formula explodes
    # exactly when the fire is hottest. Clip to (0, 0.99]: any r ≥ 0.99
    # is already deep into R1A_CAP territory, and a buggy exporter giving
    # r < 0 would otherwise invert the direction of the shed check.
    r = max(0.0, min(0.99, rejection_rate))
    mult = 1.0 + (r / (1.0 - r)) * R1A_GAIN
    mult = min(mult, R1A_CAP)
    proposed = max(math.ceil(current * mult), current + 1)
    return Proposal(
        replicas=proposed,
        rule="r1a-fire",
        reason=f"rej={rejection_rate:.4f} n={rejection_count:.0f} mult={mult:.3f} cur={current}",
    )


def steady_growth(verdicts, current, request_count=None, frac=R1B_STEP_FRAC):
    """R1b: any SLO signal VIOLATED AND ≥20 requests in the 5m SLO window.

    Step is max(1, ceil(frac × current)) — min-step-1 covers the
    zero-base corner (C5: `ceil(0 × frac) = 0` must not stall growth).
    Cooldown gating is the caller's job (Gates.up_cooldown_open).

    Evidence floor: TTFT/OTPS quantiles off a handful of requests aren't
    a distribution — p80 of N=3 IS that single sample. Require ≥20
    requests in the same 5m window the verdict was evaluated over."""
    if not any(v is Verdict.VIOLATED for v in verdicts.values()):
        return None
    if request_count is None or request_count < R1B_MIN_REQUESTS:
        return None
    step = max(1, math.ceil(current * frac))
    viol = sorted(k for k, v in verdicts.items() if v is Verdict.VIOLATED)
    return Proposal(
        replicas=current + step,
        rule="r1b-steady",
        reason=f"step=+{step} viol={viol} n={request_count:.0f}",
    )


def quiet_shed(verdicts, rejection_rate, current, frac=R1C_STEP_FRAC):
    """R1c: sustained comfort + quiet rejection → remove a step.

    Requires every declared SLO verdict COMFORTABLE — GREY is not
    good enough (R3). Rejection must be near zero (<0.1%). Shed
    gating (down-cooldown + comfort sustainment) is the caller's job
    (Gates.shed_ready); we assume it here but refuse on evidence.
    """
    if rejection_rate is None or rejection_rate >= REJECTION_OK_FLOOR:
        return None
    if not verdicts:
        return None
    if not all(v is Verdict.COMFORTABLE for v in verdicts.values()):
        return None
    step = max(1, math.ceil(current * frac))
    return Proposal(
        replicas=current - step,
        rule="r1c-shed",
        reason=f"step=-{step} rej={rejection_rate:.4f}",
    )


def evaluate(verdicts, rejection_rate, current, gates,
             rejection_count=None, request_count=None):
    """Top-down first-match. None = hold (caller keeps committed).

    R1a fires regardless of gate state — reads neither flag.
    R1b requires up_cooldown_open.
    R1c requires shed_ready (cooldown AND comfort-sustain).

    Counts are evidence floors: R1a needs ≥R1A_MIN_REJECTIONS rejections
    in the 2m window; R1b needs ≥R1B_MIN_REQUESTS requests in the 5m
    window. R1c has no floor — comfort + rejection-quiet is enough
    (a 30-min continuous streak is itself the sample-size argument)."""
    p = fire_alarm(rejection_rate, current, rejection_count)
    if p is not None:
        return p
    if gates.up_cooldown_open:
        p = steady_growth(verdicts, current, request_count)
        if p is not None:
            return p
    if gates.shed_ready:
        p = quiet_shed(verdicts, rejection_rate, current)
        if p is not None:
            return p
    return None
