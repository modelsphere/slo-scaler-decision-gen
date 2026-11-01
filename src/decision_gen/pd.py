"""PD-disagg (DisaggregatedSet) engine-shape math. Pure; no I/O.

For pdengine-typed services the unit is the *engine* — one prefill or
decode LWS group. A slice bundles `p + d` engines; n (`spec.slices`) is
deploy-time authority and the autoscaler never moves it. Bounds arrive
in engines so a CR can say "this PD service runs between 6 and 24
engines" without commiting to a P:D ratio.

Three operations live here, all total:

  - `correct_to_bounds` — same-ratio walk of engine totals into the
    CR's [min, max] window, growing (ceil) and shrinking (floor) as a
    pair. Returns None when the shape is mathematically inexpressible
    inside the window; callers skip the service rather than emit a
    bound-violating row.
  - `role_margins` — per-role SLO comfort depth. ttft speaks for
    prefill, otps for decode; multiple kinds collapse by min (the
    least-comfortable kind is the role's margin).
  - `pick_shed_role` — R1c chooses the role with the bigger margin,
    refusing to drop a role below its one-engine floor.
"""

import math
from collections import namedtuple

from decision_gen.thresholds import Direction

PdShape = namedtuple("PdShape", ["prefill", "decode", "engines"])
RoleMargins = namedtuple("RoleMargins", ["prefill", "decode"])


def shape_gpu_cost(shape, role_group_gpus):
    """GPUs a PdShape costs given per-role engine-group sizes.

    Heterogeneous role groups (e.g. prefill TP4 / decode TP8) work
    naturally — the controller charges each role at its own rate.
    """
    return (shape.prefill * role_group_gpus["prefill"]
            + shape.decode * role_group_gpus["decode"])


def trim_growth_to_cap(proposal, committed, cap):
    """Pull a bound-crossing *growth* back so est fits `cap`.

    Removes `proposal.engines - cap` engines, taken from the roles
    the proposal grew (larger increment shed first; a tie sheds
    decode). Never takes a role below its committed value — the
    sibling role's existing engines are untouched. Returns the
    trimmed PdShape *at* the cap, or `committed` when the proposal's
    own delta doesn't cover the overshoot (committed was already
    out of window — the bounds-fix path owns that case).
    """
    over = proposal.engines - cap
    if over <= 0:
        return proposal
    gp = proposal.prefill - committed.prefill
    gd = proposal.decode - committed.decode
    if gp + gd < over:
        return committed
    while over > 0:
        if gd >= gp and gd > 0:
            gd -= 1
        elif gp > 0:
            gp -= 1
        else:
            break
        over -= 1
    return _mk(committed.prefill + gp, committed.decode + gd)


def pad_shed_to_floor(proposal, committed, floor):
    """Push a floor-crossing *shed* back up so est fits `floor`.

    Returns `floor - proposal.engines` engines to the role the
    proposal shed from (the role that sits furthest below its
    committed level; a tie returns to decode). Never returns a
    role above its committed value. Returns the padded PdShape
    *at* the floor, or `committed` when the proposal's own delta
    doesn't cover the shortfall.
    """
    under = floor - proposal.engines
    if under <= 0:
        return proposal
    sp = committed.prefill - proposal.prefill
    sd = committed.decode - proposal.decode
    if sp + sd < under:
        return committed
    while under > 0:
        if sd >= sp and sd > 0:
            sd -= 1
        elif sp > 0:
            sp -= 1
        else:
            break
        under -= 1
    return _mk(committed.prefill - sp, committed.decode - sd)


def _mk(prefill, decode):
    return PdShape(int(prefill), int(decode), int(prefill) + int(decode))


def per_slice_window(engines_min, engines_max, slices):
    """Fleet-total CR bounds → per-slice window for `correct_to_bounds`.

    Slices are identical, so a fleet bound B becomes per-slice B/n;
    the window closes up to the nearest integers
    (`ceil(min/n) .. floor(max/n)`). Returns (None, None) when the
    window is empty — e.g. min = max = 7 with n = 2 slices is
    mathematically inexpressible, and the caller must skip the service
    rather than emit a bound-violating shape.
    """
    n = max(1, int(slices))
    lo = math.ceil(int(engines_min) / n)
    hi = math.floor(int(engines_max) / n)
    if lo > hi:
        return None, None
    return lo, hi


def correct_to_bounds(prefill, decode, engines_min, engines_max):
    """Bring per-slice shape (prefill, decode) inside [min, max].

    The window is in *engines of a slice* — the caller lifts
    `n × (x + y)` into per-slice numbers before the call. Per-slice
    math is the only meaningful comparison because the autoscaler
    never moves n; outside v1 it simply reports infeasible.

    The bounds window is *per-slice* engine totals (callers divide
    fleet-total bounds by `spec.slices` before calling). Rounding can
    miss the bound by a small residual; engines are walked one at a
    time, always towards the bound, taking from the role furthest
    from its ideal ratio (largest overshoot first, ties to decode so
    equal shapes resolve deterministically).

    Inexpressible shapes return None and the caller must *skip the
    service* rather than emit a bound-violating row. The residual
    walk never breaks the ≥1 floor, so a shape can only fail when the
    bound leaves no room for it — e.g. an odd bound against a 1:1
    ratio walks one engine off the pair and lands; a bound below 2
    (below the pair's floor) refuses.
    """
    E = int(prefill) + int(decode)
    mn, mx = int(engines_min), int(engines_max)
    if mn == 0 and mx == 0:
        return _mk(0, 0)
    if E == 0:
        return None
    if mn <= E <= mx:
        return _mk(prefill, decode)

    over = E > mx                     # True → shrink, False → grow
    bound = mx if over else mn
    r = bound / E
    p = max(1, (math.floor if over else math.ceil)(prefill * r))
    d = max(1, (math.floor if over else math.ceil)(decode * r))

    # Residual walk, one engine at a time, always towards the bound.
    # On grow we overshot (p + d > bound) so we remove; on shrink we
    # fell short so we add. Choose the role whose current value is
    # furthest from its proportional ideal for that bound, ties to
    # decode so equal shapes resolve the same way every time.
    def ideal_prefill():
        return prefill * r

    while p + d != bound:
        if p + d > bound:
            # Remove one: pick the role further above its ideal, floor 1.
            over_p = p - ideal_prefill()
            over_d = d - (decode * r)
            if p > 1 and (over_p >= over_d or d <= 1):
                p -= 1
            elif d > 1:
                d -= 1
            else:
                return None
        else:
            # Add one: pick the role further below its ideal.
            under_p = ideal_prefill() - p
            under_d = (decode * r) - d
            if under_p >= under_d:
                p += 1
            else:
                d += 1

    shape = _mk(p, d)
    return shape if mn <= shape.engines <= mx else None


def role_margins(ttft_readings, otps_readings, ttft_thresholds, otps_thresholds,
                 headroom):
    """Per-role comfort depth; > 1 means inside the comfort zone.

    `ttft_readings` / `otps_readings` are `{kind: (value | None, state)}`
    with state "ok" | "no_traffic" | "missing_series". Thresholds are
    parallel dicts `{kind: float}`. A role's margin is the *minimum*
    over its kinds — one tight kind caps the role. A missing or
    zero-valued role returns None, which `pick_shed_role` treats as
    infinitely uncomfortable.
    """

    def _fold(direction, readings, thresholds):
        best = None
        for kind, (value, state) in (readings or {}).items():
            thr = (thresholds or {}).get(kind)
            if thr is None or thr <= 0:
                continue
            if state == "no_traffic":
                m = math.inf
            elif state != "ok" or value is None:
                continue
            elif direction is Direction.CEILING:
                m = (thr * headroom) / value if value > 0 else math.inf
            else:
                m = (value * headroom) / thr
            best = m if best is None else min(best, m)
        return best

    return RoleMargins(
        prefill=_fold(Direction.CEILING, ttft_readings, ttft_thresholds),
        decode=_fold(Direction.FLOOR, otps_readings, otps_thresholds),
    )


def pick_shed_role(prefill, decode, margins):
    """Which role takes the R1c step. "prefill" | "decode" | None.

    Both roles must stay >= 1 — a PD service needs at least one of
    each. The role with the larger margin sheds; ties go to the role
    with more replicas (smaller relative disruption), then to decode
    as the final deterministic tie-break.
    """
    if prefill <= 1 and decode <= 1:
        return None
    mp = margins.prefill if margins.prefill is not None else -math.inf
    md = margins.decode if margins.decode is not None else -math.inf

    if prefill <= 1:
        return "decode"
    if decode <= 1:
        return "prefill"
    if mp > md:
        return "prefill"
    if md > mp:
        return "decode"
    if prefill > decode:
        return "prefill"
    if decode > prefill:
        return "decode"
    return "decode"
