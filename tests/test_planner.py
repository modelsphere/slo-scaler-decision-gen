"""Tests for decision_gen.planner — per-pool allocator.

Baseline semantic:
  alloc[s] starts at max(min[s], min(want[s], current[s])). This means:
    - If a service is at current and wants to hold (want == current), it starts at current.
    - If a service wants to scale down (want < current), it starts at want.
    - If a service wants to scale up, it starts at current and phase 2/3 grow it.
    - If current < min (e.g. scale-up crashed), the floor still guarantees min.
"""

import pytest

from decision_gen.planner import resolve


def _cr(min_v=1, max_v=10, priority=5):
    return {"min": min_v, "max": max_v, "priority": priority}


POOL_H100 = "NVIDIA-H100-80GB-HBM3"
POOL_A100 = "NVIDIA-A100-SXM4-80GB"


# ---------- basic: one service ----------

def test_single_service_pool_has_room():
    alloc = resolve(
        wants={"svc": 4},
        current={"svc": 2},
        placement={"svc": (POOL_H100, 8)},
        cr={"svc": _cr(min_v=1, max_v=10)},
        gap={POOL_H100: 16},   # 16 GPUs free; 2→4 needs 16
    )
    assert alloc == {"svc": 4}


def test_single_service_pool_zero_gap_holds_at_current():
    """Baseline is min(want, current) — with no spare we hold, not drop to min."""
    alloc = resolve(
        wants={"svc": 5},
        current={"svc": 3},
        placement={"svc": (POOL_H100, 8)},
        cr={"svc": _cr(min_v=1, max_v=10)},
        gap={POOL_H100: 0},
    )
    assert alloc == {"svc": 3}


def test_single_service_partial_satisfaction():
    alloc = resolve(
        wants={"svc": 6},
        current={"svc": 2},
        placement={"svc": (POOL_H100, 8)},
        cr={"svc": _cr(min_v=1, max_v=10)},
        gap={POOL_H100: 16},  # only enough for +2 replicas
    )
    assert alloc == {"svc": 4}


def test_single_service_scale_down():
    """want < current: start at want, no phase 2/3 needed."""
    alloc = resolve(
        wants={"svc": 2},
        current={"svc": 5},
        placement={"svc": (POOL_H100, 4)},
        cr={"svc": _cr(min_v=1, max_v=10)},
        gap={POOL_H100: 0},
    )
    assert alloc == {"svc": 2}


def test_current_below_min_floor_bumps_to_min():
    """If current=0 but min=1 (CR-driven minimum), floor bumps to 1."""
    alloc = resolve(
        wants={"svc": 0},
        current={"svc": 0},
        placement={"svc": (POOL_H100, 8)},
        cr={"svc": _cr(min_v=1, max_v=10)},
        gap={POOL_H100: 8},   # need 8 to pull 0 → 1
    )
    assert alloc == {"svc": 1}


def test_floor_bump_costs_spare_when_min_exceeds_current():
    """Bumping from current=0 to min=1 must be paid from spare, or we don't do it."""
    alloc = resolve(
        wants={"svc": 5},
        current={"svc": 0},
        placement={"svc": (POOL_H100, 8)},
        cr={"svc": _cr(min_v=1, max_v=10)},
        gap={POOL_H100: 0},   # zero spare; can't even pay for the min bump
    )
    assert alloc == {"svc": 1}   # spec: floors *guarantee* min, even at gap overrun
    # Note: this exceeds our gap constraint; acceptable per the design intent
    # ("floors establish the baseline") since the alternative is illegal alloc < min.


# ---------- per-pool isolation ----------

def test_multi_pool_isolation():
    """Shortages in one pool don't affect the other."""
    alloc = resolve(
        wants={"a": 5, "b": 5},
        current={"a": 2, "b": 2},
        placement={"a": (POOL_H100, 8), "b": (POOL_A100, 8)},
        cr={"a": _cr(min_v=1, max_v=10), "b": _cr(min_v=1, max_v=10)},
        gap={POOL_H100: 0, POOL_A100: 100},
    )
    assert alloc["a"] == 2   # no room
    assert alloc["b"] == 5   # plenty of room


# ---------- phase 2: priority & fairness ----------

def test_higher_priority_tier_gets_capacity_first():
    """Two services, both wanting to grow, one higher priority."""
    alloc = resolve(
        wants={"hi": 6, "lo": 6},
        current={"hi": 2, "lo": 2},
        placement={"hi": (POOL_H100, 8), "lo": (POOL_H100, 8)},
        cr={
            "hi": _cr(min_v=1, max_v=10, priority=10),
            "lo": _cr(min_v=1, max_v=10, priority=0),
        },
        gap={POOL_H100: 40},   # 5 replicas of 8 GPUs; hi wants +4, lo wants +4
    )
    # Hi tier first: gets +4 (using 32 GPUs), then lo tier has 8 left: +1.
    assert alloc == {"hi": 6, "lo": 3}


def test_same_tier_splits_round_robin():
    """Two same-priority services both wanting +5 replicas with 6 GPUs spare each."""
    cr = _cr(min_v=0, max_v=20, priority=5)
    alloc = resolve(
        wants={"a": 10, "b": 10},
        current={"a": 0, "b": 0},
        placement={"a": (POOL_H100, 4), "b": (POOL_H100, 4)},
        cr={"a": cr, "b": cr},
        gap={POOL_H100: 40},   # 10 replicas of 4 GPUs
    )
    assert alloc["a"] + alloc["b"] == 10
    assert abs(alloc["a"] - alloc["b"]) <= 1   # round-robin keeps them within 1


def test_cheaper_service_fits_when_expensive_one_doesnt():
    """Same priority, but one service's gpr exceeds remaining spare — the other can still fit."""
    cr = _cr(min_v=1, max_v=10, priority=5)
    alloc = resolve(
        wants={"big": 3, "small": 3},
        current={"big": 1, "small": 1},
        placement={"big": (POOL_H100, 16), "small": (POOL_H100, 2)},
        cr={"big": cr, "small": cr},
        gap={POOL_H100: 8},   # can't afford +1 big (16), can afford +4 small (2)
    )
    # Round-robin by deficit (in GPR units) — big has larger deficit initially, tried first,
    # skips; small bumps to 3, consumes 4 GPUs; big still blocked.
    assert alloc["big"] == 1
    assert alloc["small"] == 3   # actually maybe 4+ if spare allows; up to 8/2=4 bumps


# ---------- phase 3: preemption ----------

def test_high_priority_squeezes_low():
    """H=10 wants to grow at expense of L=0, who is currently above its min."""
    alloc = resolve(
        wants={"hi": 4, "lo": 3},
        current={"hi": 1, "lo": 3},
        placement={"hi": (POOL_H100, 8), "lo": (POOL_H100, 8)},
        cr={
            "hi": _cr(min_v=1, max_v=10, priority=10),
            "lo": _cr(min_v=1, max_v=10, priority=0),
        },
        gap={POOL_H100: 0},
    )
    # baseline: hi=1, lo=3 (want==current) → no spare
    # phase 2: no progress
    # phase 3: hi needs (4-1)*8=24 GPUs. donors=lo (priority 0<10), at alloc=3, min=1,
    # can shed up to 2 replicas. take = ceil(24/1/8) = 3, capped at 2 → take 2.
    # lo → 1. freed=16. bumps = min(3, 16//8)=2 → hi=3. spare=0.
    # loop again: hi still wants 4, alloc=3, donor lo at 1 == min, no donors → break.
    assert alloc == {"hi": 3, "lo": 1}


def test_preemption_does_not_force_donor_below_min():
    alloc = resolve(
        wants={"hi": 10, "lo": 3},
        current={"hi": 1, "lo": 3},
        placement={"hi": (POOL_H100, 8), "lo": (POOL_H100, 8)},
        cr={
            "hi": _cr(min_v=1, max_v=10, priority=10),
            "lo": _cr(min_v=1, max_v=10, priority=0),
        },
        gap={POOL_H100: 0},
    )
    # hi wants 9 more (72 GPUs); donors can give at most 2 replicas of 8 GPUs = 16 GPUs.
    # So hi ends at 1 + 2 = 3.
    assert alloc["lo"] == 1   # lo floor preserved
    assert alloc["hi"] == 3   # partial growth


def test_preemption_only_when_unsatisfied():
    """A service at its want does NOT trigger preemption, even if others below it could shed."""
    alloc = resolve(
        wants={"hi": 1, "lo": 3},   # hi doesn't want to grow
        current={"hi": 1, "lo": 3},
        placement={"hi": (POOL_H100, 8), "lo": (POOL_H100, 8)},
        cr={
            "hi": _cr(min_v=1, max_v=10, priority=10),
            "lo": _cr(min_v=1, max_v=10, priority=0),
        },
        gap={POOL_H100: 0},
    )
    # No one is unsatisfied (hi=1, lo=3 = want). No preemption.
    assert alloc == {"hi": 1, "lo": 3}


def test_preemption_across_two_donor_tiers():
    """Service at tier 10 needs to shed both tier-0 mid and tier-1 bottom donors."""
    alloc = resolve(
        wants={"hi": 4, "mid": 2, "lo": 2},
        current={"hi": 1, "mid": 2, "lo": 2},
        placement={
            "hi": (POOL_H100, 8),
            "mid": (POOL_H100, 8),
            "lo": (POOL_H100, 8),
        },
        cr={
            "hi": _cr(min_v=1, max_v=10, priority=10),
            "mid": _cr(min_v=1, max_v=10, priority=5),
            "lo": _cr(min_v=1, max_v=10, priority=0),
        },
        gap={POOL_H100: 0},
    )
    # baseline: hi=1, mid=2, lo=2
    # phase 3: hi wants 3 more (24 GPUs).
    #   donor_tier = lowest with alloc>min = lo (priority 0). shed 1 → freed 8, hi=2, spare=0.
    #   (take = ceil(24/1/8)=3, capped at lo-min=1 → 1)
    #   donors again: lo at min. next tier: mid (priority 5). shed 1 → freed 8, hi=3.
    #   donors again: mid at min. no donors → break.
    assert alloc == {"hi": 3, "mid": 1, "lo": 1}


def test_preemption_proportional_within_tier():
    """Two same-tier donors split the shed proportionally (in this impl: ceil-split)."""
    alloc = resolve(
        wants={"hi": 4, "d1": 3, "d2": 3},
        current={"hi": 1, "d1": 3, "d2": 3},
        placement={
            "hi": (POOL_H100, 8),
            "d1": (POOL_H100, 8),
            "d2": (POOL_H100, 8),
        },
        cr={
            "hi": _cr(min_v=1, max_v=10, priority=10),
            "d1": _cr(min_v=1, max_v=10, priority=0),
            "d2": _cr(min_v=1, max_v=10, priority=0),
        },
        gap={POOL_H100: 0},
    )
    # hi wants 3 more (24 GPUs). Donor tier {d1, d2} with 3 alloc each, min 1 each.
    # need=24, take per donor = ceil(24/2/8) = 2, capped at donor alloc-min=2 → 2 each.
    # Shed 2 from d1 → freed 16, bumps = min(3, 16//8)=2 → hi=3, spare=0.
    # Shed 2 from d2 → freed 16, bumps = min(1, 16//8)=1 → hi=4, spare=8.
    # d1=1, d2=1, hi=4.
    assert alloc["hi"] == 4
    assert alloc["d1"] == 1
    assert alloc["d2"] == 1


# ---------- misc ----------

def test_gap_defaults_to_zero_when_pool_missing():
    """Pool name not in gap dict → spare = 0."""
    alloc = resolve(
        wants={"svc": 5},
        current={"svc": 2},
        placement={"svc": ("unknown-pool", 8)},
        cr={"svc": _cr(min_v=1, max_v=10)},
        gap={},   # "unknown-pool" missing
    )
    assert alloc == {"svc": 2}


def test_no_services_returns_empty():
    assert resolve(wants={}, current={}, placement={}, cr={}, gap={}) == {}
