"""Pure per-service scaling rule. No I/O, no clock (passed in), no
cross-service state.

Given one service's CR spec + observed signals + current replicas +
last-change timestamp, returns the desired replica count for this tick.

Design invariants:
  - scale-up is two-gear: multiplicative on rejection spike, additive
    on SLO violation
  - scale-down requires: cooldown elapsed *and* both SLOs comfortably
    met *and* rejection rate near zero
  - NaN input → "no signal" → hold, not violation
  - output is clamped to [min, max] before planner ever sees it
"""

import logging
import math
import os

log = logging.getLogger(__name__)


def env_float(name, default):
    return float(os.environ.get(name, default))


def env_int(name, default):
    return int(os.environ.get(name, default))


SCALE_UP_MULTIPLIER = env_float("SCALE_UP_MULTIPLIER", 1.5)
SCALE_UP_STEP = env_int("SCALE_UP_STEP", 1)
SCALE_DOWN_COOLDOWN_S = env_int("SCALE_DOWN_COOLDOWN_S", 3600)
REJECTION_THRESHOLD = env_float("REJECTION_THRESHOLD", 0.01)
REJECTION_OK_FLOOR = env_float("REJECTION_OK_FLOOR", 0.001)
SLO_HEADROOM = env_float("SLO_HEADROOM", 0.5)

DEFAULT_MIN = 1
DEFAULT_MAX = 1


def is_no_signal(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


def _metrics_list(block):
    """Extract `block.metrics` (a list), tolerating absent blocks."""
    if not block:
        return []
    if isinstance(block, dict):
        return block.get("metrics", []) or []
    return block or []


def slo_violated(metrics_spec, observed):
    """metrics_spec = [{'type':'p95','threshold':2000.0}, ...]
    observed     = {'p95': 2500.0, 'avg': 800.0, ...}
    True if any pair's observed exceeds its threshold.
    Missing/NaN observed → False (no signal is not a violation)."""
    for m in _metrics_list(metrics_spec):
        val = observed.get(m["type"])
        if is_no_signal(val):
            continue
        if val > m["threshold"]:
            return True
    return False


def slo_comfortably_met(metrics_spec, observed):
    """Every pair's observed < threshold × SLO_HEADROOM.
    Missing/NaN observed → False (can't confirm comfortable).

    An empty metrics list is trivially comfortable — nothing to violate."""
    metrics = _metrics_list(metrics_spec)
    if not metrics:
        return True
    for m in metrics:
        val = observed.get(m["type"])
        if is_no_signal(val):
            return False
        if not (val < m["threshold"] * SLO_HEADROOM):
            return False
    return True


def _min_max(cr_spec):
    mn = cr_spec.get("minimumDeployment", {}) or {}
    mx = cr_spec.get("maximumDeployment", {}) or {}
    mn_v = int(mn.get("value", DEFAULT_MIN)) if mn.get("value") is not None else DEFAULT_MIN
    mx_v = int(mx.get("value", DEFAULT_MAX)) if mx.get("value") is not None else max(mn_v, DEFAULT_MAX)
    # Defensive per §5.1: malformed CR (min > max) → clamp, don't crash.
    if mn_v > mx_v:
        log.warning("CR min>max (%s > %s); clamping max=min", mn_v, mx_v)
        mx_v = mn_v
    return mn_v, mx_v


def decide(cr_spec, signals, current_replicas, last_change_at, now):
    """Pure function. Returns (desired, reason).

    Args:
      cr_spec:          parsed LLMSLORequirement spec dict
      signals:          {
          'ttft':           {'p50':s,'p80':s,...,'avg':s | None},
          'otps':           {'p50':t,'p80':t,...,'avg':t | None},
          'rejection_rate': r in [0,1] | None,
      }
      current_replicas: live count (int; not None — controller handles
                        missing signal by falling back before calling us)
      last_change_at:   epoch seconds of last committed change
      now:              epoch seconds

    Returns:
      (desired, reason) — desired is clamped to [min, max]; reason is a
      short string for logging.
    """
    mn, mx = _min_max(cr_spec)
    ttft_block = (cr_spec.get("ttft") or {}).get("default")
    otps_block = (cr_spec.get("otps") or {}).get("default")
    ttft_sig = signals.get("ttft") or {}
    otps_sig = signals.get("otps") or {}
    rej = signals.get("rejection_rate")

    # Rule 1: rejection spike → multiplicative up.
    # NaN rejection doesn't trigger (is_no_signal check).
    if not is_no_signal(rej) and rej > REJECTION_THRESHOLD:
        desired = max(math.ceil(current_replicas * SCALE_UP_MULTIPLIER), current_replicas + 1)
        reason = f"scale-up-rejection rej={rej:.4f}"
    # Rule 2: SLO violation → additive up.
    elif slo_violated(ttft_block, ttft_sig) or slo_violated(otps_block, otps_sig):
        desired = current_replicas + SCALE_UP_STEP
        reason = "scale-up-slo"
    # Rule 3: cooldown OK *and* comfortable *and* quiet → −1.
    elif (now - last_change_at) >= SCALE_DOWN_COOLDOWN_S:
        if not is_no_signal(rej) and rej >= REJECTION_OK_FLOOR:
            reason = f"hold-rejection-above-floor rej={rej:.4f}"
            desired = current_replicas
        elif (slo_comfortably_met(ttft_block, ttft_sig)
              and slo_comfortably_met(otps_block, otps_sig)):
            desired = current_replicas - 1
            reason = "scale-down"
        else:
            desired = current_replicas
            reason = "hold-not-comfortable"
    else:
        desired = current_replicas
        reason = f"hold-cooldown remaining={int(SCALE_DOWN_COOLDOWN_S - (now - last_change_at))}s"

    clamped = max(mn, min(mx, desired))
    if clamped != desired:
        reason += f" clamp[{mn},{mx}]"
    return clamped, reason
