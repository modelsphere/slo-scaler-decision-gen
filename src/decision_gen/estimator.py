"""Pure per-service scaling rule. No I/O, no clock (passed in), no
cross-service state.

Given one service's CR spec + observed signals + current replicas +
last-change timestamp, returns the desired replica count for this tick.

Direction semantics (per-signal): TTFT is a latency ceiling — violated
when `observed > threshold`. OTPS is a throughput floor — violated when
`observed < threshold`. "Comfortable" is the same-side safe zone:
ceilings must sit below `threshold * SLO_HEADROOM`, floors above
`threshold / SLO_HEADROOM`, which makes scale-down stickier than
scale-up and prevents flapping.

Design invariants:
  - scale-up is two-gear: multiplicative on rejection spike, additive
    on SLO violation
  - rule 1 (rejection) is a fire alarm and fires regardless of cooldown;
    rule 2 (SLO violation) needs the 30-min up-cooldown; rule 3 (shed)
    needs the 60-min down-cooldown
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


SCALE_UP_MULTIPLIER_CAP = env_float("SCALE_UP_MULTIPLIER_CAP", 1.5)
SCALE_UP_MULTIPLIER_GAIN = env_float("SCALE_UP_MULTIPLIER_GAIN", 2.0)
SCALE_UP_STEP_FRAC = env_float("SCALE_UP_STEP_FRAC", 0.15)
SCALE_UP_STEP_MIN = env_int("SCALE_UP_STEP_MIN", 1)
SCALE_UP_COOLDOWN_S = env_int("SCALE_UP_COOLDOWN_S", 1800)
SCALE_DOWN_STEP_FRAC = env_float("SCALE_DOWN_STEP_FRAC", 0.15)
SCALE_DOWN_STEP_MIN = env_int("SCALE_DOWN_STEP_MIN", 1)
SCALE_DOWN_COOLDOWN_S = env_int("SCALE_DOWN_COOLDOWN_S", 3600)
SCALE_DOWN_COMFORT_S = env_int("SCALE_DOWN_COMFORT_S", 1800)
REJECTION_THRESHOLD = env_float("REJECTION_THRESHOLD", 0.05)
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


def slo_violated(metrics_spec, observed, direction="ceiling"):
    """True if any pair's observed crosses its threshold in the bad direction.
    direction 'ceiling' (TTFT): violated when val > threshold.
    direction 'floor' (OTPS):   violated when val < threshold.
    Missing/NaN observed → False (no signal is not a violation)."""
    for m in _metrics_list(metrics_spec):
        val = observed.get(m["type"])
        if is_no_signal(val):
            continue
        if direction == "floor":
            if val < m["threshold"]:
                return True
        else:
            if val > m["threshold"]:
                return True
    return False


def slo_comfortably_met(metrics_spec, observed, direction="ceiling"):
    """True when every pair is on the safe side with full headroom.
    'ceiling': val < threshold * SLO_HEADROOM (must be far below).
    'floor':   val > threshold / SLO_HEADROOM (must be far above).
    Missing/NaN observed → False (can't confirm comfortable).

    An empty metrics list is trivially comfortable — nothing to violate."""
    metrics = _metrics_list(metrics_spec)
    if not metrics:
        return True
    for m in metrics:
        val = observed.get(m["type"])
        if is_no_signal(val):
            return False
        if direction == "floor":
            if not (val > m["threshold"] / SLO_HEADROOM):
                return False
        else:
            if not (val < m["threshold"] * SLO_HEADROOM):
                return False
    return True


def slo_not_comfortable(metrics_spec, observed, direction="ceiling"):
    """Names of metric types that are NOT comfortable (used for log
    decomposition). Returns [] when fully comfortable or when metrics
    list is empty."""
    bad = []
    for m in _metrics_list(metrics_spec):
        val = observed.get(m["type"])
        thr = m.get("threshold")
        if val is None or (isinstance(val, float) and math.isnan(val)) or thr is None:
            bad.append(m["type"])
            continue
        if direction == "floor":
            if not (val > thr / SLO_HEADROOM):
                bad.append(m["type"])
        else:
            if not (val < thr * SLO_HEADROOM):
                bad.append(m["type"])
    return bad


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


def decide(cr_spec, signals, current_replicas, last_change_at, now, comfortable_since=None):
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
      comfortable_since: epoch seconds when the current unbroken run of
                        fully-comfortable ticks began, or None if the
                        service isn't currently comfortable. The
                        controller maintains this per-service; rule 3
                        requires it ≥ SCALE_DOWN_COMFORT_S ago.

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

    # Rule 1: rejection spike → multiplicative up. Fire-alarm path —
    # ungated by cooldown; a spike mid-cooldown still scales up.
    # Magnitude is sized to the deficit we'd need to serve the rejected
    # traffic: accepted fraction is 1−r, deficit ratio is r/(1−r), and
    # we overshoot it by GAIN to converge faster than the drain rate.
    if not is_no_signal(rej) and rej >= REJECTION_THRESHOLD:
        mult = 1.0 + (rej / (1.0 - rej)) * SCALE_UP_MULTIPLIER_GAIN
        mult = min(mult, SCALE_UP_MULTIPLIER_CAP)
        desired = max(math.ceil(current_replicas * mult), current_replicas + 1)
        reason = f"scale-up-rejection rej={rej:.4f} mult={mult:.3f}"
    # Rule 2: SLO violation → additive up, gated by the up-cooldown.
    # Step is ~15% of current so large pools don't crawl +1 at a time.
    elif (now - last_change_at) >= SCALE_UP_COOLDOWN_S and (
            slo_violated(ttft_block, ttft_sig, "ceiling")
            or slo_violated(otps_block, otps_sig, "floor")):
        step = max(SCALE_UP_STEP_MIN, math.ceil(current_replicas * SCALE_UP_STEP_FRAC))
        desired = current_replicas + step
        reason = f"scale-up-slo step={step}"
    # Rule 3: down-cooldown OK *and* sustained comfort *and* quiet → −step.
    # Step shapes rule 2's: ~15% of current so large pools converge in
    # hours, not days. Overshoot risk is bounded by the rejection spike
    # path (rule 1) which is ungated and recovers immediately. Sustained
    # comfort (SCALE_DOWN_COMFORT_S consecutive comfortable ticks) prevents
    # shed off a single quiet tick after a hot hour.
    elif (now - last_change_at) >= SCALE_DOWN_COOLDOWN_S:
        ttft_comfort = slo_comfortably_met(ttft_block, ttft_sig, "ceiling")
        otps_comfort = slo_comfortably_met(otps_block, otps_sig, "floor")
        quiet = is_no_signal(rej) or rej < REJECTION_OK_FLOOR
        if not quiet:
            reason = f"hold-rejection-above-floor rej={rej:.4f}"
            desired = current_replicas
        elif not (ttft_comfort and otps_comfort):
            bad = []
            if not ttft_comfort:
                bad = [f"ttft({','.join(slo_not_comfortable(ttft_block, ttft_sig, 'ceiling'))})"]
            if not otps_comfort:
                bad.append(f"otps({','.join(slo_not_comfortable(otps_block, otps_sig, 'floor'))})")
            desired = current_replicas
            reason = f"hold-not-comfortable {'+'.join(bad)}"
        elif comfortable_since is None or (now - comfortable_since) < SCALE_DOWN_COMFORT_S:
            comfort_elapsed = 0 if comfortable_since is None else int(now - comfortable_since)
            desired = current_replicas
            reason = f"hold-comfort-window {comfort_elapsed}s<{SCALE_DOWN_COMFORT_S}s"
        else:
            step = max(SCALE_DOWN_STEP_MIN, math.ceil(current_replicas * SCALE_DOWN_STEP_FRAC))
            desired = current_replicas - step
            reason = f"scale-down step={step}"
    else:
        desired = current_replicas
        reason = f"hold-cooldown remaining={int(SCALE_DOWN_COOLDOWN_S - (now - last_change_at))}s"

    clamped = max(mn, min(mx, desired))
    if clamped != desired:
        reason += f" clamp[{mn},{mx}]"
    idle_s = max(0, now - last_change_at) if last_change_at else 0
    comfort_s = max(0, now - comfortable_since) if comfortable_since else 0
    log.debug(
        "decide service=%s cur=%d idle=%ds comfort=%ds ttft[%s] otps[%s] rej=%s → est=%d (%s)",
        cr_spec.get("serviceId", "?"), current_replicas, idle_s, comfort_s,
        _fmt_signals(signals.get("ttft"), ttft_block, "ceiling"),
        _fmt_signals(signals.get("otps"), otps_block, "floor"),
        _fmt_val(signals.get("rejection_rate")),
        clamped, reason,
    )
    return clamped, reason


def _fmt_val(v):
    if v is None:
        return "–"
    if isinstance(v, float) and math.isnan(v):
        return "nan"
    return f"{v:g}"


def _fmt_signals(d, block, direction):
    """Format one signal block for the decide log line with a 3-state
    verdict per metric:
      ✗ violated   — crossed threshold in the bad direction
      ~ ok-not-comfy — on the safe side of threshold but not past the
                       comfort margin (no scale-up, but not provable
                       shed either)
      ✓ comfortable — deep on the safe side of the comfort margin
    Missing observed → '–·miss(thr=N,comfort=M)'  (counts as not-comfy).
    No threshold declared for a kind → bare 'kind=value'.

    Comfort margin comes from SLO_HEADROOM (default 0.5):
      ceiling: comfortable iff val < thr × 0.5
      floor:   comfortable iff val > thr / 0.5 = 2× thr
    """
    if not d:
        return "–"
    thresholds = {}
    for m in _metrics_list(block):
        if isinstance(m, dict):
            thresholds[m.get("type")] = m.get("threshold")
    parts = []
    for k in sorted(d.keys()):
        v = d[k]
        thr = thresholds.get(k)
        if thr is None:
            parts.append(f"{k}={_fmt_val(v)}")
            continue
        if direction == "ceiling":
            comfort_threshold = thr * SLO_HEADROOM
            cmp_op = "<"
        else:
            comfort_threshold = thr / SLO_HEADROOM
            cmp_op = ">"
        if is_no_signal(v):
            parts.append(f"{k}=–·miss(thr={thr:g},comfy{cmp_op}{comfort_threshold:g})")
            continue
        violated = (v > thr) if direction == "ceiling" else (v < thr)
        if direction == "ceiling":
            comfortable = v < comfort_threshold
        else:
            comfortable = v > comfort_threshold
        if v < thr:
            truthy = "<"
        elif v > thr:
            truthy = ">"
        else:
            truthy = "="
        if violated:
            v_word, v_thr, v_cmp = "✗", thr, truthy
        elif comfortable:
            v_word, v_thr, v_cmp = "✓", comfort_threshold, ("<" if direction == "ceiling" else ">")
        else:
            v_word, v_thr, v_cmp = "~", comfort_threshold, ("≥" if direction == "ceiling" else "≤")
        parts.append(f"{k}={_fmt_val(v)}{v_cmp}{v_thr:g}{v_word}")
    return " ".join(parts)
