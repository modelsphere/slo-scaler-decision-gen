"""SLO threshold semantics: direction, violation, comfort.

The single comparison site for the whole codebase (P1 / arch §3.2).
No `value > threshold` may appear anywhere else; if a caller thinks it
needs one, it belongs here instead.

R2: TTFT is a ceiling (violated when value goes ABOVE threshold);
OTPS is a floor (violated when value drops BELOW). v1 shipped
ceiling-on-both (I3) and green tests didn't notice because they were
written against the same wrong rule. The new `Direction` type makes
that mistake unrepresentable at the call site.

R3: three states, not two. GREY is "on the safe side but not deep
enough" — not a violation, not proof of comfort. Shed paths only
consume COMFORTABLE.

C14: strict comparisons everywhere. `value == threshold` is neither
violation nor comfort.

Reading-state guard: `classify` only accepts values that have been
confirmed OK upstream (signals.Reading.state == "ok"). NaN / absent
never enter these functions; they are *no evidence*, not verdicts.
"""

from enum import Enum


class Direction(Enum):
    CEILING = "ceiling"   # violated iff value > threshold   (e.g. TTFT)
    FLOOR   = "floor"     # violated iff value < threshold   (e.g. OTPS)


class Verdict(Enum):
    VIOLATED = "violated"
    GREY = "grey"
    COMFORTABLE = "comfortable"


# Declared once; adding a metric kind is a table entry, not new branches.
METRIC_DIRECTION = {
    "ttft": Direction.CEILING,
    "otps": Direction.FLOOR,
}


def violated(direction, value, threshold):
    if direction is Direction.CEILING:
        return value > threshold
    return value < threshold


def comfortable(direction, value, threshold, headroom):
    if direction is Direction.CEILING:
        return value < threshold * headroom
    return value > threshold / headroom


def classify(direction, value, threshold, headroom):
    if violated(direction, value, threshold):
        return Verdict.VIOLATED
    if comfortable(direction, value, threshold, headroom):
        return Verdict.COMFORTABLE
    return Verdict.GREY
