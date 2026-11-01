"""`GET /decisions` wire serialization — the only owner of the format.

Any change to what goes on the wire is a one-function diff in this
file and must be explicit in review (D7: unchanged since v1). Everything
else treats decisions as plain (ns, service) → int maps, plus the
additive `shape` block for pdengine services.
"""

import logging

API_VERSION = "llmscaling.inference.x-k8s.io/v1alpha1"

log = logging.getLogger(__name__)


def _is_pd(value):
    return hasattr(value, "prefill") and hasattr(value, "decode")


def _pd_shape_block(pd_spec, committed_shape):
    """The `shape.pd` block for a pdengine-typed service.

    `pd_spec` is what the DS is actually running (slices, per-role
    engines per slice, group sizes). `committed_shape` is the
    pd.PdShape we are recommending this tick. The two are always
    per-slice; n is never moved by the autoscaler.
    """
    return {
        "slices": pd_spec.slices,
        "prefill": {
            "replicas": committed_shape.prefill,
            "liveReplicas": pd_spec.prefill,
        },
        "decode": {
            "replicas": committed_shape.decode,
            "liveReplicas": pd_spec.decode,
        },
    }


def to_wire(committed, pd_specs=None):
    """{(ns, service): replicas} → the /decisions payload dict.

    `committed` maps to ints for replica/job services, or to
    pd.PdShape for pdengine ones. The row's currency is decided
    from the VALUE's type — a `PdShape` committed renders the
    `shape.pd` block using the PDSpec the controller last
    resolved for the service (`pd_specs[key]`); an int stays
    byte-identical to v1. A PdShape-committed row with no
    matching PDSpec is omitted from the wire, not fake-filled —
    the controller's `_pd_specs` bookkeeping is the only place
    that should ever let that happen."""
    pd_specs = pd_specs or {}
    rows = []
    for (ns, svc), value in sorted(committed.items()):
        row = {
            "namespace": ns,
            "serviceId": svc,
        }
        if _is_pd(value):
            pd_spec = pd_specs.get((ns, svc))
            if pd_spec is None:
                log.error(
                    "%s/%s: pdengine committed but no resolved PDSpec; "
                    "omitting decision row rather than emit a null block",
                    ns, svc,
                )
                continue
            # `active` mirrors today's slice count for replica-
            # observing consumers, with one exception: an all-zero
            # decision (scale-to-zero) reports active 0 so the
            # "off" signal is unambiguous.
            active = (pd_spec.slices) if value.engines > 0 else 0
            row["replicas"] = {"active": active}
            row["shape"] = {"pd": _pd_shape_block(pd_spec, value)}
        else:
            row["replicas"] = {"active": value}
        rows.append(row)
    return {"apiVersion": API_VERSION, "decisions": rows}
