"""`GET /decisions` wire serialization — the only owner of the format.

Any change to what goes on the wire is a one-function diff in this
file and must be explicit in review (D7: unchanged since v1). Everything
else treats decisions as plain (ns, service) → int maps, plus the
additive `shape` block for pdengine services.
"""

API_VERSION = "llmscaling.inference.x-k8s.io/v1alpha1"


def _pd_shape_block(pd_spec, committed_shape):
    """The `shape.pd` block for a pdengine-typed service.

    `pd_spec` is Placement.pd — what the live DS is actually running
    (slices, per-role engines per slice, group sizes). `committed_shape`
    is the pd.PdShape we are recommending this tick. The two are
    always per-slice; n is never moved by the autoscaler.
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


def to_wire(committed, pd_placements=None):
    """{(ns, service): replicas} → the /decisions payload dict.

    `committed` maps to ints for replica/job services, or to
    pd.PdShape for pdengine ones; `pd_placements[k]` is the Placement
    whose `.pd` block the shape was committed against (and the only
    place slices/live-shape metadata comes from). Non-PD rows stay
    byte-identical to v1.
    """
    pd_placements = pd_placements or {}
    rows = []
    for (ns, svc), value in sorted(committed.items()):
        row = {
            "namespace": ns,
            "serviceId": svc,
        }
        placement = pd_placements.get((ns, svc))
        if placement is not None and placement.pd is not None:
            # `value` is the committed PdShape. `active` mirrors today's
            # slice count for replica-observing consumers, with one
            # exception: an all-zero decision (scale-to-zero) reports
            # active 0 so the "off" signal is unambiguous.
            active = placement.pd.slices if value.engines > 0 else 0
            row["replicas"] = {"active": active}
            row["shape"] = {"pd": _pd_shape_block(placement.pd, value)}
        else:
            row["replicas"] = {"active": value}
        rows.append(row)
    return {"apiVersion": API_VERSION, "decisions": rows}
