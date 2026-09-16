"""`GET /decisions` wire serialization — the only owner of the format.

Any change to what goes on the wire is a one-function diff in this
file and must be explicit in review (D7: unchanged since v1). Everything
else treats decisions as plain (ns, service) → int maps.
"""

API_VERSION = "llmscaling.inference.x-k8s.io/v1alpha1"


def to_wire(committed):
    """{(ns, service): replicas} → the /decisions payload dict."""
    return {
        "apiVersion": API_VERSION,
        "decisions": [
            {
                "namespace": ns,
                "serviceId": svc,
                "replicas": {"active": n},
            }
            for (ns, svc), n in sorted(committed.items())
        ],
    }
