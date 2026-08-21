"""Fixtures for decision_gen tests.

The controller collaborators (`slo_store`, `k8s_state`, `metrics.Prometheus`)
are replaced by dict-backed fakes. The controller itself runs in-process on
a caller-provided clock so tests are deterministic.
"""

from __future__ import annotations

import pytest


class FakeSLOStore:
    """Dict-backed SLO cache. Matches `slo_store.SLOStore.snapshot()` shape."""

    def __init__(self, specs=None):
        self._specs = dict(specs or {})

    def snapshot(self):
        return dict(self._specs)

    def set(self, namespace, service_id, spec):
        self._specs[(namespace, service_id)] = spec

    def delete(self, namespace, service_id):
        self._specs.pop((namespace, service_id), None)


class FakeProm:
    """Callable-backed Prometheus stub.

    Configure per-signal callbacks:

        prom.ttft_fn = lambda ns, svc, kind: 42.0
        prom.otps_fn = lambda ns, svc, kind: None
        prom.rejection_rate_fn = lambda ns, svc: 0.005
        prom.current_replicas_fn = lambda ns, svc: 2

    Defaults return None (no signal).
    """

    def __init__(self):
        self.ttft_fn = lambda ns, svc, kind: None
        self.otps_fn = lambda ns, svc, kind: None
        self.rejection_rate_fn = lambda ns, svc: None
        self.current_replicas_fn = lambda ns, svc: None
        self.calls = []

    def ttft(self, namespace, service_id, kind):
        self.calls.append(("ttft", namespace, service_id, kind))
        return self.ttft_fn(namespace, service_id, kind)

    def otps(self, namespace, service_id, kind):
        self.calls.append(("otps", namespace, service_id, kind))
        return self.otps_fn(namespace, service_id, kind)

    def rejection_rate(self, namespace, service_id):
        self.calls.append(("rejection_rate", namespace, service_id))
        return self.rejection_rate_fn(namespace, service_id)

    def current_replicas(self, namespace, service_id):
        self.calls.append(("current_replicas", namespace, service_id))
        return self.current_replicas_fn(namespace, service_id)


class FakeK8sState:
    """Stub `K8sState`. Provides `resolve_placement` and `pool_capacity`."""

    def __init__(self):
        self.placements = {}   # {(ns, svc): Placement | None}
        self.capacity = {}     # {pool: allocatable_gpus}
        self.placement_calls = []

    def resolve_placement(self, namespace, service_id):
        self.placement_calls.append((namespace, service_id))
        return self.placements.get((namespace, service_id))

    def pool_capacity(self):
        return dict(self.capacity)


@pytest.fixture
def fake_slo_store():
    return FakeSLOStore()


@pytest.fixture
def fake_prom():
    return FakeProm()


@pytest.fixture
def fake_k8s():
    return FakeK8sState()


@pytest.fixture
def cr_spec():
    """A minimal sane CR spec; override fields per test as needed."""
    return {
        "serviceId": "svc",
        "priority": 5,
        "minimumDeployment": {"type": "replica", "value": 1},
        "maximumDeployment": {"type": "replica", "value": 4},
        "ttft": {"default": {"metrics": [{"type": "p80", "threshold": 20.0}]}},
        "otps": {"default": {"metrics": [{"type": "p80", "threshold": 30.0}]}},
    }
