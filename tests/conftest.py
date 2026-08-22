"""Fixtures for decision_gen tests.

Controller collaborators (`slo_store`, `k8s_state`, `signals.Signals`)
are replaced by fakes. The controller runs in-process on a caller-
provided clock so tests are deterministic.
"""

import pytest

from decision_gen.signals import Reading


class FakeSLOStore:
    """Dict-backed SLO cache. Matches `slo_store.SLOStore.snapshot()`."""

    def __init__(self, specs=None):
        self._specs = dict(specs or {})

    def snapshot(self):
        return dict(self._specs)

    def set(self, namespace, service_id, spec):
        self._specs[(namespace, service_id)] = spec

    def delete(self, namespace, service_id):
        self._specs.pop((namespace, service_id), None)


class FakeSignals:
    """Returns `Reading` named-tuples. Defaults to missing_series so a
    test that doesn't configure a signal exercises the hold path."""

    def __init__(self):
        self.ttft_fn = {}
        self.otps_fn = {}
        self.rejection_val = None
        self.replicas_ready_val = None
        self.calls = []

    def _wrap(self, v):
        if isinstance(v, Reading):
            return v
        return Reading(v, "ok") if v is not None else Reading(None, "missing_series")

    def ttft(self, ns, svc, kind):
        self.calls.append(("ttft", ns, svc, kind))
        return self._wrap(self.ttft_fn.get((ns, svc, kind)))

    def otps(self, ns, svc, kind):
        self.calls.append(("otps", ns, svc, kind))
        return self._wrap(self.otps_fn.get((ns, svc, kind)))

    def rejection_rate(self, ns, svc):
        self.calls.append(("rejection_rate", ns, svc))
        return self._wrap(self.rejection_val)

    def replicas_ready(self, ns, svc):
        self.calls.append(("replicas_ready", ns, svc))
        return self._wrap(self.replicas_ready_val)

    def set_ttft(self, ns, svc, kind, value):
        self.ttft_fn[(ns, svc, kind)] = value

    def set_otps(self, ns, svc, kind, value):
        self.otps_fn[(ns, svc, kind)] = value

    def set_rejection(self, value):
        self.rejection_val = value

    def set_replicas_ready(self, value):
        self.replicas_ready_val = value


class FakeK8sState:
    def __init__(self):
        self.placements = {}   # {(ns, svc): Placement | None}
        self.capacity = {}     # {pool: allocatable_gpus}

    def resolve_placement(self, namespace, service_id):
        return self.placements.get((namespace, service_id))

    def pool_capacity(self):
        return dict(self.capacity)


class FakeClock:
    """Deterministic now(). Tests advance() between ticks."""

    def __init__(self, start=1_000_000.0):
        self.t = start

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


@pytest.fixture
def fake_slo():
    return FakeSLOStore()


@pytest.fixture
def fake_signals():
    return FakeSignals()


@pytest.fixture
def fake_k8s():
    return FakeK8sState()


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def cr_spec():
    return {
        "serviceId": "svc",
        "priority": 5,
        "minimumDeployment": {"type": "replica", "value": 1},
        "maximumDeployment": {"type": "replica", "value": 8},
        "ttft": {"default": {"metrics": [{"type": "p80", "threshold": 20.0}]}},
        "otps": {"default": {"metrics": [{"type": "p80", "threshold": 30.0}]}},
    }
