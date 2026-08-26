"""Tests for decision_gen.slo_store — watch stream -> snapshot."""

import threading
import time
from unittest.mock import MagicMock

import pytest

from decision_gen.slo_store import SLOStore


def _cr_event(etype, namespace, service_id, rv=1, extra_spec=None):
    spec = {"serviceId": service_id, "priority": 5}
    if extra_spec:
        spec.update(extra_spec)
    return {
        "type": etype,
        "object": {
            "apiVersion": "inference.x-k8s.io/v1alpha1",
            "kind": "LLMSLORequirement",
            "metadata": {
                "name": service_id,
                "namespace": namespace,
                "resourceVersion": str(rv),
            },
            "spec": spec,
        },
    }


class FakeWatch:
    """Scriptable watch that replays a list of event batches.

    Each batch is iterated; the stream() generator ends after the last
    batch is exhausted (simulating stream close / timeout)."""

    def __init__(self, batches):
        self.batches = list(batches)
        self._stopped = False

    def stream(self, *args, **kwargs):
        for batch in self.batches:
            for ev in batch:
                if self._stopped:
                    return
                yield ev

    def stop(self):
        self._stopped = True


def _wait_for(pred, timeout_s=2.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return False


# ---------- tests ----------

def test_synced_set_on_bookmark():
    """Canonical sync signal. Watch streams ADDED, ADDED, ..., BOOKMARK.
    Sync must not fire until the bookmark: the bug this test guards is
    'first tick sees only 1 of N CRs because the watch was mid-list'."""
    w = FakeWatch([[
        _cr_event("ADDED", "kimi", "kimi-k25"),
        _cr_event("ADDED", "modelforge", "fallback-modelforge-01"),
        {"type": "BOOKMARK", "object": {}},
    ]])
    store = SLOStore(api=MagicMock(), watch_factory=lambda: w)
    assert not store.synced()
    store.start()
    try:
        assert _wait_for(store.synced)
        # Both CRs must be visible when synced fires.
        assert len(store.snapshot()) == 2
        assert store.wait_synced(timeout=0) is True
    finally:
        store.stop()


def test_not_synced_until_bookmark():
    """Specifically: first ADDED alone doesn't sync. The race this test
    pins produced `tick start: 1 CRs` instead of 2."""

    class BlockingWatch:
        """Yields the first ADDED, then blocks until the test releases it,
        then yields the rest."""
        def __init__(self):
            self.release = threading.Event()

        def stream(self, *a, **kw):
            yield _cr_event("ADDED", "kimi", "kimi-k25")
            self.release.wait(2)
            yield _cr_event("ADDED", "modelforge", "fallback-modelforge-01")
            yield {"type": "BOOKMARK", "object": {}}

        def stop(self):
            self.release.set()   # unblock so the thread can exit

    w = BlockingWatch()
    store = SLOStore(api=MagicMock(), watch_factory=lambda: w)
    store.start()
    try:
        assert _wait_for(lambda: len(store.snapshot()) == 1)
        assert not store.synced(), "synced too early — first ADDED set it"
        w.release.set()
        assert _wait_for(store.synced)
        assert len(store.snapshot()) == 2
    finally:
        store.stop()


def test_synced_set_on_bookmark_no_crs():
    """Empty CR set: just BOOKMARK. Watch has still synced — 'no CRs exist'
    is no longer ambiguous with 'haven't seen the list yet'."""
    w = FakeWatch([[{"type": "BOOKMARK", "object": {}}]])
    store = SLOStore(api=MagicMock(), watch_factory=lambda: w)
    assert not store.synced()
    store.start()
    try:
        assert _wait_for(store.synced)
        assert store.snapshot() == {}
    finally:
        store.stop()


def test_synced_on_modified_without_bookmark():
    """Graceful fallback for servers that don't honor allow_watch_bookmarks:
    a non-ADDED event necessarily means the initial list has ended."""
    w = FakeWatch([[
        _cr_event("ADDED", "kimi", "kimi-k25"),
        _cr_event("MODIFIED", "kimi", "kimi-k25", rv=2),
    ]])
    store = SLOStore(api=MagicMock(), watch_factory=lambda: w)
    store.start()
    try:
        assert _wait_for(store.synced)
    finally:
        store.stop()


def test_added_then_snapshot():
    w = FakeWatch([[
        _cr_event("ADDED", "kimi", "kimi-k25"),
        _cr_event("ADDED", "modelforge", "fallback-modelforge-01"),
    ]])
    store = SLOStore(api=MagicMock(), watch_factory=lambda: w)
    store.start()
    try:
        assert _wait_for(lambda: len(store.snapshot()) == 2)
        snap = store.snapshot()
        assert ("kimi", "kimi-k25") in snap
        assert ("modelforge", "fallback-modelforge-01") in snap
        assert snap[("kimi", "kimi-k25")]["priority"] == 5
    finally:
        store.stop()


def test_modified_overwrites():
    w = FakeWatch([[
        _cr_event("ADDED",    "kimi", "kimi-k25", extra_spec={"priority": 5}),
        _cr_event("MODIFIED", "kimi", "kimi-k25", rv=2, extra_spec={"priority": 9}),
    ]])
    store = SLOStore(api=MagicMock(), watch_factory=lambda: w)
    store.start()
    try:
        assert _wait_for(
            lambda: store.snapshot().get(("kimi", "kimi-k25"), {}).get("priority") == 9
        )
    finally:
        store.stop()


def test_deleted_removes():
    w = FakeWatch([[
        _cr_event("ADDED",   "kimi", "kimi-k25"),
        _cr_event("DELETED", "kimi", "kimi-k25"),
    ]])
    store = SLOStore(api=MagicMock(), watch_factory=lambda: w)
    store.start()
    try:
        assert _wait_for(lambda: len(store.snapshot()) == 0)
    finally:
        store.stop()


def test_multiple_namespaces_independent():
    w = FakeWatch([[
        _cr_event("ADDED", "kimi", "kimi-k25"),
        _cr_event("ADDED", "modelforge", "fallback-modelforge-01"),
        _cr_event("DELETED", "kimi", "kimi-k25"),
    ]])
    store = SLOStore(api=MagicMock(), watch_factory=lambda: w)
    store.start()
    try:
        assert _wait_for(lambda: len(store.snapshot()) == 1)
        assert ("modelforge", "fallback-modelforge-01") in store.snapshot()
    finally:
        store.stop()


def test_empty_initial_snapshot():
    w = FakeWatch([[]])
    store = SLOStore(api=MagicMock(), watch_factory=lambda: w)
    store.start()
    try:
        time.sleep(0.05)
        assert store.snapshot() == {}
    finally:
        store.stop()


def test_error_event_triggers_reconnect():
    """ERROR event raises inside the watcher; the loop reconnects."""
    batches = [
        [_cr_event("ADDED", "kimi", "kimi-k25")],
        [{"type": "ERROR", "object": {"message": "gone"}}],
        [_cr_event("ADDED", "kimi", "k2", extra_spec={"priority": 7})],
    ]
    w = FakeWatch(batches)
    store = SLOStore(api=MagicMock(), watch_factory=lambda: w)
    store.start()
    try:
        # Even after the ERROR, the next ADDED should eventually land.
        # The reconnect loop calls stream again, which starts over from
        # the first batch since FakeWatch just replays.
        assert _wait_for(lambda: ("kimi", "kimi-k25") in store.snapshot())
    finally:
        store.stop()


def test_bookmark_ignored():
    w = FakeWatch([[
        _cr_event("ADDED", "kimi", "kimi-k25"),
        {"type": "BOOKMARK", "object": {}},
    ]])
    store = SLOStore(api=MagicMock(), watch_factory=lambda: w)
    store.start()
    try:
        assert _wait_for(lambda: len(store.snapshot()) == 1)
    finally:
        store.stop()


def test_missing_service_id_skipped(caplog):
    w = FakeWatch([[
        {"type": "ADDED", "object": {
            "metadata": {"namespace": "kimi", "name": "no-service-id"},
            "spec": {},
        }},
    ]])
    store = SLOStore(api=MagicMock(), watch_factory=lambda: w)
    store.start()
    try:
        time.sleep(0.05)
        assert store.snapshot() == {}
    finally:
        store.stop()
