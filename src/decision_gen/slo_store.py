"""In-memory cache of LLMSLORequirement CRs.

Watches the CRD server-side and maintains a thread-safe snapshot. The
controller only calls `snapshot()`; no watch callbacks fire into
controller code.

Reconnect policy: capped exponential backoff (1s → 60s). Cached CRs
stay usable across reconnects. BOOKMARK / ERROR events are tolerated;
ERROR triggers a reconnect.
"""

import logging
import threading
import time

from kubernetes import client, watch

log = logging.getLogger(__name__)


GROUP = "inference.x-k8s.io"
VERSION = "v1alpha1"
PLURAL = "llmslorequirements"

_BACKOFF_INITIAL = 1.0
_BACKOFF_MAX = 60.0


class SLOStore:
    def __init__(self, api=None, watch_factory=None):
        """Dependency-injectable for tests.

        `api`            kubernetes.client.CustomObjectsApi (constructed
                         on first start() if None).
        `watch_factory`  callable returning a kubernetes.watch.Watch
                         (or compatible). Tests pass a fake that yields
                         a scripted event sequence.
        """
        self._lock = threading.Lock()
        self._crs = {}   # (namespace, service_id) -> {spec dict, resource_version}

        self._api = api
        self._watch_factory = watch_factory
        self._stop = threading.Event()
        self._synced = threading.Event()    # set after first watch event of any kind
        self._thread = None
        self._started = False

    # ---------- lifecycle ----------

    def start(self):
        if self._started:
            return
        self._started = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="slo-store",
        )
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    # ---------- consumer API ----------

    def snapshot(self):
        """Point-in-time copy: {(ns, service_id): spec}."""
        with self._lock:
            return {k: v["spec"] for k, v in self._crs.items()}

    def get(self, namespace, service_id):
        with self._lock:
            e = self._crs.get((namespace, service_id))
            return e["spec"] if e else None

    def synced(self):
        return self._synced.is_set()

    def wait_synced(self, timeout):
        """Block until the watch has delivered its first event, or `timeout`
        seconds elapse. Returns True iff synced (False = hit the timeout).
        Any event counts — BOOKMARK on an empty CR set is still contact with
        the API server and is enough to know "the world I'm about to tick
        over isn't just pre-watch emptiness"."""
        return self._synced.wait(timeout)

    # ---------- watcher ----------

    def _get_api(self):
        if self._api is None:
            t = time.monotonic()
            self._api = client.CustomObjectsApi()
            log.info("k8s api init: CustomObjectsApi(slo) %.3fs", time.monotonic() - t)
        return self._api

    def _make_watch(self):
        if self._watch_factory is not None:
            return self._watch_factory()
        return watch.Watch()

    def _run(self):
        """Event loop with reconnect. Errors only WARN-log; we keep retrying."""
        backoff = _BACKOFF_INITIAL
        while not self._stop.is_set():
            try:
                self._stream_once()
                backoff = _BACKOFF_INITIAL   # clean exit → reset
            except Exception:
                log.exception("slo_store watch errored; reconnecting")
            if self._stop.is_set():
                break
            self._stop.wait(backoff)
            backoff = min(_BACKOFF_MAX, backoff * 2)

    def _stream_once(self):
        """One watch session. Yields until the stream ends or errors."""
        api = self._get_api()
        w = self._make_watch()
        t_first_event = time.monotonic()
        n_events = 0
        try:
            for event in w.stream(
                api.list_cluster_custom_object,
                GROUP, VERSION, PLURAL,
                # Ask for a BOOKMARK after the initial list — that's the
                # canonical "list complete" signal we set `_synced` on.
                allow_watch_bookmarks=True,
                timeout_seconds=300,
            ):
                if n_events == 0:
                    log.info("slo_store first event: %.3fs", time.monotonic() - t_first_event)
                n_events += 1
                if self._stop.is_set():
                    break
                self._apply(event)
        finally:
            try:
                w.stop()
            except Exception:
                pass

    def _apply(self, event):
        etype = event.get("type") if isinstance(event, dict) else None
        obj = event.get("object") if isinstance(event, dict) else None
        # `_synced` means the initial watch list has fully arrived. ADDED events
        # during the list are still arriving — setting synced on the first one
        # was the bug. Signal: BOOKMARK (canonical k8s ≥1.15 marker), or any
        # non-ADDED type (MODIFIED/DELETED/ERROR can only fire after the list
        # has been delivered → list is a subset of what we've already applied).
        if etype == "BOOKMARK":
            if not self._synced.is_set():
                log.info("slo_store initial list complete (bookmark)")
            self._synced.set()
            return
        if etype == "ERROR":
            if not self._synced.is_set():
                log.info("slo_store initial list ended (ERROR event; treating as synced)")
            self._synced.set()
        elif etype in ("MODIFIED", "DELETED") and not self._synced.is_set():
            log.info("slo_store initial list ended (first %s event; treating as synced)", etype)
            self._synced.set()

        if etype in ("ADDED", "MODIFIED"):
            if not isinstance(obj, dict):
                return
            md = obj.get("metadata", {})
            spec = obj.get("spec", {}) or {}
            ns = md.get("namespace")
            svc = spec.get("serviceId")
            rv = md.get("resourceVersion")
            if not ns or not svc:
                log.warning("slo_store: skipping CR missing ns/serviceId: %r", obj)
                return
            with self._lock:
                self._crs[(ns, svc)] = {"spec": spec, "resource_version": rv}
        elif etype == "DELETED":
            md = obj.get("metadata", {}) if isinstance(obj, dict) else {}
            spec = obj.get("spec", {}) if isinstance(obj, dict) else {}
            ns, svc = md.get("namespace"), spec.get("serviceId")
            if ns and svc:
                with self._lock:
                    self._crs.pop((ns, svc), None)
        elif etype == "ERROR":
            log.warning("slo_store: watch ERROR event: %r", obj)
            raise RuntimeError("watch ERROR event")
        # BOOKMARK / unknowns: ignore.
