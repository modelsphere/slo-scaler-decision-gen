"""Smoke test: server.py serves /healthz, /readyz, /decisions from a snapshot."""

import json
import threading
import urllib.request

from decision_gen import server


def _find_free_port():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start(snapshot):
    port = _find_free_port()
    httpd = server.ThreadingHTTPServer(
        ("127.0.0.1", port), server.make_handler(lambda: snapshot)
    )
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, port


def _get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as resp:
        return resp.status, json.loads(resp.read())


def test_health_and_decisions_on_empty_snapshot():
    snapshot = {"apiVersion": "llmscaling.inference.x-k8s.io/v1alpha1", "decisions": []}
    httpd, port = _start(snapshot)
    try:
        assert _get(port, "/healthz") == (200, {"status": "ok"})
        assert _get(port, "/readyz") == (200, {"status": "ok"})
        status, body = _get(port, "/decisions")
        assert status == 200
        assert body["apiVersion"] == "llmscaling.inference.x-k8s.io/v1alpha1"
        assert body["decisions"] == []
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_decisions_filtering():
    snapshot = {
        "apiVersion": "llmscaling.inference.x-k8s.io/v1alpha1",
        "decisions": [
            {"namespace": "kimi", "serviceId": "kimi-k25", "replicas": {"active": 2}},
            {"namespace": "modelforge", "serviceId": "fallback-modelforge-01",
             "replicas": {"active": 1}},
        ],
    }
    httpd, port = _start(snapshot)
    try:
        _, body = _get(port, "/decisions?namespace=kimi")
        assert [d["serviceId"] for d in body["decisions"]] == ["kimi-k25"]

        _, body = _get(port, "/decisions?serviceId=fallback-modelforge-01")
        assert [d["namespace"] for d in body["decisions"]] == ["modelforge"]

        _, body = _get(port, "/api/v1alpha1/decisions")
        assert len(body["decisions"]) == 2
    finally:
        httpd.shutdown()
        httpd.server_close()
