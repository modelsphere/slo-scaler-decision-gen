"""HTTP surface. Read-only handlers over the controller snapshot.

Routes:
  GET /decisions                    — full snapshot, ?namespace=, ?serviceId= filters
  GET /api/v1alpha1/decisions       — alias
  GET /healthz                      — liveness (200 once the HTTP surface is up)
  GET /readyz                       — readiness (503 until the controller's first tick)

/readyz returning 503 pre-tick keeps a fresh pod out of the Service
endpoints, so consumers hitting the Service URL don't get the empty
boot snapshot. /decisions itself answers 503 for the same window so
consumers bypassing kube-proxy see "not ready yet" rather than a
well-formed but empty decision list (a real answer that's a lie).
"""

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

log = logging.getLogger(__name__)


def make_handler(snapshot_fn, ready_fn=lambda: True):
    """Factory. snapshot_fn() → decision payload dict. ready_fn() → bool."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "DecisionGen/0.1"

        def do_GET(self):
            path = urlparse(self.path).path.rstrip("/")
            if path in ("/health", "/healthz"):
                self._respond(200, {"status": "ok"})
            elif path == "/readyz":
                if ready_fn():
                    self._respond(200, {"status": "ok"})
                else:
                    self._respond(503, {"status": "not_ready", "reason": "no tick yet"})
            elif path in ("/decisions", "/api/v1alpha1/decisions"):
                if not ready_fn():
                    return self._respond(
                        503, {"status": "not_ready", "reason": "no tick yet"})
                params = parse_qs(urlparse(self.path).query)
                payload = snapshot_fn()
                ns_filter = params.get("namespace", [None])[0]
                svc_filter = params.get("serviceId", [None])[0]
                decisions = [
                    d for d in payload["decisions"]
                    if (ns_filter in (None, d["namespace"])
                        and svc_filter in (None, d["serviceId"]))
                ]
                self._respond(200, {**payload, "decisions": decisions})
            else:
                self._respond(404, {"error": "not found"})

        def _respond(self, status, body):
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, fmt, *args):
            log.info("http %s %s", self.address_string(), fmt % args)

    return Handler


def serve(port, snapshot, ready=lambda: True):
    """Block forever. Binds on all interfaces."""
    server = ThreadingHTTPServer(("0.0.0.0", port), make_handler(snapshot, ready))
    log.info("decision-gen listening on :%d", port)
    try:
        server.serve_forever()
    finally:
        server.server_close()
