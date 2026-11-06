"""A stand-in decision-gen that serves `/decisions` from a local file.

Dev only: it is not in the package or the image.

For clusters where the real decision server cannot size a service, such as a
mock model whose pods request no GPU and so resolve to no placement. The HTTP
surface and the wire format are the real server's (server.py, snapshot.py);
only where the counts come from differs.

The file is checked on every request and re-read when it changed, so an edit
is served from the next request on. An edit that does not parse is logged and
the last good decisions stay served. Until one good read, /decisions and
/readyz answer 503.

  decisions:
    - namespace: models
      serviceId: glm-53
      replicas: 2

Run it from a checkout with decision-gen installed (make dev):

  python3 hack/mock_decision_gen.py [path]

The path defaults to $MOCK_CONFIG, then mock.yaml beside this script; PORT to
8080. In a cluster, the decision-gen image has everything it imports: mount this
file and the config from a ConfigMap and run it with python3.
"""

import logging
import os
import sys
import threading

import yaml

from decision_gen import snapshot as wire
from decision_gen.server import serve

log = logging.getLogger("decision_gen.mock")

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mock.yaml")
_MISSING = "missing"


class ConfigError(ValueError):
    pass


def parse(text):
    """The file's text → {(namespace, serviceId): replicas}."""
    doc = yaml.safe_load(text)
    if doc is None:
        return {}
    if not isinstance(doc, dict):
        raise ConfigError("the file must be a mapping with a `decisions` list")
    _no_unknown(doc, {"decisions"}, "the file")
    entries = doc.get("decisions")
    if entries is None:
        return {}
    if not isinstance(entries, list):
        raise ConfigError("decisions must be a list")
    out = {}
    for i, e in enumerate(entries):
        where = f"decisions[{i}]"
        if not isinstance(e, dict):
            raise ConfigError(f"{where} must be a mapping")
        _no_unknown(e, {"namespace", "serviceId", "replicas"}, where)
        ns, svc, n = e.get("namespace"), e.get("serviceId"), e.get("replicas")
        for field, v in (("namespace", ns), ("serviceId", svc)):
            if not isinstance(v, str) or not v:
                raise ConfigError(f"{where}.{field} must be a non-empty string")
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ConfigError(f"{where}.replicas must be an integer >= 0")
        if (ns, svc) in out:
            raise ConfigError(f"{where}: {ns}/{svc} is listed twice")
        out[(ns, svc)] = n
    return out


def _no_unknown(mapping, allowed, where):
    unknown = set(mapping) - allowed
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {', '.join(sorted(map(str, unknown)))}")


class FileDecisions:
    """The decisions in one file, re-read whenever it changes."""

    def __init__(self, path):
        self.path = path
        self._lock = threading.Lock()
        # What the file looked like at the last read, good or bad, so a broken
        # file is read and logged once rather than on every request.
        self._seen = None
        self._committed = None      # last good read; None until there is one

    def ready(self):
        with self._lock:
            self._refresh()
            return self._committed is not None

    def snapshot(self):
        with self._lock:
            self._refresh()
            return wire.to_wire(self._committed or {})

    def _refresh(self):
        try:
            st = os.stat(self.path)
        except OSError as e:
            if self._seen != _MISSING:
                self._seen = _MISSING
                log.warning("%s: %s; %s", self.path, e.strerror, self._keeping())
            return
        # A ConfigMap volume swaps a symlink, which changes the inode.
        seen = (st.st_mtime_ns, st.st_size, st.st_ino)
        if seen == self._seen:
            return
        self._seen = seen
        try:
            with open(self.path, encoding="utf-8") as f:
                committed = parse(f.read())
        except (OSError, UnicodeDecodeError, yaml.YAMLError, ConfigError) as e:
            log.error("%s: %s; %s", self.path, e, self._keeping())
            return
        self._committed = committed
        log.info("%s: %s", self.path, "; ".join(
            f"{ns}/{svc}={n}" for (ns, svc), n in sorted(committed.items())) or "no decisions")

    def _keeping(self):
        if self._committed is None:
            return "not ready until it reads"
        return "serving the last good decisions"


def main(argv=None):
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    argv = sys.argv[1:] if argv is None else argv
    path = argv[0] if argv else os.environ.get("MOCK_CONFIG", DEFAULT_CONFIG)
    port = int(os.environ.get("PORT", 8080))

    decisions = FileDecisions(path)
    decisions.ready()       # read once now, so the log shows the starting state
    serve(port=port, snapshot=decisions.snapshot, ready=decisions.ready,
          not_ready_reason=f"no valid decisions file at {path}")


if __name__ == "__main__":
    main()
