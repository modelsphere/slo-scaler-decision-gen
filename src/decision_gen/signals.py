"""Prometheus client — the only owner of PromQL templates.

Renamed from `metrics` to mark the narrower role: this is our
signal-semantics boundary, not a general metrics client. Every query
returns a typed `Reading`, never a bare float-or-None. R5-table and
C13 require NaN and absent to reach the decision layer *distinctly*;
collapsing both to None is the v1 ambiguity.

Contract rules (§3 of the requirements; verified against live Prom —
see commit history for curl transcripts):

  - No unit conversions anywhere (I2). CR thresholds and series share
    units by declaration.
  - No label filters beyond `service=` (I1/C11). In particular, never
    `stream="true"` — live exporters omit that label entirely for
    some services. `sum by(le)` aggregates past whatever labels exist.
  - 5m rate windows for TTFT / OTPS sums and histograms (halves
    tick-to-tick variance without making rule-1 sluggish at our 60s
    tick); 2m for rejection.
  - `histogram_quantile` NaN means "no requests in window" — a real
    state, not an error. Zero rows means the series is absent.
  - These two reach the caller as distinct `state` values.
"""

import logging
import math
from collections import namedtuple

import requests

log = logging.getLogger(__name__)

_QUANTILES = {"p50": 0.50, "p80": 0.80, "p90": 0.90, "p95": 0.95, "p99": 0.99}


# One signal observation. `value` is a float when state == 'ok', else
# None. Consumers branch on state first:
#   ok              → have a number
#   no_traffic      → histogram_quantile returned NaN (zero requests
#                     in window). Trustworthy quiescence.
#   missing_series  → 0 rows / transport error / bad shape. Cannot
#                     trust; hold decisions that need this signal.
Reading = namedtuple("Reading", ["value", "state"])


class Signals:
    def __init__(self, base_url, timeout_s=5):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    # ---------- primitive ----------

    def _query(self, promql, label):
        """Instant query → Reading."""
        tag = f"[{label}] " if label else ""
        try:
            resp = requests.get(
                f"{self.base_url}/api/v1/query",
                params={"query": promql},
                timeout=self.timeout_s,
            )
        except requests.RequestException as e:
            log.warning("prom http error | %s%s: %s", tag, promql, e)
            return Reading(None, "missing_series")
        if resp.status_code != 200:
            log.warning("prom %d | %s%s: %s",
                        resp.status_code, tag, promql, resp.text[:200])
            return Reading(None, "missing_series")
        try:
            rows = resp.json()["data"]["result"]
        except (ValueError, KeyError) as e:
            log.warning("prom bad json | %s%s: %s", tag, promql, e)
            return Reading(None, "missing_series")
        if not rows:
            log.debug("prom 0 rows | %s%s", tag, promql)
            return Reading(None, "missing_series")
        if len(rows) > 1:
            log.warning("prom >1 row | %s%s (%d rows)", tag, promql, len(rows))
            return Reading(None, "missing_series")
        try:
            v = float(rows[0]["value"][1])
        except (TypeError, ValueError, KeyError, IndexError) as e:
            log.warning("prom bad value | %s%s: %s", tag, promql, e)
            return Reading(None, "missing_series")

        if math.isnan(v):
            log.debug("prom %s→ NaN (no traffic) | %s", tag, promql)
            return Reading(None, "no_traffic")
        log.debug("prom %s→ %g | %s", tag, v, promql)
        return Reading(v, "ok")

    # ---------- signal templates ----------

    @staticmethod
    def _svc(namespace, service_id):
        return f"{namespace}/{service_id}"

    def _histogram(self, series, namespace, service_id, kind):
        svc = self._svc(namespace, service_id)
        if kind == "avg":
            return (
                f'sum(rate({series}_sum{{service="{svc}"}}[5m]))'
                f' / '
                f'sum(rate({series}_count{{service="{svc}"}}[5m]))'
            )
        q = _QUANTILES[kind]
        return (
            f'histogram_quantile({q}, '
            f'sum by(le)(rate({series}_bucket{{service="{svc}"}}[5m])))'
        )

    def ttft(self, namespace, service_id, kind):
        """TTFT in seconds. kind: 'avg' | 'p50' | 'p80' | ... | 'p99'."""
        return self._query(
            self._histogram("bodylog_ttft_seconds", namespace, service_id, kind),
            label=f"ttft.{kind}",
        )

    def otps(self, namespace, service_id, kind):
        """OTPS in tokens/sec/request. Same return contract as ttft."""
        return self._query(
            self._histogram("bodylog_output_tok_per_second", namespace, service_id, kind),
            label=f"otps.{kind}",
        )

    def rejection_rate(self, namespace, service_id):
        """429s ÷ total requests over 2m, in [0, 1]."""
        svc = self._svc(namespace, service_id)
        return self._query(
            f'sum(rate(openresty_rejected_total{{service="{svc}"}}[2m]))'
            f' / '
            f'clamp_min(sum(rate(bodylog_requests_total{{service="{svc}",backend!="(none)"}}[2m])), 0.001)',
            label="rej",
        )

    def replicas_ready(self, namespace, service_id):
        """Physical ready replicas (bodylog_service_replicas_ready).

        Fractional values from HA exporters are rounded (C15)."""
        svc = self._svc(namespace, service_id)
        r = self._query(
            f'bodylog_service_replicas_ready{{service="{svc}"}}',
            label="replicas",
        )
        if r.state != "ok":
            return r
        return Reading(int(r.value + 0.5), "ok")
