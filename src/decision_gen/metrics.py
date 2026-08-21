"""Prometheus client. Translates high-level signal names into PromQL,
executes queries, returns plain floats.

Only this module knows the PromQL templates. Everything upstream
serves plain values (floats, NaN, or None for "no series").

Templates follow `docs/prometheus_explore.md` §4:
  - TTFT histogram is stream-only (stream="true"); non-stream rows are
    misrecorded exporters-side.
  - Unit for `_seconds` series is seconds; we multiply by 1000 for the
    estimator's millisecond thresholds.
  - `NaN` from histogram_quantile means "no requests in window"; we
    surface it as float('nan') so the caller can distinguish NaN from
    "series absent" (None).
"""

import logging
import math

import requests

log = logging.getLogger(__name__)

_QUANTILES = {"p50": 0.50, "p80": 0.80, "p90": 0.90, "p95": 0.95, "p99": 0.99}


class Prometheus:
    def __init__(self, base_url, timeout_s=5):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def query_scalar(self, promql):
        """Instant query → float | None.

        Returns:
          float            single-row value (may be nan if the PromQL
                           expression itself evaluated to NaN)
          None             0 rows, HTTP error, or unexpected shape
        """
        try:
            resp = requests.get(
                f"{self.base_url}/api/v1/query",
                params={"query": promql},
                timeout=self.timeout_s,
            )
        except requests.RequestException as e:
            log.warning("prom http error for %r: %s", promql, e)
            return None
        if resp.status_code != 200:
            log.warning("prom %d for %r: %s", resp.status_code, promql, resp.text[:200])
            return None
        try:
            rows = resp.json()["data"]["result"]
        except (ValueError, KeyError) as e:
            log.warning("prom bad json for %r: %s", promql, e)
            return None
        if not rows:
            return None
        if len(rows) > 1:
            log.warning("prom >1 row for %r (%d); absent", promql, len(rows))
            return None
        try:
            return float(rows[0]["value"][1])
        except (TypeError, ValueError, KeyError, IndexError) as e:
            log.warning("prom bad value for %r: %s", promql, e)
            return None

    @staticmethod
    def _svc(namespace, service_id):
        return f'{namespace}/{service_id}'

    def _ttft_promql(self, namespace, service_id, kind):
        svc = self._svc(namespace, service_id)
        if kind == "avg":
            return (
                f'sum(rate(bodylog_ttft_seconds_sum{{service="{svc}",stream="true"}}[5m]))'
                f' / '
                f'sum(rate(bodylog_ttft_seconds_count{{service="{svc}",stream="true"}}[5m]))'
            )
        q = _QUANTILES[kind]
        return (
            f'histogram_quantile({q}, '
            f'sum by(le)(rate(bodylog_ttft_seconds_bucket{{service="{svc}",stream="true"}}[5m])))'
        )

    def _otps_promql(self, namespace, service_id, kind):
        svc = self._svc(namespace, service_id)
        if kind == "avg":
            return (
                f'sum(rate(bodylog_output_tok_per_second_sum{{service="{svc}"}}[5m]))'
                f' / '
                f'sum(rate(bodylog_output_tok_per_second_count{{service="{svc}"}}[5m]))'
            )
        q = _QUANTILES[kind]
        return (
            f'histogram_quantile({q}, '
            f'sum by(le)(rate(bodylog_output_tok_per_second_bucket{{service="{svc}"}}[5m])))'
        )

    def ttft(self, namespace, service_id, kind):
        """TTFT for the given kind ('avg' | 'p50' | 'p80' | 'p90' | 'p95' | 'p99').

        Return is **milliseconds**. CR thresholds are in ms; the PromQL
        series is seconds. Returns None on transport failure, NaN on a
        queryable-but-empty series (no requests in window)."""
        v = self.query_scalar(self._ttft_promql(namespace, service_id, kind))
        if v is None:
            return None
        if math.isnan(v):
            return v
        return v * 1000.0

    def otps(self, namespace, service_id, kind):
        """OTPS in tokens/sec/request. Same return contract as ttft()."""
        return self.query_scalar(self._otps_promql(namespace, service_id, kind))

    def rejection_rate(self, namespace, service_id):
        """429s ÷ total requests over 1m, in [0, 1]. None on failure."""
        svc = self._svc(namespace, service_id)
        q = (
            f'sum(rate(openresty_rejected_total{{service="{svc}"}}[1m]))'
            f' / '
            f'clamp_min(sum(rate(bodylog_requests_total{{service="{svc}",backend!="(none)"}}[1m])), 0.001)'
        )
        return self.query_scalar(q)

    def current_replicas(self, namespace, service_id):
        """Live ready replicas (bodylog_service_replicas_ready).
        None on failure or missing series."""
        svc = self._svc(namespace, service_id)
        v = self.query_scalar(f'bodylog_service_replicas_ready{{service="{svc}"}}')
        if v is None or math.isnan(v):
            return None
        return int(v + 0.5)
