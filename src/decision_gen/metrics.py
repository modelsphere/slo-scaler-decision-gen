"""Prometheus client. Translates high-level signal names into PromQL,
executes queries, returns plain floats.

Only this module knows the PromQL templates. Everything upstream
serves plain values (floats, NaN, or None for "no series").

Templates follow `docs/prometheus_explore.md` §4, with one deviation:
  - We do NOT filter on the `stream` label. Live Prom shows kimi and
    modelforge emit TTFT buckets with no `stream` label at all (their
    exporters omit it), and `sum by(le)` already aggregates past any
    label that is present. Filtering on stream="true" therefore
    returned 0 rows for both services. See commit that lands this
    comment for the curl that proved it.
  - Units are passed through unchanged. `bodylog_ttft_seconds_*` is in
    seconds, `bodylog_output_tok_per_second_*` is tokens/s, rejection
    is a fraction in [0,1], replicas are a count. CR thresholds are
    declared in the same unit as the series they name — no conversion
    anywhere in this module.
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

    def query_scalar(self, promql, label=""):
        """Instant query → float | None.

        label: short tag (e.g. "ttft.p80") prepended to log lines so the
        result isn't an anonymous number.

        Returns:
          float            single-row value (may be nan if the PromQL
                           expression itself evaluated to NaN)
          None             0 rows, HTTP error, or unexpected shape
        """
        label_prefix = f"[{label}] " if label else ""
        try:
            resp = requests.get(
                f"{self.base_url}/api/v1/query",
                params={"query": promql},
                timeout=self.timeout_s,
            )
        except requests.RequestException as e:
            log.warning("prom http error | %s%s: %s", label_prefix, promql, e)
            return None
        if resp.status_code != 200:
            log.warning("prom %d | %s%s: %s", resp.status_code, label_prefix, promql, resp.text[:200])
            return None
        try:
            rows = resp.json()["data"]["result"]
        except (ValueError, KeyError) as e:
            log.warning("prom bad json | %s%s: %s", label_prefix, promql, e)
            return None
        if not rows:
            log.debug("prom 0 rows | %s%s", label_prefix, promql)
            return None
        if len(rows) > 1:
            log.warning("prom >1 row | %s%s (%d rows)", label_prefix, promql, len(rows))
            return None
        try:
            v = float(rows[0]["value"][1])
        except (TypeError, ValueError, KeyError, IndexError) as e:
            log.warning("prom bad value | %s%s: %s", label_prefix, promql, e)
            return None
        log.debug("prom %s→ %g | %s", label_prefix, v, promql)
        return v

    @staticmethod
    def _svc(namespace, service_id):
        return f'{namespace}/{service_id}'

    def _ttft_promql(self, namespace, service_id, kind):
        svc = self._svc(namespace, service_id)
        if kind == "avg":
            return (
                f'sum(rate(bodylog_ttft_seconds_sum{{service="{svc}"}}[5m]))'
                f' / '
                f'sum(rate(bodylog_ttft_seconds_count{{service="{svc}"}}[5m]))'
            )
        q = _QUANTILES[kind]
        return (
            f'histogram_quantile({q}, '
            f'sum by(le)(rate(bodylog_ttft_seconds_bucket{{service="{svc}"}}[5m])))'
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
        """TTFT in **seconds**, for the given kind ('avg' | 'p50' | ... | 'p99').

        Returns None on transport failure, NaN on a queryable-but-empty
        series (no requests in window). The PromQL series is seconds;
        CR thresholds are seconds; no conversion."""
        return self.query_scalar(
            self._ttft_promql(namespace, service_id, kind),
            label=f"ttft.{kind}",
        )

    def otps(self, namespace, service_id, kind):
        """OTPS in tokens/sec/request. Same return contract as ttft()."""
        return self.query_scalar(
            self._otps_promql(namespace, service_id, kind),
            label=f"otps.{kind}",
        )

    def rejection_rate(self, namespace, service_id):
        """429s ÷ total requests over 2m, in [0, 1]. None on failure.

        2m window: halves the tick-to-tick variance of a 1m window
        without making rule-1 (rejection spike) feel sluggish at our
        60s tick. See estimator for the rule this feeds."""
        svc = self._svc(namespace, service_id)
        q = (
            f'sum(rate(openresty_rejected_total{{service="{svc}"}}[2m]))'
            f' / '
            f'clamp_min(sum(rate(bodylog_requests_total{{service="{svc}",backend!="(none)"}}[2m])), 0.001)'
        )
        return self.query_scalar(q, label="rej")

    def current_replicas(self, namespace, service_id):
        """Live ready replicas (bodylog_service_replicas_ready).
        None on failure or missing series."""
        svc = self._svc(namespace, service_id)
        v = self.query_scalar(
            f'bodylog_service_replicas_ready{{service="{svc}"}}',
            label="replicas",
        )
        if v is None or math.isnan(v):
            return None
        return int(v + 0.5)
