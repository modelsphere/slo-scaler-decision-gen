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
    some services.
  - TTFT / OTPS quantiles read the NATIVE histogram form of the series
    (`histogram_quantile(q, sum(rate(series{...}[5m])))`), not classic
    `_bucket` + `by(le)`. Classic buckets cap at le=10s / le=400 tok/s —
    exactly the overload region where the reading matters most — and
    overestimate by ~30-40% at p80 (measured live 2026-08-26: TTFT
    classic 0.77s vs native 0.54s; OTPS classic 320 vs native 249 t/s).
    `avg` still uses `_sum` / `_count` (exposed identically by both forms).
  - OTPS quantile inversion: the metric is a FLOOR — CR 'p80' means
    "80% of requests must exceed the threshold speed", i.e. violated by
    the SLOW tail. On a tokens/sec histogram that is quantile 1−0.8=0.2,
    not 0.8 (p80 of tok/s is the fast head). TTFT is a CEILING and uses
    the quantile as declared.
  - `replicas_ready` uses `max by(service)(avg_over_time(raw[2m]))`:
    the `avg_over_time(2m)` smooths over scrape gaps — observed live
    2026-08-26: instant vector of one service vanished entirely during
    drain while 2m-avg still returned a value, so a missing instant
    doesn't knock `physical` to None and lose IN_DRAIN detection.
    `max by(service)` then collapses pod/instance/route so a second
    exporter pod (HA or rolling restart) can't produce >1 row, which
    this client treats as missing_series.
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
            log.info("prom 0 rows | %s%s", tag, promql)
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
            log.info("prom %s→ NaN (no traffic) | %s", tag, promql)
            return Reading(None, "no_traffic")
        log.info("prom %s→ %g | %s", tag, v, promql)
        return Reading(v, "ok")

    # ---------- signal templates ----------

    @staticmethod
    def _svc(namespace, service_id):
        return f"{namespace}/{service_id}"

    def _histogram(self, series, namespace, service_id, kind, invert=False):
        svc = self._svc(namespace, service_id)
        if kind == "avg":
            return (
                f'sum(rate({series}_sum{{service="{svc}"}}[5m]))'
                f' / '
                f'sum(rate({series}_count{{service="{svc}"}}[5m]))'
            )
        q = _QUANTILES[kind]
        if invert:
            q = 1.0 - q
        # Native histogram: no _bucket suffix, no `by(le)` — bucketless
        # form, no le-cap distortion. See module docstring.
        return (
            f'histogram_quantile({q}, '
            f'sum(rate({series}{{service="{svc}"}}[5m])))'
        )

    def ttft(self, namespace, service_id, kind):
        """TTFT in seconds (ceiling). kind: 'avg' | 'p50' | ... | 'p99'."""
        return self._query(
            self._histogram("bodylog_ttft_seconds", namespace, service_id, kind),
            label=f"ttft.{kind}",
        )

    def otps(self, namespace, service_id, kind):
        """OTPS in tokens/sec/request (floor). A CR 'pN' declares N% of
        requests must beat the threshold — the slow tail — so the native
        quantile is inverted: p80 → 0.2. 'avg' is unaffected."""
        return self._query(
            self._histogram("bodylog_output_tok_per_second", namespace, service_id, kind,
                            invert=True),
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

        `avg_over_time(2m)` smooths over single-scrape gaps (an instant
        vector can vanish for one 30s scrape during exporter restarts,
        and None here turns off IN_DRAIN detection in direction.phase_of).
        `max by(service)` then collapses pod/instance/route so HA or
        rolling restarts of the exporter can't produce >1 series. Values
        are rounded after the collapse (C15)."""
        svc = self._svc(namespace, service_id)
        r = self._query(
            f'max by(service)(avg_over_time('
            f'bodylog_service_replicas_ready{{service="{svc}"}}[2m]))',
            label="replicas",
        )
        if r.state != "ok":
            return r
        return Reading(int(r.value + 0.5), "ok")
