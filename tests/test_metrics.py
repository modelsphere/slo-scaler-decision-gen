"""Tests for decision_gen.metrics — PromQL templates and return shapes."""

from unittest.mock import MagicMock, patch

from decision_gen.metrics import Prometheus


def _prom():
    return Prometheus(base_url="http://prom:9090", timeout_s=1)


def _resp(rows, status=200):
    r = MagicMock()
    r.status_code = status
    r.text = ""
    r.json.return_value = {"status": "success", "data": {"resultType": "vector", "result": rows}}
    return r


def _row(value):
    return {"metric": {}, "value": [1700000000, str(value)]}


# ---------- PromQL template shape ----------

def test_ttft_avg_template():
    with patch("decision_gen.metrics.requests.get", return_value=_resp([_row(0.25)])) as get:
        v = _prom().ttft("kimi", "kimi-k25", "avg")
    assert v == 250.0
    q = get.call_args.kwargs["params"]["query"]
    assert 'service="kimi/kimi-k25"' in q
    assert 'stream="true"' in q
    assert 'bodylog_ttft_seconds_sum' in q
    assert 'bodylog_ttft_seconds_count' in q
    assert '[5m]' in q


def test_ttft_percentile_template():
    with patch("decision_gen.metrics.requests.get", return_value=_resp([_row(0.035)])) as get:
        v = _prom().ttft("kimi", "kimi-k25", "p95")
    assert v == 35.0
    q = get.call_args.kwargs["params"]["query"]
    assert q.startswith("histogram_quantile(0.95,")
    assert 'bodylog_ttft_seconds_bucket' in q
    assert 'sum by(le)' in q
    assert 'service="kimi/kimi-k25"' in q
    assert 'stream="true"' in q


def test_otps_avg_template():
    with patch("decision_gen.metrics.requests.get", return_value=_resp([_row(28.4)])) as get:
        v = _prom().otps("kimi", "kimi-k25", "avg")
    assert v == 28.4
    q = get.call_args.kwargs["params"]["query"]
    assert 'bodylog_output_tok_per_second_sum' in q
    assert 'bodylog_output_tok_per_second_count' in q
    assert 'service="kimi/kimi-k25"' in q
    # OTPS has no stream filter — it's already "per request"
    assert 'stream=' not in q


def test_otps_percentile_template():
    with patch("decision_gen.metrics.requests.get", return_value=_resp([_row(20.0)])) as get:
        v = _prom().otps("kimi", "kimi-k25", "p80")
    assert v == 20.0
    q = get.call_args.kwargs["params"]["query"]
    assert q.startswith("histogram_quantile(0.8,")
    assert 'bodylog_output_tok_per_second_bucket' in q


def test_rejection_rate_template():
    with patch("decision_gen.metrics.requests.get", return_value=_resp([_row(0.005)])) as get:
        v = _prom().rejection_rate("kimi", "kimi-k25")
    assert abs(v - 0.005) < 1e-12
    q = get.call_args.kwargs["params"]["query"]
    assert 'openresty_rejected_total' in q
    assert 'clamp_min' in q
    assert 'backend!="(none)"' in q
    assert '[1m]' in q


def test_current_replicas_template():
    with patch("decision_gen.metrics.requests.get", return_value=_resp([_row(2)])) as get:
        v = _prom().current_replicas("kimi", "kimi-k25")
    assert v == 2
    q = get.call_args.kwargs["params"]["query"]
    assert q == 'bodylog_service_replicas_ready{service="kimi/kimi-k25"}'


# ---------- Value handling ----------

def test_zero_rows_returns_none():
    with patch("decision_gen.metrics.requests.get", return_value=_resp([])):
        assert _prom().current_replicas("kimi", "unknown") is None


def test_multi_row_returns_none_and_logs(caplog):
    with patch("decision_gen.metrics.requests.get",
               return_value=_resp([_row(1), _row(2)])):
        assert _prom().query_scalar("some_metric") is None


def test_http_error_returns_none():
    with patch("decision_gen.metrics.requests.get", return_value=_resp([], status=500)):
        assert _prom().query_scalar("some_metric") is None


def test_request_exception_returns_none():
    import requests
    with patch("decision_gen.metrics.requests.get",
               side_effect=requests.ConnectionError("boom")):
        assert _prom().query_scalar("some_metric") is None


def test_nan_string_returns_nan():
    with patch("decision_gen.metrics.requests.get", return_value=_resp([_row("NaN")])):
        v = _prom().query_scalar("some_metric")
    import math
    assert math.isnan(v)


def test_ttft_nan_passes_through():
    """histogram_quantile NaN must surface as NaN (not None, not exception)."""
    import math
    with patch("decision_gen.metrics.requests.get", return_value=_resp([_row("NaN")])):
        v = _prom().ttft("kimi", "kimi-k25", "p95")
    assert math.isnan(v)


def test_ttft_converts_seconds_to_milliseconds():
    """The PromQL series is in seconds; estimator wants ms."""
    with patch("decision_gen.metrics.requests.get", return_value=_resp([_row(0.020)])):
        assert _prom().ttft("kimi", "kimi-k25", "p80") == 20.0


def test_otps_does_not_convert_units():
    """OTPS is already in the right unit (tokens/s per request)."""
    with patch("decision_gen.metrics.requests.get", return_value=_resp([_row(30.0)])):
        assert _prom().otps("kimi", "kimi-k25", "p80") == 30.0


def test_current_replicas_nan_returns_none():
    """NaN on replicas_ready is nonsense — treat as missing."""
    with patch("decision_gen.metrics.requests.get", return_value=_resp([_row("NaN")])):
        assert _prom().current_replicas("kimi", "kimi-k25") is None


def test_current_replicas_rounds_fractional():
    """Prom gauges can be fractional (HA exporters, etc.) — round to int."""
    with patch("decision_gen.metrics.requests.get", return_value=_resp([_row(2.4)])):
        assert _prom().current_replicas("kimi", "kimi-k25") == 2
