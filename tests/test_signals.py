"""PromQL the signals send."""

from decision_gen import signals as signals_mod
from decision_gen.signals import Signals


class _Resp:
    status_code = 200
    text = ""

    def json(self):
        return {"data": {"result": [{"value": [0, "1"]}]}}


def test_replicas_ready_reads_the_service_level_role_only(monkeypatch):
    sent = []

    def fake_get(url, params, timeout):
        sent.append(params["query"])
        return _Resp()

    monkeypatch.setattr(signals_mod.requests, "get", fake_get)
    assert Signals("http://prom").replicas_ready("ns", "svc").value == 1
    assert 'bodylog_service_replicas_ready{service="ns/svc", role=~"|router"}' in sent[0]


def test_log_staleness_reads_the_exporter_freshness_gauge(monkeypatch):
    sent = []

    def fake_get(url, params, timeout):
        sent.append(params["query"])
        return _Resp()

    monkeypatch.setattr(signals_mod.requests, "get", fake_get)
    assert Signals("http://prom").log_staleness().value == 1.0
    assert sent == ["time() - max(bodylog_exporter_last_ts_seconds)"]
