"""decision-gen-mock: /decisions from a file, re-read when it changes."""

import importlib.util
import json
import logging
import os
import pathlib
import re
import threading
import urllib.error
import urllib.request

import pytest

from decision_gen import server, snapshot

_spec = importlib.util.spec_from_file_location(
    "mock_decision_gen", pathlib.Path(__file__).parent.parent / "hack" / "mock_decision_gen.py")
mock = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mock)
ConfigError, FileDecisions, parse = mock.ConfigError, mock.FileDecisions, mock.parse


def _start(decisions):
    httpd = server.ThreadingHTTPServer(
        ("127.0.0.1", 0),
        server.make_handler(decisions.snapshot, decisions.ready, "no valid decisions file"),
    )
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]


def _get(port, path):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _counts(port, path="/decisions"):
    status, body = _get(port, path)
    assert status == 200, body
    assert body["apiVersion"] == snapshot.API_VERSION
    return {(d["namespace"], d["serviceId"]): d["replicas"]["active"] for d in body["decisions"]}


def _doc(*entries):
    return "decisions:\n" + "".join(
        f"  - {{namespace: {ns}, serviceId: {svc}, replicas: {n}}}\n" for ns, svc, n in entries)


def _write_in_place(path, text):
    """Overwrite the same inode, and move mtime on: two saves inside one clock
    tick would otherwise look unchanged, which a person editing never does."""
    before = os.stat(path).st_mtime_ns if os.path.exists(path) else 0
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    os.utime(path, ns=(before + 1_000_000, before + 1_000_000))


@pytest.fixture
def served(tmp_path):
    path = tmp_path / "mock.yaml"
    httpd, port = _start(FileDecisions(str(path)))
    yield path, port
    httpd.shutdown()
    httpd.server_close()


# ---------- parse ----------

def test_parse_reads_entries():
    assert parse(_doc(("models", "a", 2), ("other", "a", 0))) == {
        ("models", "a"): 2, ("other", "a"): 0}


@pytest.mark.parametrize("text", ["", "decisions:\n", "decisions: []\n"])
def test_parse_empty_is_no_decisions(text):
    assert parse(text) == {}


@pytest.mark.parametrize("text, message", [
    ("- a\n", "must be a mapping"),
    ("decision: []\n", "unknown key(s) decision"),
    ("decisions: {}\n", "decisions must be a list"),
    ("decisions: [x]\n", "decisions[0] must be a mapping"),
    (_doc(("models", "a", -1)), "replicas must be an integer >= 0"),
    (_doc(("models", "a", "true")), "replicas must be an integer >= 0"),
    (_doc(("models", "a", "'2'")), "replicas must be an integer >= 0"),
    (_doc(("''", "a", 1)), "namespace must be a non-empty string"),
    ("decisions:\n  - {namespace: models, replicas: 1}\n", "serviceId must be a non-empty string"),
    ("decisions:\n  - {namespace: models, serviceId: a, replica: 1}\n", "unknown key(s) replica"),
    (_doc(("models", "a", 1), ("models", "a", 2)), "models/a is listed twice"),
])
def test_parse_refuses(text, message):
    with pytest.raises(ConfigError, match=re.escape(message)):
        parse(text)


# ---------- serving ----------

def test_not_ready_until_the_file_reads(served):
    path, port = served
    for p in ("/readyz", "/decisions"):
        assert _get(port, p) == (503, {"status": "not_ready", "reason": "no valid decisions file"})

    path.write_text("decisions: [oops\n")
    assert _get(port, "/readyz")[0] == 503

    _write_in_place(path, _doc(("models", "a", 2)))
    assert _get(port, "/readyz") == (200, {"status": "ok"})
    assert _counts(port) == {("models", "a"): 2}


def test_an_edit_is_served_on_the_next_request(served):
    path, port = served
    path.write_text(_doc(("models", "a", 2)))
    assert _counts(port) == {("models", "a"): 2}

    _write_in_place(path, _doc(("models", "a", 3)))
    assert _counts(port) == {("models", "a"): 3}

    # An editor that saves by renaming a new file over the old one.
    tmp = path.with_suffix(".tmp")
    tmp.write_text(_doc(("models", "a", 1), ("models", "b", 0)))
    os.replace(tmp, path)
    assert _counts(port) == {("models", "a"): 1, ("models", "b"): 0}


def test_a_configmap_update_is_picked_up(tmp_path):
    """A ConfigMap volume points the file at ..data/<key> and updates by
    swapping the ..data symlink to a new directory."""
    for d, n in (("ts1", 2), ("ts2", 5)):
        (tmp_path / d).mkdir()
        (tmp_path / d / "mock.yaml").write_text(_doc(("models", "a", n)))
    os.symlink("ts1", tmp_path / "..data")
    os.symlink("..data/mock.yaml", tmp_path / "mock.yaml")
    httpd, port = _start(FileDecisions(str(tmp_path / "mock.yaml")))
    try:
        assert _counts(port) == {("models", "a"): 2}
        os.symlink("ts2", tmp_path / "..data_tmp")
        os.replace(tmp_path / "..data_tmp", tmp_path / "..data")
        assert _counts(port) == {("models", "a"): 5}
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_a_broken_edit_keeps_the_last_good_decisions(served, caplog):
    path, port = served
    path.write_text(_doc(("models", "a", 2)))
    assert _counts(port) == {("models", "a"): 2}

    with caplog.at_level(logging.ERROR, logger="decision_gen.mock"):
        _write_in_place(path, _doc(("models", "a", -1)))
        assert _counts(port) == {("models", "a"): 2}
        assert _counts(port) == {("models", "a"): 2}
    errors = [r for r in caplog.records if "replicas must be" in r.getMessage()]
    assert len(errors) == 1, "a broken file is reported once, not on every request"

    path.unlink()
    assert _counts(port) == {("models", "a"): 2}

    path.write_text(_doc(("models", "a", 4)))
    assert _counts(port) == {("models", "a"): 4}


def test_filters_are_the_real_servers(served):
    path, port = served
    path.write_text(_doc(("models", "a", 2), ("models", "b", 3), ("other", "a", 1)))
    assert _counts(port, "/decisions?namespace=models&serviceId=a") == {("models", "a"): 2}
    assert _counts(port, "/api/v1alpha1/decisions?serviceId=a") == {("models", "a"): 2, ("other", "a"): 1}
    assert _counts(port, "/decisions?serviceId=missing") == {}
