# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import http.server
import threading
from contextlib import contextmanager

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


ENGINE_EXPOSITION = """\
# HELP sglang:num_running_reqs The number of running requests.
# TYPE sglang:num_running_reqs gauge
sglang:num_running_reqs{model_name="test-model"} 3.0
# HELP sglang:gen_throughput The generation throughput (token/s).
# TYPE sglang:gen_throughput gauge
sglang:gen_throughput{model_name="test-model"} 42.5
# HELP python_gc_objects_collected_total Objects collected during gc.
# TYPE python_gc_objects_collected_total counter
python_gc_objects_collected_total{generation="0"} 17.0
"""


@contextmanager
def _serving(payload: bytes, status: int = 200):
    """Run a throwaway HTTP server returning ``payload`` on any path."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - http.server API
            self.send_response(status)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()


def _collect_names(collector) -> list[str]:
    return [family.name for family in collector.collect()]


def test_engine_metrics_url_derives_metrics_path():
    from dynamo.openai_backend.engine_metrics import engine_metrics_url

    assert engine_metrics_url("http://127.0.0.1:30000/v1") == (
        "http://127.0.0.1:30000/metrics"
    )
    assert engine_metrics_url("http://engine:8080") == "http://engine:8080/metrics"


@pytest.mark.parametrize("bad", ["", "127.0.0.1:30000/v1", "/v1", "ftp://host/v1"])
def test_engine_metrics_url_rejects_non_http_urls(bad):
    """A relative URL would make urlopen read a local file path."""
    from dynamo.openai_backend.engine_metrics import engine_metrics_url

    with pytest.raises(ValueError):
        engine_metrics_url(bad)


def test_collector_yields_engine_families():
    from dynamo.openai_backend.engine_metrics import EngineMetricsCollector

    with _serving(ENGINE_EXPOSITION.encode()) as origin:
        names = _collect_names(EngineMetricsCollector(f"{origin}/metrics"))

    assert "sglang:num_running_reqs" in names
    assert "sglang:gen_throughput" in names


def test_collector_is_empty_when_engine_unreachable():
    """An engine started without --enable-metrics must not break the scrape."""
    from dynamo.openai_backend.engine_metrics import EngineMetricsCollector

    collector = EngineMetricsCollector("http://127.0.0.1:1/metrics", timeout=0.25)

    assert _collect_names(collector) == []


def test_collector_is_empty_on_error_response():
    from dynamo.openai_backend.engine_metrics import EngineMetricsCollector

    with _serving(b"nope", status=503) as origin:
        collector = EngineMetricsCollector(f"{origin}/metrics", timeout=0.5)

        assert _collect_names(collector) == []


def test_collector_does_not_raise_on_malformed_exposition():
    from dynamo.openai_backend.engine_metrics import EngineMetricsCollector

    with _serving(b"# TYPE broken\nnot a metric line\n") as origin:
        collector = EngineMetricsCollector(f"{origin}/metrics", timeout=0.5)

        # Whatever the parser makes of this, it must not escape the collector.
        _collect_names(collector)


def test_collector_ignores_ambient_proxy_settings(monkeypatch):
    """The engine is local; a pod-wide proxy must not intercept the scrape."""
    from dynamo.openai_backend.engine_metrics import EngineMetricsCollector

    monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")

    with _serving(ENGINE_EXPOSITION.encode()) as origin:
        names = _collect_names(EngineMetricsCollector(f"{origin}/metrics"))

    assert "sglang:num_running_reqs" in names
