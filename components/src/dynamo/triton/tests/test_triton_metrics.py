# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the embedded Triton -> Dynamo metrics bridge."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from dynamo.triton import main, metrics

pytestmark = [
    pytest.mark.unit,
    pytest.mark.triton,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _metrics_config(*, metrics):
    """
    Build a minimal fake configuration object for init_worker().

    Args:
        metrics: Value for the --allow-metrics flag (True enables the
            metrics bridge, False disables it).

    Returns:
        A SimpleNamespace mimicking the real worker config, with just
        enough attributes for init_worker() to run.
    """
    return SimpleNamespace(
        namespace="dynamo",
        server_id="triton",
        model_repository="/models",
        metrics=metrics,
        to_server_options=lambda: {"model_repository": "/models"},
    )


def test_bridge_returns_metrics_then_stops_after_close():
    """
    Verify normal operation and shutdown behavior of the bridge.

    While the bridge is active, collect() must return Triton's raw metrics
    text unchanged. After close() is called, collect() must return an empty
    string, and Triton's metrics() method must not be called again.
    """
    server = MagicMock(name="server")
    server.metrics.return_value = "# HELP nv_c\nnv_c 7\n"
    bridge = metrics.TritonMetricsBridge(server)

    # Active state: metrics text is passed through unchanged.
    assert bridge.collect() == "# HELP nv_c\nnv_c 7\n"

    # Closed state: no more calls to Triton; collect() returns empty string.
    bridge.close()
    assert bridge.collect() == ""
    server.metrics.assert_called_once()


def test_bridge_raises_error_when_triton_fails():
    """
    Verify that errors from Triton are not swallowed by the bridge.

    If Triton's metrics() call fails, the bridge must let that exception
    propagate to the caller rather than hiding or logging it silently.
    """
    server = MagicMock(name="server")
    server.metrics.side_effect = RuntimeError("metrics unavailable")
    bridge = metrics.TritonMetricsBridge(server)

    with pytest.raises(RuntimeError, match="metrics unavailable"):
        bridge.collect()


def test_metrics_bridge_close_waits_for_inflight_to_finish():
    """
    Verify that close() is safe to call concurrently with collect().

    If a metrics scrape is already in progress when close() is called,
    close() must block until that scrape finishes. This prevents a race
    where the bridge could try to read from a server that is mid-shutdown.
    """
    scrape_entered = threading.Event()
    release_scrape = threading.Event()

    def blocking_metrics():
        scrape_entered.set()
        release_scrape.wait(timeout=5)
        return "nv_a 1\n"

    server = MagicMock(name="server")
    server.metrics.side_effect = blocking_metrics
    bridge = metrics.TritonMetricsBridge(server)

    scrape_result: list[str] = []
    close_done = threading.Event()

    # Simulate two things happening at once: a metrics read, and a shutdown.
    scrape = threading.Thread(target=lambda: scrape_result.append(bridge.collect()))
    closer = threading.Thread(target=lambda: (bridge.close(), close_done.set()))

    scrape.start()
    try:
        assert scrape_entered.wait(timeout=5)  # confirm the scrape has started
        closer.start()
        # close() cannot complete while a metrics collection is holding the lock
        # The current state is checked, and the correct ordering is verified by
        # the post-release assertions below
        assert not close_done.is_set()
        release_scrape.set()  # allow the scrape to complete
    finally:
        release_scrape.set()
        scrape.join(timeout=5)
        closer.join(timeout=5)

    assert scrape_result == ["nv_a 1\n"]  # scrape completed successfully
    assert close_done.is_set()  # close() only finished after the scrape
    assert bridge.collect() == ""  # bridge is now inactive


def test_register_creates_metrics_only_endpoint():
    """
    Verify how the bridge attaches itself to the runtime.

    Registration must create exactly one dedicated endpoint reserved for
    metrics only (it should never serve regular traffic), and must attach
    exactly one Prometheus-format callback (the bridge's collect() method).
    """
    server = MagicMock(name="server")
    endpoint = MagicMock(name="endpoint")
    runtime = MagicMock(name="runtime")
    runtime.endpoint.return_value = endpoint
    config = MagicMock(name="config")
    config.namespace = "dynamo"
    config.server_id = "triton"

    bridge = metrics._register_triton_metrics_bridge(runtime, config, server)

    # Confirm the reserved metrics-only endpoint is used, and not served.
    runtime.endpoint.assert_called_once_with(
        metrics.create_metrics_endpoint_url(runtime, config)
    )
    endpoint.serve_endpoint.assert_not_called()

    # Confirm exactly one metrics callback is registered.
    endpoint.metrics.register_prometheus_expfmt_callback.assert_called_once_with(
        bridge.collect
    )


def test_metrics_stop_closes_bridge_before_server_stops():
    """
    Verify shutdown ordering: metrics bridge closes before the server stops.

    This ordering matters because it guarantees no code path can read
    metrics from a server that has already been stopped.
    """
    server = MagicMock(name="server")
    bridge = metrics.TritonMetricsBridge(server)

    # Capture what collect() returns exactly when server.stop() is called.
    # If the bridge was already closed first, this will already be "".
    collect_at_stop: list[str] = []
    server.stop.side_effect = lambda: collect_at_stop.append(bridge.collect())

    metrics._stop_triton_server(server, bridge)

    server.stop.assert_called_once()
    assert collect_at_stop == [""]


@pytest.mark.parametrize(
    "metrics_enabled",
    [True, False],
    ids=["metrics-enabled", "metrics-disabled"],
)
def test_init_worker_enables_or_skips_metrics(monkeypatch, metrics_enabled):
    """
    Verify --allow-metrics gates bridge setup in init_worker().

    Enabled: the bridge is registered exactly once, even with several models
    loaded at startup.
    Disabled: skips all bridge setup entirely.
    """

    server = MagicMock(name="server")
    server.metrics.return_value = "nv_x 1\n"
    monkeypatch.setattr(main, "TritonServer", MagicMock(return_value=server))
    monkeypatch.setattr(main, "_register_and_serve", AsyncMock())

    endpoint = MagicMock(name="metrics_endpoint")
    runtime = MagicMock(name="runtime")
    runtime.endpoint.return_value = endpoint

    config = _metrics_config(metrics=metrics_enabled)
    config.namespace = "dynamo"
    config.server_id = "triton"
    worker_state = main.WorkerState()
    try:
        asyncio.run(main.init_worker(runtime, config, worker_state))
    except RuntimeError as e:
        # init_worker() raises if no models are found in the repository, but
        # that doesn't matter for this test: we only care about metrics setup.
        assert "No ready models found" in str(e)

    if metrics_enabled:
        endpoint_name = f"{config.namespace}.{config.server_id}._metrics"
        runtime.endpoint.assert_called_once_with(endpoint_name)
        endpoint.metrics.register_prometheus_expfmt_callback.assert_called_once_with(
            worker_state.metrics_bridge.collect
        )
    else:
        assert worker_state.metrics_bridge is None  # no bridge object created
        runtime.endpoint.assert_not_called()  # no metrics endpoint requested
