# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``EmbeddingMultiWorkerDispatchPayload``.

These tests exercise the baseline-snapshot / final-snapshot diff logic
that backs the multi-worker dispatch checks — without needing a GPU or
a real vLLM worker. We monkeypatch ``requests.get`` to return canned
``/metrics`` responses that mimic the
``dynamo_component_requests_total`` counter rising as requests land.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest

from tests.utils import payloads as payloads_mod
from tests.utils.payloads import EmbeddingMultiWorkerDispatchPayload

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]


def _metrics_text(value: float) -> str:
    """Minimal Prometheus exposition with the counter we care about."""
    return (
        "# HELP dynamo_component_requests_total Total requests handled\n"
        "# TYPE dynamo_component_requests_total counter\n"
        f'dynamo_component_requests_total{{model="m"}} {value}\n'
    )


class _FakeMetricsServer:
    """Sequence of canned ``/metrics`` responses keyed by port.

    Construct with the per-port value sequence. Each call to ``get(port)``
    advances and returns the next value. Calling more times than entries
    were registered raises ``IndexError`` so over-scraping is caught.
    """

    def __init__(self, port_sequences: dict[int, list[float]]) -> None:
        self._port_sequences = port_sequences
        self._port_index: dict[int, int] = dict.fromkeys(port_sequences, 0)

    def get(self, port: int) -> str:
        idx = self._port_index[port]
        value = self._port_sequences[port][idx]
        self._port_index[port] = idx + 1
        return _metrics_text(value)


@pytest.fixture
def patch_requests_get(monkeypatch):
    """Return a callable that installs a fake ``requests.get`` driven by
    a ``_FakeMetricsServer``.
    """

    def install(server: _FakeMetricsServer):
        def fake_get(url: str, timeout: float = 0) -> Any:
            # url looks like ``http://localhost:<port>/metrics``
            port = int(url.split(":")[-1].split("/")[0])
            text = server.get(port)
            resp = MagicMock()
            resp.raise_for_status.return_value = None
            resp.text = text
            return resp

        monkeypatch.setattr(payloads_mod.requests, "get", fake_get)

    return install


def _mock_response() -> Any:
    """Return a MagicMock that looks like a successful /v1/embeddings response.

    Only ``raise_for_status`` and ``json`` are touched by
    ``EmbeddingPayload.extract_embeddings`` (which is what
    ``response_handler`` delegates to).
    """
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {
        "object": "list",
        "data": [{"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0}],
    }
    return resp


def _make_payload(
    *,
    ports: tuple[int, int] = (8081, 8082),
    expected_indices: set[int],
    repeat_count: int = 3,
    min_total_delta: int = 0,
) -> EmbeddingMultiWorkerDispatchPayload:
    return EmbeddingMultiWorkerDispatchPayload(
        body={"model": "m", "input": "x"},
        expected_response=["Generated 1 embeddings with dimension"],
        expected_log=[],
        repeat_count=repeat_count,
        host="localhost",
        system_ports=list(ports),
        expected_worker_indices_with_delta=expected_indices,
        min_total_delta=min_total_delta,
        settle_seconds=0.0,
    )


def _drive_payload(
    payload: EmbeddingMultiWorkerDispatchPayload,
    *,
    response_text: str = "Generated 1 embeddings with dimension 3",
) -> None:
    """Invoke validate() ``payload.repeat_count`` times, mirroring what the
    test harness loop does for each request iteration.
    """
    for _ in range(payload.repeat_count):
        payload.validate(_mock_response(), response_text)


# ── Tests ──────────────────────────────────────────────────────────────────


def test_same_model_load_balance_passes_when_both_workers_increment(
    patch_requests_get,
):
    """Both workers' counters rise → expected_indices={0, 1} passes."""
    # baseline scrape (repeat 1), then final scrape (repeat 3); repeat 2 is a no-op
    patch_requests_get(
        _FakeMetricsServer(
            {
                8081: [5.0, 12.0],  # +7 on worker A
                8082: [3.0, 11.0],  # +8 on worker B
            }
        )
    )
    payload = _make_payload(expected_indices={0, 1})
    _drive_payload(payload)


def test_same_model_load_balance_fails_when_only_one_worker_increments(
    patch_requests_get,
):
    """One worker idle while the other absorbs all traffic → assert fires."""
    patch_requests_get(
        _FakeMetricsServer(
            {
                8081: [5.0, 25.0],  # +20 on A
                8082: [3.0, 3.0],  # +0 on B (idle)
            }
        )
    )
    payload = _make_payload(expected_indices={0, 1})
    with pytest.raises(AssertionError, match="Expected worker indices"):
        _drive_payload(payload)


def test_multi_model_dispatch_passes_when_only_target_worker_increments(
    patch_requests_get,
):
    """Model-A traffic should appear only on worker A (index 0)."""
    patch_requests_get(
        _FakeMetricsServer(
            {
                8081: [5.0, 15.0],  # +10 on A
                8082: [3.0, 3.0],  # +0 on B — correct!
            }
        )
    )
    payload = _make_payload(expected_indices={0}, min_total_delta=10)
    _drive_payload(payload)


def test_multi_model_dispatch_fails_when_wrong_worker_gets_traffic(
    patch_requests_get,
):
    """If model-A traffic leaks onto worker B, the index check catches it."""
    patch_requests_get(
        _FakeMetricsServer(
            {
                8081: [5.0, 15.0],  # +10 on A
                8082: [3.0, 4.0],  # +1 on B — leak!
            }
        )
    )
    payload = _make_payload(expected_indices={0})
    with pytest.raises(AssertionError, match="Expected worker indices"):
        _drive_payload(payload)


def test_min_total_delta_lower_bound(patch_requests_get):
    """min_total_delta enforces a per-burst floor on summed delta."""
    patch_requests_get(
        _FakeMetricsServer(
            {
                8081: [0.0, 2.0],  # +2
                8082: [0.0, 1.0],  # +1 → sum = 3
            }
        )
    )
    payload = _make_payload(
        expected_indices={0, 1},
        min_total_delta=10,  # floor 10, actual 3 — should fail
    )
    with pytest.raises(AssertionError, match="total delta"):
        _drive_payload(payload)


def test_baseline_snapshot_taken_on_first_repeat(patch_requests_get):
    """Baseline is taken after the first request — earlier traffic on the
    workers should not be subtracted from this burst's delta.
    """
    # Worker A starts already at 100 (e.g. a prior burst's leftover), then
    # increments to 105 by the end of this burst. Worker B stays flat at 50.
    patch_requests_get(
        _FakeMetricsServer(
            {
                8081: [100.0, 105.0],  # +5
                8082: [50.0, 50.0],  # +0
            }
        )
    )
    payload = _make_payload(
        expected_indices={0},  # only A should show delta
        min_total_delta=5,
    )
    _drive_payload(payload)
