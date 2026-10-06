# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping
from typing import Any, Optional

import pytest

from dynamo.planner.core.types import (
    BatchDispatcherFeedback,
    BatchJobDemand,
    PoolTrafficDemand,
)
from dynamo.planner.environment.batch import (
    BatchGatewayJobSource,
    BatchSchedulingCollector,
    LlmdAsyncOpenMetricsSource,
    LlmdAsyncPrometheusSource,
    OpenMetricsOnlineTrafficSource,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


class _FakeResponse:
    def __init__(self, payload: object) -> None:
        self._payload = payload

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        traceback: Optional[object],
    ) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    async def json(self) -> object:
        return self._payload

    async def text(self) -> str:
        if not isinstance(self._payload, str):
            raise TypeError("fake response payload is not text")
        return self._payload


class _FakeSession:
    def __init__(
        self, responses: Mapping[tuple[str, tuple[tuple[str, str], ...]], object]
    ):
        self._responses = dict(responses)
        self.calls: list[
            tuple[str, tuple[tuple[str, str], ...], Mapping[str, str]]
        ] = []

    def get(
        self,
        url: str,
        *,
        params: Optional[Mapping[str, str]],
        headers: Mapping[str, str],
    ) -> _FakeResponse:
        normalized_params = tuple(sorted((params or {}).items()))
        self.calls.append((url, normalized_params, dict(headers)))
        key = (url, normalized_params)
        if key not in self._responses:
            raise AssertionError(f"unexpected HTTP request: {key!r}")
        return _FakeResponse(self._responses[key])


class _FakeOpenMetricsSession:
    def __init__(self, payloads: list[str | BaseException]) -> None:
        self._payloads = list(payloads)
        self.calls: list[str] = []
        self.timeouts: list[float | None] = []

    def get(self, url: str, *, timeout: Optional[object] = None) -> _FakeResponse:
        self.calls.append(url)
        self.timeouts.append(getattr(timeout, "total", None))
        if not self._payloads:
            raise AssertionError(f"unexpected HTTP request: {url!r}")
        payload = self._payloads.pop(0)
        if isinstance(payload, BaseException):
            raise payload
        return _FakeResponse(payload)


class _SequenceClock:
    def __init__(self, values: list[float]) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


def _openmetrics(*samples: str, active_requests: int | float = 0) -> str:
    return (
        "\n".join(
            (
                "process_start_time_seconds 1",
                f"dynamo_frontend_active_requests {active_requests}",
                *samples,
            )
        )
        + "\n"
    )


@pytest.mark.asyncio
async def test_openmetrics_online_traffic_warms_up_then_reports_counter_rate() -> None:
    metrics_url = "http://frontend.example/metrics"
    session = _FakeOpenMetricsSession(
        [
            _openmetrics(
                'dynamo_frontend_requests_started_total{request_type="chat",streaming="true"} 100',
                'dynamo_frontend_requests_started_total{request_type="chat",streaming="false"} 900',
            ),
            _openmetrics(
                'dynamo_frontend_requests_started_total{request_type="chat",streaming="true"} 130',
                'dynamo_frontend_requests_started_total{request_type="chat",streaming="false"} 990',
                active_requests=7,
            ),
        ]
    )
    source = OpenMetricsOnlineTrafficSource(
        pool_id="pool-a",
        metrics_url=metrics_url,
        session=session,  # type: ignore[arg-type]
        match_labels={"request_type": "chat", "streaming": "true"},
        monotonic_clock=_SequenceClock([0.0, 1.0, 11.0, 12.0]),
        wall_clock=_SequenceClock([10.0, 20.0]),
    )

    with pytest.raises(RuntimeError, match="warming up"):
        await source.collect_online_traffic(observed_at_s=10.0)

    traffic = await source.collect_online_traffic(observed_at_s=20.0)

    assert traffic == [
        PoolTrafficDemand(
            observed_at_s=20.0,
            pool_id="pool-a",
            online_offered_rps=3.0,
            frontend_active_requests=7,
        )
    ]
    assert session.calls == [metrics_url, metrics_url]
    assert session.timeouts == [5.0, 5.0]


@pytest.mark.asyncio
async def test_openmetrics_online_traffic_counter_reset_rewarms_baseline() -> None:
    session = _FakeOpenMetricsSession(
        [
            _openmetrics("dynamo_frontend_requests_started_total 100"),
            _openmetrics("dynamo_frontend_requests_started_total 5"),
            _openmetrics("dynamo_frontend_requests_started_total 25"),
        ]
    )
    source = OpenMetricsOnlineTrafficSource(
        pool_id="pool-a",
        metrics_url="http://frontend.example/metrics",
        session=session,  # type: ignore[arg-type]
        match_labels={},
        monotonic_clock=_SequenceClock([0.0, 1.0, 11.0, 12.0, 17.0, 18.0]),
        wall_clock=_SequenceClock([10.0, 20.0, 25.0]),
    )

    with pytest.raises(RuntimeError, match="warming up"):
        await source.collect_online_traffic(observed_at_s=10.0)
    with pytest.raises(RuntimeError, match="counter reset"):
        await source.collect_online_traffic(observed_at_s=20.0)

    traffic = await source.collect_online_traffic(observed_at_s=25.0)

    assert traffic[0].online_offered_rps == 4.0
    assert traffic[0].frontend_active_requests == 0


@pytest.mark.asyncio
async def test_openmetrics_online_traffic_absent_labelset_is_anchored_zero() -> None:
    session = _FakeOpenMetricsSession(
        [
            _openmetrics("unrelated_counter_total 1"),
            _openmetrics("unrelated_counter_total 2"),
        ]
    )
    source = OpenMetricsOnlineTrafficSource(
        pool_id="pool-a",
        metrics_url="http://frontend.example/metrics",
        session=session,  # type: ignore[arg-type]
        match_labels={},
        monotonic_clock=_SequenceClock([0.0, 1.0, 11.0, 12.0]),
        wall_clock=_SequenceClock([10.0, 20.0]),
    )

    with pytest.raises(RuntimeError, match="warming up"):
        await source.collect_online_traffic(observed_at_s=10.0)
    traffic = await source.collect_online_traffic(observed_at_s=20.0)

    assert traffic[0].online_offered_rps == 0.0
    assert traffic[0].frontend_active_requests == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("active_sample", "error_match"),
    [
        ("dynamo_frontend_active_requests -1", "non-negative"),
        ("dynamo_frontend_active_requests 1.5", "integer"),
    ],
)
async def test_openmetrics_online_traffic_rejects_invalid_active_request_gauge(
    active_sample: str, error_match: str
) -> None:
    source = OpenMetricsOnlineTrafficSource(
        pool_id="pool-a",
        metrics_url="http://frontend.example/metrics",
        session=_FakeOpenMetricsSession(  # type: ignore[arg-type]
            [
                "process_start_time_seconds 1\n"
                "dynamo_frontend_requests_started_total 100\n"
                f"{active_sample}\n"
            ]
        ),
        match_labels={},
        monotonic_clock=_SequenceClock([0.0, 1.0]),
        wall_clock=_SequenceClock([10.0]),
    )

    with pytest.raises(ValueError, match=error_match):
        await source.collect_online_traffic(observed_at_s=10.0)


@pytest.mark.asyncio
async def test_openmetrics_online_traffic_rejects_duplicate_active_request_gauge() -> (
    None
):
    payload = "\n".join(
        (
            "process_start_time_seconds 1",
            "dynamo_frontend_requests_started_total 100",
            "dynamo_frontend_active_requests 1",
            "dynamo_frontend_active_requests 2",
        )
    )
    source = OpenMetricsOnlineTrafficSource(
        pool_id="pool-a",
        metrics_url="http://frontend.example/metrics",
        session=_FakeOpenMetricsSession([payload + "\n"]),  # type: ignore[arg-type]
        match_labels={},
        monotonic_clock=_SequenceClock([0.0, 1.0]),
        wall_clock=_SequenceClock([10.0]),
    )

    with pytest.raises(ValueError, match="exactly one series"):
        await source.collect_online_traffic(observed_at_s=10.0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "active_samples",
    [
        # A freshly started frontend registers the GaugeVec but exports no
        # child until the model serves its first inference request.
        "",
        # Other models have created their children; ours has not yet.
        'dynamo_frontend_active_requests{model="model-b"} 3\n',
    ],
)
async def test_openmetrics_online_traffic_lazily_absent_gauge_keeps_rate(
    active_samples: str,
) -> None:
    session = _FakeOpenMetricsSession(
        [
            "process_start_time_seconds 1\n"
            'dynamo_frontend_requests_started_total{model="model-a"} 100\n'
            f"{active_samples}",
            "process_start_time_seconds 1\n"
            'dynamo_frontend_requests_started_total{model="model-a"} 150\n'
            f"{active_samples}",
        ]
    )
    source = OpenMetricsOnlineTrafficSource(
        pool_id="pool-a",
        metrics_url="http://frontend.example/metrics",
        session=session,  # type: ignore[arg-type]
        match_labels={"model": "model-a"},
        monotonic_clock=_SequenceClock([0.0, 1.0, 11.0, 12.0]),
        wall_clock=_SequenceClock([10.0, 20.0]),
    )

    with pytest.raises(RuntimeError, match="warming up"):
        await source.collect_online_traffic(observed_at_s=10.0)
    traffic = await source.collect_online_traffic(observed_at_s=20.0)

    # The valid rate survives so admission can open; concurrency is unknown,
    # which the policy never treats as idle.
    assert traffic[0].online_offered_rps == pytest.approx(5.0)
    assert traffic[0].frontend_active_requests is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "active_sample",
    [
        "dynamo_frontend_active_requests 1",
        'dynamo_frontend_active_requests{model="model-a",endpoint="chat"} 1',
    ],
)
async def test_openmetrics_online_traffic_rejects_wrong_active_request_labels(
    active_sample: str,
) -> None:
    source = OpenMetricsOnlineTrafficSource(
        pool_id="pool-a",
        metrics_url="http://frontend.example/metrics",
        session=_FakeOpenMetricsSession(  # type: ignore[arg-type]
            [
                "process_start_time_seconds 1\n"
                'dynamo_frontend_requests_started_total{model="model-a"} 100\n'
                f"{active_sample}\n"
            ]
        ),
        match_labels={"model": "model-a"},
        monotonic_clock=_SequenceClock([0.0, 1.0]),
        wall_clock=_SequenceClock([10.0]),
    )

    with pytest.raises(ValueError, match="does not match selector keys"):
        await source.collect_online_traffic(observed_at_s=10.0)


@pytest.mark.asyncio
async def test_openmetrics_online_traffic_uses_model_for_active_request_gauge() -> None:
    source = OpenMetricsOnlineTrafficSource(
        pool_id="pool-a",
        metrics_url="http://frontend.example/metrics",
        session=_FakeOpenMetricsSession(  # type: ignore[arg-type]
            [
                "process_start_time_seconds 1\n"
                'dynamo_frontend_requests_started_total{model="model-a",request_type="stream"} 100\n'
                'dynamo_frontend_active_requests{model="model-a"} 0\n'
                'dynamo_frontend_active_requests{model="model-b"} 9\n',
                "process_start_time_seconds 1\n"
                'dynamo_frontend_requests_started_total{model="model-a",request_type="stream"} 110\n'
                'dynamo_frontend_active_requests{model="model-a"} 3\n'
                'dynamo_frontend_active_requests{model="model-b"} 8\n',
            ]
        ),
        match_labels={"model": "model-a", "request_type": "stream"},
        monotonic_clock=_SequenceClock([0.0, 1.0, 11.0, 12.0]),
        wall_clock=_SequenceClock([10.0, 20.0]),
    )

    with pytest.raises(RuntimeError, match="warming up"):
        await source.collect_online_traffic(observed_at_s=10.0)
    traffic = await source.collect_online_traffic(observed_at_s=20.0)

    assert traffic[0].frontend_active_requests == 3


@pytest.mark.asyncio
async def test_openmetrics_online_traffic_missing_series_rewarms_baseline() -> None:
    session = _FakeOpenMetricsSession(
        [
            _openmetrics("dynamo_frontend_requests_started_total 100"),
            _openmetrics("unrelated_counter_total 1"),
            _openmetrics("dynamo_frontend_requests_started_total 150"),
            _openmetrics("dynamo_frontend_requests_started_total 170"),
        ]
    )
    source = OpenMetricsOnlineTrafficSource(
        pool_id="pool-a",
        metrics_url="http://frontend.example/metrics",
        session=session,  # type: ignore[arg-type]
        match_labels={},
        monotonic_clock=_SequenceClock([0.0, 1.0, 11.0, 12.0, 21.0, 22.0, 32.0, 33.0]),
        wall_clock=_SequenceClock([10.0, 20.0, 30.0, 40.0]),
    )

    with pytest.raises(RuntimeError, match="warming up"):
        await source.collect_online_traffic(observed_at_s=10.0)
    with pytest.raises(ValueError, match="disappeared"):
        await source.collect_online_traffic(observed_at_s=20.0)
    with pytest.raises(RuntimeError, match="warming up"):
        await source.collect_online_traffic(observed_at_s=30.0)

    traffic = await source.collect_online_traffic(observed_at_s=40.0)

    assert traffic[0].online_offered_rps == 2.0


@pytest.mark.asyncio
async def test_openmetrics_online_traffic_requires_exporter_anchor() -> None:
    source = OpenMetricsOnlineTrafficSource(
        pool_id="pool-a",
        metrics_url="http://frontend.example/metrics",
        session=_FakeOpenMetricsSession(["unrelated_counter_total 1\n"]),  # type: ignore[arg-type]
        match_labels={},
        monotonic_clock=_SequenceClock([0.0, 1.0]),
        wall_clock=_SequenceClock([10.0]),
    )

    with pytest.raises(ValueError, match="process_start_time_seconds"):
        await source.collect_online_traffic(observed_at_s=10.0)


@pytest.mark.asyncio
async def test_openmetrics_online_traffic_scrape_gap_rewarms_baseline() -> None:
    source = OpenMetricsOnlineTrafficSource(
        pool_id="pool-a",
        metrics_url="http://frontend.example/metrics",
        session=_FakeOpenMetricsSession(  # type: ignore[arg-type]
            [
                _openmetrics("dynamo_frontend_requests_started_total 100"),
                TimeoutError("scrape timed out"),
                _openmetrics("dynamo_frontend_requests_started_total 150"),
                _openmetrics("dynamo_frontend_requests_started_total 170"),
            ]
        ),
        match_labels={},
        monotonic_clock=_SequenceClock([0.0, 1.0, 11.0, 21.0, 22.0, 32.0, 33.0]),
        wall_clock=_SequenceClock([10.0, 30.0, 40.0]),
    )

    with pytest.raises(RuntimeError, match="warming up"):
        await source.collect_online_traffic(observed_at_s=10.0)
    with pytest.raises(TimeoutError, match="scrape timed out"):
        await source.collect_online_traffic(observed_at_s=20.0)
    with pytest.raises(RuntimeError, match="warming up"):
        await source.collect_online_traffic(observed_at_s=30.0)

    traffic = await source.collect_online_traffic(observed_at_s=40.0)

    assert traffic[0].online_offered_rps == 2.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_windows", "expected_rate"),
    [
        ([0.0, 9.0, 10.0, 10.1], 10.0),
        ([0.0, 1.0, 10.0, 19.0], 10.0 / 9.0),
    ],
)
async def test_openmetrics_online_rate_uses_conservative_scrape_windows(
    request_windows: list[float], expected_rate: float
) -> None:
    source = OpenMetricsOnlineTrafficSource(
        pool_id="pool-a",
        metrics_url="http://frontend.example/metrics",
        session=_FakeOpenMetricsSession(  # type: ignore[arg-type]
            [
                _openmetrics("dynamo_frontend_requests_started_total 100"),
                _openmetrics("dynamo_frontend_requests_started_total 110"),
            ]
        ),
        match_labels={},
        monotonic_clock=_SequenceClock(request_windows),
        wall_clock=_SequenceClock([100.0, 119.0]),
    )

    with pytest.raises(RuntimeError, match="warming up"):
        await source.collect_online_traffic(observed_at_s=1.0)
    traffic = await source.collect_online_traffic(observed_at_s=2.0)

    assert traffic[0].online_offered_rps == pytest.approx(expected_rate)
    assert traffic[0].observed_at_s == 119.0


@pytest.mark.asyncio
async def test_openmetrics_online_exporter_restart_rewarms_overtaken_counter() -> None:
    source = OpenMetricsOnlineTrafficSource(
        pool_id="pool-a",
        metrics_url="http://frontend.example/metrics",
        session=_FakeOpenMetricsSession(  # type: ignore[arg-type]
            [
                "process_start_time_seconds 1\n"
                "dynamo_frontend_requests_started_total 100\n"
                "dynamo_frontend_active_requests 4\n",
                "process_start_time_seconds 2\n"
                "dynamo_frontend_requests_started_total 150\n"
                "dynamo_frontend_active_requests 2\n",
                "process_start_time_seconds 2\n"
                "dynamo_frontend_requests_started_total 170\n"
                "dynamo_frontend_active_requests 1\n",
            ]
        ),
        match_labels={},
        monotonic_clock=_SequenceClock([0.0, 1.0, 11.0, 12.0, 22.0, 23.0]),
        wall_clock=_SequenceClock([10.0, 20.0, 30.0]),
    )

    with pytest.raises(RuntimeError, match="warming up"):
        await source.collect_online_traffic(observed_at_s=10.0)
    with pytest.raises(RuntimeError, match="exporter identity changed"):
        await source.collect_online_traffic(observed_at_s=20.0)
    traffic = await source.collect_online_traffic(observed_at_s=30.0)

    assert traffic[0].online_offered_rps == 2.0
    assert traffic[0].frontend_active_requests == 1


@pytest.mark.asyncio
async def test_llmd_async_openmetrics_collects_queue_rate_and_valid_cap() -> None:
    metrics_url = "http://llmd-async.example/metrics"
    session = _FakeOpenMetricsSession(
        [
            _openmetrics(
                'llm_d_async_async_broker_backlog{pool_name="pool-a"} 7',
                'llm_d_async_async_broker_backlog_source_available{pool_name="pool-a"} 1',
                'llm_d_async_async_queue_depth{pool_name="pool-a"} 3',
                'llm_d_async_async_inflight_requests{pool_name="pool-a"} 2',
                'llm_d_async_async_dispatched_requests_total{pool_name="pool-a"} 100',
                'llm_d_async_async_drain_limit_lease_valid{pool_name="pool-a"} 1',
                'llm_d_async_async_drain_limit_rps{pool_name="pool-a"} 4.5',
                'llm_d_async_async_drain_limit_valid_until_seconds{pool_name="pool-a"} 200',
            ),
            _openmetrics(
                'llm_d_async_async_broker_backlog{pool_name="pool-a"} 8',
                'llm_d_async_async_broker_backlog_source_available{pool_name="pool-a"} 1',
                'llm_d_async_async_queue_depth{pool_name="pool-a"} 3',
                'llm_d_async_async_inflight_requests{pool_name="pool-a"} 4',
                'llm_d_async_async_dispatched_requests_total{pool_name="pool-a"} 125',
                'llm_d_async_async_drain_limit_lease_valid{pool_name="pool-a"} 1',
                'llm_d_async_async_drain_limit_rps{pool_name="pool-a"} 6',
                'llm_d_async_async_drain_limit_valid_until_seconds{pool_name="pool-a"} 200',
            ),
        ]
    )
    source = LlmdAsyncOpenMetricsSource(
        pools=["pool-a"],
        metrics_url=metrics_url,
        session=session,  # type: ignore[arg-type]
        monotonic_clock=_SequenceClock([0.0, 1.0, 11.0, 12.0]),
        wall_clock=_SequenceClock([100.0, 110.0]),
    )

    with pytest.raises(RuntimeError, match="warming up"):
        await source.collect_dispatcher_feedback(observed_at_s=100.0)
    feedback = await source.collect_dispatcher_feedback(observed_at_s=110.0)

    assert feedback == [
        BatchDispatcherFeedback(
            observed_at_s=110.0,
            pool_id="pool-a",
            observation_window_s=10.0,
            queued_requests=11,
            inflight_requests=4,
            actual_dispatch_rps=2.5,
            applied_max_admission_rps=6.0,
        )
    ]
    assert session.calls == [metrics_url, metrics_url]
    assert session.timeouts == [5.0, 5.0]


def _llmd_dispatch_metrics(
    dispatched: int | None,
    *,
    availability: int = 1,
) -> str:
    samples = [
        'llm_d_async_async_broker_backlog{pool_name="pool-a"} 0',
        (
            "llm_d_async_async_broker_backlog_source_available"
            f'{{pool_name="pool-a"}} {availability}'
        ),
    ]
    if dispatched is not None:
        samples.append(
            "llm_d_async_async_dispatched_requests_total"
            f'{{pool_name="pool-a"}} {dispatched}'
        )
    return _openmetrics(*samples)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("gap", "error_type", "error"),
    [
        (
            _llmd_dispatch_metrics(120, availability=0),
            ValueError,
            "availability.*exactly 1",
        ),
        (RuntimeError("scrape failed"), RuntimeError, "scrape failed"),
    ],
)
async def test_llmd_async_openmetrics_gap_rewarms_dispatch_baseline(
    gap: str | BaseException,
    error_type: type[Exception],
    error: str,
) -> None:
    scrape_gap = isinstance(gap, BaseException)
    source = LlmdAsyncOpenMetricsSource(
        pools=["pool-a"],
        metrics_url="http://llmd-async.example/metrics",
        session=_FakeOpenMetricsSession(  # type: ignore[arg-type]
            [
                _llmd_dispatch_metrics(100),
                gap,
                _llmd_dispatch_metrics(150),
                _llmd_dispatch_metrics(170),
            ]
        ),
        monotonic_clock=_SequenceClock(
            (
                [0.0, 1.0, 11.0, 21.0, 22.0, 32.0, 33.0]
                if scrape_gap
                else [0.0, 1.0, 11.0, 12.0, 21.0, 22.0, 32.0, 33.0]
            )
        ),
        wall_clock=_SequenceClock(
            [10.0, 30.0, 40.0] if scrape_gap else [10.0, 20.0, 30.0, 40.0]
        ),
    )

    with pytest.raises(RuntimeError, match="warming up"):
        await source.collect_dispatcher_feedback(observed_at_s=10.0)
    with pytest.raises(error_type, match=error):
        await source.collect_dispatcher_feedback(observed_at_s=20.0)

    with pytest.raises(RuntimeError, match="warming up"):
        await source.collect_dispatcher_feedback(observed_at_s=30.0)
    following = await source.collect_dispatcher_feedback(observed_at_s=40.0)

    assert following[0].actual_dispatch_rps == 2.0
    assert following[0].observation_window_s == 10.0


@pytest.mark.asyncio
async def test_llmd_async_openmetrics_accepts_lazy_zero_dispatch_counter() -> None:
    source = LlmdAsyncOpenMetricsSource(
        pools=["pool-a"],
        metrics_url="http://llmd-async.example/metrics",
        session=_FakeOpenMetricsSession(  # type: ignore[arg-type]
            [_llmd_dispatch_metrics(None), _llmd_dispatch_metrics(None)]
        ),
        monotonic_clock=_SequenceClock([0.0, 1.0, 11.0, 12.0]),
        wall_clock=_SequenceClock([10.0, 20.0]),
    )

    with pytest.raises(RuntimeError, match="warming up"):
        await source.collect_dispatcher_feedback(observed_at_s=10.0)
    feedback = await source.collect_dispatcher_feedback(observed_at_s=20.0)

    assert feedback == [
        BatchDispatcherFeedback(
            observed_at_s=20.0,
            pool_id="pool-a",
            observation_window_s=10.0,
            queued_requests=0,
            inflight_requests=0,
            actual_dispatch_rps=0.0,
            applied_max_admission_rps=None,
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("lease_valid", "valid_until_s"),
    [(0, 200), (1, 100)],
)
async def test_llmd_async_openmetrics_hides_invalid_or_expired_cap(
    lease_valid: int,
    valid_until_s: int,
) -> None:
    payload = _openmetrics(
        'llm_d_async_async_broker_backlog{pool_name="pool-a"} 0',
        'llm_d_async_async_broker_backlog_source_available{pool_name="pool-a"} 1',
        'llm_d_async_async_dispatched_requests_total{pool_name="pool-a"} 0',
        f'llm_d_async_async_drain_limit_lease_valid{{pool_name="pool-a"}} {lease_valid}',
        'llm_d_async_async_drain_limit_rps{pool_name="pool-a"} 9',
        f'llm_d_async_async_drain_limit_valid_until_seconds{{pool_name="pool-a"}} {valid_until_s}',
    )
    session = _FakeOpenMetricsSession([payload, payload])
    source = LlmdAsyncOpenMetricsSource(
        pools=["pool-a"],
        metrics_url="http://llmd-async.example/metrics",
        session=session,  # type: ignore[arg-type]
        monotonic_clock=_SequenceClock([0.0, 1.0, 11.0, 12.0]),
        wall_clock=_SequenceClock([90.0, 100.0]),
    )

    with pytest.raises(RuntimeError, match="warming up"):
        await source.collect_dispatcher_feedback(observed_at_s=90.0)
    feedback = await source.collect_dispatcher_feedback(observed_at_s=100.0)

    assert feedback[0].applied_max_admission_rps is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "error"),
    [
        (
            _openmetrics(
                'llm_d_async_async_broker_backlog_source_available{pool_name="pool-a"} 1'
            ),
            "required OpenMetrics sample.*broker_backlog",
        ),
        (
            _openmetrics(
                'llm_d_async_async_broker_backlog{pool_name="pool-a"} 0',
                'llm_d_async_async_broker_backlog_source_available{pool_name="pool-a"} 0',
            ),
            "availability.*exactly 1",
        ),
        (
            _openmetrics('llm_d_async_async_broker_backlog{pool_name="pool-a"} 0'),
            "availability.*exactly 1",
        ),
    ],
)
async def test_llmd_async_openmetrics_requires_available_broker_anchor(
    payload: str,
    error: str,
) -> None:
    source = LlmdAsyncOpenMetricsSource(
        pools=["pool-a"],
        metrics_url="http://llmd-async.example/metrics",
        session=_FakeOpenMetricsSession([payload]),  # type: ignore[arg-type]
        monotonic_clock=_SequenceClock([0.0, 1.0]),
        wall_clock=_SequenceClock([100.0]),
    )

    with pytest.raises(ValueError, match=error):
        await source.collect_dispatcher_feedback(observed_at_s=100.0)


def _batch(
    job_id: str,
    status: str,
    *,
    total: int,
    completed: int,
    failed: int,
    expires_at: object,
    created_at: object = 1_000,
    completion_window: object = "24h",
    pool: str = "pool-a",
    work_class: str = "chat-small",
) -> dict[str, Any]:
    return {
        "id": job_id,
        "status": status,
        "expires_at": expires_at,
        "created_at": created_at,
        "completion_window": completion_window,
        "metadata": {"pool": pool, "work_class": work_class},
        "request_counts": {
            "total": total,
            "completed": completed,
            "failed": failed,
        },
    }


def _source_for_listed_batch(job: Mapping[str, Any]) -> BatchGatewayJobSource:
    base_url = "http://batch.example"
    responses: dict[tuple[str, tuple[tuple[str, str], ...]], object] = {
        (
            f"{base_url}/v1/batches",
            (("after", "0"), ("limit", "100")),
        ): {"data": [job], "has_more": False}
    }
    job_id = job.get("id")
    if isinstance(job_id, str):
        responses[(f"{base_url}/v1/batches/{job_id}", ())] = job
    session = _FakeSession(responses)
    return BatchGatewayJobSource(
        base_url=base_url,
        session=session,  # type: ignore[arg-type]
        pool_resolver=lambda _job: "pool-a",
        work_class_resolver=lambda _job: "chat",
    )


@pytest.mark.asyncio
async def test_batch_gateway_paginates_and_refreshes_nonterminal_jobs() -> None:
    base_url = "http://batch.example"
    listed_active = _batch(
        "batch/a b",
        "in_progress",
        total=10,
        completed=1,
        failed=0,
        expires_at=2_000,
    )
    listed_terminal = _batch(
        "batch-done",
        "completed",
        total=4,
        completed=4,
        failed=0,
        expires_at=1_900,
    )
    listed_validating = _batch(
        "batch-new",
        "validating",
        total=0,
        completed=0,
        failed=0,
        expires_at=None,
        pool="pool-b",
        work_class="responses-large",
    )
    detailed_active = _batch(
        "batch/a b",
        "in_progress",
        total=10,
        completed=7,
        failed=1,
        expires_at=2_100,
    )
    detailed_validating = _batch(
        "batch-new",
        "in_progress",
        total=8,
        completed=2,
        failed=0,
        expires_at=2_200,
        pool="pool-b",
        work_class="responses-large",
    )
    responses = {
        (
            f"{base_url}/v1/batches",
            (("after", "0"), ("limit", "2")),
        ): {"data": [listed_active, listed_terminal], "has_more": True},
        (
            f"{base_url}/v1/batches",
            (("after", "2"), ("limit", "2")),
        ): {"data": [listed_validating], "has_more": False},
        (f"{base_url}/v1/batches/batch%2Fa%20b", ()): detailed_active,
        (f"{base_url}/v1/batches/batch-new", ()): detailed_validating,
    }
    session = _FakeSession(responses)
    source = BatchGatewayJobSource(
        base_url=f"{base_url}/",
        session=session,  # type: ignore[arg-type]
        pool_resolver=lambda job: job["metadata"]["pool"],
        work_class_resolver=lambda job: job["metadata"]["work_class"],
        headers={"Authorization": "Bearer test-only"},
        page_size=2,
    )

    jobs = await source.collect_batch_jobs(observed_at_s=1_800.0)

    assert [job.job_id for job in jobs] == [
        "batch/a b",
        "batch-done",
        "batch-new",
    ]
    assert jobs[0].completed_requests == 7
    assert jobs[0].failed_requests == 1
    assert jobs[0].remaining_requests == 2
    assert jobs[0].deadline_at_s == 2_100.0
    assert jobs[1].completed_requests == 4
    assert jobs[2].pool_id == "pool-b"
    assert jobs[2].work_class == "responses-large"
    assert jobs[2].total_requests == 8
    requested_urls = [call[0] for call in session.calls]
    assert requested_urls.count(f"{base_url}/v1/batches/batch-done") == 0
    assert all(
        headers == {"Authorization": "Bearer test-only"}
        for _url, _params, headers in session.calls
    )


@pytest.mark.asyncio
async def test_batch_gateway_rejects_nonprogressing_pagination() -> None:
    base_url = "http://batch.example"
    session = _FakeSession(
        {
            (
                f"{base_url}/v1/batches",
                (("after", "0"), ("limit", "100")),
            ): {"data": [], "has_more": True}
        }
    )
    source = BatchGatewayJobSource(
        base_url=base_url,
        session=session,  # type: ignore[arg-type]
        pool_resolver=lambda _job: "pool-a",
        work_class_resolver=lambda _job: "chat",
    )

    with pytest.raises(ValueError, match="has_more=true with an empty page"):
        await source.collect_batch_jobs(observed_at_s=1.0)


@pytest.mark.asyncio
async def test_batch_gateway_rejects_unknown_status_before_using_stale_counts() -> None:
    base_url = "http://batch.example"
    unknown = _batch(
        "batch-unknown",
        "mystery",
        total=10,
        completed=0,
        failed=0,
        expires_at=None,
    )
    session = _FakeSession(
        {
            (
                f"{base_url}/v1/batches",
                (("after", "0"), ("limit", "100")),
            ): {"data": [unknown], "has_more": False}
        }
    )
    source = BatchGatewayJobSource(
        base_url=base_url,
        session=session,  # type: ignore[arg-type]
        pool_resolver=lambda _job: "pool-a",
        work_class_resolver=lambda _job: "chat",
    )

    with pytest.raises(ValueError, match="unknown status"):
        await source.collect_batch_jobs(observed_at_s=1.0)


@pytest.mark.asyncio
async def test_batch_gateway_prefers_explicit_expiry_over_derivation_fields() -> None:
    job = _batch(
        "batch-explicit",
        "completed",
        total=1,
        completed=1,
        failed=0,
        expires_at=2_000,
        created_at=-1,
        completion_window="not-a-duration",
    )

    jobs = await _source_for_listed_batch(job).collect_batch_jobs(observed_at_s=1.0)

    assert jobs[0].deadline_at_s == 2_000.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "completion_window, duration_ns",
    [
        ("1h2m3s4ms5us6µs7ns", 3_723_004_011_007),
        ("0.5h", 1_800_000_000_000),
        ("+1.25ms", 1_250_000),
        ("1μs", 1_000),
        ("1.s", 1_000_000_000),
        (".5s", 500_000_000),
        ("1ns", 1),
        ("2562047h47m16.854775807s", (1 << 63) - 1),
    ],
)
async def test_batch_gateway_derives_expiry_from_go_duration(
    completion_window: str, duration_ns: int
) -> None:
    created_at = 1
    job = _batch(
        "batch-derived",
        "completed",
        total=1,
        completed=1,
        failed=0,
        expires_at=None,
        created_at=created_at,
        completion_window=completion_window,
    )

    jobs = await _source_for_listed_batch(job).collect_batch_jobs(observed_at_s=1.0)

    expected_deadline_s = created_at + duration_ns / 1_000_000_000
    assert jobs[0].deadline_at_s == pytest.approx(expected_deadline_s)
    assert jobs[0].deadline_at_s <= expected_deadline_s


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "completion_window",
    [
        "",
        "0",
        "0s",
        "0.1ns",
        "-1s",
        "1e3s",
        "١s",
        "1h 2m",
        "24hours",
        "2562047h47m16.854775808s",
    ],
)
async def test_batch_gateway_rejects_invalid_or_nonpositive_completion_window(
    completion_window: str,
) -> None:
    job = _batch(
        "batch-invalid-window",
        "completed",
        total=1,
        completed=1,
        failed=0,
        expires_at=None,
        completion_window=completion_window,
    )

    with pytest.raises(ValueError, match="completion_window|Go duration"):
        await _source_for_listed_batch(job).collect_batch_jobs(observed_at_s=1.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("created_at", [None, -1, True, "1000", 1 << 63])
async def test_batch_gateway_rejects_invalid_created_at(created_at: object) -> None:
    job = _batch(
        "batch-invalid-created",
        "completed",
        total=1,
        completed=1,
        failed=0,
        expires_at=None,
        created_at=created_at,
        completion_window="1h",
    )

    with pytest.raises(ValueError, match="created_at"):
        await _source_for_listed_batch(job).collect_batch_jobs(observed_at_s=1.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("expires_at", [-1, True, "2000", 1 << 63])
async def test_batch_gateway_rejects_invalid_explicit_expiry_without_fallback(
    expires_at: object,
) -> None:
    job = _batch(
        "batch-invalid-explicit",
        "completed",
        total=1,
        completed=1,
        failed=0,
        expires_at=expires_at,
        created_at=1_000,
        completion_window="1h",
    )

    with pytest.raises(ValueError, match="expires_at"):
        await _source_for_listed_batch(job).collect_batch_jobs(observed_at_s=1.0)


@pytest.mark.asyncio
async def test_batch_gateway_requires_completion_window_for_derived_expiry() -> None:
    job = _batch(
        "batch-missing-window",
        "completed",
        total=1,
        completed=1,
        failed=0,
        expires_at=None,
    )
    del job["completion_window"]

    with pytest.raises(ValueError, match="completion_window"):
        await _source_for_listed_batch(job).collect_batch_jobs(observed_at_s=1.0)


@pytest.mark.asyncio
async def test_batch_gateway_rejects_derived_deadline_timestamp_overflow() -> None:
    job = _batch(
        "batch-overflow",
        "completed",
        total=1,
        completed=1,
        failed=0,
        expires_at=None,
        created_at=(1 << 63) - 1,
        completion_window="1s",
    )

    with pytest.raises(ValueError, match="derived deadline overflows"):
        await _source_for_listed_batch(job).collect_batch_jobs(observed_at_s=1.0)


@pytest.mark.asyncio
async def test_batch_gateway_uses_declared_count_while_active_total_is_zero() -> None:
    job = _batch(
        "batch-count-fallback",
        "in_progress",
        total=0,
        completed=0,
        failed=0,
        expires_at=2_000,
    )
    job["metadata"]["planner_request_count"] = "37"

    jobs = await _source_for_listed_batch(job).collect_batch_jobs(observed_at_s=1.0)

    assert jobs[0].total_requests == 37
    assert jobs[0].remaining_requests == 37


@pytest.mark.asyncio
async def test_batch_gateway_rejects_active_zero_total_without_declaration() -> None:
    job = _batch(
        "batch-count-missing",
        "validating",
        total=0,
        completed=0,
        failed=0,
        expires_at=2_000,
    )

    with pytest.raises(ValueError, match="total=0.*planner_request_count"):
        await _source_for_listed_batch(job).collect_batch_jobs(observed_at_s=1.0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "declaration",
    ["0", "-1", "+1", "01", " 1", "1.0", 1, True, None, str(1 << 63)],
)
async def test_batch_gateway_rejects_invalid_planner_request_count(
    declaration: object,
) -> None:
    job = _batch(
        "batch-count-invalid",
        "in_progress",
        total=0,
        completed=0,
        failed=0,
        expires_at=2_000,
    )
    job["metadata"]["planner_request_count"] = declaration

    with pytest.raises(ValueError, match="planner_request_count"):
        await _source_for_listed_batch(job).collect_batch_jobs(observed_at_s=1.0)


@pytest.mark.asyncio
async def test_batch_gateway_rejects_declared_and_live_count_mismatch() -> None:
    job = _batch(
        "batch-count-mismatch",
        "in_progress",
        total=36,
        completed=1,
        failed=0,
        expires_at=2_000,
    )
    job["metadata"]["planner_request_count"] = "37"

    with pytest.raises(ValueError, match="does not match.*total=36"):
        await _source_for_listed_batch(job).collect_batch_jobs(observed_at_s=1.0)


@pytest.mark.asyncio
async def test_batch_gateway_accepts_matching_declared_and_live_count() -> None:
    job = _batch(
        "batch-count-match",
        "in_progress",
        total=37,
        completed=1,
        failed=0,
        expires_at=2_000,
    )
    job["metadata"]["planner_request_count"] = "37"

    jobs = await _source_for_listed_batch(job).collect_batch_jobs(observed_at_s=1.0)

    assert jobs[0].total_requests == 37
    assert jobs[0].remaining_requests == 36


class _JobSource:
    def __init__(self, jobs: list[BatchJobDemand]) -> None:
        self.jobs = jobs
        self.observed_at_s: Optional[float] = None

    async def collect_batch_jobs(self, *, observed_at_s: float) -> list[BatchJobDemand]:
        self.observed_at_s = observed_at_s
        return self.jobs


class _TrafficSource:
    def __init__(self, traffic: list[PoolTrafficDemand]) -> None:
        self.traffic = traffic
        self.observed_at_s: Optional[float] = None

    async def collect_online_traffic(
        self, *, observed_at_s: float
    ) -> list[PoolTrafficDemand]:
        self.observed_at_s = observed_at_s
        return self.traffic


class _FeedbackSource:
    def __init__(self, feedback: list[BatchDispatcherFeedback]) -> None:
        self.feedback = feedback
        self.observed_at_s: Optional[float] = None

    async def collect_dispatcher_feedback(
        self, *, observed_at_s: float
    ) -> list[BatchDispatcherFeedback]:
        self.observed_at_s = observed_at_s
        return self.feedback


class _FailingSource:
    def __init__(self, error: BaseException) -> None:
        self.error = error

    async def collect_batch_jobs(self, *, observed_at_s: float) -> list[BatchJobDemand]:
        raise self.error

    async def collect_online_traffic(
        self, *, observed_at_s: float
    ) -> list[PoolTrafficDemand]:
        raise self.error

    async def collect_dispatcher_feedback(
        self, *, observed_at_s: float
    ) -> list[BatchDispatcherFeedback]:
        raise self.error


class _HangingOptionalSource:
    def __init__(self) -> None:
        self.started = {name: asyncio.Event() for name in ("traffic", "feedback")}
        self.cancelled = {name: asyncio.Event() for name in ("traffic", "feedback")}

    async def _hang(self, name: str) -> None:
        self.started[name].set()
        try:
            await asyncio.Event().wait()
        finally:
            self.cancelled[name].set()

    async def collect_online_traffic(
        self, *, observed_at_s: float
    ) -> list[PoolTrafficDemand]:
        await self._hang("traffic")
        raise AssertionError("unreachable")

    async def collect_dispatcher_feedback(
        self, *, observed_at_s: float
    ) -> list[BatchDispatcherFeedback]:
        await self._hang("feedback")
        raise AssertionError("unreachable")


def _valid_active_job(*, observed_at_s: float = 1.0) -> BatchJobDemand:
    return BatchJobDemand(
        observed_at_s=observed_at_s,
        pool_id="pool-a",
        job_id="job-a",
        status="in_progress",
        total_requests=10,
        completed_requests=3,
        failed_requests=1,
        deadline_at_s=2_000.0,
        work_class="chat",
    )


@pytest.mark.asyncio
async def test_composite_collector_uses_one_sample_timestamp() -> None:
    observed_at_s = 1_234.5
    job_source = _JobSource(
        [
            BatchJobDemand(
                observed_at_s=observed_at_s,
                pool_id="pool-a",
                job_id="job-a",
                status="in_progress",
                total_requests=10,
                completed_requests=3,
                failed_requests=1,
                deadline_at_s=2_000.0,
                work_class="chat",
            )
        ]
    )
    traffic_source = _TrafficSource(
        [
            PoolTrafficDemand(
                observed_at_s=observed_at_s,
                pool_id="pool-a",
                online_offered_rps=5.0,
            )
        ]
    )
    feedback_source = _FeedbackSource(
        [
            BatchDispatcherFeedback(
                observed_at_s=observed_at_s,
                pool_id="pool-a",
                observation_window_s=30.0,
                queued_requests=6,
                inflight_requests=2,
                actual_dispatch_rps=1.5,
                applied_max_admission_rps=2.0,
            )
        ]
    )
    collector = BatchSchedulingCollector(
        batch_jobs=job_source,
        online_traffic=traffic_source,
        dispatcher_feedback=feedback_source,
        clock=lambda: observed_at_s,
    )

    observation = await collector.collect()

    assert job_source.observed_at_s == observed_at_s
    assert traffic_source.observed_at_s == observed_at_s
    assert feedback_source.observed_at_s == observed_at_s
    assert observation.job_demands == job_source.jobs
    assert observation.pool_traffic == traffic_source.traffic
    assert observation.dispatcher_feedback == feedback_source.feedback


@pytest.mark.asyncio
async def test_composite_collector_omits_invalid_optional_traffic() -> None:
    traffic = PoolTrafficDemand(
        observed_at_s=1.0,
        pool_id="pool-a",
        online_offered_rps=1.0,
    )
    traffic.online_offered_rps = math.nan
    job = _valid_active_job()
    collector = BatchSchedulingCollector(
        batch_jobs=_JobSource([job]),
        online_traffic=_TrafficSource([traffic]),
        dispatcher_feedback=_FeedbackSource([]),
        clock=lambda: 1.0,
    )

    observation = await collector.collect()

    assert observation.job_demands == [job]
    assert observation.pool_traffic == []


@pytest.mark.asyncio
async def test_composite_collector_preserves_jobs_when_traffic_fails() -> None:
    job = _valid_active_job()
    feedback = BatchDispatcherFeedback(
        observed_at_s=1.0,
        pool_id="pool-a",
        observation_window_s=10.0,
        queued_requests=6,
        inflight_requests=2,
        actual_dispatch_rps=1.5,
        applied_max_admission_rps=2.0,
    )
    collector = BatchSchedulingCollector(
        batch_jobs=_JobSource([job]),
        online_traffic=_FailingSource(RuntimeError("traffic unavailable")),
        dispatcher_feedback=_FeedbackSource([feedback]),
        clock=lambda: 1.0,
    )

    observation = await collector.collect()

    assert observation.job_demands == [job]
    assert observation.pool_traffic == []
    assert observation.dispatcher_feedback == [feedback]


@pytest.mark.asyncio
async def test_composite_collector_preserves_jobs_and_traffic_when_feedback_fails() -> (
    None
):
    job = _valid_active_job()
    traffic = PoolTrafficDemand(
        observed_at_s=1.0,
        pool_id="pool-a",
        online_offered_rps=0.0,
    )
    collector = BatchSchedulingCollector(
        batch_jobs=_JobSource([job]),
        online_traffic=_TrafficSource([traffic]),
        dispatcher_feedback=_FailingSource(RuntimeError("feedback unavailable")),
        clock=lambda: 1.0,
    )

    observation = await collector.collect()

    assert observation.job_demands == [job]
    assert observation.pool_traffic == [traffic]
    assert observation.dispatcher_feedback == []


@pytest.mark.asyncio
async def test_composite_collector_bounds_and_cancels_stalled_optional_sources() -> (
    None
):
    job = _valid_active_job()
    optional = _HangingOptionalSource()
    collector = BatchSchedulingCollector(
        batch_jobs=_JobSource([job]),
        online_traffic=optional,
        dispatcher_feedback=optional,
        clock=lambda: 1.0,
        optional_source_timeout_seconds=0.01,
    )

    observation = await asyncio.wait_for(collector.collect(), timeout=1.0)

    assert observation.job_demands == [job]
    assert observation.pool_traffic == []
    assert observation.dispatcher_feedback == []
    assert all(event.is_set() for event in optional.started.values())
    assert all(event.is_set() for event in optional.cancelled.values())


@pytest.mark.asyncio
async def test_composite_collector_propagates_gateway_failure() -> None:
    collector = BatchSchedulingCollector(
        batch_jobs=_FailingSource(RuntimeError("gateway unavailable")),
        online_traffic=_TrafficSource([]),
        dispatcher_feedback=_FeedbackSource([]),
        clock=lambda: 1.0,
    )

    with pytest.raises(RuntimeError, match="gateway unavailable"):
        await collector.collect()


@pytest.mark.asyncio
async def test_composite_collector_does_not_degrade_cancellation() -> None:
    collector = BatchSchedulingCollector(
        batch_jobs=_JobSource([]),
        online_traffic=_FailingSource(asyncio.CancelledError()),
        dispatcher_feedback=_FeedbackSource([]),
        clock=lambda: 1.0,
    )

    with pytest.raises(asyncio.CancelledError):
        await collector.collect()


@pytest.mark.asyncio
async def test_composite_collector_revalidates_mutated_contract_objects() -> None:
    job = BatchJobDemand(
        observed_at_s=1.0,
        pool_id="pool-a",
        job_id="job-a",
        status="in_progress",
        total_requests=10,
        completed_requests=3,
        failed_requests=1,
        deadline_at_s=2_000.0,
        work_class="chat",
    )
    job.completed_requests = 11
    collector = BatchSchedulingCollector(
        batch_jobs=_JobSource([job]),
        online_traffic=_TrafficSource([]),
        dispatcher_feedback=_FeedbackSource([]),
        clock=lambda: 1.0,
    )

    with pytest.raises(ValueError, match="must not exceed"):
        await collector.collect()


def _prometheus_vector(value: object) -> list[dict[str, object]]:
    return [{"metric": {}, "value": [1_234.0, value]}]


@pytest.mark.asyncio
async def test_llmd_async_prometheus_source_collects_and_escapes_pool() -> None:
    pool_id = 'batch"\\\n\t'
    queries: list[str] = []

    async def query(promql: str) -> object:
        queries.append(promql)
        if "timestamp(" in promql:
            return _prometheus_vector("2")
        if "broker_backlog_source_available" in promql:
            return _prometheus_vector("1")
        if "broker_backlog" in promql:
            return _prometheus_vector("11")
        if "inflight_requests" in promql:
            return _prometheus_vector("3")
        if "dispatched_requests_total" in promql:
            return _prometheus_vector("2.75")
        if "drain_limit_rps" in promql:
            return _prometheus_vector("4.5")
        if "drain_limit_lease_valid" in promql:
            return _prometheus_vector("1")
        if "drain_limit_valid_until_seconds" in promql:
            return _prometheus_vector("1")
        raise AssertionError(f"unexpected query: {promql}")

    source = LlmdAsyncPrometheusSource(
        pools=[pool_id],
        query=query,
        observation_window_s=45,
        max_sample_age_s=5,
    )

    feedback = await source.collect_dispatcher_feedback(observed_at_s=1_234.0)

    assert feedback == [
        BatchDispatcherFeedback(
            observed_at_s=1_232.0,
            pool_id=pool_id,
            observation_window_s=45.0,
            queued_requests=11,
            inflight_requests=3,
            actual_dispatch_rps=2.75,
            applied_max_admission_rps=4.5,
        )
    ]
    assert len(queries) == 8
    escaped_selector = 'pool_name="batch\\"\\\\\\n\\t"'
    assert all(escaped_selector in promql for promql in queries)
    assert any("[45s]" in promql for promql in queries)
    queued_query = next(
        promql
        for promql in queries
        if "broker_backlog" in promql
        and "broker_backlog_source_available" not in promql
    )
    assert "max by (queue_id, queue_name, pool_name)" in queued_query
    assert "async_queue_depth" in queued_query
    assert "async_broker_backlog" in queued_query.split("or vector(0)")[0]
    assert "or vector(0)" in queued_query
    availability_query = next(
        promql
        for promql in queries
        if "broker_backlog_source_available" in promql and "timestamp(" not in promql
    )
    assert availability_query.startswith("min(")
    assert "or vector(0)" not in availability_query
    sample_age_query = next(promql for promql in queries if "timestamp(" in promql)
    assert "time() - min(timestamp(" in sample_age_query
    assert all(
        "or vector(0)" in promql
        for promql in queries
        if "inflight_requests" in promql or "dispatched_requests_total" in promql
    )
    expiry_query = next(
        promql for promql in queries if "drain_limit_valid_until_seconds" in promql
    )
    assert "> bool time()" in expiry_query
    assert "or vector(0)" in expiry_query
    lease_query = next(
        promql for promql in queries if "drain_limit_lease_valid" in promql
    )
    assert "or vector(0)" in lease_query


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "lease_valid, lease_unexpired",
    [(0.0, 1.0), (1.0, 0.0)],
)
async def test_llmd_async_prometheus_source_maps_invalid_lease_to_unreported_cap(
    lease_valid: float, lease_unexpired: float
) -> None:
    queries: list[str] = []

    async def query(promql: str) -> object:
        queries.append(promql)
        if "timestamp(" in promql:
            return 0.25
        if "broker_backlog_source_available" in promql:
            return 1.0
        if "broker_backlog" in promql:
            return 0.0
        if "inflight_requests" in promql:
            return 0.0
        if "dispatched_requests_total" in promql:
            return 0.0
        if "drain_limit_rps" in promql:
            raise AssertionError("inactive leases must not query the cap")
        if "drain_limit_lease_valid" in promql:
            assert "or vector(0)" in promql
            return lease_valid
        if "drain_limit_valid_until_seconds" in promql:
            assert "or vector(0)" in promql
            return lease_unexpired
        raise AssertionError(f"unexpected query: {promql}")

    source = LlmdAsyncPrometheusSource(pools=["pool-a"], query=query)

    feedback = await source.collect_dispatcher_feedback(observed_at_s=1.0)

    assert feedback[0].applied_max_admission_rps is None
    assert not any("drain_limit_rps" in promql for promql in queries)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "missing_metric",
    ["drain_limit_lease_valid", "drain_limit_valid_until_seconds"],
)
async def test_llmd_async_prometheus_source_maps_missing_lease_signal_to_no_cap(
    missing_metric: str,
) -> None:
    queries: list[str] = []

    async def query(promql: str) -> object:
        queries.append(promql)
        if "timestamp(" in promql:
            return 0.25
        if "broker_backlog_source_available" in promql:
            return 1.0
        if "broker_backlog" in promql:
            return 0.0
        if "inflight_requests" in promql:
            return 0.0
        if "dispatched_requests_total" in promql:
            return 0.0
        if "drain_limit_rps" in promql:
            raise AssertionError("a missing lease signal must not query the cap")
        if "drain_limit_lease_valid" in promql:
            assert "or vector(0)" in promql
            return 0.0 if missing_metric in promql else 1.0
        if "drain_limit_valid_until_seconds" in promql:
            assert "or vector(0)" in promql
            return 0.0 if missing_metric in promql else 1.0
        raise AssertionError(f"unexpected query: {promql}")

    source = LlmdAsyncPrometheusSource(pools=["pool-a"], query=query)

    feedback = await source.collect_dispatcher_feedback(observed_at_s=1.0)

    assert feedback[0].applied_max_admission_rps is None
    assert not any("drain_limit_rps" in promql for promql in queries)


@pytest.mark.asyncio
@pytest.mark.parametrize("availability", [0.0, []])
async def test_llmd_async_prometheus_source_requires_backlog_availability(
    availability: object,
) -> None:
    async def query(promql: str) -> object:
        if "broker_backlog_source_available" in promql:
            return availability
        if "drain_limit_lease_valid" in promql:
            return 1.0
        if "drain_limit_valid_until_seconds" in promql:
            return 1.0
        return 0.0

    source = LlmdAsyncPrometheusSource(pools=["pool-a"], query=query)

    with pytest.raises(ValueError, match="availability|exactly one sample"):
        await source.collect_dispatcher_feedback(observed_at_s=1.0)


@pytest.mark.asyncio
async def test_llmd_async_prometheus_source_accepts_sync_client_envelopes() -> None:
    request_timeouts: list[float] = []

    class _PrometheusClient:
        def custom_query(self, promql: str, *, timeout: float) -> object:
            request_timeouts.append(timeout)
            if "timestamp(" in promql:
                value = "0.25"
            elif "drain_limit_lease_valid" in promql:
                value = "1"
            elif "drain_limit_valid_until_seconds" in promql:
                value = "1"
            elif "drain_limit_rps" in promql:
                value = "2.5"
            elif "broker_backlog_source_available" in promql:
                value = "1"
            elif "broker_backlog" in promql:
                value = "7"
            elif "inflight_requests" in promql:
                value = "2"
            elif "dispatched_requests_total" in promql:
                value = "1.25"
            else:
                raise AssertionError(f"unexpected query: {promql}")
            return {
                "status": "success",
                "data": {"result": _prometheus_vector(value)},
            }

    source = LlmdAsyncPrometheusSource(
        pools=["pool-a"],
        query=_PrometheusClient(),
        request_timeout_seconds=2.5,
    )

    feedback = await source.collect_dispatcher_feedback(observed_at_s=1.0)

    assert feedback[0].queued_requests == 7
    assert feedback[0].inflight_requests == 2
    assert feedback[0].actual_dispatch_rps == 1.25
    assert feedback[0].applied_max_admission_rps == 2.5
    assert request_timeouts == [2.5] * 8


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("sample_age", "error"),
    [
        (-0.1, "non-negative"),
        (60.1, "exceeding"),
        ([], "exactly one sample"),
    ],
)
async def test_llmd_async_prometheus_source_rejects_untrustworthy_sample_age(
    sample_age: object,
    error: str,
) -> None:
    async def query(promql: str) -> object:
        if "timestamp(" in promql:
            return sample_age
        if "broker_backlog_source_available" in promql:
            return 1.0
        if "drain_limit_lease_valid" in promql:
            return 0.0
        if "drain_limit_valid_until_seconds" in promql:
            return 0.0
        return 0.0

    source = LlmdAsyncPrometheusSource(
        pools=["pool-a"], query=query, max_sample_age_s=60.0
    )

    with pytest.raises(ValueError, match=error):
        await source.collect_dispatcher_feedback(observed_at_s=1_234.0)


@pytest.mark.asyncio
async def test_llmd_async_prometheus_source_fails_on_missing_or_fractional_counts() -> (
    None
):
    async def missing_query(_promql: str) -> object:
        return []

    missing_source = LlmdAsyncPrometheusSource(pools=["pool-a"], query=missing_query)
    with pytest.raises(ValueError, match="exactly one sample"):
        await missing_source.collect_dispatcher_feedback(observed_at_s=1.0)

    async def fractional_query(promql: str) -> object:
        if "broker_backlog_source_available" in promql:
            return 1.0
        if "broker_backlog" in promql:
            return 1.5
        if "drain_limit_lease_valid" in promql:
            return 1.0
        return 0.0

    fractional_source = LlmdAsyncPrometheusSource(
        pools=["pool-a"], query=fractional_query
    )
    with pytest.raises(ValueError, match="queued_requests.*integer"):
        await fractional_source.collect_dispatcher_feedback(observed_at_s=1.0)
