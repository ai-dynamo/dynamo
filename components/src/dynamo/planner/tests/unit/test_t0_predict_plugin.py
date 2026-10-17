# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Traffic sampling and forecast contract; no model downloads required."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Sequence

import pytest

from dynamo.planner.examples.external_plugin.t0_beta import predictor
from dynamo.planner.plugins.proto.v1 import plugin_pb2 as pb

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


class Clock:
    now: float = 0.0

    def __call__(self) -> float:
        return self.now


def request(count: float = 10.0, duration: float = 60.0) -> pb.PredictStageRequest:
    return pb.PredictStageRequest(
        context=pb.PipelineContext(
            request_id=str(count),
            observations=pb.ObservationData(
                traffic=pb.TrafficMetrics(
                    duration_s=duration, num_req=count, isl=100, osl=20
                )
            ),
        )
    )


class Forecast:
    def __call__(
        self, history: Sequence[tuple[float, float, float]], quantile: float
    ) -> tuple[float, float, float]:
        # Catch incorrect history ordering, truncation, units, or quantile forwarding.
        assert list(history) == [(20.0, 100.0, 20.0), (30.0, 100.0, 20.0)]
        assert quantile == 0.9
        return (90.0, 150.0, 25.0)


@pytest.mark.asyncio
async def test_idle_windows_keep_last_known_request_shape() -> None:
    clock = Clock()

    def forecast(
        history: Sequence[tuple[float, float, float]], quantile: float
    ) -> tuple[float, float, float]:
        assert list(history) == [(10.0, 100.0, 20.0), (0.0, 100.0, 20.0)]
        return (10.0, 100.0, 20.0)

    plugin = predictor.T0Predictor(forecast, min_history=2, clock=clock)
    await plugin.predict(request(10))
    clock.now = 60
    idle = request(0)
    idle.context.observations.traffic.isl = 0
    idle.context.observations.traffic.osl = 0
    result = await plugin.predict(idle)
    assert result.predictions.predicted_isl == 100


@pytest.mark.asyncio
async def test_idle_startup_waits_for_a_known_request_shape() -> None:
    plugin = predictor.T0Predictor(Forecast(), min_history=2)
    idle = request(0)
    idle.context.observations.traffic.isl = 0
    idle.context.observations.traffic.osl = 0
    result = await plugin.predict(idle)
    assert result.reason == "idle_without_history"


@pytest.mark.asyncio
async def test_old_samples_are_evicted() -> None:
    clock = Clock()

    def forecast(
        history: Sequence[tuple[float, float, float]], quantile: float
    ) -> tuple[float, float, float]:
        return (sum(row[0] for row in history), 100.0, 20.0)

    plugin = predictor.T0Predictor(
        forecast, context_length=2, min_history=2, clock=clock
    )
    await plugin.predict(request(10))
    for now, count, expected in [
        (60.0, 20.0, 30.0),
        (120.0, 30.0, 50.0),
        (180.0, 40.0, 70.0),
    ]:
        clock.now = now
        result = await plugin.predict(request(count))
        assert result.predictions.predicted_num_req == expected


@pytest.mark.asyncio
async def test_forecast_uses_bounded_history_and_leaves_metadata_to_builtin() -> None:
    clock = Clock()
    plugin = predictor.T0Predictor(
        Forecast(), context_length=2, min_history=2, clock=clock
    )
    warmup = await plugin.predict(request(10))
    assert not warmup.HasField("predictions")
    # Ignore a mismatched window rather than treating request counts as rates.
    clock.now = 60
    assert not (await plugin.predict(request(15, duration=30))).HasField("predictions")
    clock.now = 120
    assert not (await plugin.predict(request(20))).HasField("predictions")
    clock.now = 180
    result = await plugin.predict(request(30))
    assert result.predictions.predicted_num_req == 90.0
    assert result.predictions.predicted_isl == 150.0
    assert result.predictions.predicted_osl == 25.0
    assert not result.predictions.HasField("predicted_kv_hit_rate")
    assert not result.predictions.HasField("predicted_accept_length")
    assert not result.final


@pytest.mark.asyncio
async def test_repeated_tick_and_gap_do_not_create_fake_history() -> None:
    clock = Clock()
    plugin = predictor.T0Predictor(
        Forecast(), context_length=2, min_history=2, clock=clock
    )
    await plugin.predict(request(10))
    assert not (await plugin.predict(request(10))).HasField("predictions")
    clock.now = 180
    assert not (await plugin.predict(request(20))).HasField("predictions")
    clock.now = 240
    assert (await plugin.predict(request(30))).predictions.predicted_num_req == 90


@pytest.mark.parametrize("count", [float("nan"), float("inf"), -1.0])
@pytest.mark.asyncio
async def test_invalid_observations_do_not_reach_model(count: float) -> None:
    def unexpected(
        history: Sequence[tuple[float, float, float]], quantile: float
    ) -> tuple[float, float, float]:
        pytest.fail("invalid observation reached model")

    plugin = predictor.T0Predictor(unexpected, min_history=2)
    assert not (await plugin.predict(request(count))).HasField("predictions")


@pytest.mark.asyncio
async def test_negative_predictions_are_clamped() -> None:
    clock = Clock()
    plugin = predictor.T0Predictor(
        lambda history, quantile: (-3.0, -2.0, -1.0), min_history=2, clock=clock
    )
    await plugin.predict(request(10))
    clock.now = 60
    result = await plugin.predict(request(20))
    assert result.predictions.predicted_num_req == 0
    assert result.predictions.predicted_isl == 1
    assert result.predictions.predicted_osl == 1


@pytest.mark.asyncio
async def test_nonfinite_model_output_fails_for_planner_fallback() -> None:
    clock = Clock()
    plugin = predictor.T0Predictor(
        lambda history, quantile: (float("nan"), 100.0, 20.0),
        min_history=2,
        clock=clock,
    )
    await plugin.predict(request(10))
    clock.now = 60
    with pytest.raises(ValueError, match="finite"):
        await plugin.predict(request(20))


@pytest.mark.asyncio
async def test_model_failure_is_visible_and_later_ticks_can_recover() -> None:
    clock = Clock()
    calls: int = 0

    def failing(
        history: Sequence[tuple[float, float, float]], quantile: float
    ) -> tuple[float, float, float]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("model unavailable")
        return (70.0, 100.0, 20.0)

    plugin = predictor.T0Predictor(failing, min_history=2, clock=clock)
    await plugin.predict(request(10))
    clock.now = 60
    with pytest.raises(RuntimeError, match="model unavailable"):
        await plugin.predict(request(20))
    clock.now = 120
    assert (await plugin.predict(request(30))).predictions.predicted_num_req == 70.0


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_cancelled_rpc_does_not_start_overlapping_inference() -> None:
    clock = Clock()
    started = threading.Event()
    release = threading.Event()

    def blocking(
        history: Sequence[tuple[float, float, float]], quantile: float
    ) -> tuple[float, float, float]:
        started.set()
        if not release.wait(timeout=5):
            raise TimeoutError("test did not release inference")
        return (40.0, 100.0, 20.0)

    plugin = predictor.T0Predictor(blocking, min_history=2, clock=clock)
    await plugin.predict(request(10))
    clock.now = 60
    task = asyncio.create_task(plugin.predict(request(20)))
    try:
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        clock.now = 120
        result = await plugin.predict(request(30))
        assert result.reason == "inference_busy"
        assert not result.HasField("predictions")
    finally:
        release.set()
        await plugin.close()
    clock.now = 180
    assert (await plugin.predict(request(40))).predictions.predicted_num_req == 40.0


@pytest.mark.parametrize(
    "construct",
    [
        lambda: predictor.T0Predictor(Forecast(), quantile=0.0),
        lambda: predictor.T0Predictor(Forecast(), quantile=float("nan")),
        lambda: predictor.T0Predictor(Forecast(), interval_seconds=0),
        lambda: predictor.T0Predictor(Forecast(), interval_seconds=float("inf")),
        lambda: predictor.T0Predictor(Forecast(), min_history=1),
        lambda: predictor.T0Predictor(Forecast(), context_length=1),
    ],
)
def test_invalid_config_fails_at_startup(
    construct: Callable[[], predictor.T0Predictor]
) -> None:
    with pytest.raises(ValueError):
        construct()
