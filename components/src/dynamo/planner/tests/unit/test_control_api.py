# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.control_api import MinimumEndpointUnavailableError, _build_app
from dynamo.planner.core.base import NativePlannerBase
from dynamo.planner.environment.state import DeploymentState
from dynamo.planner.errors import DeploymentValidationError

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


class _Environment:
    def __init__(
        self,
        *,
        prefill_gpus: int = 2,
        decode_gpus: int = 1,
        prefill_watts: int | None = None,
        decode_watts: int | None = None,
    ) -> None:
        self.shutdown_calls = 0
        self.state = DeploymentState()
        self.state.prefill.num_gpus = prefill_gpus
        self.state.decode.num_gpus = decode_gpus
        self.state.prefill.power_watts_per_replica = prefill_watts
        self.state.decode.power_watts_per_replica = decode_watts

    def deployment_state(self) -> DeploymentState:
        return self.state

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        self.shutdown_calls += 1


def _planner(
    mode: str,
    *,
    prefill_watts: int | None = None,
    decode_watts: int | None = None,
    **overrides,
) -> NativePlannerBase:
    values = {
        "namespace": "test-ns",
        "environment": "virtual",
        "mode": mode,
        "max_gpu_budget": 20,
        "metric_reporting_prometheus_port": 0,
        "live_dashboard_port": 0,
        "control_api_port": 0,
    }
    values.update(overrides)
    config = PlannerConfig(**values)
    with patch("dynamo.planner.core.base.PlannerPrometheusMetrics"):
        return NativePlannerBase(
            None,
            config,
            _Environment(
                prefill_watts=prefill_watts,
                decode_watts=decode_watts,
            ),
        )


@pytest.mark.asyncio
async def test_disagg_get_and_partial_patch_are_mode_shaped_and_atomic():
    planner = _planner("disagg", min_endpoint=2, prefill_min_endpoint=3)

    assert await planner.get_min_endpoints() == {
        "mode": "disagg",
        "prefill_min_endpoint": 3,
        "decode_min_endpoint": 2,
    }
    assert await planner.patch_min_endpoints({"decode_min_endpoint": 4}) == {
        "mode": "disagg",
        "prefill_min_endpoint": 3,
        "decode_min_endpoint": 4,
    }
    assert planner.config.prefill_min_endpoint == 3
    assert planner.config.decode_min_endpoint == 4


@pytest.mark.asyncio
async def test_agg_runtime_patch_updates_legacy_min_endpoint():
    planner = _planner("agg", min_endpoint=2)

    assert await planner.patch_min_endpoints({"min_endpoint": 5}) == {
        "mode": "agg",
        "min_endpoint": 5,
    }
    assert planner.config.min_endpoint == 5

    assert await planner.patch_min_endpoints({"min_endpoint": 0}) == {
        "mode": "agg",
        "min_endpoint": 0,
    }
    assert planner.config.min_endpoint == 0


@pytest.mark.asyncio
async def test_http_validation_and_budget_rejection_leave_config_unchanged():
    from aiohttp.test_utils import TestClient, TestServer

    planner = _planner("disagg", max_gpu_budget=8)
    client = TestClient(TestServer(_build_app(planner)))
    await client.start_server()
    try:
        response = await client.get("/v1/min-endpoints")
        assert response.status == 200
        assert await response.json() == {
            "mode": "disagg",
            "prefill_min_endpoint": 1,
            "decode_min_endpoint": 1,
        }

        for payload in ({}, {"unknown": 2}):
            response = await client.patch("/v1/min-endpoints", json=payload)
            assert response.status == 400

        for payload in (
            {"prefill_min_endpoint": None},
            {"prefill_min_endpoint": 0},
            {"prefill_min_endpoint": "2"},
            {"prefill_min_endpoint": True},
            {"prefill_min_endpoint": 2.5},
        ):
            response = await client.patch("/v1/min-endpoints", json=payload)
            assert response.status == 422

        response = await client.patch(
            "/v1/min-endpoints", json={"prefill_min_endpoint": 4}
        )
        assert response.status == 422
        assert planner.config.prefill_min_endpoint is None

        response = await client.patch(
            "/v1/min-endpoints", json={"decode_min_endpoint": 3}
        )
        assert response.status == 200
        assert planner.config.decode_min_endpoint == 3
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_http_rejects_fields_inactive_for_mode():
    from aiohttp.test_utils import TestClient, TestServer

    planner = _planner("agg")
    client = TestClient(TestServer(_build_app(planner)))
    await client.start_server()
    try:
        response = await client.patch(
            "/v1/min-endpoints", json={"decode_min_endpoint": 2}
        )
        assert response.status == 422
        assert planner.config.min_endpoint == 1
    finally:
        await client.close()


def test_startup_validation_rejects_minimum_gpu_footprint():
    planner = _planner("disagg", max_gpu_budget=2)

    with pytest.raises(DeploymentValidationError, match="requires 3 GPUs"):
        planner._validate_min_endpoint_budgets_at_startup()


@pytest.mark.asyncio
async def test_runtime_patch_rejects_minimum_power_footprint():
    planner = _planner(
        "disagg",
        environment="kubernetes",
        enable_power_awareness=True,
        total_gpu_power_limit=300,
        prefill_watts=100,
        decode_watts=100,
    )

    with pytest.raises(ValueError, match="exceeding total_gpu_power_limit"):
        await planner.patch_min_endpoints({"prefill_min_endpoint": 3})
    assert planner.config.prefill_min_endpoint is None


@pytest.mark.asyncio
async def test_runtime_patch_waits_for_tick_lock():
    planner = _planner("agg")

    async with planner._config_lock:
        patch_task = asyncio.create_task(
            planner.patch_min_endpoints({"min_endpoint": 2})
        )
        await asyncio.sleep(0)
        assert not patch_task.done()

    assert await patch_task == {"mode": "agg", "min_endpoint": 2}


@pytest.mark.asyncio
async def test_runtime_api_returns_503_when_decision_lock_times_out():
    from aiohttp.test_utils import TestClient, TestServer

    planner = _planner("agg")
    client = TestClient(TestServer(_build_app(planner)))
    await client.start_server()
    try:
        with patch(
            "dynamo.planner.core.base._CONFIG_LOCK_TIMEOUT_SECONDS", 0
        ), pytest.raises(MinimumEndpointUnavailableError):
            async with planner._config_lock:
                await planner.get_min_endpoints()

        with patch("dynamo.planner.core.base._CONFIG_LOCK_TIMEOUT_SECONDS", 0):
            async with planner._config_lock:
                get_response = await client.get("/v1/min-endpoints")
                patch_response = await client.patch(
                    "/v1/min-endpoints", json={"min_endpoint": 2}
                )
        assert get_response.status == 503
        assert patch_response.status == 503
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_shutdown_finalizes_diagnostics_once():
    planner = _planner("agg")
    planner._recorder = MagicMock()

    await planner._shutdown_runtime()
    await planner._shutdown_runtime()

    planner._recorder.finalize.assert_called_once_with()


@pytest.mark.asyncio
async def test_control_api_port_zero_disables_server_startup():
    planner = _planner("agg", control_api_port=0)
    planner._bootstrap_regression = AsyncMock()
    planner._bootstrap_engine_plugins_if_needed = AsyncMock()

    with patch(
        "dynamo.planner.core.base.start_control_api", new_callable=AsyncMock
    ) as start_api:
        await planner._async_init()

    start_api.assert_not_awaited()
    await planner._shutdown_runtime()


@pytest.mark.asyncio
async def test_async_init_budget_failure_shuts_down_environment():
    planner = _planner("disagg", max_gpu_budget=2)

    with pytest.raises(DeploymentValidationError, match="requires 3 GPUs"):
        await planner._async_init()

    assert planner.environment.shutdown_calls == 1


@pytest.mark.asyncio
async def test_async_init_environment_failure_still_runs_shutdown():
    planner = _planner("agg")
    planner.environment.initialize = AsyncMock(
        side_effect=RuntimeError("partial initialization failed")
    )

    with pytest.raises(RuntimeError, match="partial initialization failed"):
        await planner._async_init()

    assert planner.environment.shutdown_calls == 1


@pytest.mark.asyncio
async def test_async_init_control_api_failure_disables_api_only():
    planner = _planner("agg", control_api_port=9086)
    planner._bootstrap_regression = AsyncMock()
    planner._bootstrap_engine_plugins_if_needed = AsyncMock()

    with patch(
        "dynamo.planner.core.base.start_control_api",
        new_callable=AsyncMock,
        side_effect=OSError("address already in use"),
    ):
        await planner._async_init()

    assert planner._control_api_runner is None
    assert planner.environment.shutdown_calls == 0
    await planner._shutdown_runtime()
    assert planner.environment.shutdown_calls == 1
