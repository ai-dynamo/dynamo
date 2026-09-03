# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.control_api import _build_app, _MinimumEndpointUnavailableError
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
        prefill_gpu_cost: int | None = None,
        decode_gpu_cost: int | None = None,
        prefill_watts: int | None = None,
        decode_watts: int | None = None,
    ) -> None:
        self.shutdown_calls = 0
        self.state = DeploymentState()
        self.state.prefill.num_gpus = prefill_gpus
        self.state.decode.num_gpus = decode_gpus
        self.state.prefill.gpus_per_replica = prefill_gpu_cost
        self.state.decode.gpus_per_replica = decode_gpu_cost
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
    prefill_gpu_cost: int | None = None,
    decode_gpu_cost: int | None = None,
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
                prefill_gpu_cost=prefill_gpu_cost,
                decode_gpu_cost=decode_gpu_cost,
            ),
        )


@pytest.mark.asyncio
async def test_disagg_get_and_partial_patch_are_mode_shaped_and_atomic():
    planner = _planner("disagg", min_endpoint=2, prefill_min_endpoint=3)

    assert await planner.get_min_endpoints() == {
        "mode": "disagg",
        "prefill_min_endpoint": 3,
        "decode_min_endpoint": 2,
        "min_gpu_budget": -1,
        "max_gpu_budget": 20,
    }
    assert await planner.patch_min_endpoints({"decode_min_endpoint": 4}) == {
        "mode": "disagg",
        "prefill_min_endpoint": 3,
        "decode_min_endpoint": 4,
        "min_gpu_budget": -1,
        "max_gpu_budget": 20,
    }
    assert planner.config.prefill_min_endpoint == 3
    assert planner.config.decode_min_endpoint == 4


@pytest.mark.asyncio
async def test_agg_runtime_patch_updates_min_endpoint():
    planner = _planner("agg", min_endpoint=2)

    assert await planner.patch_min_endpoints({"min_endpoint": 5}) == {
        "mode": "agg",
        "min_endpoint": 5,
        "min_gpu_budget": -1,
        "max_gpu_budget": 20,
    }
    assert planner.config.min_endpoint == 5

    assert await planner.patch_min_endpoints({"min_endpoint": 0}) == {
        "mode": "agg",
        "min_endpoint": 0,
        "min_gpu_budget": -1,
        "max_gpu_budget": 20,
    }
    assert planner.config.min_endpoint == 0


@pytest.mark.asyncio
async def test_runtime_patch_updates_gpu_budgets_only():
    planner = _planner("disagg")

    assert await planner.patch_min_endpoints(
        {"min_gpu_budget": 12, "max_gpu_budget": 24}
    ) == {
        "mode": "disagg",
        "prefill_min_endpoint": 1,
        "decode_min_endpoint": 1,
        "min_gpu_budget": 12,
        "max_gpu_budget": 24,
    }
    assert planner.config.min_gpu_budget == 12
    assert planner.config.max_gpu_budget == 24


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_http_patch_updates_gpu_budgets_and_returns_complete_configuration():
    from aiohttp.test_utils import TestClient, TestServer

    planner = _planner("disagg")
    client = TestClient(TestServer(_build_app(planner)))
    await client.start_server()
    try:
        response = await client.patch(
            "/v1/min-endpoints",
            json={"min_gpu_budget": 16, "max_gpu_budget": 64},
        )

        assert response.status == 200
        assert await response.json() == {
            "mode": "disagg",
            "prefill_min_endpoint": 1,
            "decode_min_endpoint": 1,
            "min_gpu_budget": 16,
            "max_gpu_budget": 64,
        }
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_runtime_patch_atomically_updates_endpoints_and_gpu_budgets():
    planner = _planner(
        "disagg", prefill_gpu_cost=1, decode_gpu_cost=1, max_gpu_budget=20
    )

    response = await planner.patch_min_endpoints(
        {
            "prefill_min_endpoint": 8,
            "decode_min_endpoint": 8,
            "min_gpu_budget": 16,
            "max_gpu_budget": 64,
        }
    )

    assert response == {
        "mode": "disagg",
        "prefill_min_endpoint": 8,
        "decode_min_endpoint": 8,
        "min_gpu_budget": 16,
        "max_gpu_budget": 64,
    }
    assert planner.config.prefill_min_endpoint == 8
    assert planner.config.decode_min_endpoint == 8
    assert planner.config.min_gpu_budget == 16
    assert planner.config.max_gpu_budget == 64


@pytest.mark.asyncio
async def test_runtime_patch_rejects_invalid_gpu_budget_band_without_mutation():
    planner = _planner("disagg", min_endpoint=2)

    with pytest.raises(ValueError, match="min_gpu_budget=24 exceeds max_gpu_budget=16"):
        await planner.patch_min_endpoints(
            {
                "prefill_min_endpoint": 4,
                "decode_min_endpoint": 4,
                "min_gpu_budget": 24,
                "max_gpu_budget": 16,
            }
        )

    assert planner.config.prefill_min_endpoint is None
    assert planner.config.decode_min_endpoint is None
    assert planner.config.min_gpu_budget == -1
    assert planner.config.max_gpu_budget == 20


@pytest.mark.asyncio
async def test_runtime_patch_uses_prospective_max_for_endpoint_footprint():
    planner = _planner(
        "disagg", prefill_gpu_cost=1, decode_gpu_cost=1, max_gpu_budget=20
    )

    with pytest.raises(
        ValueError, match="requires 16 GPUs, exceeding max_gpu_budget=15"
    ):
        await planner.patch_min_endpoints(
            {
                "prefill_min_endpoint": 8,
                "decode_min_endpoint": 8,
                "max_gpu_budget": 15,
            }
        )

    assert planner.config.prefill_min_endpoint is None
    assert planner.config.decode_min_endpoint is None
    assert planner.config.max_gpu_budget == 20


@pytest.mark.asyncio
async def test_runtime_patch_supports_disabling_each_gpu_budget():
    planner = _planner("disagg", min_gpu_budget=12, max_gpu_budget=20)

    response = await planner.patch_min_endpoints(
        {"min_gpu_budget": -1, "max_gpu_budget": -1}
    )

    assert response["min_gpu_budget"] == -1
    assert response["max_gpu_budget"] == -1


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["prefill", "decode", "agg"])
async def test_gpu_budget_fields_are_active_in_every_mode(mode):
    planner = _planner(mode)

    response = await planner.patch_min_endpoints({"max_gpu_budget": 24})

    assert response["max_gpu_budget"] == 24
    assert planner.config.max_gpu_budget == 24


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
            "min_gpu_budget": -1,
            "max_gpu_budget": 8,
        }

        for payload in (
            {},
            {"unknown": 2},
            {"min_gpu_budget": 4, "unknown": 2},
        ):
            response = await client.patch("/v1/min-endpoints", json=payload)
            assert response.status == 400
            assert planner.config.min_gpu_budget == -1

        for payload in (
            {"prefill_min_endpoint": None},
            {"prefill_min_endpoint": 0},
            {"prefill_min_endpoint": "2"},
            {"prefill_min_endpoint": True},
            {"prefill_min_endpoint": 2.5},
            {"min_gpu_budget": -2},
            {"max_gpu_budget": -2},
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
@pytest.mark.timeout(5)
@pytest.mark.parametrize(
    "mode,inactive_field",
    [
        ("disagg", "min_endpoint"),
        ("prefill", "decode_min_endpoint"),
        ("decode", "prefill_min_endpoint"),
        ("agg", "decode_min_endpoint"),
    ],
)
async def test_http_rejects_inactive_endpoint_without_applying_budget(
    mode, inactive_field
):
    from aiohttp.test_utils import TestClient, TestServer

    planner = _planner(mode)
    client = TestClient(TestServer(_build_app(planner)))
    await client.start_server()
    try:
        response = await client.patch(
            "/v1/min-endpoints",
            json={inactive_field: 2, "max_gpu_budget": 32},
        )
        assert response.status == 422
        assert planner.config.max_gpu_budget == 20
    finally:
        await client.close()


def test_startup_validation_rejects_minimum_gpu_footprint():
    planner = _planner("disagg", max_gpu_budget=2)

    with pytest.raises(DeploymentValidationError, match="requires 3 GPUs"):
        planner._validate_min_endpoint_budgets_at_startup()


def test_startup_validation_rejects_invalid_gpu_budget_band():
    planner = _planner("disagg", min_gpu_budget=21, max_gpu_budget=20)

    with pytest.raises(
        DeploymentValidationError,
        match="min_gpu_budget=21 exceeds max_gpu_budget=20",
    ):
        planner._validate_min_endpoint_budgets_at_startup()


def test_startup_validation_charges_sidecar_gpu_cost():
    planner = _planner(
        "disagg",
        max_gpu_budget=7,
        prefill_gpu_cost=3,
        decode_gpu_cost=5,
    )

    with pytest.raises(DeploymentValidationError, match="requires 8 GPUs"):
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

    assert await patch_task == {
        "mode": "agg",
        "min_endpoint": 2,
        "min_gpu_budget": -1,
        "max_gpu_budget": 20,
    }


@pytest.mark.asyncio
async def test_runtime_api_returns_503_when_decision_lock_times_out():
    from aiohttp.test_utils import TestClient, TestServer

    planner = _planner("agg")
    client = TestClient(TestServer(_build_app(planner)))
    await client.start_server()
    try:
        with (
            patch("dynamo.planner.core.base._CONFIG_LOCK_TIMEOUT_SECONDS", 0),
            pytest.raises(_MinimumEndpointUnavailableError),
        ):
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
async def test_shutdown_retrieves_completed_effect_submission_error(caplog):
    planner = _planner("agg")

    async def fail_submission():
        raise RuntimeError("late submission failure")

    submission_task = asyncio.create_task(fail_submission())
    await asyncio.sleep(0)
    assert submission_task.done()
    planner._effect_submission_task = submission_task

    await planner._shutdown_runtime()

    assert planner._effect_submission_task is None
    assert "Failed pending planner effect submission" in caplog.text
    assert "late submission failure" in caplog.text


@pytest.mark.asyncio
async def test_control_api_port_zero_disables_server_startup():
    planner = _planner("agg", control_api_port=0)
    planner._bootstrap_regression = AsyncMock()
    planner._bootstrap_engine_plugins_if_needed = AsyncMock()

    with patch(
        "dynamo.planner.core.base._start_control_api", new_callable=AsyncMock
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
        "dynamo.planner.core.base._start_control_api",
        new_callable=AsyncMock,
        side_effect=OSError("address already in use"),
    ):
        await planner._async_init()

    assert planner._control_api_runner is None
    assert planner.environment.shutdown_calls == 0
    await planner._shutdown_runtime()
    assert planner.environment.shutdown_calls == 1
