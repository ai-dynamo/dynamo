# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment power-config load policy (init-only static caps).

``PlannerEnvironmentImpl._load_static_power_caps_at_startup`` reads DGD-owned
caps once after worker readiness and fails closed on malformed or infeasible
values. The caps are static for the planner lifetime: ``refresh()`` never
re-reads or drift-checks the DGD power annotation — a cap change takes effect
only after a worker rollout plus a Planner restart.
"""

from unittest.mock import Mock

import pytest
from kubernetes.client import ApiException

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.environment.base import PlannerEnvironmentImpl
from dynamo.planner.errors import DeploymentValidationError, PowerAnnotationInvalidError
from dynamo.planner.monitoring.dgd_services import ComponentPowerConfig

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def _cfg(role, gpu_power_limit_watts, gpus_per_replica=1):
    return ComponentPowerConfig(
        component_name=role,
        role=role,
        gpu_power_limit_watts=gpu_power_limit_watts,
        gpus_per_replica=gpus_per_replica,
    )


def _env(
    controller,
    *,
    enable_power=True,
    budget=10000,
    min_endpoint=1,
    require_prefill=True,
    require_decode=True,
):
    config = PlannerConfig.model_construct(
        backend="vllm",
        namespace="ns",
        environment="kubernetes",
        enable_power_awareness=enable_power,
        total_gpu_power_limit=budget,
        min_endpoint=min_endpoint,
    )
    return PlannerEnvironmentImpl(
        config=config,
        controller=controller,
        require_prefill=require_prefill,
        require_decode=require_decode,
    )


def _controller(prefill_watts=700, decode_watts=1200, *, gpus_per_replica=1):
    controller = Mock()
    controller.get_component_power_configs = Mock(
        return_value=(
            _cfg("prefill", prefill_watts, gpus_per_replica),
            _cfg("decode", decode_watts, gpus_per_replica),
        )
    )
    return controller


def test_init_populates_power_watts():
    env = _env(_controller(700, 1200))
    env._load_static_power_caps_at_startup()
    state = env.deployment_state()
    assert state.prefill.power_watts_per_replica == 700
    assert state.decode.power_watts_per_replica == 1200


def test_init_failure_raises_deployment_validation_error():
    controller = _controller()
    controller.get_component_power_configs.side_effect = PowerAnnotationInvalidError(
        "VllmDecodeWorker", "0"
    )
    env = _env(controller)
    with pytest.raises(DeploymentValidationError):
        env._load_static_power_caps_at_startup()


def test_init_infeasible_minimum_footprint_raises():
    env = _env(_controller(700, 1200), budget=1000)
    with pytest.raises(DeploymentValidationError, match="Infeasible power budget"):
        env._load_static_power_caps_at_startup()


def test_awareness_off_is_a_noop():
    controller = _controller()
    env = _env(controller, enable_power=False)
    env._load_static_power_caps_at_startup()
    controller.get_component_power_configs.assert_not_called()
    assert env.deployment_state().prefill.power_watts_per_replica is None


def test_init_transient_apiexception_fails_closed():
    controller = _controller()
    controller.get_component_power_configs.side_effect = ApiException(
        status=503, reason="Service Unavailable"
    )
    env = _env(controller)
    with pytest.raises(DeploymentValidationError):
        env._load_static_power_caps_at_startup()


def test_init_requires_power_capable_connector():
    controller = Mock(spec=[])
    env = _env(controller)
    with pytest.raises(DeploymentValidationError, match="get_component_power_configs"):
        env._load_static_power_caps_at_startup()


def test_restart_adopts_current_cap_not_stale_max():
    fresh_env = _env(_controller(500, 1200))
    fresh_env._load_static_power_caps_at_startup()
    state = fresh_env.deployment_state()
    assert state.prefill.power_watts_per_replica == 500


@pytest.mark.asyncio
async def test_refresh_does_not_reread_power_annotation():
    """Static-cap contract: caps are read once at startup and ``refresh()``
    never re-resolves the DGD power annotation. A cap change takes effect only
    after a worker rollout plus a Planner restart, so refresh() must not call
    ``get_component_power_configs`` nor drift-check against startup values."""
    from unittest.mock import AsyncMock

    controller = Mock()
    controller.get_gpu_counts = Mock(return_value=(1, 1))
    controller.get_component_power_configs = Mock(
        return_value=(_cfg("prefill", 700), _cfg("decode", 1200))
    )
    controller.get_actual_worker_counts = AsyncMock(return_value=(1, 1, True))
    controller.get_model_name = Mock(return_value="test-model")

    env = _env(controller)
    env._load_static_power_caps_at_startup()
    assert env.deployment_state().decode.power_watts_per_replica == 1200
    controller.get_component_power_configs.reset_mock()

    # Even if the DGD annotation would now resolve differently, refresh() does
    # not re-read it: no second resolve, no drift error, cap unchanged.
    controller.get_component_power_configs.return_value = (
        _cfg("prefill", 999),
        _cfg("decode", 999),
    )
    await env.refresh()

    controller.get_component_power_configs.assert_not_called()
    assert env.deployment_state().decode.power_watts_per_replica == 1200


def test_call_with_optional_deployment_forwards_when_accepted():
    seen = {}

    def accepts(*, require_prefill=True, deployment=None):
        seen["deployment"] = deployment
        seen["require_prefill"] = require_prefill
        return (1, 1)

    out = PlannerEnvironmentImpl._call_with_optional_deployment(
        accepts, deployment={"dgd": True}, require_prefill=False
    )
    assert out == (1, 1)
    assert seen == {"deployment": {"dgd": True}, "require_prefill": False}


def test_call_with_optional_deployment_skips_when_unsupported():
    seen = {}

    def legacy(*, require_prefill=True):
        seen["require_prefill"] = require_prefill
        return (2, 2)

    out = PlannerEnvironmentImpl._call_with_optional_deployment(
        legacy, deployment={"dgd": True}, require_prefill=True
    )
    assert out == (2, 2)
    assert seen == {"require_prefill": True}


def test_call_with_optional_deployment_propagates_inner_typeerror():
    """A TypeError raised inside the connector must not be swallowed/retried."""

    def broken(*, require_prefill=True, deployment=None):
        del require_prefill, deployment
        raise TypeError("NoneType is not subscriptable")

    with pytest.raises(TypeError, match="NoneType is not subscriptable"):
        PlannerEnvironmentImpl._call_with_optional_deployment(
            broken, deployment={"dgd": True}, require_prefill=True
        )


@pytest.mark.asyncio
async def test_initialize_caches_caps_from_settled_snapshot_not_lagging_get():
    """Gen-2 lower cap must come from the settled wait snapshot.

    Production race: annotation-only update bumps metadata.generation while
    observedGeneration and worker status still describe gen 1. Counts look
    stable, so a post-wait DGD GET could return the lower annotation before
    Pods roll. initialize() must cache watts from the same snapshot the
    settled wait returned — never a later GET of a lagging generation.
    """
    from unittest.mock import AsyncMock

    lagging = {
        "metadata": {"generation": 2},
        "status": {"observedGeneration": 1},
        "spec": {"components": []},
    }
    settled = {
        "metadata": {"generation": 2},
        "status": {"observedGeneration": 2},
        "spec": {"components": []},
    }
    seen = {"power_deployments": [], "gpu_deployments": []}

    def get_power_configs(
        *,
        require_prefill=True,
        require_decode=True,
        deployment=None,
        prefill_component_name=None,
        decode_component_name=None,
    ):
        del (
            require_prefill,
            require_decode,
            prefill_component_name,
            decode_component_name,
        )
        seen["power_deployments"].append(deployment)
        return (_cfg("prefill", 300), _cfg("decode", 300))

    def get_gpu_counts(*, require_prefill=True, require_decode=True, deployment=None):
        del require_prefill, require_decode
        seen["gpu_deployments"].append(deployment)
        return (1, 1)

    controller = Mock()
    controller.async_init = AsyncMock()
    controller.validate_deployment = AsyncMock()
    controller.wait_for_settled_graph_deployment = AsyncMock(return_value=settled)
    controller.get_graph_deployment = Mock(return_value=lagging)
    controller.get_gpu_counts = get_gpu_counts
    controller.get_actual_worker_counts = AsyncMock(return_value=(1, 1, True))
    controller.get_model_name = Mock(return_value="test-model")
    controller.get_component_power_configs = get_power_configs

    env = _env(controller, budget=10000)
    fpm = Mock()
    fpm.async_init = AsyncMock()
    env.fpm_provider = fpm

    await env.initialize()

    controller.wait_for_settled_graph_deployment.assert_awaited_once_with(
        include_planner=False
    )
    assert seen["power_deployments"][0] is settled
    assert seen["gpu_deployments"][0] is settled
    assert env.deployment_state().decode.power_watts_per_replica == 300
