# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock

import pytest

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.environment.base import PlannerEnvironmentImpl
from dynamo.planner.errors import DeploymentValidationError
from dynamo.planner.monitoring.worker_info import (
    WorkerInfo,
    build_worker_info_from_defaults,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def _config(**overrides) -> PlannerConfig:
    values = {
        "namespace": "base-ns",
        "backend": "vllm",
        "mode": "disagg",
        "environment": "kubernetes",
        "prefill_engine_num_gpu": 2,
        "decode_engine_num_gpu": 4,
    }
    values.update(overrides)
    return PlannerConfig.model_construct(**values)


def _controller() -> MagicMock:
    controller = MagicMock()
    controller.async_init = AsyncMock()
    controller.validate_deployment = AsyncMock()
    controller.wait_for_deployment_ready = AsyncMock()
    controller.get_worker_info.side_effect = lambda sub_component_type, backend: (
        build_worker_info_from_defaults(backend, sub_component_type)
    )
    controller.get_gpu_counts.return_value = (2, 4)
    controller.get_actual_worker_counts = AsyncMock(return_value=(2, 3, True))
    controller.get_model_name.return_value = "test-model"
    return controller


def _fpm_provider() -> MagicMock:
    provider = MagicMock()
    provider.async_init = AsyncMock()
    provider.refresh = AsyncMock()
    provider.shutdown = AsyncMock()
    return provider


@pytest.mark.asyncio
async def test_initialize_uses_backend_names_and_resolves_namespace_before_state():
    order = []
    controller = _controller()
    controller.wait_for_deployment_ready = AsyncMock(
        side_effect=lambda **kwargs: order.append("ready")
    )
    controller.get_gpu_counts.side_effect = lambda **kwargs: (
        order.append("gpu") or (2, 4)
    )

    namespace_source = MagicMock()
    namespace_source.runtime_namespace.return_value = "base-ns-workerhash"
    namespace_source.refresh_runtime_namespace = AsyncMock(
        side_effect=lambda: order.append("namespace") or True
    )
    fpm_provider = _fpm_provider()
    fpm_provider.async_init = AsyncMock(
        side_effect=lambda namespace: order.append(f"fpm:{namespace}")
    )
    environment = PlannerEnvironmentImpl(
        config=_config(),
        controller=controller,
        require_prefill=True,
        require_decode=True,
        fpm_provider=fpm_provider,
        runtime_namespace_source=namespace_source,
    )

    await environment.initialize()

    controller.validate_deployment.assert_awaited_once_with(
        prefill_component_name="VllmPrefillWorker",
        decode_component_name="VllmDecodeWorker",
        require_prefill=True,
        require_decode=True,
    )
    assert order.index("ready") < order.index("namespace") < order.index("gpu")
    assert order.index("gpu") < order.index("fpm:base-ns-workerhash")
    fpm_provider.async_init.assert_awaited_once_with("base-ns-workerhash")


@pytest.mark.asyncio
async def test_replica_expected_count_tracks_only_stable_observations():
    controller = _controller()
    controller.get_actual_worker_counts = AsyncMock(
        side_effect=[(2, 3, True), (4, 5, False)]
    )
    environment = PlannerEnvironmentImpl(
        config=_config(),
        controller=controller,
        require_prefill=True,
        require_decode=True,
    )
    environment.deployment_state().prefill.info = WorkerInfo(k8s_name="prefill")
    environment.deployment_state().decode.info = WorkerInfo(k8s_name="decode")

    await environment._refresh_replica_counts()

    assert environment.deployment_state().prefill.replicas.active == 2
    assert environment.deployment_state().prefill.replicas.expected == 2
    assert environment.deployment_state().prefill.replicas.scaling is False
    assert environment.deployment_state().decode.replicas.active == 3
    assert environment.deployment_state().decode.replicas.expected == 3
    assert environment.deployment_state().decode.replicas.scaling is False

    await environment._refresh_replica_counts()

    assert environment.deployment_state().prefill.replicas.active == 4
    assert environment.deployment_state().prefill.replicas.expected is None
    assert environment.deployment_state().prefill.replicas.scaling is True
    assert environment.deployment_state().decode.replicas.active == 5
    assert environment.deployment_state().decode.replicas.expected is None
    assert environment.deployment_state().decode.replicas.scaling is True


def test_gpu_discovery_validation_error_falls_back_without_mutating_config():
    config = _config(prefill_engine_num_gpu=2, decode_engine_num_gpu=4)
    controller = _controller()
    controller.get_gpu_counts.side_effect = DeploymentValidationError(
        ["DGD does not declare GPU resources"]
    )
    environment = PlannerEnvironmentImpl(
        config=config,
        controller=controller,
        require_prefill=True,
        require_decode=True,
    )

    environment._refresh_gpu_counts()

    assert environment.deployment_state().prefill.num_gpus == 2
    assert environment.deployment_state().decode.num_gpus == 4
    assert config.prefill_engine_num_gpu == 2
    assert config.decode_engine_num_gpu == 4


def test_gpu_discovery_failure_retains_last_observed_state():
    config = _config(prefill_engine_num_gpu=None, decode_engine_num_gpu=None)
    controller = _controller()
    controller.get_gpu_counts.side_effect = DeploymentValidationError(
        ["temporary DGD lookup failure"]
    )
    environment = PlannerEnvironmentImpl(
        config=config,
        controller=controller,
        require_prefill=True,
        require_decode=True,
    )
    environment.deployment_state().prefill.num_gpus = 2
    environment.deployment_state().decode.num_gpus = 4

    environment._refresh_gpu_counts()

    assert environment.deployment_state().prefill.num_gpus == 2
    assert environment.deployment_state().decode.num_gpus == 4


@pytest.mark.parametrize(
    ("require_prefill", "require_decode", "missing_field"),
    [
        (True, False, "prefill_engine_num_gpu"),
        (False, True, "decode_engine_num_gpu"),
    ],
)
def test_gpu_refresh_validates_required_widths(
    require_prefill, require_decode, missing_field
):
    config = _config(
        prefill_engine_num_gpu=None,
        decode_engine_num_gpu=None,
    )
    controller = _controller()
    controller.get_gpu_counts.return_value = (None, None)
    environment = PlannerEnvironmentImpl(
        config=config,
        controller=controller,
        require_prefill=require_prefill,
        require_decode=require_decode,
    )

    with pytest.raises(DeploymentValidationError, match=missing_field):
        environment._refresh_gpu_counts()
