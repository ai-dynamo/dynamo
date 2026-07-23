# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment power-config load policy (init-only static caps).

``PlannerEnvironmentImpl._load_static_power_caps_at_startup`` reads DGD-owned
caps once after worker readiness and fails closed on malformed or infeasible
values. Runtime refresh does not re-adopt caps, but re-resolves topology via
the same power-config path to fail closed if GPUs-per-replica drifts.
"""

import os
from contextlib import contextmanager
from unittest.mock import Mock, patch

import pytest
from kubernetes.client import ApiException

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.connectors.kubernetes import KubernetesConnector
from dynamo.planner.core.util import deployment_state_changed
from dynamo.planner.environment.base import PlannerEnvironmentImpl
from dynamo.planner.environment.state import DeploymentState
from dynamo.planner.errors import DeploymentValidationError, PowerAnnotationInvalidError
from dynamo.planner.monitoring.dgd_services import (
    POWER_ANNOTATION_KEY,
    ComponentPowerConfig,
)

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


def _worker(name, *, comp_type, watts="300", gpus="1"):
    return {
        "name": name,
        "type": comp_type,
        "replicas": 1,
        "podTemplate": {
            "metadata": {"annotations": {POWER_ANNOTATION_KEY: watts}},
            "spec": {
                "containers": [
                    {
                        "name": "main",
                        "resources": {"limits": {"nvidia.com/gpu": gpus}},
                    }
                ]
            },
        },
    }


def _dgd(*components):
    return {
        "metadata": {"name": "power-topology-test"},
        "spec": {"components": list(components)},
    }


@contextmanager
def _k8s_env(dgd, *, require_prefill, require_decode):
    """Real KubernetesConnector over a mocked apiserver returning ``dgd``."""
    with (
        patch("dynamo.planner.connectors.clients.kubernetes_api.config"),
        patch("dynamo.planner.connectors.clients.kubernetes_api.client.CoreV1Api"),
        patch(
            "dynamo.planner.connectors.clients.kubernetes_api.client.CustomObjectsApi"
        ) as custom,
        patch.dict(os.environ, {"DYN_PARENT_DGD_K8S_NAME": "power-topology-test"}),
    ):
        custom_api = custom.return_value
        custom_api.get_namespaced_custom_object.return_value = dgd
        connector = KubernetesConnector("test-ns")
        env = _env(
            connector,
            require_prefill=require_prefill,
            require_decode=require_decode,
        )
        yield env, custom_api


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


def test_deployment_state_changed_on_watts_change():
    old = DeploymentState()
    new = DeploymentState()
    old.decode.power_watts_per_replica = 1200
    new.decode.power_watts_per_replica = 1000
    assert deployment_state_changed(old, new, False, True) is True


def test_runtime_gpus_per_replica_change_fails_closed():
    """A worker rollout that changes GPUs/replica must not run against the
    startup-cached wattage; the runtime guard fails closed and demands a
    restart instead of projecting the budget from stale per-replica watts."""
    controller = Mock()
    controller.get_component_power_configs = Mock(
        side_effect=[
            (_cfg("prefill", 300, gpus_per_replica=1), _cfg("decode", 300, 1)),
            (_cfg("prefill", 300, gpus_per_replica=4), _cfg("decode", 300, 1)),
        ]
    )
    env = _env(controller)
    env._load_static_power_caps_at_startup()
    # Caps stay at the startup-adopted values; only the fingerprint is checked.
    assert env.deployment_state().prefill.power_watts_per_replica == 300

    with pytest.raises(DeploymentValidationError, match="power configuration changed"):
        env._assert_power_config_static()
    assert env.deployment_state().prefill.power_watts_per_replica == 300


def test_runtime_topology_unchanged_passes():
    env = _env(_controller(700, 1200, gpus_per_replica=2))
    env._load_static_power_caps_at_startup()
    env._assert_power_config_static()


def test_topology_guard_noop_when_awareness_off():
    controller = _controller()
    env = _env(controller, enable_power=False)
    env._load_static_power_caps_at_startup()
    env._assert_power_config_static()
    controller.get_component_power_configs.assert_not_called()


def test_topology_guard_fails_closed_when_re_resolve_fails():
    controller = Mock()
    controller.get_component_power_configs = Mock(
        side_effect=[
            (_cfg("prefill", 300, 1), _cfg("decode", 300, 1)),
            ApiException(status=503, reason="Service Unavailable"),
        ]
    )
    env = _env(controller)
    env._load_static_power_caps_at_startup()
    with pytest.raises(DeploymentValidationError, match="re-verify"):
        env._assert_power_config_static()


def test_topology_guard_sees_agg_generic_worker_gpu_change():
    """Aggregate ``type: worker`` is invisible to ``get_gpu_counts``; the
    power-path guard must still fail closed when that worker's GPUs change."""
    dgd_v1 = _dgd(_worker("VllmWorker", comp_type="worker", watts="300", gpus="1"))
    dgd_v2 = _dgd(_worker("VllmWorker", comp_type="worker", watts="300", gpus="4"))
    with _k8s_env(dgd_v1, require_prefill=False, require_decode=True) as (
        env,
        custom_api,
    ):
        env._load_static_power_caps_at_startup()
        assert env.deployment_state().decode.power_watts_per_replica == 300

        # Prove the shared GPU path cannot see this worker (the old blind spot).
        with pytest.raises(DeploymentValidationError):
            env.controller.get_gpu_counts(require_prefill=False, require_decode=True)

        custom_api.get_namespaced_custom_object.return_value = dgd_v2
        with pytest.raises(DeploymentValidationError, match="Restart the Planner"):
            env._assert_power_config_static()
        # Caps must not be re-adopted from the new topology.
        assert env.deployment_state().decode.power_watts_per_replica == 300


def test_topology_guard_sees_named_generic_worker_gpu_change():
    """Named disagg ``type: worker`` components need explicit-name resolution;
    the power-path guard must observe a GPU-resource change on those roles."""
    dgd_v1 = _dgd(
        _worker("VllmPrefillWorker", comp_type="worker", watts="350", gpus="1"),
        _worker("VllmDecodeWorker", comp_type="worker", watts="300", gpus="1"),
    )
    dgd_v2 = _dgd(
        _worker("VllmPrefillWorker", comp_type="worker", watts="350", gpus="1"),
        _worker("VllmDecodeWorker", comp_type="worker", watts="300", gpus="4"),
    )
    with _k8s_env(dgd_v1, require_prefill=True, require_decode=True) as (
        env,
        custom_api,
    ):
        env._load_static_power_caps_at_startup()
        assert env.deployment_state().prefill.power_watts_per_replica == 350
        assert env.deployment_state().decode.power_watts_per_replica == 300

        with pytest.raises(DeploymentValidationError):
            env.controller.get_gpu_counts(require_prefill=True, require_decode=True)

        custom_api.get_namespaced_custom_object.return_value = dgd_v2
        with pytest.raises(DeploymentValidationError, match="Restart the Planner"):
            env._assert_power_config_static()
        assert env.deployment_state().decode.power_watts_per_replica == 300


def test_guard_sees_cap_only_annotation_change():
    """A cap-only DGD annotation edit (same GPUs/replica) must still fail closed.

    After the worker rollout settles on the higher cap, the Planner would
    otherwise resume scale-up against the lower startup-cached
    ``power_watts_per_replica`` and admit an over-budget projection.
    """
    dgd_v1 = _dgd(
        _worker("VllmPrefillWorker", comp_type="prefill", watts="300", gpus="4"),
        _worker("VllmDecodeWorker", comp_type="decode", watts="300", gpus="4"),
    )
    dgd_v2 = _dgd(
        _worker("VllmPrefillWorker", comp_type="prefill", watts="400", gpus="4"),
        _worker("VllmDecodeWorker", comp_type="decode", watts="300", gpus="4"),
    )
    with _k8s_env(dgd_v1, require_prefill=True, require_decode=True) as (
        env,
        custom_api,
    ):
        env._load_static_power_caps_at_startup()
        # 300 W × 4 GPUs = 1200 W/replica at startup.
        assert env.deployment_state().prefill.power_watts_per_replica == 1200

        custom_api.get_namespaced_custom_object.return_value = dgd_v2
        with pytest.raises(DeploymentValidationError, match="Restart the Planner"):
            env._assert_power_config_static()
        # Must not silently adopt the new 400 W × 4 = 1600 W projection.
        assert env.deployment_state().prefill.power_watts_per_replica == 1200
        assert env.deployment_state().prefill.power_gpu_limit_watts == 300


def test_topology_guard_fails_closed_without_startup_fingerprint():
    """Awareness on without initialize()/startup load must not silently skip."""
    env = _env(_controller())
    with pytest.raises(
        DeploymentValidationError, match="fingerprint was never captured"
    ):
        env._assert_power_config_static()


@pytest.mark.asyncio
async def test_refresh_shares_one_dgd_fetch_for_gpu_and_power():
    """When power awareness is on, GPU counts and the fingerprint guard share
    one ``get_graph_deployment`` result (no second DGD GET for power)."""
    from unittest.mock import AsyncMock

    deployment = {"shared": True}
    controller = Mock()
    controller.get_graph_deployment = Mock(return_value=deployment)
    controller.get_gpu_counts = Mock(return_value=(1, 1))
    controller.get_component_power_configs = Mock(
        return_value=(_cfg("prefill", 700), _cfg("decode", 1200))
    )
    controller.get_actual_worker_counts = AsyncMock(return_value=(1, 1, True))
    controller.get_model_name = Mock(return_value="test-model")

    env = _env(controller)
    env._load_static_power_caps_at_startup()
    controller.get_graph_deployment.reset_mock()
    controller.get_gpu_counts.reset_mock()
    controller.get_component_power_configs.reset_mock()

    await env.refresh()

    controller.get_graph_deployment.assert_called_once()
    controller.get_gpu_counts.assert_called_once()
    assert controller.get_gpu_counts.call_args.kwargs["deployment"] is deployment
    controller.get_component_power_configs.assert_called_once()
    assert (
        controller.get_component_power_configs.call_args.kwargs["deployment"]
        is deployment
    )
