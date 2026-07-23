# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment power-config load policy (init-only static caps).

``PlannerEnvironmentImpl._load_static_power_caps_at_startup`` reads DGD-owned
caps once after worker readiness and fails closed on malformed or infeasible
values. Runtime refresh does not re-read caps.
"""

from unittest.mock import Mock

import pytest
from kubernetes.client import ApiException

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.util import deployment_state_changed
from dynamo.planner.environment.base import PlannerEnvironmentImpl
from dynamo.planner.environment.state import DeploymentState
from dynamo.planner.errors import DeploymentValidationError, PowerAnnotationInvalidError
from dynamo.planner.monitoring.dgd_services import ComponentPowerConfig

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def _cfg(role, watts_per_replica):
    return ComponentPowerConfig(
        component_name=role,
        role=role,
        gpu_power_limit_watts=watts_per_replica,
        gpus_per_replica=1,
    )


def _env(controller, *, enable_power=True, budget=10000, min_endpoint=1):
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
        require_prefill=True,
        require_decode=True,
    )


def _controller(prefill_watts=700, decode_watts=1200):
    controller = Mock()
    controller.get_component_power_configs = Mock(
        return_value=(_cfg("prefill", prefill_watts), _cfg("decode", decode_watts))
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


def test_deployment_state_changed_on_watts_change():
    old = DeploymentState()
    new = DeploymentState()
    old.decode.power_watts_per_replica = 1200
    new.decode.power_watts_per_replica = 1000
    assert deployment_state_changed(old, new, False, True) is True


def test_runtime_gpus_per_replica_change_fails_closed():
    """A worker rollout that changes per-pod GPUs must not run against the
    startup-cached wattage; the runtime guard fails closed and demands a
    restart instead of projecting the budget from stale per-replica watts."""
    env = _env(_controller(700, 1200))
    state = env.deployment_state()
    state.prefill.num_gpus = 1
    state.decode.num_gpus = 1
    env._load_static_power_caps_at_startup()

    state.prefill.num_gpus = 4  # rollout bumped per-pod GPUs 1 -> 4
    with pytest.raises(DeploymentValidationError, match="topology changed"):
        env._assert_power_topology_static()


def test_runtime_topology_unchanged_passes():
    env = _env(_controller(700, 1200))
    state = env.deployment_state()
    state.prefill.num_gpus = 2
    state.decode.num_gpus = 4
    env._load_static_power_caps_at_startup()

    env._assert_power_topology_static()


def test_topology_guard_noop_when_awareness_off():
    env = _env(_controller(), enable_power=False)
    state = env.deployment_state()
    state.prefill.num_gpus = 1
    state.decode.num_gpus = 1
    env._load_static_power_caps_at_startup()

    state.prefill.num_gpus = 8
    env._assert_power_topology_static()
