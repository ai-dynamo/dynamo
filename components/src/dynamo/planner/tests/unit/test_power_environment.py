# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment power-config refresh policy (Phase 2b).

``PlannerEnvironmentImpl._refresh_power_configs`` is strict at initialization
(any resolve failure aborts startup) and conservative at runtime (a malformed
or changed cap keeps the last-good watts, holds the per-role maximum, and
latches the deployment-scoped scale-up block). Also covers the
``deployment_state_changed`` power comparisons and the restart-adoption path.
"""

from unittest.mock import AsyncMock, Mock

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


# ---------------------------------------------------------------------------
# Initialization: strict / fail-closed
# ---------------------------------------------------------------------------


def test_init_populates_power_watts():
    env = _env(_controller(700, 1200))
    env._refresh_power_configs(is_initialization=True)
    state = env.deployment_state()
    assert state.prefill.power_watts_per_replica == 700
    assert state.decode.power_watts_per_replica == 1200
    assert state.power_scale_up_blocked is False


def test_init_failure_raises_deployment_validation_error():
    controller = _controller()
    controller.get_component_power_configs.side_effect = PowerAnnotationInvalidError(
        "VllmDecodeWorker", "0"
    )
    env = _env(controller)
    with pytest.raises(DeploymentValidationError):
        env._refresh_power_configs(is_initialization=True)


def test_init_infeasible_minimum_footprint_raises():
    # min_endpoint 1 of prefill(700) + decode(1200) = 1900 W > 1000 W budget.
    env = _env(_controller(700, 1200), budget=1000)
    with pytest.raises(DeploymentValidationError, match="Infeasible power budget"):
        env._refresh_power_configs(is_initialization=True)


def test_awareness_off_is_a_noop():
    controller = _controller()
    env = _env(controller, enable_power=False)
    env._refresh_power_configs(is_initialization=True)
    controller.get_component_power_configs.assert_not_called()
    assert env.deployment_state().prefill.power_watts_per_replica is None


# ---------------------------------------------------------------------------
# Runtime: conservative
# ---------------------------------------------------------------------------


def test_runtime_malformed_keeps_last_good_and_blocks():
    controller = _controller(700, 1200)
    env = _env(controller)
    env._refresh_power_configs(is_initialization=True)

    # Runtime refresh now fails to resolve.
    controller.get_component_power_configs.side_effect = PowerAnnotationInvalidError(
        "VllmDecodeWorker", "bad"
    )
    env._refresh_power_configs(is_initialization=False)

    state = env.deployment_state()
    assert state.prefill.power_watts_per_replica == 700  # last good kept
    assert state.decode.power_watts_per_replica == 1200
    assert state.power_scale_up_blocked is True
    assert state.power_scale_up_blocked_reason


def test_runtime_increased_cap_holds_max_and_blocks():
    controller = _controller(700, 1200)
    env = _env(controller)
    env._refresh_power_configs(is_initialization=True)

    # Prefill cap increases at runtime.
    controller.get_component_power_configs.return_value = (
        _cfg("prefill", 900),
        _cfg("decode", 1200),
    )
    env._refresh_power_configs(is_initialization=False)

    state = env.deployment_state()
    assert state.prefill.power_watts_per_replica == 900  # max(700, 900)
    assert state.power_scale_up_blocked is True


def test_runtime_decreased_cap_keeps_larger_and_blocks():
    controller = _controller(700, 1200)
    env = _env(controller)
    env._refresh_power_configs(is_initialization=True)

    # Decode cap decreases at runtime — keep the larger (conservative) value.
    controller.get_component_power_configs.return_value = (
        _cfg("prefill", 700),
        _cfg("decode", 900),
    )
    env._refresh_power_configs(is_initialization=False)

    state = env.deployment_state()
    assert state.decode.power_watts_per_replica == 1200  # max(1200, 900)
    assert state.power_scale_up_blocked is True


def test_runtime_unchanged_cap_does_not_block():
    controller = _controller(700, 1200)
    env = _env(controller)
    env._refresh_power_configs(is_initialization=True)
    env._refresh_power_configs(is_initialization=False)  # same values
    assert env.deployment_state().power_scale_up_blocked is False


def test_runtime_transient_apiexception_keeps_last_good_and_blocks():
    """A transient apiserver error at runtime must not terminate refresh — keep
    last-good caps and block scale-up instead of propagating."""
    controller = _controller(700, 1200)
    env = _env(controller)
    env._refresh_power_configs(is_initialization=True)

    controller.get_component_power_configs.side_effect = ApiException(
        status=500, reason="Server Error"
    )
    env._refresh_power_configs(is_initialization=False)  # must not raise

    state = env.deployment_state()
    assert state.prefill.power_watts_per_replica == 700
    assert state.decode.power_watts_per_replica == 1200
    assert state.power_scale_up_blocked is True


def test_init_transient_apiexception_fails_closed():
    """At startup a transient apiserver error is fatal (fail closed)."""
    controller = _controller()
    controller.get_component_power_configs.side_effect = ApiException(
        status=503, reason="Service Unavailable"
    )
    env = _env(controller)
    with pytest.raises(DeploymentValidationError):
        env._refresh_power_configs(is_initialization=True)


def test_restart_adopts_current_cap_not_stale_max():
    """A cold start (fresh env) after a worker-ready transition adopts the
    current DGD-desired cap directly — not a stale max(old, new) that only a
    continuously-running process would hold."""
    # Prior process ended blocked at a conservative 900; a new process starts
    # fresh and the DGD now settles at 500.
    fresh_env = _env(_controller(500, 1200))
    fresh_env._refresh_power_configs(is_initialization=True)
    state = fresh_env.deployment_state()
    assert state.prefill.power_watts_per_replica == 500  # adopted, not max'd
    assert state.power_scale_up_blocked is False


@pytest.mark.asyncio
async def test_runtime_refresh_keeps_last_good_when_replica_get_fails():
    """Full ``refresh()`` must not abort the tick when replica counts fail.

    Power caps resolve first; a later transport error on
    ``get_actual_worker_counts`` keeps last-good inventory and latches
    scale-up suppression instead of raising out of the planner loop.
    """
    controller = _controller(700, 1200)
    controller.get_gpu_counts = Mock(return_value=(2, 4))
    controller.get_model_name = Mock(return_value="m")
    controller.get_actual_worker_counts = AsyncMock(
        side_effect=ConnectionError("apiserver unreachable")
    )
    env = _env(controller)
    # Seed last-good replica counts as if a prior successful refresh ran.
    state = env.deployment_state()
    state.prefill.replicas.active = 2
    state.prefill.replicas.expected = 2
    state.decode.replicas.active = 2
    state.decode.replicas.expected = 2
    env._refresh_power_configs(is_initialization=True)
    assert state.power_scale_up_blocked is False

    await env.refresh()

    state = env.deployment_state()
    assert state.prefill.replicas.active == 2  # last-good retained
    assert state.decode.replicas.active == 2
    assert state.power_scale_up_blocked is True
    assert "replica count refresh failed" in state.power_scale_up_blocked_reason


@pytest.mark.asyncio
async def test_runtime_refresh_survives_non_api_exception_on_gpu_counts():
    """urllib3-style transport errors (not ApiException) must still fall back."""
    controller = _controller(700, 1200)
    controller.get_gpu_counts = Mock(side_effect=TimeoutError("read timed out"))
    controller.get_actual_worker_counts = AsyncMock(return_value=(2, 2, True))
    controller.get_model_name = Mock(return_value="m")
    env = _env(controller)
    state = env.deployment_state()
    state.prefill.num_gpus = 2
    state.decode.num_gpus = 4
    env._refresh_power_configs(is_initialization=True)
    assert state.power_scale_up_blocked is False

    await env.refresh()

    state = env.deployment_state()
    assert state.prefill.num_gpus == 2
    assert state.decode.num_gpus == 4
    assert state.prefill.power_watts_per_replica == 700
    assert state.power_scale_up_blocked is True
    assert "GPU count refresh failed" in state.power_scale_up_blocked_reason


# ---------------------------------------------------------------------------
# deployment_state_changed power comparisons
# ---------------------------------------------------------------------------


def test_deployment_state_changed_on_watts_change():
    old = DeploymentState()
    new = DeploymentState()
    old.decode.power_watts_per_replica = 1200
    new.decode.power_watts_per_replica = 1000
    assert deployment_state_changed(old, new, False, True) is True


def test_deployment_state_changed_on_blocked_flag_with_unchanged_watts():
    old = DeploymentState()
    new = DeploymentState()
    for s in (old, new):
        s.prefill.power_watts_per_replica = 700
        s.decode.power_watts_per_replica = 1200
    new.power_scale_up_blocked = True
    new.power_scale_up_blocked_reason = "cap changed"
    # Watts identical, but the blocked transition must still register.
    assert deployment_state_changed(old, new, True, True) is True
