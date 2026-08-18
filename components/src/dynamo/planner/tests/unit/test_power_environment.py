# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment power-config load policy (init-only static caps).

``PlannerEnvironmentImpl._load_static_power_caps_at_startup`` reads DGD-owned
caps once after worker readiness and fails closed on malformed or infeasible
values. Caps are startup-static for the Planner lifetime: ``refresh()`` never
re-reads or re-adopts them. DGD admission rejects annotation or topology tuple
changes; adopting new values requires a replacement DGD and Planner process.
"""

from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from kubernetes.client import ApiException

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.connectors.base import (
    DynamoGraphPowerBudgetSnapshot,
    is_power_aware_connector,
)
from dynamo.planner.environment.base import PlannerEnvironmentImpl
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


def _power_controller(**kwargs):
    """Minimal mock accepted by is_power_aware_connector().

    All four protocol methods must be set explicitly on the instance so
    inspect.getattr_static can find them and callable() returns True.
    """
    controller = Mock(**kwargs)
    controller.get_graph_deployment = Mock()
    controller.wait_for_settled_graph_deployment = AsyncMock()
    controller.get_power_aware_worker_counts = AsyncMock(return_value=(1, 1, True))
    return controller


def _controller(prefill_watts=700, decode_watts=1200, *, gpus_per_replica=1):
    controller = _power_controller()
    controller.get_component_power_configs = Mock(
        return_value=(
            _cfg("prefill", prefill_watts, gpus_per_replica),
            _cfg("decode", decode_watts, gpus_per_replica),
        )
    )
    return controller


def _with_transactional_policy(controller, *, budget_watts, min_endpoint):
    controller.get_power_budget_snapshot = Mock(
        return_value=_dgpb_snapshot(budget_watts, min_endpoint)
    )
    controller.consume_power_budget_snapshot = Mock()
    return controller


def _dgpb_snapshot(budget_watts=5000, min_endpoint=1):
    return DynamoGraphPowerBudgetSnapshot(
        budget_watts=budget_watts,
        min_endpoint=min_endpoint,
        phase="Idle",
        inventory_epoch=1,
        rollout_in_progress=False,
        components=(),
        conditions=(),
    )


def _refresh_controller(prefill_cfg, decode_cfg):
    controller = _power_controller()
    controller.get_gpu_counts = Mock(return_value=(1, 1))
    controller.get_component_power_configs = Mock(
        return_value=(prefill_cfg, decode_cfg)
    )
    controller.get_actual_worker_counts = AsyncMock(return_value=(1, 1, True))
    controller.get_model_name = Mock(return_value="test-model")
    return controller


def _worker(name, *, comp_type, watts="300", gpus="1", node_count=None):
    component = {
        "name": name,
        "type": comp_type,
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
    if node_count is not None:
        component["multinode"] = {"nodeCount": node_count}
    return component


def _dgd(*components):
    return {"spec": {"components": list(components)}}


def test_is_power_aware_connector_rejects_bare_magic_mock():
    assert is_power_aware_connector(MagicMock()) is False


def test_is_power_aware_connector_accepts_valid_connector():
    connector = Mock()
    connector.get_graph_deployment = Mock()
    connector.get_component_power_configs = Mock()
    connector.wait_for_settled_graph_deployment = AsyncMock()
    connector.get_power_aware_worker_counts = AsyncMock()
    assert is_power_aware_connector(connector) is True


def test_is_power_aware_connector_rejects_missing_method():
    connector = Mock()
    connector.get_graph_deployment = Mock()
    connector.get_component_power_configs = Mock()
    connector.get_power_aware_worker_counts = AsyncMock()
    # wait_for_settled_graph_deployment absent from __dict__
    assert is_power_aware_connector(connector) is False


def test_is_power_aware_connector_rejects_non_callable_attribute():
    connector = Mock()
    connector.get_graph_deployment = Mock()
    connector.get_component_power_configs = Mock()
    connector.wait_for_settled_graph_deployment = "not_a_function"
    connector.get_power_aware_worker_counts = AsyncMock()
    assert is_power_aware_connector(connector) is False


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


def test_init_uses_cached_dgpb_policy_instead_of_inactive_planner_fields():
    controller = _with_transactional_policy(
        _controller(700, 1200), budget_watts=5000, min_endpoint=1
    )
    env = _env(controller, budget=1000, min_endpoint=9)

    env._load_static_power_caps_at_startup()

    assert env.config.total_gpu_power_limit == 5000
    assert env.config.min_endpoint == 1
    controller.get_power_budget_snapshot.assert_called_once_with()
    controller.consume_power_budget_snapshot.assert_not_called()


def test_init_fails_when_cached_dgpb_policy_is_infeasible():
    controller = _with_transactional_policy(
        _controller(700, 1200), budget_watts=1000, min_endpoint=1
    )
    env = _env(controller, budget=10000, min_endpoint=1)

    with pytest.raises(
        DeploymentValidationError,
        match=r"min_endpoint=1.*total_gpu_power_limit=1000W",
    ):
        env._load_static_power_caps_at_startup()


def test_init_static_connector_keeps_planner_policy():
    env = _env(_controller(700, 1200), budget=5000, min_endpoint=2)

    env._load_static_power_caps_at_startup()

    assert env.config.total_gpu_power_limit == 5000
    assert env.config.min_endpoint == 2


@pytest.mark.asyncio
async def test_startup_discovers_dgpb_before_legacy_settlement_wait():
    controller = _controller()
    snapshot_cache = {"value": None}
    snapshot = _dgpb_snapshot()
    deployment = _dgd(
        _worker("prefill", comp_type="prefill"),
        _worker("decode", comp_type="decode"),
    )

    controller.get_power_budget_snapshot = Mock(
        side_effect=lambda: snapshot_cache["value"]
    )
    controller.consume_power_budget_snapshot = Mock()

    async def observe_dgpb():
        snapshot_cache["value"] = snapshot
        return 1, 1, False

    controller.get_power_aware_worker_counts = AsyncMock(side_effect=observe_dgpb)
    controller.get_graph_deployment.return_value = deployment
    env = _env(controller)

    observed = await env._wait_for_startup_dgd_snapshot()

    assert observed is deployment
    controller.get_power_aware_worker_counts.assert_awaited_once_with()
    controller.wait_for_settled_graph_deployment.assert_not_awaited()
    controller.get_graph_deployment.assert_called_once_with()


@pytest.mark.asyncio
async def test_refresh_starts_dgpb_cycle_before_observing_counts():
    controller = _refresh_controller(_cfg("prefill", 700), _cfg("decode", 1200))
    events = []
    controller.get_power_budget_snapshot = Mock(return_value=_dgpb_snapshot())
    controller.consume_power_budget_snapshot = Mock(
        side_effect=lambda: events.append("consume")
    )

    async def observe_counts(**_kwargs):
        events.append("observe")
        return 1, 1, True

    controller.get_power_aware_worker_counts = AsyncMock(side_effect=observe_counts)
    env = _env(controller)

    await env.refresh()

    assert events == ["consume", "observe"]


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
    with pytest.raises(DeploymentValidationError, match="PowerAwareConnector"):
        env._load_static_power_caps_at_startup()


def test_init_required_prefill_config_none_raises():
    """A connector returning None prefill config when prefill is required must fail closed."""
    controller = _power_controller()
    controller.get_component_power_configs = Mock(
        return_value=(None, _cfg("decode", 300))
    )
    env = _env(controller, require_prefill=True, require_decode=True)
    with pytest.raises(DeploymentValidationError, match="prefill"):
        env._load_static_power_caps_at_startup()


def test_init_required_decode_config_none_raises():
    """A connector returning None decode config when decode is required must fail closed."""
    controller = _power_controller()
    controller.get_component_power_configs = Mock(
        return_value=(_cfg("prefill", 700), None)
    )
    env = _env(controller, require_prefill=True, require_decode=True)
    with pytest.raises(DeploymentValidationError, match="decode"):
        env._load_static_power_caps_at_startup()


@pytest.mark.asyncio
async def test_initialize_power_on_requires_power_aware_connector():
    """Power on + connector missing a PowerAwareConnector method must raise immediately.

    is_power_aware_connector() uses inspect.getattr_static, which bypasses
    Mock's __getattr__. A method not explicitly set on the mock instance is
    invisible to the check, so omitting wait_for_settled_graph_deployment
    triggers DeploymentValidationError rather than silently falling through
    to replica-count readiness and skipping observedGeneration / pod-annotation
    convergence.
    """
    controller = Mock()
    controller.async_init = AsyncMock()
    controller.validate_deployment = AsyncMock()
    controller.get_graph_deployment = Mock()
    controller.get_component_power_configs = Mock(
        return_value=(_cfg("prefill", 700), _cfg("decode", 300))
    )
    controller.get_power_aware_worker_counts = AsyncMock()
    # wait_for_settled_graph_deployment intentionally absent from __dict__:
    # is_power_aware_connector(controller) → False.

    env = _env(controller)
    fpm = Mock()
    fpm.async_init = AsyncMock()
    env.fpm_provider = fpm

    with pytest.raises(DeploymentValidationError, match="PowerAwareConnector"):
        await env.initialize()


def test_restart_adopts_current_cap_not_stale_max():
    fresh_env = _env(_controller(500, 1200))
    fresh_env._load_static_power_caps_at_startup()
    state = fresh_env.deployment_state()
    assert state.prefill.power_watts_per_replica == 500


@pytest.mark.asyncio
async def test_refresh_does_not_reread_power_annotation():
    """refresh() must not call get_component_power_configs after startup.

    Caps are startup-static. Per Ted's review contract, the runtime tick loop
    must not issue a DGD GET to re-resolve or drift-check the annotation —
    annotation changes take effect only after a worker rollout plus Planner
    restart.
    """
    controller = _refresh_controller(_cfg("prefill", 700), _cfg("decode", 1200))
    env = _env(controller)
    env._load_static_power_caps_at_startup()
    assert env.deployment_state().decode.power_watts_per_replica == 1200
    controller.get_component_power_configs.reset_mock()

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
    controller.get_power_aware_worker_counts = AsyncMock(return_value=(1, 1, True))
    controller.get_model_name = Mock(return_value="test-model")
    controller.get_component_power_configs = get_power_configs

    env = _env(controller, budget=10000)
    fpm = Mock()
    fpm.async_init = AsyncMock()
    env.fpm_provider = fpm

    await env.initialize()

    controller.wait_for_settled_graph_deployment.assert_awaited_once_with(
        include_planner=False,
        require_prefill=True,
        require_decode=True,
        prefill_component_name="VllmPrefillWorker",
        decode_component_name="VllmDecodeWorker",
    )
    assert seen["power_deployments"][0] is settled
    assert seen["gpu_deployments"][0] is settled
    assert env.deployment_state().decode.power_watts_per_replica == 300


@pytest.mark.asyncio
async def test_initialize_power_disabled_skips_settled_backing_wait():
    """Power-off init must not take the pod-annotation settled path.

    Every real Kubernetes connector exposes wait_for_settled_graph_deployment;
    using it when enable_power_awareness=False would list Pods for annotation
    verification and can 403 or time out for custom service accounts that only
    had the pre-power RBAC surface.
    """
    backing_lookups = []

    async def wait_settled(*, include_planner=True):
        del include_planner
        backing_lookups.append("settled")
        raise AssertionError("settled/backing wait must not run when power is off")

    controller = Mock()
    controller.async_init = AsyncMock()
    controller.validate_deployment = AsyncMock()
    controller.wait_for_settled_graph_deployment = wait_settled
    controller.wait_for_deployment_ready = AsyncMock()
    controller.get_gpu_counts = Mock(return_value=(1, 1))
    controller.get_actual_worker_counts = AsyncMock(return_value=(1, 1, True))
    controller.get_model_name = Mock(return_value="test-model")
    controller.get_component_power_configs = Mock(
        side_effect=AssertionError("power configs must not be resolved when off")
    )
    controller.get_graph_deployment = Mock(
        side_effect=AssertionError("shared DGD GET is power-gated")
    )

    env = _env(controller, enable_power=False)
    fpm = Mock()
    fpm.async_init = AsyncMock()
    env.fpm_provider = fpm

    await env.initialize()

    controller.wait_for_deployment_ready.assert_awaited_once_with(include_planner=False)
    assert backing_lookups == []
    assert env.deployment_state().prefill.power_watts_per_replica is None
