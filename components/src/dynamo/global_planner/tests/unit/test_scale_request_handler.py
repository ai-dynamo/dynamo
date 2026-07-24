# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ScaleRequestHandler."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dynamo.global_planner.orchestrator import PoolIntent
from dynamo.global_planner.scale_handler import ScaleRequestHandler
from dynamo.planner import KubernetesConnector, SubComponentType, TargetReplica
from dynamo.planner.connectors.protocol import ScaleRequest
from dynamo.planner.errors import DynamoGraphDeploymentNotReadyError

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
    pytest.mark.filterwarnings("ignore::pydantic.warnings.PydanticDeprecatedSince20"),
]

# TODO(global-planner): Add invariant coverage for HA/shared state, cancellation
# and intermediate PATCH failure, out-of-bound recovery, selected-only tolerance,
# and duplicate-target execution; patch discovery by default in handler unit
# tests and repair the concurrent-insert test so its _read_pools monkeypatch
# accepts role_hints and actually runs instead of having TypeError swallowed.


@pytest.fixture
def mock_runtime():
    """Create a mock DistributedRuntime."""
    return MagicMock()


@pytest.mark.asyncio
async def test_handler_authorization_success(mock_runtime):
    """Test handler authorizes requests from managed namespaces."""
    handler = ScaleRequestHandler(
        runtime=mock_runtime, managed_namespaces=["app-ns"], k8s_namespace="default"
    )

    request = ScaleRequest(
        caller_namespace="app-ns",
        graph_deployment_name="my-dgd",
        k8s_namespace="default",
        target_replicas=[
            TargetReplica(
                sub_component_type=SubComponentType.PREFILL, desired_replicas=3
            )
        ],
    )

    # Mock KubernetesConnector
    with patch(
        "dynamo.global_planner.kubernetes_capacity_manager.KubernetesConnector"
    ) as mock_connector_cls:
        mock_connector = AsyncMock()
        mock_connector_cls.return_value = mock_connector
        mock_connector._async_init = AsyncMock()
        mock_connector.set_component_replicas = AsyncMock()
        mock_connector.kube_api = MagicMock()
        mock_connector.kube_api.get_graph_deployment = MagicMock(
            return_value=_dgd_spec(prefill_replicas=3, decode_replicas=5)
        )

        # Process request (pass as dict to match endpoint behavior)
        results = []
        async for response in handler.scale_request(request.model_dump()):
            results.append(response)

        assert len(results) == 1
        response = results[0]
        assert response["status"] == "success"
        assert "Scaled" in response["message"]
        assert response["current_replicas"]["prefill"] == 3
        assert response["current_replicas"]["decode"] == 5


@pytest.mark.asyncio
async def test_handler_authorization_failure(mock_runtime):
    """Test handler rejects requests from unauthorized namespaces."""
    handler = ScaleRequestHandler(
        runtime=mock_runtime,
        managed_namespaces=["authorized-ns"],
        k8s_namespace="default",
    )

    request = ScaleRequest(
        caller_namespace="unauthorized-ns",
        graph_deployment_name="my-dgd",
        k8s_namespace="default",
        target_replicas=[
            TargetReplica(
                sub_component_type=SubComponentType.PREFILL, desired_replicas=3
            )
        ],
    )

    # Process request
    results = []
    async for response in handler.scale_request(request.model_dump()):
        results.append(response)

    assert len(results) == 1
    response = results[0]
    assert response["status"] == "error"
    assert "not authorized" in response["message"]
    assert response["current_replicas"] == {}


@pytest.mark.asyncio
async def test_handler_multiple_dgds(mock_runtime):
    """Test handler creates separate connectors for different DGDs (and caches them)."""
    handler = ScaleRequestHandler(
        runtime=mock_runtime, managed_namespaces=["app-ns"], k8s_namespace="default"
    )

    request1 = ScaleRequest(
        caller_namespace="app-ns",
        graph_deployment_name="dgd-1",
        k8s_namespace="default",
        target_replicas=[
            TargetReplica(
                sub_component_type=SubComponentType.PREFILL, desired_replicas=2
            )
        ],
    )

    request2 = ScaleRequest(
        caller_namespace="app-ns",
        graph_deployment_name="dgd-2",  # Different DGD
        k8s_namespace="default",
        target_replicas=[
            TargetReplica(
                sub_component_type=SubComponentType.PREFILL, desired_replicas=4
            )
        ],
    )

    with patch(
        "dynamo.global_planner.kubernetes_capacity_manager.KubernetesConnector"
    ) as mock_connector_cls:
        mock_connector = AsyncMock()
        mock_connector_cls.return_value = mock_connector
        mock_connector._async_init = AsyncMock()
        mock_connector.set_component_replicas = AsyncMock()
        mock_connector.kube_api = MagicMock()
        mock_connector.kube_api.get_graph_deployment = MagicMock(
            return_value={"spec": {"components": []}}
        )

        # Process both requests
        async for _ in handler.scale_request(request1.model_dump()):
            pass
        async for _ in handler.scale_request(request2.model_dump()):
            pass

        # Verify two connectors were created
        assert "default/dgd-1" in handler.orchestrator.capacity_manager.connectors
        assert "default/dgd-2" in handler.orchestrator.capacity_manager.connectors
        assert mock_connector_cls.call_count == 2


@pytest.mark.asyncio
async def test_handler_error_handling(mock_runtime):
    """Test handler error handling during scaling."""
    handler = ScaleRequestHandler(
        runtime=mock_runtime, managed_namespaces=["app-ns"], k8s_namespace="default"
    )

    request = ScaleRequest(
        caller_namespace="app-ns",
        graph_deployment_name="my-dgd",
        k8s_namespace="default",
        target_replicas=[
            TargetReplica(
                sub_component_type=SubComponentType.PREFILL, desired_replicas=3
            )
        ],
    )

    with patch(
        "dynamo.global_planner.kubernetes_capacity_manager.KubernetesConnector"
    ) as mock_connector_cls:
        mock_connector = AsyncMock()
        mock_connector_cls.return_value = mock_connector
        mock_connector._async_init = AsyncMock()
        # Simulate error during scaling
        mock_connector.set_component_replicas = AsyncMock(
            side_effect=Exception("Scaling failed")
        )

        # Process request (pass as dict to match endpoint behavior)
        results = []
        async for response in handler.scale_request(request.model_dump()):
            results.append(response)

        assert len(results) == 1
        response = results[0]
        assert response["status"] == "error"
        assert "Scaling failed" in response["message"]


@pytest.mark.asyncio
async def test_populate_connectors_explicit_mode(mock_runtime):
    """Test _populate_k8s_connectors only creates connectors for managed DGDs."""
    handler = ScaleRequestHandler(
        runtime=mock_runtime,
        managed_namespaces=["default-model-a"],
        k8s_namespace="default",
        max_total_gpus=-1,  # Don't trigger discovery in __init__
    )

    with (
        patch(
            "dynamo.global_planner.kubernetes_capacity_manager.KubernetesAPI"
        ) as mock_kube_cls,
        patch(
            "dynamo.global_planner.kubernetes_capacity_manager.KubernetesConnector"
        ) as mock_connector_cls,
    ):
        mock_kube = MagicMock()
        mock_kube_cls.return_value = mock_kube
        mock_kube.list_graph_deployments.return_value = [
            {"metadata": {"name": "model-a"}},
            {"metadata": {"name": "model-b"}},  # Not in managed set
            {"metadata": {"name": "gp-ctrl"}},  # Not in managed set
        ]
        mock_connector_cls.return_value = MagicMock()

        handler.orchestrator.capacity_manager.discover(
            handler.orchestrator.managed_deployments
        )

        # Only model-a should be discovered
        assert "default/model-a" in handler.orchestrator.capacity_manager.connectors
        assert "default/model-b" not in handler.orchestrator.capacity_manager.connectors
        assert "default/gp-ctrl" not in handler.orchestrator.capacity_manager.connectors
        assert mock_connector_cls.call_count == 1


@pytest.mark.asyncio
async def test_populate_connectors_implicit_mode(mock_runtime):
    """Test _populate_k8s_connectors creates connectors for all DGDs in implicit mode."""
    handler = ScaleRequestHandler(
        runtime=mock_runtime,
        managed_namespaces=None,
        k8s_namespace="default",
        max_total_gpus=-1,  # Don't trigger discovery in __init__
    )

    with (
        patch(
            "dynamo.global_planner.kubernetes_capacity_manager.KubernetesAPI"
        ) as mock_kube_cls,
        patch(
            "dynamo.global_planner.kubernetes_capacity_manager.KubernetesConnector"
        ) as mock_connector_cls,
    ):
        mock_kube = MagicMock()
        mock_kube_cls.return_value = mock_kube
        mock_kube.list_graph_deployments.return_value = [
            {"metadata": {"name": "model-a"}},
            {"metadata": {"name": "model-b"}},
        ]
        mock_connector_cls.return_value = MagicMock()

        handler.orchestrator.capacity_manager.discover(
            handler.orchestrator.managed_deployments
        )

        # All DGDs should be discovered
        assert "default/model-a" in handler.orchestrator.capacity_manager.connectors
        assert "default/model-b" in handler.orchestrator.capacity_manager.connectors
        assert mock_connector_cls.call_count == 2


@pytest.mark.asyncio
async def test_handler_blocking_mode(mock_runtime):
    """Test handler respects blocking mode."""
    handler = ScaleRequestHandler(
        runtime=mock_runtime, managed_namespaces=["app-ns"], k8s_namespace="default"
    )

    request = ScaleRequest(
        caller_namespace="app-ns",
        graph_deployment_name="my-dgd",
        k8s_namespace="default",
        target_replicas=[
            TargetReplica(
                sub_component_type=SubComponentType.PREFILL, desired_replicas=3
            )
        ],
        blocking=True,  # Request blocking mode
    )

    with patch(
        "dynamo.global_planner.kubernetes_capacity_manager.KubernetesConnector"
    ) as mock_connector_cls:
        mock_connector = AsyncMock()
        mock_connector_cls.return_value = mock_connector
        mock_connector._async_init = AsyncMock()
        mock_connector.set_component_replicas = AsyncMock()
        mock_connector.kube_api = MagicMock()
        mock_connector.kube_api.get_graph_deployment = MagicMock(
            return_value={"spec": {"components": []}}
        )

        # Process request (pass as dict to match endpoint behavior)
        async for _ in handler.scale_request(request.model_dump()):
            pass

        # Verify blocking=True was passed to connector
        mock_connector.set_component_replicas.assert_called_once()
        call_args = mock_connector.set_component_replicas.call_args
        assert call_args[1]["blocking"] is True


# ---------------------------------------------------------------------------- #
# Helpers for arbitration tests                                                #
# ---------------------------------------------------------------------------- #


def _component(name, replicas, gpu, ctype, node_count=None):
    component = {
        "name": name,
        "type": ctype,
        "replicas": replicas,
        "podTemplate": {
            "spec": {
                "containers": [
                    {"name": "main", "resources": {"limits": {"nvidia.com/gpu": gpu}}}
                ]
            }
        },
    }
    if node_count is not None:
        component["multinode"] = {"nodeCount": node_count}
    return component


def _dgd_spec(
    prefill_replicas,
    decode_replicas,
    prefill_gpu=1,
    decode_gpu=1,
    prefill_node_count=None,
    decode_node_count=None,
):
    """Build a v1beta1 DGD spec with prefill and decode components."""
    return {
        "spec": {
            "components": [
                _component(
                    "prefill-svc",
                    prefill_replicas,
                    prefill_gpu,
                    "prefill",
                    prefill_node_count,
                ),
                _component(
                    "decode-svc",
                    decode_replicas,
                    decode_gpu,
                    "decode",
                    decode_node_count,
                ),
            ]
        }
    }


def _worker_dgd_spec(
    replicas, gpu=1, component_name="worker-svc", component_type="worker"
):
    """Build a v1beta1 DGD spec with one generic worker component."""
    component = {
        "name": component_name,
        "replicas": replicas,
        "podTemplate": {
            "spec": {
                "containers": [
                    {"name": "main", "resources": {"limits": {"nvidia.com/gpu": gpu}}}
                ]
            }
        },
    }
    if component_type is not None:
        component["type"] = component_type
    return {"spec": {"components": [component]}}


def _two_worker_dgd_spec(replicas=1, gpu=1):
    """Build a DGD with two generic workers that can resolve to one role."""
    deployment = _worker_dgd_spec(replicas=replicas, gpu=gpu, component_name="worker-a")
    deployment["spec"]["components"].extend(
        _worker_dgd_spec(replicas=replicas, gpu=gpu, component_name="worker-b")["spec"][
            "components"
        ]
    )
    return deployment


def _install_connector(handler, dgd_key, dgd_spec_dict, parent_dgd_name="my-dgd"):
    """Attach a mocked KubernetesConnector to the handler for one DGD."""
    connector = AsyncMock()
    connector._async_init = AsyncMock()
    connector.set_component_replicas = AsyncMock()
    connector.parent_dgd_name = parent_dgd_name
    connector.kube_api = MagicMock()
    connector.kube_api.get_graph_deployment = MagicMock(return_value=dgd_spec_dict)
    handler.orchestrator.capacity_manager.connectors[dgd_key] = connector
    return connector


def _install_real_connector(handler, dgd_key, dgd_spec_dict, parent_dgd_name="my-dgd"):
    """Attach a real KubernetesConnector backed by a mocked Kubernetes API."""
    connector = KubernetesConnector.__new__(KubernetesConnector)
    connector.parent_dgd_name = parent_dgd_name
    connector.graph_deployment_name = parent_dgd_name
    connector.raise_not_ready = True
    connector.kube_api = MagicMock()
    connector.kube_api.get_graph_deployment.return_value = dgd_spec_dict
    connector.kube_api.is_deployment_ready.return_value = True
    connector.kube_api.wait_for_graph_deployment_ready = AsyncMock()
    handler.orchestrator.capacity_manager.connectors[dgd_key] = connector
    return connector


def _scale_req(
    dgd="my-dgd",
    k8s_ns="default",
    caller_ns="app-ns",
    prefill=None,
    decode=None,
    prefill_component_name=None,
    decode_component_name=None,
):
    """Build a ScaleRequest with one or both pool targets set."""
    targets = []
    if prefill is not None:
        targets.append(
            TargetReplica(
                sub_component_type=SubComponentType.PREFILL,
                component_name=prefill_component_name,
                desired_replicas=prefill,
            )
        )
    if decode is not None:
        targets.append(
            TargetReplica(
                sub_component_type=SubComponentType.DECODE,
                component_name=decode_component_name,
                desired_replicas=decode,
            )
        )
    return ScaleRequest(
        caller_namespace=caller_ns,
        graph_deployment_name=dgd,
        k8s_namespace=k8s_ns,
        target_replicas=targets,
    )


async def _run(handler, request):
    """Drive the async generator returned by scale_request, collect responses."""
    results = []
    async for resp in handler.scale_request(request.model_dump()):
        results.append(resp)
    return results


# ---------------------------------------------------------------------------- #
# min_total_gpus / arbitration tests                                           #
# ---------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_min_total_gpus_disabled_by_default(mock_runtime):
    """With min_total_gpus=-1 (default), scale-downs are unaffected by any floor."""
    handler = ScaleRequestHandler(
        runtime=mock_runtime,
        managed_namespaces=["default-my-dgd"],
        k8s_namespace="default",
    )
    assert handler.min_total_gpus == -1

    connector = _install_connector(
        handler, "default/my-dgd", _dgd_spec(prefill_replicas=3, decode_replicas=3)
    )
    req = _scale_req(caller_ns="default-my-dgd", prefill=1)  # scale prefill down
    results = await _run(handler, req)
    assert results[0]["status"] == "success"
    connector.set_component_replicas.assert_called_once()


@pytest.mark.asyncio
async def test_scale_down_denied_when_breaches_floor_and_no_pair(mock_runtime):
    """Floor is 6, currently at 6 (3 prefill + 3 decode), prefill scale-down denied."""
    handler = ScaleRequestHandler(
        runtime=mock_runtime,
        managed_namespaces=["default-my-dgd"],
        k8s_namespace="default",
        min_total_gpus=6,
        max_total_gpus=6,
    )
    connector = _install_connector(
        handler, "default/my-dgd", _dgd_spec(prefill_replicas=3, decode_replicas=3)
    )
    req = _scale_req(caller_ns="default-my-dgd", prefill=2)  # want to drop to 2
    results = await _run(handler, req)
    # Soft-deny status: budget breach is normal in fixed-total mode, not a fault.
    assert results[0]["status"] == "rejected"
    assert (
        "budget breach" in results[0]["message"].lower()
        or "below floor" in results[0]["message"].lower()
    )
    connector.set_component_replicas.assert_not_called()


@pytest.mark.asyncio
async def test_scale_down_paired_with_pending_scale_up_in_same_dgd(mock_runtime):
    """Floor=max=6, prefill scale-down is paired with a cached decode scale-up intent."""
    handler = ScaleRequestHandler(
        runtime=mock_runtime,
        managed_namespaces=["default-my-dgd"],
        k8s_namespace="default",
        min_total_gpus=6,
        max_total_gpus=6,
    )
    connector = _install_connector(
        handler, "default/my-dgd", _dgd_spec(prefill_replicas=3, decode_replicas=3)
    )

    # Pre-seed cache: decode wants to go to 4 (scale up by 1)
    handler.orchestrator._intent_cache["default/my-dgd/decode"] = PoolIntent(
        last_desired=4, last_seen_at=time.time()
    )

    # Prefill wants to go to 2 (scale down by 1)
    req = _scale_req(caller_ns="default-my-dgd", prefill=2)
    results = await _run(handler, req)
    assert results[0]["status"] == "success"

    # Both pools applied in one K8s call
    connector.set_component_replicas.assert_called_once()
    call_targets = connector.set_component_replicas.call_args[0][0]
    sub_types = {t.sub_component_type.value: t.desired_replicas for t in call_targets}
    assert sub_types == {"prefill": 2, "decode": 4}


@pytest.mark.asyncio
async def test_scale_down_paired_across_different_dgd(mock_runtime):
    """Cross-DGD pairing: DGD-A's scale-down pairs with DGD-B's cached scale-up.
    Both sides are applied via separate (non-atomic) connector calls."""
    handler = ScaleRequestHandler(
        runtime=mock_runtime,
        managed_namespaces=["default-dgd-a", "default-dgd-b"],
        k8s_namespace="default",
        min_total_gpus=12,
        max_total_gpus=12,
    )
    connector_a = _install_connector(
        handler,
        "default/dgd-a",
        _dgd_spec(prefill_replicas=3, decode_replicas=3),
        parent_dgd_name="dgd-a",
    )
    connector_b = _install_connector(
        handler,
        "default/dgd-b",
        _dgd_spec(prefill_replicas=3, decode_replicas=3),
        parent_dgd_name="dgd-b",
    )

    # Decode in DGD-B wants to scale up — should pair across DGDs with DGD-A's scale-down
    handler.orchestrator._intent_cache["default/dgd-b/decode"] = PoolIntent(
        last_desired=4, last_seen_at=time.time()
    )

    # Prefill in DGD-A wants to scale down. total standalone = 5+6 = 11 (< min=12);
    # paired = 5+7 = 12 (exactly at floor), so the pair should apply.
    req = _scale_req(dgd="dgd-a", caller_ns="default-dgd-a", prefill=2)
    results = await _run(handler, req)
    assert results[0]["status"] == "success"
    # Both connectors called — cross-DGD transfer is two separate patches
    connector_a.set_component_replicas.assert_called_once()
    connector_b.set_component_replicas.assert_called_once()
    a_targets = {
        t.sub_component_type.value: t.desired_replicas
        for t in connector_a.set_component_replicas.call_args[0][0]
    }
    b_targets = {
        t.sub_component_type.value: t.desired_replicas
        for t in connector_b.set_component_replicas.call_args[0][0]
    }
    assert a_targets == {"prefill": 2}
    assert b_targets == {"decode": 4}


@pytest.mark.asyncio
async def test_pair_packs_both_partners_when_both_fit(mock_runtime):
    """Multi-partner packing: when same-DGD AND cross-DGD partners both fit
    within the budget band, BOTH are applied (the user's "larger groups of
    scaling decisions" property). Same-DGD partners are merged into the
    request's DGD's atomic patch; cross-DGD partners get their own patch."""
    handler = ScaleRequestHandler(
        runtime=mock_runtime,
        managed_namespaces=["default-dgd-a", "default-dgd-b"],
        k8s_namespace="default",
        # Band [11, 13] so both +1-GPU partners fit strictly below the
        # ceiling. (Under a fixed min==max==12 cap, strict ceiling would
        # admit only one of the two scale-up partners.)
        min_total_gpus=11,
        max_total_gpus=13,
    )
    connector_a = _install_connector(
        handler,
        "default/dgd-a",
        _dgd_spec(prefill_replicas=3, decode_replicas=3),
        parent_dgd_name="dgd-a",
    )
    connector_b = _install_connector(
        handler,
        "default/dgd-b",
        _dgd_spec(prefill_replicas=3, decode_replicas=3),
        parent_dgd_name="dgd-b",
    )
    # Both DGD-A's decode and DGD-B's decode have pending scale-up intents
    # (each +1 GPU). Cluster baseline = 12; standalone request lands at 11
    # (at floor). Pack both partners → 11+1+1 = 13 (at ceiling, strict).
    handler.orchestrator._intent_cache["default/dgd-a/decode"] = PoolIntent(
        last_desired=4, last_seen_at=time.time()
    )
    handler.orchestrator._intent_cache["default/dgd-b/decode"] = PoolIntent(
        last_desired=4, last_seen_at=time.time()
    )
    req = _scale_req(dgd="dgd-a", caller_ns="default-dgd-a", prefill=2)
    results = await _run(handler, req)
    assert results[0]["status"] == "success"
    connector_a.set_component_replicas.assert_called_once()
    connector_b.set_component_replicas.assert_called_once()
    a_targets = {
        t.sub_component_type.value: t.desired_replicas
        for t in connector_a.set_component_replicas.call_args[0][0]
    }
    b_targets = {
        t.sub_component_type.value: t.desired_replicas
        for t in connector_b.set_component_replicas.call_args[0][0]
    }
    # DGD-A patch combines the request's prefill change with the same-DGD
    # decode partner (one atomic call).
    assert a_targets == {"prefill": 2, "decode": 4}
    # DGD-B's cross-DGD partner is a separate patch.
    assert b_targets == {"decode": 4}


@pytest.mark.asyncio
async def test_cross_dgd_pair_second_patch_failure_self_corrects(mock_runtime, caplog):
    """If the second K8s patch in a cross-DGD pair fails, the first stays
    applied and a loud error is logged; no rollback, no crash."""
    import logging as _logging

    handler = ScaleRequestHandler(
        runtime=mock_runtime,
        managed_namespaces=["default-dgd-a", "default-dgd-b"],
        k8s_namespace="default",
        min_total_gpus=12,
        max_total_gpus=12,
    )
    connector_a = _install_connector(
        handler,
        "default/dgd-a",
        _dgd_spec(prefill_replicas=3, decode_replicas=3),
        parent_dgd_name="dgd-a",
    )
    connector_b = _install_connector(
        handler,
        "default/dgd-b",
        _dgd_spec(prefill_replicas=3, decode_replicas=3),
        parent_dgd_name="dgd-b",
    )
    # DGD-B's decode is a pending partner
    handler.orchestrator._intent_cache["default/dgd-b/decode"] = PoolIntent(
        last_desired=4, last_seen_at=time.time()
    )
    # DGD-B's K8s patch fails
    connector_b.set_component_replicas.side_effect = Exception("simulated K8s failure")

    req = _scale_req(dgd="dgd-a", caller_ns="default-dgd-a", prefill=2)
    with caplog.at_level(_logging.ERROR, logger="dynamo.global_planner.scale_handler"):
        results = await _run(handler, req)
    # Overall response: the request-side patch succeeded; response path reports
    # success because the request-side was applied. The cross-DGD partner
    # failure is logged as an error; self-correction happens on the next tick.
    connector_a.set_component_replicas.assert_called_once()
    connector_b.set_component_replicas.assert_called_once()
    # Second-patch failure should be logged at ERROR
    assert any(
        "Multi-partner transfer" in rec.message and "failed" in rec.message
        for rec in caplog.records
    )
    # Request response still reports success for the request side
    assert results[0]["status"] == "success"


@pytest.mark.asyncio
async def test_cross_dgd_pair_second_patch_not_ready_self_corrects(
    mock_runtime, caplog
):
    """If a later cross-DGD patch is skipped because the DGD is not ready,
    the first patch stays applied and the request side still reports success."""
    import logging as _logging

    handler = ScaleRequestHandler(
        runtime=mock_runtime,
        managed_namespaces=["default-dgd-a", "default-dgd-b"],
        k8s_namespace="default",
        min_total_gpus=12,
        max_total_gpus=12,
    )
    connector_a = _install_connector(
        handler,
        "default/dgd-a",
        _dgd_spec(prefill_replicas=3, decode_replicas=3),
        parent_dgd_name="dgd-a",
    )
    connector_b = _install_connector(
        handler,
        "default/dgd-b",
        _dgd_spec(prefill_replicas=3, decode_replicas=3),
        parent_dgd_name="dgd-b",
    )
    handler.orchestrator._intent_cache["default/dgd-b/decode"] = PoolIntent(
        last_desired=4, last_seen_at=time.time()
    )
    connector_b.set_component_replicas.side_effect = DynamoGraphDeploymentNotReadyError(
        "dgd-b", "default"
    )

    req = _scale_req(dgd="dgd-a", caller_ns="default-dgd-a", prefill=2)
    with caplog.at_level(
        _logging.WARNING, logger="dynamo.global_planner.scale_handler"
    ):
        results = await _run(handler, req)

    connector_a.set_component_replicas.assert_called_once()
    connector_b.set_component_replicas.assert_called_once()
    assert results[0]["status"] == "success"
    assert any(
        "not ready" in rec.message and "will self-correct" in rec.message
        for rec in caplog.records
    )


@pytest.mark.asyncio
async def test_cross_dgd_pair_first_patch_failure_yields_error(mock_runtime):
    """If the FIRST K8s patch in a cross-DGD pair fails, nothing has been
    applied yet, the second patch must not be attempted, and the response
    must surface the failure as ``error`` (not ``success``)."""
    handler = ScaleRequestHandler(
        runtime=mock_runtime,
        managed_namespaces=["default-dgd-a", "default-dgd-b"],
        k8s_namespace="default",
        min_total_gpus=12,
        max_total_gpus=12,
    )
    connector_a = _install_connector(
        handler,
        "default/dgd-a",
        _dgd_spec(prefill_replicas=3, decode_replicas=3),
        parent_dgd_name="dgd-a",
    )
    connector_b = _install_connector(
        handler,
        "default/dgd-b",
        _dgd_spec(prefill_replicas=3, decode_replicas=3),
        parent_dgd_name="dgd-b",
    )
    handler.orchestrator._intent_cache["default/dgd-b/decode"] = PoolIntent(
        last_desired=4, last_seen_at=time.time()
    )
    # Request is DGD-A prefill 3 → 2 (net_delta < 0), so the scale-down
    # side (DGD-A) is the FIRST patch by ordering rule. Make it raise.
    connector_a.set_component_replicas.side_effect = Exception(
        "simulated 409 conflict on first patch"
    )

    req = _scale_req(dgd="dgd-a", caller_ns="default-dgd-a", prefill=2)
    results = await _run(handler, req)

    # First patch attempted and raised; second patch never reached.
    connector_a.set_component_replicas.assert_called_once()
    connector_b.set_component_replicas.assert_not_called()
    # Outer handler surfaces the exception as ``error`` rather than success.
    assert results[0]["status"] == "error"
    assert "simulated 409 conflict" in results[0]["message"]


@pytest.mark.asyncio
async def test_cross_dgd_asymmetric_pair_rejected_above_ceiling(mock_runtime):
    """Cross-DGD pair whose post-transfer total would exceed ``max`` is
    rejected: tolerance does **not** relax the upper bound (max is a hard
    cluster-capacity cap)."""
    handler = ScaleRequestHandler(
        runtime=mock_runtime,
        managed_namespaces=["default-dgd-a", "default-dgd-b"],
        k8s_namespace="default",
        min_total_gpus=10,
        max_total_gpus=10,
    )
    # DGD-A: prefill=4 (1 GPU each) + decode=0 → 4 GPUs.
    # DGD-B: one agg pool reusing "decode" subComponentType with 2 GPU/worker,
    #   3 workers → 6 GPUs. Total cluster=10.
    _install_connector(
        handler,
        "default/dgd-a",
        _dgd_spec(prefill_replicas=4, decode_replicas=0, prefill_gpu=1, decode_gpu=2),
        parent_dgd_name="dgd-a",
    )
    _install_connector(
        handler,
        "default/dgd-b",
        _dgd_spec(prefill_replicas=0, decode_replicas=3, prefill_gpu=1, decode_gpu=2),
        parent_dgd_name="dgd-b",
    )
    # DGD-B decode wants +1 (+2 GPUs); DGD-A prefill request wants -1 (-1 GPU).
    # Paired total = 10 - 1 + 2 = 11 > max=10. Decode's 2 GPU/worker step can't
    # be partially applied to land at exactly 10, so the pair is denied.
    handler.orchestrator._intent_cache["default/dgd-b/decode"] = PoolIntent(
        last_desired=4, last_seen_at=time.time()
    )
    req = _scale_req(dgd="dgd-a", caller_ns="default-dgd-a", prefill=3)
    results = await _run(handler, req)
    assert results[0]["status"] == "rejected"
    handler.orchestrator.capacity_manager.connectors[
        "default/dgd-a"
    ].set_component_replicas.assert_not_called()
    handler.orchestrator.capacity_manager.connectors[
        "default/dgd-b"
    ].set_component_replicas.assert_not_called()


@pytest.mark.asyncio
async def test_intent_cache_respects_ttl(mock_runtime):
    """Stale cached intents (past TTL) are not eligible as pair partners."""
    handler = ScaleRequestHandler(
        runtime=mock_runtime,
        managed_namespaces=["default-my-dgd"],
        k8s_namespace="default",
        min_total_gpus=6,
        max_total_gpus=6,
        intent_cache_ttl_seconds=30,
    )
    connector = _install_connector(
        handler, "default/my-dgd", _dgd_spec(prefill_replicas=3, decode_replicas=3)
    )
    # Cache a decode scale-up intent that is too old
    handler.orchestrator._intent_cache["default/my-dgd/decode"] = PoolIntent(
        last_desired=4, last_seen_at=time.time() - 60
    )

    req = _scale_req(caller_ns="default-my-dgd", prefill=2)
    results = await _run(handler, req)
    assert results[0]["status"] == "rejected"
    connector.set_component_replicas.assert_not_called()


@pytest.mark.asyncio
async def test_scale_up_paired_with_pending_scale_down_when_ceiling_breached(
    mock_runtime,
):
    """Ceiling case: scale-up paired with cached opposite scale-down."""
    handler = ScaleRequestHandler(
        runtime=mock_runtime,
        managed_namespaces=["default-my-dgd"],
        k8s_namespace="default",
        min_total_gpus=6,
        max_total_gpus=6,
    )
    connector = _install_connector(
        handler, "default/my-dgd", _dgd_spec(prefill_replicas=3, decode_replicas=3)
    )
    # Prefill wants to drop by 1; cache it
    handler.orchestrator._intent_cache["default/my-dgd/prefill"] = PoolIntent(
        last_desired=2, last_seen_at=time.time()
    )

    # Decode wants to go up by 1 (would breach ceiling without pair)
    req = _scale_req(caller_ns="default-my-dgd", decode=4)
    results = await _run(handler, req)
    assert results[0]["status"] == "success"
    connector.set_component_replicas.assert_called_once()
    call_targets = connector.set_component_replicas.call_args[0][0]
    sub_types = {t.sub_component_type.value: t.desired_replicas for t in call_targets}
    assert sub_types == {"decode": 4, "prefill": 2}


@pytest.mark.asyncio
async def test_asymmetric_per_worker_gpu_pair_rejected_above_ceiling(mock_runtime):
    """Prefill=1 GPU/worker, Decode=2 GPU/worker. Paired transfer would land
    at total=11 > max=10. ``max`` is a strict cluster-capacity bound (no
    tolerance on the upper side), so the pair is denied — even though the
    overshoot is only one decode-worker's worth of GPUs."""
    handler = ScaleRequestHandler(
        runtime=mock_runtime,
        managed_namespaces=["default-my-dgd"],
        k8s_namespace="default",
        min_total_gpus=10,
        max_total_gpus=10,
    )
    # prefill: 4 replicas * 1 GPU = 4; decode: 3 * 2 = 6; total = 10
    connector = _install_connector(
        handler,
        "default/my-dgd",
        _dgd_spec(prefill_replicas=4, decode_replicas=3, prefill_gpu=1, decode_gpu=2),
    )
    # Decode wants +1 (=+2 GPUs); prefill cached intent -1 (=-1 GPU).
    # Paired total = 10 + 2 - 1 = 11 > max=10. No partial works (decode step
    # is 2 GPUs — partial-K can only be 3=current or 4=full, neither lands at
    # 10 with the -1 prefill applied).
    handler.orchestrator._intent_cache["default/my-dgd/prefill"] = PoolIntent(
        last_desired=3, last_seen_at=time.time()
    )
    req = _scale_req(caller_ns="default-my-dgd", decode=4)
    results = await _run(handler, req)
    assert results[0]["status"] == "rejected"
    connector.set_component_replicas.assert_not_called()


@pytest.mark.asyncio
async def test_asymmetric_pair_denied_if_above_ceiling(mock_runtime):
    """Paired total that exceeds ``max`` is rejected. ``max`` is a strict
    upper bound — no tolerance applied."""
    handler = ScaleRequestHandler(
        runtime=mock_runtime,
        managed_namespaces=["default-my-dgd"],
        k8s_namespace="default",
        min_total_gpus=10,
        max_total_gpus=10,
    )
    # prefill: 4 * 1 = 4; decode: 3 * 2 = 6; total = 10
    connector = _install_connector(
        handler,
        "default/my-dgd",
        _dgd_spec(prefill_replicas=4, decode_replicas=3, prefill_gpu=1, decode_gpu=2),
    )
    # Decode wants +2 (=+4 GPUs), prefill cached intent -1 (=-1 GPU).
    # Paired total = 10 + 4 - 1 = 13 > max=10. Deny.
    handler.orchestrator._intent_cache["default/my-dgd/prefill"] = PoolIntent(
        last_desired=3, last_seen_at=time.time()
    )
    req = _scale_req(caller_ns="default-my-dgd", decode=5)
    results = await _run(handler, req)
    assert results[0]["status"] == "rejected"
    connector.set_component_replicas.assert_not_called()


@pytest.mark.asyncio
async def test_asymmetric_pair_denied_if_below_tolerance(mock_runtime):
    """Symmetric floor-undershoot case for tolerance: paired transfer whose
    post-pair total falls below min - tolerance is rejected with a 'below
    floor' reason."""
    handler = ScaleRequestHandler(
        runtime=mock_runtime,
        managed_namespaces=["default-my-dgd"],
        k8s_namespace="default",
        min_total_gpus=10,
        max_total_gpus=10,
    )
    # prefill=5, decode=5, both 1 GPU/worker. total=10, tolerance=1.
    connector = _install_connector(
        handler,
        "default/my-dgd",
        _dgd_spec(prefill_replicas=5, decode_replicas=5, prefill_gpu=1, decode_gpu=1),
    )
    # Decode wants +1 (+1 GPU); request prefill wants -3 (-3 GPUs). Paired
    # total = 10 - 3 + 1 = 8. min - tolerance = 10 - 1 = 9. 8 < 9 → deny.
    handler.orchestrator._intent_cache["default/my-dgd/decode"] = PoolIntent(
        last_desired=6, last_seen_at=time.time()
    )
    req = _scale_req(caller_ns="default-my-dgd", prefill=2)
    results = await _run(handler, req)
    assert results[0]["status"] == "rejected"
    assert "below floor" in results[0]["message"].lower()
    connector.set_component_replicas.assert_not_called()


@pytest.mark.asyncio
async def test_initial_below_floor_logs_warning(mock_runtime):
    """At startup, when discovered total < min_total_gpus, a warning is
    logged. The handler does not attempt any proactive fill — soft-floor
    semantics — and the floor only blocks explicit scale-downs going forward
    (covered by the standalone-deny tests above)."""
    with (
        patch(
            "dynamo.global_planner.kubernetes_capacity_manager.KubernetesAPI"
        ) as mock_kube_cls,
        patch(
            "dynamo.global_planner.kubernetes_capacity_manager.KubernetesConnector"
        ) as mock_connector_cls,
    ):
        mock_kube = MagicMock()
        mock_kube_cls.return_value = mock_kube
        mock_kube.list_graph_deployments.return_value = [
            {"metadata": {"name": "my-dgd"}},
        ]
        mock_connector = MagicMock()
        mock_connector.parent_dgd_name = "my-dgd"
        mock_connector.kube_api = MagicMock()
        mock_connector.kube_api.get_graph_deployment = MagicMock(
            return_value=_dgd_spec(prefill_replicas=1, decode_replicas=1)
        )
        mock_connector_cls.return_value = mock_connector

        with patch("dynamo.global_planner.orchestrator.logger") as mock_logger:
            ScaleRequestHandler(
                runtime=mock_runtime,
                managed_namespaces=["default-my-dgd"],
                k8s_namespace="default",
                min_total_gpus=10,
            )
            warnings = [
                call
                for call in mock_logger.warning.call_args_list
                if call.args and "below min_total_gpus" in str(call.args[0])
            ]
            assert warnings, "Expected a warning about being below the floor"
            # No proactive fill was issued.
            mock_connector.set_component_replicas.assert_not_called()


@pytest.mark.asyncio
async def test_out_of_order_requests_pair_via_cache(mock_runtime):
    """Pool A's scale-down is denied first (no pair in cache). Then pool B's
    scale-up arrives; the denied intent is still in cache; pair executes."""
    handler = ScaleRequestHandler(
        runtime=mock_runtime,
        managed_namespaces=["default-my-dgd"],
        k8s_namespace="default",
        min_total_gpus=6,
        max_total_gpus=6,
    )
    connector = _install_connector(
        handler, "default/my-dgd", _dgd_spec(prefill_replicas=3, decode_replicas=3)
    )

    # First request: prefill wants to drop to 2 (from 3). No decode intent in cache.
    req_prefill = _scale_req(caller_ns="default-my-dgd", prefill=2)
    results_1 = await _run(handler, req_prefill)
    assert results_1[0]["status"] == "rejected"
    connector.set_component_replicas.assert_not_called()
    # Verify the intent was still cached
    assert "default/my-dgd/prefill" in handler.orchestrator._intent_cache
    assert (
        handler.orchestrator._intent_cache["default/my-dgd/prefill"].last_desired == 2
    )

    # Second request: decode wants to go up to 4. Now prefill's intent pairs.
    req_decode = _scale_req(caller_ns="default-my-dgd", decode=4)
    results_2 = await _run(handler, req_decode)
    assert results_2[0]["status"] == "success"
    connector.set_component_replicas.assert_called_once()
    call_targets = connector.set_component_replicas.call_args[0][0]
    sub_types = {t.sub_component_type.value: t.desired_replicas for t in call_targets}
    assert sub_types == {"decode": 4, "prefill": 2}


@pytest.mark.asyncio
async def test_target_dgd_not_ready_yields_rejected(mock_runtime):
    """A not-ready target DGD is a retryable no-op, not a successful scale."""
    handler = ScaleRequestHandler(
        runtime=mock_runtime,
        managed_namespaces=["app-ns"],
        k8s_namespace="default",
    )
    connector = _install_connector(
        handler,
        "default/my-dgd",
        _dgd_spec(prefill_replicas=1, decode_replicas=1),
    )
    connector.set_component_replicas.side_effect = DynamoGraphDeploymentNotReadyError(
        "my-dgd", "default"
    )

    req = _scale_req(caller_ns="app-ns", prefill=2)
    results = await _run(handler, req)

    connector.set_component_replicas.assert_called_once()
    assert results[0]["status"] == "rejected"
    assert "not ready" in results[0]["message"]
    assert results[0]["current_replicas"] == {}


@pytest.mark.asyncio
async def test_pair_preferred_over_standalone_when_both_feasible(mock_runtime):
    """If a pending opposite-direction intent is cached AND the pair is
    feasible, the pair is applied — even if standalone would have fit bounds."""
    handler = ScaleRequestHandler(
        runtime=mock_runtime,
        managed_namespaces=["default-my-dgd"],
        k8s_namespace="default",
        min_total_gpus=4,
        max_total_gpus=8,  # wide band: both standalone and pair fit
    )
    connector = _install_connector(
        handler, "default/my-dgd", _dgd_spec(prefill_replicas=3, decode_replicas=3)
    )

    # Decode has a fresh intent to scale up
    handler.orchestrator._intent_cache["default/my-dgd/decode"] = PoolIntent(
        last_desired=4, last_seen_at=time.time()
    )

    # Prefill scale-down standalone → total=5 (in bounds).
    # Paired with decode +1 → total=6 (also in bounds). Pair is preferred.
    req = _scale_req(caller_ns="default-my-dgd", prefill=2)
    results = await _run(handler, req)
    assert results[0]["status"] == "success"
    call_targets = connector.set_component_replicas.call_args[0][0]
    sub_types = {t.sub_component_type.value: t.desired_replicas for t in call_targets}
    # Decode should be in the applied targets — indicating the pair was used
    assert "decode" in sub_types and sub_types["decode"] == 4
    assert sub_types["prefill"] == 2


@pytest.mark.asyncio
async def test_cache_entry_persists_after_standalone_apply(mock_runtime):
    """After a standalone-approved request, the cache entry persists with
    last_desired equal to what was applied."""
    handler = ScaleRequestHandler(
        runtime=mock_runtime,
        managed_namespaces=["default-my-dgd"],
        k8s_namespace="default",
        min_total_gpus=0,
        max_total_gpus=100,
    )
    dgd_state = _dgd_spec(prefill_replicas=3, decode_replicas=3)
    _install_connector(handler, "default/my-dgd", dgd_state)

    req = _scale_req(caller_ns="default-my-dgd", prefill=4)
    await _run(handler, req)

    assert (
        handler.orchestrator._intent_cache["default/my-dgd/prefill"].last_desired == 4
    )


@pytest.mark.asyncio
async def test_satisfied_cached_intent_does_not_pair(mock_runtime):
    """A cached intent whose last_desired == current_k8s (satisfied) is not
    eligible as a pair partner, so a later scale-down that would breach the
    floor is denied."""
    handler = ScaleRequestHandler(
        runtime=mock_runtime,
        managed_namespaces=["default-my-dgd"],
        k8s_namespace="default",
        min_total_gpus=7,
        max_total_gpus=100,
    )
    # prefill=4, decode=3; total = 7, exactly at floor.
    dgd_state = _dgd_spec(prefill_replicas=4, decode_replicas=3)
    _install_connector(handler, "default/my-dgd", dgd_state)

    # Seed prefill's cache entry as satisfied (last_desired == current).
    handler.orchestrator._intent_cache["default/my-dgd/prefill"] = PoolIntent(
        last_desired=4, last_seen_at=time.time()
    )

    # Decode scale-down would take total to 6, below floor=7. No usable
    # partner because prefill's cached intent is satisfied.
    req = _scale_req(caller_ns="default-my-dgd", decode=2)
    results = await _run(handler, req)
    assert results[0]["status"] == "rejected"


@pytest.mark.asyncio
async def test_pair_packing_continues_past_too_small_candidate(mock_runtime):
    """When a small partner alone doesn't reach the band, the packing must
    keep adding additional partners until the band is reached. This test
    used to verify single-partner search continued to a later candidate;
    with multi-partner packing we instead apply BOTH the small and the
    large partner — accumulating to land inside the band."""
    handler = ScaleRequestHandler(
        runtime=mock_runtime,
        managed_namespaces=["default-dgd-a", "default-dgd-b"],
        k8s_namespace="default",
        min_total_gpus=12,
        max_total_gpus=12,
    )
    # DGD-A: prefill=5, decode=3 (1 GPU each) → 8 GPUs.
    # DGD-B: prefill=1, decode=3 (1 GPU each) → 4 GPUs. Cluster total = 12.
    connector_a = _install_connector(
        handler,
        "default/dgd-a",
        _dgd_spec(prefill_replicas=5, decode_replicas=3),
        parent_dgd_name="dgd-a",
    )
    connector_b = _install_connector(
        handler,
        "default/dgd-b",
        _dgd_spec(prefill_replicas=1, decode_replicas=3),
        parent_dgd_name="dgd-b",
    )
    # Two cross-DGD candidates in DGD-B:
    #  1. prefill +1 → +1 GPU.
    #  2. decode  +4 → +4 GPU.
    # Request: DGD-A prefill 5 → 1 (-4 GPU). Standalone total = 8 (below floor 12).
    # Packing ascending: prefill (+1) accepted (still below band, total=9);
    # decode +4 full would push to 13 — over the strict ceiling (max=12, no
    # upper tolerance). So decode is partially consumed to land at exactly 12.
    # Both partners are applied; decode's cached intent (7) is NOT mutated.
    handler.orchestrator._intent_cache["default/dgd-b/prefill"] = PoolIntent(
        last_desired=2, last_seen_at=time.time()
    )
    handler.orchestrator._intent_cache["default/dgd-b/decode"] = PoolIntent(
        last_desired=7, last_seen_at=time.time()
    )

    req = _scale_req(dgd="dgd-a", caller_ns="default-dgd-a", prefill=1)
    results = await _run(handler, req)
    assert results[0]["status"] == "success"
    connector_a.set_component_replicas.assert_called_once()
    connector_b.set_component_replicas.assert_called_once()
    a_targets = {
        t.sub_component_type.value: t.desired_replicas
        for t in connector_a.set_component_replicas.call_args[0][0]
    }
    b_targets = {
        t.sub_component_type.value: t.desired_replicas
        for t in connector_b.set_component_replicas.call_args[0][0]
    }
    assert a_targets == {"prefill": 1}
    # Prefill fully consumed (small partner), decode partially consumed so
    # the combined total lands at the strict ceiling.
    assert b_targets["prefill"] == 2
    assert (
        3 < b_targets["decode"] < 7
    )  # partial: between current(3) and last_desired(7)
    # Cached intent for decode NOT mutated — planner still wants 7.
    assert handler.orchestrator._intent_cache["default/dgd-b/decode"].last_desired == 7


def test_read_all_pools_tolerates_concurrent_connector_insert(mock_runtime):
    """Reproduces the worker-thread race: another coroutine may insert into
    self.connectors while _read_all_pools iterates from a thread (the
    handler offloads it via asyncio.to_thread). The snapshot must not
    raise RuntimeError on concurrent insert."""
    handler = ScaleRequestHandler(
        runtime=mock_runtime,
        managed_namespaces=["default-my-dgd"],
        k8s_namespace="default",
    )
    _install_connector(
        handler, "default/my-dgd", _dgd_spec(prefill_replicas=1, decode_replicas=1)
    )

    # _read_dgd_pools is called per item; mutate self.connectors during
    # the iteration to simulate a concurrent first-time request.
    original = handler.orchestrator.capacity_manager._read_pools

    def _mutating_read(connector):
        # Inject a new connector mid-iteration. Without the list() snapshot
        # this raises RuntimeError: dictionary changed size during iteration.
        new_conn = MagicMock()
        new_conn.parent_dgd_name = "racy-dgd"
        new_conn.kube_api = MagicMock()
        new_conn.kube_api.get_graph_deployment = MagicMock(
            return_value=_dgd_spec(prefill_replicas=1, decode_replicas=1)
        )
        handler.orchestrator.capacity_manager.connectors.setdefault(
            "default/racy-dgd", new_conn
        )
        return original(connector)

    handler.orchestrator.capacity_manager._read_pools = _mutating_read  # type: ignore[method-assign]

    # Should not raise.
    snapshot = handler.orchestrator.capacity_manager.observe()
    # The pre-existing key is in the snapshot; the mid-iteration insert
    # may or may not appear (depends on snapshot timing) but the call
    # must complete cleanly.
    assert "default/my-dgd" in snapshot


@pytest.mark.asyncio
async def test_intent_cache_clears_on_stable_signal(mock_runtime):
    """A request from a pool with desired == current effectively clears any
    prior pending intent (since the pool is now satisfied)."""
    handler = ScaleRequestHandler(
        runtime=mock_runtime,
        managed_namespaces=["default-my-dgd"],
        k8s_namespace="default",
        min_total_gpus=6,
        max_total_gpus=6,
    )
    connector = _install_connector(
        handler, "default/my-dgd", _dgd_spec(prefill_replicas=3, decode_replicas=3)
    )

    # Prefill has prior intent to scale down to 2
    handler.orchestrator._intent_cache["default/my-dgd/prefill"] = PoolIntent(
        last_desired=2, last_seen_at=time.time()
    )
    # Now prefill sends a stable signal (desired == current)
    req_stable = _scale_req(caller_ns="default-my-dgd", prefill=3)
    await _run(handler, req_stable)
    # Prefill's cached intent is now 3 (== current), i.e., satisfied
    assert (
        handler.orchestrator._intent_cache["default/my-dgd/prefill"].last_desired == 3
    )

    # Decode scale-up that would breach ceiling should no longer find a
    # partner (prefill's intent is now stable/satisfied)
    connector.set_component_replicas.reset_mock()
    req_decode_up = _scale_req(caller_ns="default-my-dgd", decode=4)
    results = await _run(handler, req_decode_up)
    assert results[0]["status"] == "rejected"


# ---------------------------------------------------------------------------- #
# Multi-partner packing (per tedzhouhk review feedback)                        #
# ---------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_pair_packing_three_pools_full_consumption(mock_runtime):
    """tedzhouhk example 1: three pools (P0..P2). P0 has a small cached
    scale-down, P1 has a larger cached scale-down. Neither alone is enough
    to satisfy the scale-up request within band; both fully consumed lands
    on the band edge."""
    handler = ScaleRequestHandler(
        runtime=mock_runtime,
        managed_namespaces=["default-p0", "default-p1", "default-p2"],
        k8s_namespace="default",
        min_total_gpus=10,
        max_total_gpus=10,
    )
    # 1 GPU/worker decode pools; cluster total 3+5+2 = 10.
    p0 = _install_connector(
        handler,
        "default/p0",
        _dgd_spec(prefill_replicas=0, decode_replicas=3),
        parent_dgd_name="p0",
    )
    p1 = _install_connector(
        handler,
        "default/p1",
        _dgd_spec(prefill_replicas=0, decode_replicas=5),
        parent_dgd_name="p1",
    )
    p2 = _install_connector(
        handler,
        "default/p2",
        _dgd_spec(prefill_replicas=0, decode_replicas=2),
        parent_dgd_name="p2",
    )
    # P0 cached: last_desired=1 → delta -2.
    # P1 cached: last_desired=1 → delta -4.
    handler.orchestrator._intent_cache["default/p0/decode"] = PoolIntent(
        last_desired=1, last_seen_at=time.time()
    )
    handler.orchestrator._intent_cache["default/p1/decode"] = PoolIntent(
        last_desired=1, last_seen_at=time.time()
    )
    # P2 request +6 (decode 2 → 8). Standalone cluster total 3+5+8 = 16.
    # Band = [9, 11] with tol=1. Pack ascending: P0 (-2) takes us to 14
    # (still above max+tol=11), P1 (-4) takes us to 10 (in band). Both
    # admitted via the "still_approaching" path then "in_band" accept.
    req = _scale_req(dgd="p2", caller_ns="default-p2", decode=8)
    results = await _run(handler, req)
    assert results[0]["status"] == "success"
    # All three connectors called.
    p0.set_component_replicas.assert_called_once()
    p1.set_component_replicas.assert_called_once()
    p2.set_component_replicas.assert_called_once()
    p0_targets = {
        t.sub_component_type.value: t.desired_replicas
        for t in p0.set_component_replicas.call_args[0][0]
    }
    p1_targets = {
        t.sub_component_type.value: t.desired_replicas
        for t in p1.set_component_replicas.call_args[0][0]
    }
    p2_targets = {
        t.sub_component_type.value: t.desired_replicas
        for t in p2.set_component_replicas.call_args[0][0]
    }
    # Both partners fully consumed (matching their cached last_desired).
    assert p0_targets["decode"] == 1
    assert p1_targets["decode"] == 1
    assert p2_targets["decode"] == 8


@pytest.mark.asyncio
async def test_pair_packing_partial_consumption_leaves_residual(mock_runtime):
    """tedzhouhk example 2: one cached partner is much larger than needed.
    Pack the smaller partner fully, then partially consume the larger so the
    combined transfer lands inside the band. The cached intent's
    ``last_desired`` is NOT mutated — the partner's residual remains pending
    in the cache so future requests can pair with it organically."""
    handler = ScaleRequestHandler(
        runtime=mock_runtime,
        managed_namespaces=["default-p0", "default-p1", "default-p2"],
        k8s_namespace="default",
        min_total_gpus=18,
        max_total_gpus=18,
    )
    p0 = _install_connector(
        handler,
        "default/p0",
        _dgd_spec(prefill_replicas=0, decode_replicas=6),
        parent_dgd_name="p0",
    )
    p1 = _install_connector(
        handler,
        "default/p1",
        _dgd_spec(prefill_replicas=0, decode_replicas=10),
        parent_dgd_name="p1",
    )
    _install_connector(
        handler,
        "default/p2",
        _dgd_spec(prefill_replicas=0, decode_replicas=2),
        parent_dgd_name="p2",
    )
    # P0 cached: last_desired=4 → delta -2.
    # P1 cached: last_desired=2 → delta -8. (Way more than needed.)
    handler.orchestrator._intent_cache["default/p0/decode"] = PoolIntent(
        last_desired=4, last_seen_at=time.time()
    )
    handler.orchestrator._intent_cache["default/p1/decode"] = PoolIntent(
        last_desired=2, last_seen_at=time.time()
    )
    # Cluster total = 6+10+2 = 18. P2 request +4 → standalone total = 22.
    # Pack P0 fully (-2) → 20. Then P1 needs partial ≤ -1 to land in [17, 19].
    # P1 partial: K must keep total ∈ [17, 19]. Adding P1 delta = (K-10)*1.
    # 20 + (K-10) ≤ 19 → K ≤ 9. K ≥ last_desired = 2, so K can be 9 (delta -1).
    # Final: P0=4 (delta -2), P1=9 (partial; planner still wants 2), P2=6 (delta +4).
    # Total = 4 + 9 + 6 = 19. In band.
    req = _scale_req(dgd="p2", caller_ns="default-p2", decode=6)
    results = await _run(handler, req)
    assert results[0]["status"] == "success"
    p0.set_component_replicas.assert_called_once()
    p1.set_component_replicas.assert_called_once()
    p0_targets = {
        t.sub_component_type.value: t.desired_replicas
        for t in p0.set_component_replicas.call_args[0][0]
    }
    p1_targets = {
        t.sub_component_type.value: t.desired_replicas
        for t in p1.set_component_replicas.call_args[0][0]
    }
    # P0 was fully consumed (delta -2)
    assert p0_targets["decode"] == 4
    # P1 was partially consumed — applied K is between current(10) and last_desired(2)
    assert 2 < p1_targets["decode"] < 10
    # The cached intent must NOT have been mutated — planner still wants 2.
    assert handler.orchestrator._intent_cache["default/p1/decode"].last_desired == 2


@pytest.mark.asyncio
async def test_pair_packing_single_partner_regression(mock_runtime):
    """Regression: when only one partner is needed and feasible, packing
    behaves like the historical single-partner search — exactly one partner
    gets included, applied via the existing intra/cross-DGD machinery."""
    handler = ScaleRequestHandler(
        runtime=mock_runtime,
        managed_namespaces=["default-my-dgd"],
        k8s_namespace="default",
        min_total_gpus=6,
        max_total_gpus=6,
    )
    connector = _install_connector(
        handler, "default/my-dgd", _dgd_spec(prefill_replicas=3, decode_replicas=3)
    )
    handler.orchestrator._intent_cache["default/my-dgd/decode"] = PoolIntent(
        last_desired=4, last_seen_at=time.time()
    )
    req = _scale_req(caller_ns="default-my-dgd", prefill=2)
    results = await _run(handler, req)
    assert results[0]["status"] == "success"
    # Single intra-DGD patch combining request + same-DGD partner.
    connector.set_component_replicas.assert_called_once()
    targets = {
        t.sub_component_type.value: t.desired_replicas
        for t in connector.set_component_replicas.call_args[0][0]
    }
    assert targets == {"prefill": 2, "decode": 4}


@pytest.mark.asyncio
async def test_pair_packing_overshooting_partner_is_partially_consumed(mock_runtime):
    """When the next candidate's FULL inclusion would push past the band's
    far edge, the algorithm tries partial consumption that lands at the
    edge. Both partners get applied — the small one fully, the big one
    partially — staying in band."""
    handler = ScaleRequestHandler(
        runtime=mock_runtime,
        managed_namespaces=["default-p0", "default-p1", "default-p2"],
        k8s_namespace="default",
        min_total_gpus=12,
        max_total_gpus=12,
    )
    _install_connector(
        handler,
        "default/p0",
        _dgd_spec(prefill_replicas=0, decode_replicas=4),
        parent_dgd_name="p0",
    )
    p1 = _install_connector(
        handler,
        "default/p1",
        _dgd_spec(prefill_replicas=0, decode_replicas=4),
        parent_dgd_name="p1",
    )
    p2 = _install_connector(
        handler,
        "default/p2",
        _dgd_spec(prefill_replicas=0, decode_replicas=4),
        parent_dgd_name="p2",
    )
    # P1 wants -1, P2 wants -4 (both cached). Request: P0 +1.
    # Cluster total 12 → standalone with request = 13 (above strict ceiling).
    # Pack P1 (-1) → total 12 (in band [11, 13]). Continue.
    # Try P2 full (-4) → would push to 8 (below floor-tol=11). Crosses band.
    # Partial of P2: K=3 lands at total 11 (lower band edge). Apply partial.
    handler.orchestrator._intent_cache["default/p1/decode"] = PoolIntent(
        last_desired=3, last_seen_at=time.time()
    )
    handler.orchestrator._intent_cache["default/p2/decode"] = PoolIntent(
        last_desired=0, last_seen_at=time.time()
    )
    req = _scale_req(dgd="p0", caller_ns="default-p0", decode=5)
    results = await _run(handler, req)
    assert results[0]["status"] == "success"
    # P1 was applied (full).
    p1.set_component_replicas.assert_called_once()
    p1_targets = {
        t.sub_component_type.value: t.desired_replicas
        for t in p1.set_component_replicas.call_args[0][0]
    }
    assert p1_targets == {"decode": 3}
    # P2 was applied with PARTIAL consumption — landed at 3 (one worker
    # shed instead of all four), keeping total at lower band edge (11).
    p2.set_component_replicas.assert_called_once()
    p2_targets = {
        t.sub_component_type.value: t.desired_replicas
        for t in p2.set_component_replicas.call_args[0][0]
    }
    assert 0 < p2_targets["decode"] < 4
    # Cached intent for P2 is NOT mutated — planner still wants 0.
    assert handler.orchestrator._intent_cache["default/p2/decode"].last_desired == 0


@pytest.mark.asyncio
async def test_pair_packing_direction_aware_order_multi_dgd(mock_runtime):
    """When the packing spans multiple DGDs, scale-DOWN DGDs must apply
    before scale-UP DGDs, freeing GPUs before new pods are submitted."""
    handler = ScaleRequestHandler(
        runtime=mock_runtime,
        managed_namespaces=["default-up-dgd", "default-down-dgd"],
        k8s_namespace="default",
        min_total_gpus=8,
        max_total_gpus=8,
    )
    up_dgd = _install_connector(
        handler,
        "default/up-dgd",
        _dgd_spec(prefill_replicas=0, decode_replicas=3),
        parent_dgd_name="up-dgd",
    )
    down_dgd = _install_connector(
        handler,
        "default/down-dgd",
        _dgd_spec(prefill_replicas=0, decode_replicas=5),
        parent_dgd_name="down-dgd",
    )
    handler.orchestrator._intent_cache["default/down-dgd/decode"] = PoolIntent(
        last_desired=3, last_seen_at=time.time()
    )
    # Track call order
    call_order: list[str] = []
    up_dgd.set_component_replicas.side_effect = lambda *a, **kw: call_order.append("up")
    down_dgd.set_component_replicas.side_effect = lambda *a, **kw: call_order.append(
        "down"
    )
    req = _scale_req(dgd="up-dgd", caller_ns="default-up-dgd", decode=5)
    results = await _run(handler, req)
    assert results[0]["status"] == "success"
    # Down DGD must apply before up DGD.
    assert call_order == ["down", "up"]


# ---------------------------------------------------------------------------- #
# v1beta1 component reading + generic-worker role hints (ported from #11990)   #
# ---------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_generic_worker_role_used_for_budget_and_readback(mock_runtime):
    handler = ScaleRequestHandler(
        runtime=mock_runtime,
        managed_namespaces=["default-my-dgd"],
        k8s_namespace="default",
        max_total_gpus=4,
    )
    connector = _install_connector(
        handler, "default/my-dgd", _worker_dgd_spec(replicas=1, gpu=2)
    )
    req = _scale_req(
        caller_ns="default-my-dgd", decode=2, decode_component_name="worker-svc"
    )
    results = await _run(handler, req)
    assert results[0]["status"] == "success"
    assert results[0]["current_replicas"] == {"decode": 1}
    connector.set_component_replicas.assert_called_once()


@pytest.mark.asyncio
async def test_generic_worker_scale_rejected_above_budget(mock_runtime):
    handler = ScaleRequestHandler(
        runtime=mock_runtime,
        managed_namespaces=["default-my-dgd"],
        k8s_namespace="default",
        max_total_gpus=2,
    )
    connector = _install_connector(
        handler, "default/my-dgd", _worker_dgd_spec(replicas=1, gpu=2)
    )
    req = _scale_req(
        caller_ns="default-my-dgd", decode=2, decode_component_name="worker-svc"
    )
    results = await _run(handler, req)
    assert results[0]["status"] == "rejected"
    connector.set_component_replicas.assert_not_called()


@pytest.mark.asyncio
async def test_generic_worker_partner_keeps_component_name(mock_runtime):
    handler = ScaleRequestHandler(
        runtime=mock_runtime,
        managed_namespaces=["default-dgd-a", "default-dgd-b"],
        k8s_namespace="default",
        min_total_gpus=9,
        max_total_gpus=9,
    )
    connector_a = _install_connector(
        handler,
        "default/dgd-a",
        _dgd_spec(prefill_replicas=3, decode_replicas=3),
        parent_dgd_name="dgd-a",
    )
    connector_b = _install_real_connector(
        handler,
        "default/dgd-b",
        _worker_dgd_spec(replicas=3),
        parent_dgd_name="dgd-b",
    )
    cm = handler.orchestrator.capacity_manager
    cm._component_roles["default/dgd-b"] = {"worker-svc": "decode"}
    handler.orchestrator._intent_cache["default/dgd-b/decode"] = PoolIntent(
        last_desired=4, last_seen_at=time.time()
    )

    results = await _run(
        handler, _scale_req(dgd="dgd-a", caller_ns="default-dgd-a", prefill=2)
    )

    assert results[0]["status"] == "success"
    connector_a.set_component_replicas.assert_called_once()
    # The partner's component name rides through to the actual scale call.
    connector_b.kube_api.update_graph_replicas.assert_called_once_with(
        "dgd-b", "worker-svc", 4
    )


@pytest.mark.asyncio
async def test_duplicate_resolved_pool_rejected_in_budget_snapshot(mock_runtime):
    handler = ScaleRequestHandler(
        runtime=mock_runtime,
        managed_namespaces=["default-my-dgd"],
        k8s_namespace="default",
        max_total_gpus=4,
    )
    connector = _install_connector(handler, "default/my-dgd", _two_worker_dgd_spec())
    handler.orchestrator.capacity_manager._component_roles["default/my-dgd"] = {
        "worker-a": "decode",
        "worker-b": "decode",
    }

    results = await _run(
        handler,
        _scale_req(
            caller_ns="default-my-dgd", decode=2, decode_component_name="worker-a"
        ),
    )

    assert results[0]["status"] == "error"
    assert "'worker-a' and 'worker-b'" in results[0]["message"]
    assert "planner pool 'decode'" in results[0]["message"]
    connector.set_component_replicas.assert_not_called()


@pytest.mark.asyncio
async def test_duplicate_resolved_pool_rejected_during_readback(mock_runtime):
    handler = ScaleRequestHandler(
        runtime=mock_runtime,
        managed_namespaces=["default-my-dgd"],
        k8s_namespace="default",
    )
    connector = _install_connector(
        handler,
        "default/my-dgd",
        _worker_dgd_spec(replicas=1, component_name="worker-a"),
    )
    # Clean read for the budget snapshot, duplicate roles for the read-back.
    connector.kube_api.get_graph_deployment.side_effect = [
        _worker_dgd_spec(replicas=1, component_name="worker-a"),
        _two_worker_dgd_spec(),
    ]
    handler.orchestrator.capacity_manager._component_roles["default/my-dgd"] = {
        "worker-a": "decode",
        "worker-b": "decode",
    }

    results = await _run(
        handler,
        _scale_req(
            caller_ns="default-my-dgd", decode=2, decode_component_name="worker-a"
        ),
    )

    assert results[0]["status"] == "error"
    assert "'worker-a' and 'worker-b'" in results[0]["message"]
    assert "planner pool 'decode'" in results[0]["message"]
    connector.set_component_replicas.assert_called_once()


@pytest.mark.asyncio
async def test_multinode_gpu_count_enforces_budget(mock_runtime):
    handler = ScaleRequestHandler(
        runtime=mock_runtime,
        managed_namespaces=["default-my-dgd"],
        k8s_namespace="default",
        max_total_gpus=12,
    )
    connector = _install_connector(
        handler,
        "default/my-dgd",
        _dgd_spec(
            prefill_replicas=1,
            decode_replicas=0,
            prefill_gpu=4,
            prefill_node_count=2,
        ),
    )
    results = await _run(handler, _scale_req(caller_ns="default-my-dgd", prefill=2))
    assert results[0]["status"] == "rejected"
    connector.set_component_replicas.assert_not_called()


@pytest.mark.asyncio
async def test_untyped_worker_in_other_dgd_counts_toward_budget(mock_runtime):
    handler = ScaleRequestHandler(
        runtime=mock_runtime,
        managed_namespaces=["default-dgd-a", "default-dgd-b"],
        k8s_namespace="default",
        max_total_gpus=6,
    )
    connector_a = _install_connector(
        handler,
        "default/dgd-a",
        _dgd_spec(prefill_replicas=1, decode_replicas=1),
        parent_dgd_name="dgd-a",
    )
    _install_connector(
        handler,
        "default/dgd-b",
        _worker_dgd_spec(replicas=2, gpu=2, component_type=None),
        parent_dgd_name="dgd-b",
    )
    results = await _run(
        handler, _scale_req(dgd="dgd-a", caller_ns="default-dgd-a", prefill=2)
    )
    assert results[0]["status"] == "rejected"
    connector_a.set_component_replicas.assert_not_called()
