# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ScaleRequestHandler."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dynamo.global_planner.scale_handler import PoolIntent, ScaleRequestHandler
from dynamo.planner import SubComponentType, TargetReplica
from dynamo.planner.connectors.protocol import ScaleRequest

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
    pytest.mark.filterwarnings("ignore::pydantic.warnings.PydanticDeprecatedSince20"),
]


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
        "dynamo.global_planner.scale_handler.KubernetesConnector"
    ) as mock_connector_cls:
        mock_connector = AsyncMock()
        mock_connector_cls.return_value = mock_connector
        mock_connector._async_init = AsyncMock()
        mock_connector.set_component_replicas = AsyncMock()
        mock_connector.kube_api = MagicMock()
        mock_connector.kube_api.get_graph_deployment = MagicMock(
            return_value={
                "spec": {
                    "services": {
                        "prefill-svc": {"subComponentType": "prefill", "replicas": 3},
                        "decode-svc": {"subComponentType": "decode", "replicas": 5},
                    }
                }
            }
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
        "dynamo.global_planner.scale_handler.KubernetesConnector"
    ) as mock_connector_cls:
        mock_connector = AsyncMock()
        mock_connector_cls.return_value = mock_connector
        mock_connector._async_init = AsyncMock()
        mock_connector.set_component_replicas = AsyncMock()
        mock_connector.kube_api = MagicMock()
        mock_connector.kube_api.get_graph_deployment = MagicMock(
            return_value={"spec": {"services": {}}}
        )

        # Process both requests
        async for _ in handler.scale_request(request1.model_dump()):
            pass
        async for _ in handler.scale_request(request2.model_dump()):
            pass

        # Verify two connectors were created
        assert "default/dgd-1" in handler.connectors
        assert "default/dgd-2" in handler.connectors
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
        "dynamo.global_planner.scale_handler.KubernetesConnector"
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


def test_managed_dgd_names_explicit(mock_runtime):
    """Test _managed_dgd_names derives DGD names from Dynamo namespaces."""
    handler = ScaleRequestHandler(
        runtime=mock_runtime,
        managed_namespaces=["my-ns-model-a", "my-ns-model-b"],
        k8s_namespace="my-ns",
    )
    names = handler._managed_dgd_names()
    assert names == {"model-a", "model-b"}


def test_managed_dgd_names_implicit(mock_runtime):
    """Test _managed_dgd_names returns None when no managed namespaces set."""
    handler = ScaleRequestHandler(
        runtime=mock_runtime,
        managed_namespaces=None,
        k8s_namespace="my-ns",
    )
    assert handler._managed_dgd_names() is None


def test_managed_dgd_names_mismatched_prefix(mock_runtime):
    """Test _managed_dgd_names warns for namespaces that don't match the k8s prefix."""
    handler = ScaleRequestHandler(
        runtime=mock_runtime,
        managed_namespaces=["other-ns-model-a", "my-ns-model-b"],
        k8s_namespace="my-ns",
    )
    names = handler._managed_dgd_names()
    # Only the matching namespace is included
    assert names == {"model-b"}


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
        patch("dynamo.global_planner.scale_handler.KubernetesAPI") as mock_kube_cls,
        patch(
            "dynamo.global_planner.scale_handler.KubernetesConnector"
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

        handler._populate_k8s_connectors()

        # Only model-a should be discovered
        assert "default/model-a" in handler.connectors
        assert "default/model-b" not in handler.connectors
        assert "default/gp-ctrl" not in handler.connectors
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
        patch("dynamo.global_planner.scale_handler.KubernetesAPI") as mock_kube_cls,
        patch(
            "dynamo.global_planner.scale_handler.KubernetesConnector"
        ) as mock_connector_cls,
    ):
        mock_kube = MagicMock()
        mock_kube_cls.return_value = mock_kube
        mock_kube.list_graph_deployments.return_value = [
            {"metadata": {"name": "model-a"}},
            {"metadata": {"name": "model-b"}},
        ]
        mock_connector_cls.return_value = MagicMock()

        handler._populate_k8s_connectors()

        # All DGDs should be discovered
        assert "default/model-a" in handler.connectors
        assert "default/model-b" in handler.connectors
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
        "dynamo.global_planner.scale_handler.KubernetesConnector"
    ) as mock_connector_cls:
        mock_connector = AsyncMock()
        mock_connector_cls.return_value = mock_connector
        mock_connector._async_init = AsyncMock()
        mock_connector.set_component_replicas = AsyncMock()
        mock_connector.kube_api = MagicMock()
        mock_connector.kube_api.get_graph_deployment = MagicMock(
            return_value={"spec": {"services": {}}}
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


def _dgd_spec(prefill_replicas, decode_replicas, prefill_gpu=1, decode_gpu=1):
    """Build a DGD deployment spec with prefill + decode services."""
    return {
        "spec": {
            "services": {
                "prefill-svc": {
                    "subComponentType": "prefill",
                    "replicas": prefill_replicas,
                    "resources": {"limits": {"gpu": prefill_gpu}},
                },
                "decode-svc": {
                    "subComponentType": "decode",
                    "replicas": decode_replicas,
                    "resources": {"limits": {"gpu": decode_gpu}},
                },
            }
        }
    }


def _install_connector(handler, dgd_key, dgd_spec_dict, parent_dgd_name="my-dgd"):
    """Attach a mocked KubernetesConnector to the handler for one DGD."""
    connector = AsyncMock()
    connector._async_init = AsyncMock()
    connector.set_component_replicas = AsyncMock()
    connector.parent_dgd_name = parent_dgd_name
    connector.kube_api = MagicMock()
    connector.kube_api.get_graph_deployment = MagicMock(return_value=dgd_spec_dict)
    handler.connectors[dgd_key] = connector
    return connector


def _scale_req(
    dgd="my-dgd",
    k8s_ns="default",
    caller_ns="app-ns",
    prefill=None,
    decode=None,
):
    """Build a ScaleRequest with one or both pool targets set."""
    targets = []
    if prefill is not None:
        targets.append(
            TargetReplica(
                sub_component_type=SubComponentType.PREFILL,
                desired_replicas=prefill,
            )
        )
    if decode is not None:
        targets.append(
            TargetReplica(
                sub_component_type=SubComponentType.DECODE,
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
    assert results[0]["status"] == "error"
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
    handler._intent_cache["default/my-dgd/decode"] = PoolIntent(
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
    handler._intent_cache["default/dgd-b/decode"] = PoolIntent(
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
async def test_same_dgd_pair_preferred_over_cross_dgd(mock_runtime):
    """When both a same-DGD and a cross-DGD partner qualify, same-DGD wins
    (single atomic K8s patch)."""
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
    # Both DGD-A's decode and DGD-B's decode have pending scale-up intents
    handler._intent_cache["default/dgd-a/decode"] = PoolIntent(
        last_desired=4, last_seen_at=time.time()
    )
    handler._intent_cache["default/dgd-b/decode"] = PoolIntent(
        last_desired=4, last_seen_at=time.time()
    )
    req = _scale_req(dgd="dgd-a", caller_ns="default-dgd-a", prefill=2)
    results = await _run(handler, req)
    assert results[0]["status"] == "success"
    # Same-DGD pair → only connector_a called (atomic), connector_b untouched
    connector_a.set_component_replicas.assert_called_once()
    connector_b.set_component_replicas.assert_not_called()
    a_targets = {
        t.sub_component_type.value: t.desired_replicas
        for t in connector_a.set_component_replicas.call_args[0][0]
    }
    assert a_targets == {"prefill": 2, "decode": 4}


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
    handler._intent_cache["default/dgd-b/decode"] = PoolIntent(
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
        "Cross-DGD pair second-patch failed" in rec.message for rec in caplog.records
    )
    # Request response still reports success for the request side
    assert results[0]["status"] == "success"


@pytest.mark.asyncio
async def test_cross_dgd_asymmetric_gpu_tolerance(mock_runtime):
    """Cross-DGD pair with different per-replica GPU counts: tolerance is
    computed from the two paired pools only."""
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
    # DGD-B's decode wants to scale up by 1 (+2 GPUs); DGD-A's prefill request
    # scales down by 1 (-1 GPU). Paired total = 3+8 = 11. tol = max(1, 2) = 2,
    # max+tol = 12. 11 <= 12 → within bounds.
    handler._intent_cache["default/dgd-b/decode"] = PoolIntent(
        last_desired=4, last_seen_at=time.time()
    )
    req = _scale_req(dgd="dgd-a", caller_ns="default-dgd-a", prefill=3)
    results = await _run(handler, req)
    assert results[0]["status"] == "success"
    assert handler.connectors["default/dgd-a"].set_component_replicas.called
    assert handler.connectors["default/dgd-b"].set_component_replicas.called


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
    handler._intent_cache["default/my-dgd/decode"] = PoolIntent(
        last_desired=4, last_seen_at=time.time() - 60
    )

    req = _scale_req(caller_ns="default-my-dgd", prefill=2)
    results = await _run(handler, req)
    assert results[0]["status"] == "error"
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
    handler._intent_cache["default/my-dgd/prefill"] = PoolIntent(
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
async def test_asymmetric_per_worker_gpu_pair_within_tolerance(mock_runtime):
    """Prefill=1 GPU/worker, Decode=2 GPU/worker. Paired transfer lands at
    total=max+tol and is accepted under tolerance."""
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
    # Decode wants +1 (=+2 GPUs), prefill has cached intent -1 (=-1 GPU).
    # Paired total = 10 + 2 - 1 = 11. max+tol = 10+2 = 12, so within bounds.
    handler._intent_cache["default/my-dgd/prefill"] = PoolIntent(
        last_desired=3, last_seen_at=time.time()
    )
    req = _scale_req(caller_ns="default-my-dgd", decode=4)
    results = await _run(handler, req)
    assert results[0]["status"] == "success"
    connector.set_component_replicas.assert_called_once()
    call_targets = connector.set_component_replicas.call_args[0][0]
    sub_types = {t.sub_component_type.value: t.desired_replicas for t in call_targets}
    assert sub_types == {"decode": 4, "prefill": 3}


@pytest.mark.asyncio
async def test_asymmetric_pair_denied_if_outside_tolerance(mock_runtime):
    """Paired total that exceeds max+tolerance is rejected."""
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
    # Paired total = 10 + 4 - 1 = 13. max+tol = 12. Deny.
    handler._intent_cache["default/my-dgd/prefill"] = PoolIntent(
        last_desired=3, last_seen_at=time.time()
    )
    req = _scale_req(caller_ns="default-my-dgd", decode=5)
    results = await _run(handler, req)
    assert results[0]["status"] == "error"
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
    handler._intent_cache["default/my-dgd/decode"] = PoolIntent(
        last_desired=6, last_seen_at=time.time()
    )
    req = _scale_req(caller_ns="default-my-dgd", prefill=2)
    results = await _run(handler, req)
    assert results[0]["status"] == "error"
    assert "below floor" in results[0]["message"].lower()
    connector.set_component_replicas.assert_not_called()


@pytest.mark.asyncio
async def test_initial_below_floor_logs_warning_but_accepts_scale_ups(mock_runtime):
    """At startup, when discovered total < min_total_gpus, a warning is logged
    and scale-ups below ceiling still pass."""
    with (
        patch("dynamo.global_planner.scale_handler.KubernetesAPI") as mock_kube_cls,
        patch(
            "dynamo.global_planner.scale_handler.KubernetesConnector"
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

        with patch("dynamo.global_planner.scale_handler.logger") as mock_logger:
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
    assert results_1[0]["status"] == "error"
    connector.set_component_replicas.assert_not_called()
    # Verify the intent was still cached
    assert "default/my-dgd/prefill" in handler._intent_cache
    assert handler._intent_cache["default/my-dgd/prefill"].last_desired == 2

    # Second request: decode wants to go up to 4. Now prefill's intent pairs.
    req_decode = _scale_req(caller_ns="default-my-dgd", decode=4)
    results_2 = await _run(handler, req_decode)
    assert results_2[0]["status"] == "success"
    connector.set_component_replicas.assert_called_once()
    call_targets = connector.set_component_replicas.call_args[0][0]
    sub_types = {t.sub_component_type.value: t.desired_replicas for t in call_targets}
    assert sub_types == {"decode": 4, "prefill": 2}


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
    handler._intent_cache["default/my-dgd/decode"] = PoolIntent(
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

    assert handler._intent_cache["default/my-dgd/prefill"].last_desired == 4


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
    handler._intent_cache["default/my-dgd/prefill"] = PoolIntent(
        last_desired=4, last_seen_at=time.time()
    )

    # Decode scale-down would take total to 6, below floor=7. No usable
    # partner because prefill's cached intent is satisfied.
    req = _scale_req(caller_ns="default-my-dgd", decode=2)
    results = await _run(handler, req)
    assert results[0]["status"] == "error"


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
    handler._intent_cache["default/my-dgd/prefill"] = PoolIntent(
        last_desired=2, last_seen_at=time.time()
    )
    # Now prefill sends a stable signal (desired == current)
    req_stable = _scale_req(caller_ns="default-my-dgd", prefill=3)
    await _run(handler, req_stable)
    # Prefill's cached intent is now 3 (== current), i.e., satisfied
    assert handler._intent_cache["default/my-dgd/prefill"].last_desired == 3

    # Decode scale-up that would breach ceiling should no longer find a
    # partner (prefill's intent is now stable/satisfied)
    connector.set_component_replicas.reset_mock()
    req_decode_up = _scale_req(caller_ns="default-my-dgd", decode=4)
    results = await _run(handler, req_decode_up)
    assert results[0]["status"] == "error"
