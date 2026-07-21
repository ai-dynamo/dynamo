# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for GlobalPlanner GPU-budget reading across DGD schemas.

Covers v1beta1 support in ``ScaleRequestHandler._read_dgd_pools``: a v1beta1
DGD exposes its workers under ``spec.components`` (not ``spec.services``), so
before the fix the GPU budget aggregated to zero and the ``--max-total-gpus``
/ ``--min-total-gpus`` bounds were silently disabled. v1beta1 is the primary
schema (the version the planner requests); v1alpha1 remains as a fallback.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dynamo.global_planner.scale_handler import ScaleRequestHandler
from dynamo.planner import SubComponentType, TargetReplica
from dynamo.planner.connectors.protocol import ScaleRequest

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def _connector(deployment):
    connector = MagicMock()
    connector.parent_dgd_name = "my-dgd"
    connector.kube_api.get_graph_deployment = MagicMock(return_value=deployment)
    return connector


def _handler():
    return ScaleRequestHandler(
        runtime=MagicMock(), managed_namespaces=["app-ns"], k8s_namespace="default"
    )


# v1alpha1 DGD: workers under spec.services, GPU at resources.limits.gpu.
V1ALPHA1 = {
    "spec": {
        "services": {
            "prefill-svc": {
                "subComponentType": "prefill",
                "replicas": 2,
                "resources": {"limits": {"gpu": "1"}},
            },
            "decode-svc": {
                "subComponentType": "decode",
                "replicas": 3,
                "resources": {"limits": {"gpu": "1"}},
            },
            # Non-worker service (no subComponentType) is not counted as a pool.
            "frontend-svc": {"replicas": 1},
        }
    }
}


def _component(name, ctype, replicas, gpu):
    limits = {"nvidia.com/gpu": str(gpu)} if gpu else {}
    return {
        "name": name,
        "type": ctype,
        "replicas": replicas,
        "podTemplate": {"spec": {"containers": [{"resources": {"limits": limits}}]}},
    }


# v1beta1 DGD: workers under spec.components, GPU at
# podTemplate.spec.containers[].resources.limits["nvidia.com/gpu"].
V1BETA1 = {
    "spec": {
        "components": [
            _component("Frontend", "frontend", 1, 0),
            _component("Planner", "planner", 1, 0),
            _component("VllmPrefillWorker", "prefill", 2, 1),
            _component("VllmDecodeWorker", "decode", 3, 1),
        ]
    }
}


def test_read_pools_v1beta1_components():
    """v1beta1 spec.components workers are read (regression: was {} -> 0 GPU)."""
    pools = _handler()._read_dgd_pools(_connector(V1BETA1))
    assert set(pools) == {"prefill", "decode"}
    assert pools["prefill"].current_replicas == 2
    assert pools["prefill"].gpu_per_replica == 1
    assert pools["decode"].current_replicas == 3
    assert pools["decode"].gpu_per_replica == 1


def test_read_pools_v1beta1_skips_non_workers():
    """Frontend/Planner components carry no GPU and are not counted as pools."""
    pools = _handler()._read_dgd_pools(_connector(V1BETA1))
    assert "frontend" not in pools
    assert "planner" not in pools


def test_read_pools_v1alpha1_fallback():
    """Legacy v1alpha1 spec.services DGDs are still read via the fallback."""
    pools = _handler()._read_dgd_pools(_connector(V1ALPHA1))
    assert set(pools) == {"prefill", "decode"}
    assert pools["prefill"].current_replicas == 2
    assert pools["prefill"].gpu_per_replica == 1
    assert pools["decode"].current_replicas == 3
    assert pools["decode"].gpu_per_replica == 1


def test_v1beta1_components_take_precedence():
    """When both schemas are present, spec.components wins and the
    spec.services fallback is not consulted."""
    alpha_services = {
        "prefill-svc": {
            "subComponentType": "prefill",
            "replicas": 9,
            "resources": {"limits": {"gpu": "8"}},
        },
    }
    both = {"spec": {**V1BETA1["spec"], "services": alpha_services}}
    pools = _handler()._read_dgd_pools(_connector(both))
    assert set(pools) == {"prefill", "decode"}
    assert pools["prefill"].current_replicas == 2  # from components, not 9
    assert pools["prefill"].gpu_per_replica == 1  # from components, not 8


def test_budget_nonzero_for_v1beta1():
    """Total GPUs across a v1beta1 DGD is the real 5 (2*1 + 3*1), not 0."""
    handler = _handler()
    handler.connectors["default/my-dgd"] = _connector(V1BETA1)
    assert handler._total_gpus_with_overrides({}) == 5


# v1beta1 aggregated DGD: both workers declared as the generic `type: worker`
# (addressable only by component name), 2*1 + 3*1 = 5 GPUs.
V1BETA1_GENERIC_WORKERS = {
    "spec": {
        "components": [
            _component("Frontend", "frontend", 1, 0),
            _component("VllmPrefillWorker", "worker", 2, 1),
            _component("VllmDecodeWorker", "worker", 3, 1),
        ]
    }
}


def test_generic_worker_components_count_toward_budget():
    """`type: worker` components are snapshotted (keyed worker/<name>) and
    their GPUs count toward the budget total."""
    handler = _handler()
    pools = handler._read_dgd_pools(_connector(V1BETA1_GENERIC_WORKERS))
    assert set(pools) == {"worker/VllmPrefillWorker", "worker/VllmDecodeWorker"}
    assert pools["worker/VllmPrefillWorker"].current_replicas == 2
    assert pools["worker/VllmDecodeWorker"].gpu_per_replica == 1
    handler.connectors["default/my-dgd"] = _connector(V1BETA1_GENERIC_WORKERS)
    assert handler._total_gpus_with_overrides({}) == 5


@pytest.mark.asyncio
async def test_ceiling_counts_generic_worker_gpus():
    """A typed scale-up is rejected when generic-worker GPUs already consume
    the ceiling headroom (regression: type: worker GPUs were invisible to
    the budget, so this request was admitted)."""
    mixed = {
        "spec": {
            "components": [
                _component("VllmPrefillWorker", "prefill", 1, 1),
                _component("VllmDecodeWorker", "decode", 1, 1),
                _component("VllmExtraWorker", "worker", 3, 1),
            ]
        }
    }
    handler = ScaleRequestHandler(
        runtime=MagicMock(),
        managed_namespaces=["app-ns"],
        k8s_namespace="default",
        max_total_gpus=6,
    )
    connector = AsyncMock()
    connector.set_component_replicas = AsyncMock()
    connector.parent_dgd_name = "my-dgd"
    connector.kube_api = MagicMock()
    connector.kube_api.get_graph_deployment = MagicMock(return_value=mixed)
    handler.connectors["default/my-dgd"] = connector

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
    # Baseline 1+1+3 = 5 GPUs; prefill 1 -> 3 lands at 7 > ceiling 6.
    # Without worker counting the total would read 4 and be admitted.
    results = [r async for r in handler.scale_request(request.model_dump())]

    assert results[0]["status"] == "rejected"
    connector.set_component_replicas.assert_not_called()


@pytest.mark.asyncio
async def test_scale_success_reports_replicas_on_v1beta1():
    """Post-scale current_replicas read works on a v1beta1 DGD.

    Regression: the success path parsed ``deployment["spec"]["services"]``
    directly, which raises KeyError on a v1beta1 DGD and turned a completed
    scale into an ERROR response.
    """
    handler = _handler()
    request = ScaleRequest(
        caller_namespace="app-ns",
        graph_deployment_name="my-dgd",
        k8s_namespace="default",
        target_replicas=[
            TargetReplica(
                sub_component_type=SubComponentType.PREFILL, desired_replicas=2
            )
        ],
    )

    with patch(
        "dynamo.global_planner.scale_handler.KubernetesConnector"
    ) as mock_connector_cls:
        mock_connector = AsyncMock()
        mock_connector_cls.return_value = mock_connector
        mock_connector.set_component_replicas = AsyncMock()
        mock_connector.parent_dgd_name = "my-dgd"
        mock_connector.kube_api = MagicMock()
        mock_connector.kube_api.get_graph_deployment = MagicMock(return_value=V1BETA1)

        results = [r async for r in handler.scale_request(request.model_dump())]

    assert len(results) == 1
    assert results[0]["status"] == "success"
    assert results[0]["current_replicas"] == {"prefill": 2, "decode": 3}
