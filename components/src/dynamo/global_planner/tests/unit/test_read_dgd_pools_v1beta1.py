# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for GlobalPlanner GPU-budget reading across DGD schemas.

Covers v1beta1 support in ``ScaleRequestHandler._read_dgd_pools``: a v1beta1
DGD exposes its workers under ``spec.components`` (not ``spec.services``), so
before the fix the GPU budget aggregated to zero and the ``--max-total-gpus``
/ ``--min-total-gpus`` bounds were silently disabled. v1beta1 is the primary
schema (the version the planner requests); v1alpha1 remains as a fallback.
"""

from unittest.mock import MagicMock

import pytest

from dynamo.global_planner.scale_handler import ScaleRequestHandler

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
