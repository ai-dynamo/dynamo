# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration gate: DGD → resolve → project → clamp, with ZERO Pod mutation.

This is the control-plane half of the power e2e ladder (``gpu_0``; hardware NVML
enforcement is a separate GPU-gated test). It wires a real ``KubernetesConnector``
over a mocked Kubernetes API, resolves DGD-owned caps, projects the power
budget, clamps an over-budget proposal through the real engine-adapter final
boundary, and asserts the mocked CoreV1 saw **no** ``patch_namespaced_pod``
call — the hard no-mutation gate. No live cluster, no GPU.
"""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from dynamo.planner.connectors.kubernetes import KubernetesConnector
from dynamo.planner.core.budget import project_watts
from dynamo.planner.core.types import (
    EngineCapabilities,
    WorkerCapabilities,
    WorkerCounts,
)
from dynamo.planner.plugins.orchestrator.engine_adapter import OrchestratorEngineAdapter

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.planner,
]

_DGD = {
    "metadata": {"name": "power-aware-example"},
    "spec": {
        "components": [
            {
                "name": "VllmPrefillWorker",
                "type": "prefill",
                "replicas": 2,
                "podTemplate": {
                    "metadata": {
                        "annotations": {"dynamo.nvidia.com/gpu-power-limit": "350"}
                    },
                    "spec": {
                        "containers": [
                            {
                                "name": "main",
                                "resources": {"limits": {"nvidia.com/gpu": "2"}},
                            }
                        ]
                    },
                },
            },
            {
                "name": "VllmDecodeWorker",
                "type": "decode",
                "replicas": 2,
                "podTemplate": {
                    "metadata": {
                        "annotations": {"dynamo.nvidia.com/gpu-power-limit": "300"}
                    },
                    "spec": {
                        "containers": [
                            {
                                "name": "main",
                                "resources": {"limits": {"nvidia.com/gpu": "4"}},
                            }
                        ]
                    },
                },
            },
        ]
    },
}


@pytest.fixture
def k8s():
    """A KubernetesConnector whose apiserver client is fully mocked."""
    with (
        patch("dynamo.planner.connectors.clients.kubernetes_api.config"),
        patch(
            "dynamo.planner.connectors.clients.kubernetes_api.client.CoreV1Api"
        ) as core,
        patch(
            "dynamo.planner.connectors.clients.kubernetes_api.client.CustomObjectsApi"
        ) as custom,
        patch.dict(os.environ, {"DYN_PARENT_DGD_K8S_NAME": "power-aware-example"}),
    ):
        custom_api = custom.return_value
        custom_api.get_namespaced_custom_object.return_value = _DGD
        connector = KubernetesConnector("test-ns")
        yield connector, core


def _bare_adapter(capabilities):
    adapter = object.__new__(OrchestratorEngineAdapter)
    adapter._config = SimpleNamespace(
        enable_power_awareness=True,
        total_gpu_power_limit=5000,
        min_endpoint=1,
        min_gpu_budget=-1,
        max_gpu_budget=-1,
        mode="disagg",
    )
    adapter._capabilities = capabilities
    return adapter


def test_resolve_project_clamp_never_patches_pods(k8s):
    connector, core_v1_cls = k8s

    # 1. Resolve DGD-owned caps (read-only GET, no Pod write).
    prefill_cfg, decode_cfg = connector.get_component_power_configs()
    assert prefill_cfg.watts_per_replica == 700  # 2 GPU × 350 W
    assert decode_cfg.watts_per_replica == 1200  # 4 GPU × 300 W

    # 2. Project the budget at the initial replica counts (2 prefill, 2 decode).
    projected = project_watts(
        2, 2, prefill_cfg.watts_per_replica, decode_cfg.watts_per_replica
    )
    assert projected == 2 * 700 + 2 * 1200  # 3800 W of a 5000 W budget

    # 3. Clamp an over-budget scale-up proposal through the real final boundary.
    caps = WorkerCapabilities(
        prefill=EngineCapabilities(
            num_gpu=2, power_watts_per_replica=prefill_cfg.watts_per_replica
        ),
        decode=EngineCapabilities(
            num_gpu=4, power_watts_per_replica=decode_cfg.watts_per_replica
        ),
        power_scale_up_blocked=False,
    )
    adapter = _bare_adapter(caps)
    # Stable deployment: pin expected == ready so this exercises the power
    # proportional clamp, not the rollout hold (which keys off expected=None).
    wc = WorkerCounts(
        ready_num_prefill=2,
        ready_num_decode=2,
        expected_num_prefill=2,
        expected_num_decode=2,
    )
    new_p, new_d = adapter._apply_final_budget(4, 4, wc)  # 4*700 + 4*1200 = 7600 > 5000
    assert new_p is not None and new_d is not None
    assert (new_p, new_d) != (4, 4)  # proportional clamp actually ran
    assert new_p * 700 + new_d * 1200 <= 5000  # clamped to fit

    # 4. HARD GATE: the connector never even instantiates a CoreV1 (Pod) client
    #    across the whole flow — it categorically cannot patch a Pod — and the
    #    write surface does not exist to be called.
    core_v1_cls.assert_not_called()
    assert not hasattr(connector.kube_api, "patch_pod_annotation")
    assert not hasattr(connector.kube_api, "remove_pod_annotation")
    assert not hasattr(connector, "get_component_pods")
