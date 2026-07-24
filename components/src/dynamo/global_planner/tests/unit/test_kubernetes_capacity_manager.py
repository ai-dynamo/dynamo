# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for KubernetesCapacityManager — the K8s observe/scale backend.

Pool state is read from the v1beta1 DGD component model (``spec.components[]``).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dynamo.global_planner.capacity_manager import PoolSpec
from dynamo.global_planner.kubernetes_capacity_manager import KubernetesCapacityManager
from dynamo.planner import SubComponentType, TargetReplica

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def _component(name, replicas, gpu=1, ctype=None, node_count=None):
    component = {
        "name": name,
        "replicas": replicas,
        "podTemplate": {
            "spec": {
                "containers": [
                    {"name": "main", "resources": {"limits": {"nvidia.com/gpu": gpu}}}
                ]
            }
        },
    }
    if ctype is not None:
        component["type"] = ctype
    if node_count is not None:
        component["multinode"] = {"nodeCount": node_count}
    return component


def _dgd_spec(
    prefill_replicas,
    decode_replicas,
    prefill_gpu=1,
    decode_gpu=1,
    prefill_node_count=None,
):
    """A v1beta1 DGD spec with typed prefill and decode components."""
    return {
        "spec": {
            "components": [
                _component(
                    "prefill-svc",
                    prefill_replicas,
                    gpu=prefill_gpu,
                    ctype="prefill",
                    node_count=prefill_node_count,
                ),
                _component(
                    "decode-svc", decode_replicas, gpu=decode_gpu, ctype="decode"
                ),
            ]
        }
    }


def _worker_dgd_spec(replicas, gpu=1, name="worker-svc", ctype="worker"):
    """A v1beta1 DGD spec with a single generic worker component."""
    return {"spec": {"components": [_component(name, replicas, gpu=gpu, ctype=ctype)]}}


def _install_connector(cm, key, spec, parent_dgd_name="my-dgd"):
    connector = MagicMock()
    connector.parent_dgd_name = parent_dgd_name
    connector.set_component_replicas = AsyncMock()
    connector.kube_api = MagicMock()
    connector.kube_api.get_graph_deployment = MagicMock(return_value=spec)
    cm.connectors[key] = connector
    return connector


# ---------------------------------------------------------------------------- #
# Deployment-name derivation (K8s operator convention)                         #
# ---------------------------------------------------------------------------- #


def test_managed_deployment_names_explicit():
    cm = KubernetesCapacityManager("my-ns")
    assert cm._managed_deployment_names({"my-ns-model-a", "my-ns-model-b"}) == {
        "model-a",
        "model-b",
    }


def test_managed_deployment_names_implicit():
    cm = KubernetesCapacityManager("my-ns")
    assert cm._managed_deployment_names(None) is None


def test_managed_deployment_names_mismatched_prefix():
    cm = KubernetesCapacityManager("my-ns")
    # Only the caller matching the cluster prefix contributes a deployment name.
    assert cm._managed_deployment_names({"other-ns-model-a", "my-ns-model-b"}) == {
        "model-b"
    }


# ---------------------------------------------------------------------------- #
# Discovery                                                                    #
# ---------------------------------------------------------------------------- #


def test_discover_explicit_mode():
    cm = KubernetesCapacityManager("default")
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
            {"metadata": {"name": "gp-ctrl"}},
        ]
        mock_connector_cls.return_value = MagicMock()

        # Managed callers are namespaces; the deployment name is derived.
        discovered = cm.discover({"default-model-a"})

        assert discovered == ["model-a"]
        assert "default/model-a" in cm.connectors
        assert "default/model-b" not in cm.connectors
        assert mock_connector_cls.call_count == 1


def test_discover_implicit_mode():
    cm = KubernetesCapacityManager("default")
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

        discovered = cm.discover(None)

        assert set(discovered) == {"model-a", "model-b"}
        assert "default/model-a" in cm.connectors
        assert "default/model-b" in cm.connectors
        assert mock_connector_cls.call_count == 2


def test_discover_tolerates_api_failure():
    cm = KubernetesCapacityManager("default")
    with patch(
        "dynamo.global_planner.kubernetes_capacity_manager.KubernetesAPI",
        side_effect=RuntimeError("no cluster"),
    ):
        # Best-effort: must not raise, returns nothing discovered.
        assert cm.discover(None) == []
    assert cm.connectors == {}


# ---------------------------------------------------------------------------- #
# Registration / scale                                                        #
# ---------------------------------------------------------------------------- #


def test_ensure_participant_idempotent():
    cm = KubernetesCapacityManager("default")
    with patch(
        "dynamo.global_planner.kubernetes_capacity_manager.KubernetesConnector"
    ) as mock_connector_cls:
        mock_connector_cls.return_value = MagicMock()
        cm.ensure_participant(
            "default/my-dgd",
            caller_name="app-ns",
            namespace="default",
            deployment_name="my-dgd",
        )
        cm.ensure_participant(
            "default/my-dgd",
            caller_name="app-ns",
            namespace="default",
            deployment_name="my-dgd",
        )
        assert mock_connector_cls.call_count == 1
        assert cm.participant_exists("default/my-dgd")
        assert not cm.participant_exists("default/other")


@pytest.mark.asyncio
async def test_scale_calls_connector():
    cm = KubernetesCapacityManager("default")
    connector = _install_connector(cm, "default/my-dgd", _dgd_spec(1, 1))
    targets = [
        TargetReplica(sub_component_type=SubComponentType.PREFILL, desired_replicas=3)
    ]
    await cm.scale("default/my-dgd", targets, blocking=True)
    connector.set_component_replicas.assert_awaited_once_with(targets, blocking=True)


# ---------------------------------------------------------------------------- #
# Observe / current_replicas (v1beta1 component reading)                       #
# ---------------------------------------------------------------------------- #


def test_observe_parses_typed_components():
    cm = KubernetesCapacityManager("default")
    _install_connector(
        cm,
        "default/my-dgd",
        _dgd_spec(prefill_replicas=2, decode_replicas=3, decode_gpu=2),
    )
    pools = cm.observe()["default/my-dgd"]
    assert pools["prefill"] == PoolSpec(
        sub_type="prefill",
        current_replicas=2,
        gpu_per_replica=1,
        component_name="prefill-svc",
    )
    assert pools["decode"] == PoolSpec(
        sub_type="decode",
        current_replicas=3,
        gpu_per_replica=2,
        component_name="decode-svc",
    )


def test_observe_multinode_scales_gpu_count():
    cm = KubernetesCapacityManager("default")
    _install_connector(
        cm, "default/my-dgd", _dgd_spec(1, 1, prefill_gpu=2, prefill_node_count=2)
    )
    # gpu_per_replica = main-container GPUs (2) × nodeCount (2) = 4.
    assert cm.observe()["default/my-dgd"]["prefill"].gpu_per_replica == 4


def test_observe_generic_worker_keyed_by_name_then_role_hint():
    cm = KubernetesCapacityManager("default")
    _install_connector(cm, "default/w", _worker_dgd_spec(replicas=2, gpu=2))

    # No hint yet: the worker still counts, keyed by its component name.
    pools = cm.observe()["default/w"]
    assert "worker-svc" in pools and pools["worker-svc"].gpu_per_replica == 2

    # After its Planner sends a role hint, the worker maps to that pool.
    cm.remember_roles(
        "default/w",
        [
            TargetReplica(
                sub_component_type=SubComponentType.DECODE,
                component_name="worker-svc",
                desired_replicas=2,
            )
        ],
    )
    pools = cm.observe()["default/w"]
    assert "worker-svc" not in pools
    assert pools["decode"].component_name == "worker-svc"


def test_observe_untyped_worker_with_gpu_counts_toward_budget():
    cm = KubernetesCapacityManager("default")
    _install_connector(cm, "default/w", _worker_dgd_spec(replicas=2, gpu=2, ctype=None))
    pools = cm.observe()["default/w"]
    assert pools["worker-svc"].gpu_per_replica == 2


def test_observe_tolerates_read_failure():
    cm = KubernetesCapacityManager("default")
    good = _install_connector(
        cm, "default/good", _dgd_spec(1, 1), parent_dgd_name="good"
    )
    bad = MagicMock()
    bad.parent_dgd_name = "bad"
    bad.kube_api = MagicMock()
    bad.kube_api.get_graph_deployment = MagicMock(side_effect=RuntimeError("boom"))
    cm.connectors["default/bad"] = bad

    snapshot = cm.observe()
    assert snapshot["default/good"]["prefill"].current_replicas == 1
    assert snapshot["default/bad"] == {}  # tolerated, empty
    assert good.kube_api.get_graph_deployment.called


def test_observe_require_complete_raises_on_read_failure():
    cm = KubernetesCapacityManager("default")
    bad = MagicMock()
    bad.parent_dgd_name = "bad"
    bad.kube_api = MagicMock()
    bad.kube_api.get_graph_deployment = MagicMock(side_effect=RuntimeError("boom"))
    cm.connectors["default/bad"] = bad
    with pytest.raises(RuntimeError):
        cm.observe(require_complete=True)


def test_current_replicas():
    cm = KubernetesCapacityManager("default")
    _install_connector(
        cm, "default/my-dgd", _dgd_spec(prefill_replicas=2, decode_replicas=5)
    )
    assert cm.current_replicas("default/my-dgd") == {"prefill": 2, "decode": 5}
