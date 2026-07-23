# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for KubernetesCapacityManager — the K8s observe/actuate backend."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dynamo.global_planner.capacity_manager import (
    KubernetesCapacityManager,
    PoolSpec,
)
from dynamo.planner import SubComponentType, TargetReplica

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def _dgd_spec(prefill_replicas, decode_replicas, prefill_gpu=1, decode_gpu=1):
    return {
        "spec": {
            "services": {
                "prefill": {
                    "subComponentType": "prefill",
                    "replicas": prefill_replicas,
                    "resources": {"limits": {"gpu": prefill_gpu}},
                },
                "decode": {
                    "subComponentType": "decode",
                    "replicas": decode_replicas,
                    "resources": {"limits": {"gpu": decode_gpu}},
                },
            }
        }
    }


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
        patch("dynamo.global_planner.capacity_manager.KubernetesAPI") as mock_kube_cls,
        patch(
            "dynamo.global_planner.capacity_manager.KubernetesConnector"
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
        patch("dynamo.global_planner.capacity_manager.KubernetesAPI") as mock_kube_cls,
        patch(
            "dynamo.global_planner.capacity_manager.KubernetesConnector"
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
        "dynamo.global_planner.capacity_manager.KubernetesAPI",
        side_effect=RuntimeError("no cluster"),
    ):
        # Best-effort: must not raise, returns nothing discovered.
        assert cm.discover(None) == []
    assert cm.connectors == {}


# ---------------------------------------------------------------------------- #
# Registration / observe / actuate                                            #
# ---------------------------------------------------------------------------- #


def test_ensure_participant_idempotent():
    cm = KubernetesCapacityManager("default")
    with patch(
        "dynamo.global_planner.capacity_manager.KubernetesConnector"
    ) as mock_connector_cls:
        mock_connector_cls.return_value = MagicMock()
        cm.ensure_participant(
            "default/my-dgd",
            caller_name="app-ns",
            cluster_name="default",
            deployment_name="my-dgd",
        )
        cm.ensure_participant(
            "default/my-dgd",
            caller_name="app-ns",
            cluster_name="default",
            deployment_name="my-dgd",
        )
        assert mock_connector_cls.call_count == 1
        assert cm.knows("default/my-dgd")
        assert not cm.knows("default/other")


def test_observe_parses_pools():
    cm = KubernetesCapacityManager("default")
    _install_connector(
        cm,
        "default/my-dgd",
        _dgd_spec(prefill_replicas=2, decode_replicas=3, decode_gpu=2),
    )
    snapshot = cm.observe()
    pools = snapshot["default/my-dgd"]
    assert pools["prefill"] == PoolSpec("prefill", 2, 1)
    assert pools["decode"] == PoolSpec("decode", 3, 2)


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
    assert snapshot["default/good"]["prefill"] == PoolSpec("prefill", 1, 1)
    assert snapshot["default/bad"] == {}  # tolerated, empty
    assert good.kube_api.get_graph_deployment.called


def test_current_replicas():
    cm = KubernetesCapacityManager("default")
    _install_connector(
        cm, "default/my-dgd", _dgd_spec(prefill_replicas=2, decode_replicas=5)
    )
    assert cm.current_replicas("default/my-dgd") == {"prefill": 2, "decode": 5}


@pytest.mark.asyncio
async def test_actuate_calls_connector():
    cm = KubernetesCapacityManager("default")
    connector = _install_connector(cm, "default/my-dgd", _dgd_spec(1, 1))
    targets = [
        TargetReplica(sub_component_type=SubComponentType.PREFILL, desired_replicas=3)
    ]
    await cm.actuate("default/my-dgd", targets, blocking=True)
    connector.set_component_replicas.assert_awaited_once_with(targets, blocking=True)
