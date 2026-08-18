# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
from unittest.mock import Mock, patch

import pytest

from dynamo.planner.connectors.kubernetes import KubernetesConnector
from dynamo.planner.errors import DeploymentValidationError

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def _component(
    name: str,
    *,
    committed: int,
    updated: int,
    ready: int,
    terminating: int = 0,
) -> dict:
    return {
        "name": name,
        "replicaStatus": {
            "replicas": ready,
            "updatedReplicas": updated,
            "readyReplicas": ready,
        },
        "terminatingReplicas": terminating,
        "_committed": committed,
    }


def _dgpb(*components: dict, rollout: bool = False) -> dict:
    rows = []
    committed = {}
    for component in components:
        row = dict(component)
        committed[row["name"]] = row.pop("_committed")
        rows.append(row)
    return {
        "metadata": {"name": "test-graph"},
        "spec": {"budgetWatts": 4200, "policy": {"minEndpoint": 2}},
        "status": {
            "inventoryEpoch": 17,
            "phase": "Idle",
            "committedReplicaTargets": committed,
            "components": rows,
            "rolloutInProgress": rollout,
            "conditions": [
                {
                    "type": "PowerInfeasible",
                    "status": "False",
                    "reason": "Feasible",
                }
            ],
        },
    }


@pytest.fixture
def connector_and_api(monkeypatch):
    api = Mock()
    api.get_graph_power_budget = Mock()
    api.get_graph_deployment = Mock()
    api.list_pods_for_graph = Mock(return_value=[])
    api.partition_pods_by_component = Mock(return_value={})
    monkeypatch.setattr(
        "dynamo.planner.connectors.kubernetes.KubernetesAPI",
        Mock(return_value=api),
    )
    with patch.dict(os.environ, {"DYN_PARENT_DGD_K8S_NAME": "test-graph"}):
        return KubernetesConnector("test-dynamo-namespace"), api


@pytest.mark.asyncio
async def test_transactional_tick_uses_one_dgpb_snapshot(connector_and_api):
    connector, api = connector_and_api
    api.get_graph_power_budget.return_value = _dgpb(
        _component("prefill", committed=2, updated=2, ready=2),
        _component("decode", committed=4, updated=4, ready=4),
    )

    first = await connector.get_power_aware_worker_counts("prefill", "decode")
    second = await connector.get_power_aware_worker_counts("prefill", "decode")

    assert first == second == (2, 4, True)
    api.get_graph_power_budget.assert_called_once_with("test-graph")
    api.get_graph_deployment.assert_not_called()
    api.list_pods_for_graph.assert_not_called()

    snapshot = connector.get_power_budget_snapshot()
    assert snapshot is not None
    assert snapshot.budget_watts == 4200
    assert snapshot.min_endpoint == 2
    assert snapshot.inventory_epoch == 17
    assert snapshot.phase == "Idle"
    assert snapshot.conditions[0]["reason"] == "Feasible"

    connector.consume_power_budget_snapshot()
    await connector.get_power_aware_worker_counts("prefill", "decode")
    assert api.get_graph_power_budget.call_count == 2


@pytest.mark.asyncio
async def test_transactional_rollout_signal_holds_all_roles(connector_and_api):
    connector, api = connector_and_api
    api.get_graph_power_budget.return_value = _dgpb(
        _component("prefill", committed=2, updated=2, ready=2),
        _component("decode", committed=4, updated=4, ready=4),
        rollout=True,
    )

    assert await connector.get_power_aware_worker_counts("prefill", "decode") == (
        2,
        4,
        False,
    )


@pytest.mark.asyncio
async def test_transactional_component_convergence_uses_committed_vector(
    connector_and_api,
):
    connector, api = connector_and_api
    api.get_graph_power_budget.return_value = _dgpb(
        _component("prefill", committed=3, updated=2, ready=2),
        _component("decode", committed=4, updated=4, ready=4),
    )

    assert await connector.get_power_aware_worker_counts("prefill", "decode") == (
        2,
        4,
        False,
    )


@pytest.mark.asyncio
async def test_static_dgd_retains_legacy_inventory_path(connector_and_api):
    connector, api = connector_and_api
    api.get_graph_power_budget.return_value = None
    api.get_graph_deployment.return_value = {
        "metadata": {"name": "test-graph"},
        "spec": {
            "components": [
                {"name": "prefill", "type": "prefill", "replicas": 2},
                {"name": "decode", "type": "decode", "replicas": 4},
            ]
        },
    }
    api.partition_pods_by_component.return_value = {}
    api.get_service_replica_status.side_effect = [(2, True), (4, True)]
    api.has_terminating_pods.return_value = False
    api.is_rolling_update_blocking_settlement.return_value = (False, "")

    assert await connector.get_power_aware_worker_counts("prefill", "decode") == (
        2,
        4,
        True,
    )
    api.get_graph_deployment.assert_called_once_with("test-graph")
    api.list_pods_for_graph.assert_called_once_with("test-graph")


@pytest.mark.asyncio
async def test_transactional_dgd_never_downgrades_when_dgpb_is_missing(
    connector_and_api,
):
    connector, api = connector_and_api
    api.get_graph_power_budget.return_value = None
    api.get_graph_deployment.return_value = {
        "metadata": {
            "name": "test-graph",
            "annotations": {
                "dynamo.nvidia.com/power-control-mode": "transactional-replica-fence"
            },
        }
    }

    with pytest.raises(DeploymentValidationError, match="has no DGPB"):
        await connector.get_power_aware_worker_counts("prefill", "decode")
    api.list_pods_for_graph.assert_not_called()


@pytest.mark.asyncio
async def test_observed_transactional_dgpb_missing_retries_fail_closed(
    connector_and_api,
):
    connector, api = connector_and_api
    api.get_graph_power_budget.return_value = _dgpb(
        _component("decode", committed=1, updated=1, ready=1)
    )
    await connector.get_power_aware_worker_counts(None, "decode")
    connector.consume_power_budget_snapshot()
    api.get_graph_power_budget.return_value = None

    with pytest.raises(DeploymentValidationError, match="now missing"):
        await connector.get_power_aware_worker_counts(None, "decode")
    with pytest.raises(DeploymentValidationError, match="now missing"):
        await connector.get_power_aware_worker_counts(None, "decode")

    assert api.get_graph_power_budget.call_count == 3
    api.get_graph_deployment.assert_not_called()


@pytest.mark.asyncio
async def test_transactional_policy_log_is_emitted_once(connector_and_api, caplog):
    connector, api = connector_and_api
    api.get_graph_power_budget.return_value = _dgpb(
        _component("decode", committed=1, updated=1, ready=1)
    )

    with caplog.at_level(logging.INFO):
        await connector.get_power_aware_worker_counts(None, "decode")
        connector.consume_power_budget_snapshot()
        await connector.get_power_aware_worker_counts(None, "decode")

    records = [
        record
        for record in caplog.records
        if "PlannerConfig.total_gpu_power_limit" in record.getMessage()
    ]
    assert len(records) == 1
    assert "budgetWatts=4200 minEndpoint=2" in records[0].getMessage()
