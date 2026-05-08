# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for ScaleRequestHandler → GlobalPlannerMetrics wiring
(PR 8 sub-task 8-4).

Runs the real handler with a fresh ``CollectorRegistry`` and asserts
each request branch (authorized-ok, not-authorized, no-operation,
budget-exceeded, exception, success) emits the expected counter label
combination plus a latency observation.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prometheus_client import CollectorRegistry

from dynamo.global_planner.scale_handler import ScaleRequestHandler
from dynamo.planner import SubComponentType, TargetReplica
from dynamo.planner.connectors.protocol import ScaleRequest
from dynamo.planner.monitoring.planner_metrics import GlobalPlannerMetrics

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
    pytest.mark.filterwarnings("ignore::pydantic.warnings.PydanticDeprecatedSince20"),
]


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_runtime():
    return MagicMock()


@pytest.fixture
def metrics():
    return GlobalPlannerMetrics(registry=CollectorRegistry())


def _counter(metrics, *, result, reason):
    return metrics.global_scale_request_total.labels(
        result=result, reason=reason
    )._value.get()


def _latency_count(metrics, *, result):
    samples = list(metrics.global_scale_request_latency_seconds.collect())[0].samples
    for s in samples:
        if s.name.endswith("_count") and s.labels.get("result") == result:
            return s.value
    return 0.0


def _gauge(metrics, *, dgd_name):
    return metrics.global_managed_dgd_gpus.labels(dgd_name=dgd_name)._value.get()


def _make_request(caller="app-ns", dgd="my-dgd"):
    return ScaleRequest(
        caller_namespace=caller,
        graph_deployment_name=dgd,
        k8s_namespace="default",
        target_replicas=[
            TargetReplica(
                sub_component_type=SubComponentType.PREFILL, desired_replicas=3
            )
        ],
    )


# ---------------------------------------------------------------------------
# Authorized ok / success
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_success_path_emits_success_counter_and_gauge(
    mock_runtime, metrics
):
    handler = ScaleRequestHandler(
        runtime=mock_runtime,
        managed_namespaces=["app-ns"],
        k8s_namespace="default",
        metrics=metrics,
    )

    request = _make_request()
    with patch(
        "dynamo.global_planner.scale_handler.KubernetesConnector"
    ) as mock_cls:
        mock_connector = AsyncMock()
        mock_cls.return_value = mock_connector
        mock_connector.set_component_replicas = AsyncMock()
        mock_connector.kube_api = MagicMock()
        # Deployment shape: prefill svc with gpu limit, 3 replicas → 6 GPUs.
        mock_connector.kube_api.get_graph_deployment = MagicMock(
            return_value={
                "spec": {
                    "services": {
                        "prefill": {
                            "subComponentType": "prefill",
                            "replicas": 3,
                            "resources": {"limits": {"gpu": 2}},
                        },
                    }
                }
            }
        )

        async for _ in handler.scale_request(request.model_dump()):
            pass

    assert (
        _counter(metrics, result="success", reason=GlobalPlannerMetrics.REASON_SUCCESS)
        == 1
    )
    assert _latency_count(metrics, result="success") == 1.0
    # 3 replicas × 2 GPUs = 6
    assert _gauge(metrics, dgd_name="my-dgd") == 6.0


# ---------------------------------------------------------------------------
# Not authorized
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unauthorized_emits_not_authorized_reason(mock_runtime, metrics):
    handler = ScaleRequestHandler(
        runtime=mock_runtime,
        managed_namespaces=["only-this-ns"],
        k8s_namespace="default",
        metrics=metrics,
    )
    request = _make_request(caller="other-ns")
    results = []
    async for response in handler.scale_request(request.model_dump()):
        results.append(response)
    assert results[0]["status"] == "error"
    assert (
        _counter(
            metrics,
            result="error",
            reason=GlobalPlannerMetrics.REASON_NOT_AUTHORIZED,
        )
        == 1
    )
    assert _latency_count(metrics, result="error") == 1.0


# ---------------------------------------------------------------------------
# No-operation mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_operation_emits_success_with_no_operation_reason(
    mock_runtime, metrics
):
    handler = ScaleRequestHandler(
        runtime=mock_runtime,
        managed_namespaces=["app-ns"],
        k8s_namespace="default",
        no_operation=True,
        metrics=metrics,
    )
    async for _ in handler.scale_request(_make_request().model_dump()):
        pass
    assert (
        _counter(
            metrics,
            result="success",
            reason=GlobalPlannerMetrics.REASON_NO_OPERATION,
        )
        == 1
    )


# ---------------------------------------------------------------------------
# Budget exceeded
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_budget_exceeded_emits_budget_exceeded_reason(
    mock_runtime, metrics
):
    with patch(
        "dynamo.global_planner.scale_handler.KubernetesAPI"
    ) as mock_kube_api_cls:
        mock_kube_api_cls.return_value.list_graph_deployments.return_value = []

        handler = ScaleRequestHandler(
            runtime=mock_runtime,
            managed_namespaces=["app-ns"],
            k8s_namespace="default",
            max_total_gpus=0,  # any request will exceed
            metrics=metrics,
        )

    # Patch connector cache so _calculate_total_gpus_after_request reads
    # the incoming request's target — simplest way to force a budget
    # overrun without wiring a real deployment.
    fake_connector = MagicMock()
    fake_connector.parent_dgd_name = "my-dgd"
    fake_connector.kube_api = MagicMock()
    fake_connector.kube_api.get_graph_deployment.return_value = {
        "spec": {
            "services": {
                "prefill": {
                    "subComponentType": "prefill",
                    "replicas": 3,
                    "resources": {"limits": {"gpu": 1}},
                }
            }
        }
    }
    handler.connectors = {"default/my-dgd": fake_connector}

    results = []
    async for response in handler.scale_request(_make_request().model_dump()):
        results.append(response)
    assert results[0]["status"] == "error"
    assert "GPU budget exceeded" in results[0]["message"]
    assert (
        _counter(
            metrics,
            result="error",
            reason=GlobalPlannerMetrics.REASON_BUDGET_EXCEEDED,
        )
        == 1
    )


# ---------------------------------------------------------------------------
# Exception path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exception_path_emits_exception_reason(mock_runtime, metrics):
    handler = ScaleRequestHandler(
        runtime=mock_runtime,
        managed_namespaces=["app-ns"],
        k8s_namespace="default",
        metrics=metrics,
    )
    with patch(
        "dynamo.global_planner.scale_handler.KubernetesConnector"
    ) as mock_cls:
        mock_cls.side_effect = RuntimeError("boom")
        results = []
        async for response in handler.scale_request(_make_request().model_dump()):
            results.append(response)
    assert results[0]["status"] == "error"
    assert (
        _counter(
            metrics,
            result="error",
            reason=GlobalPlannerMetrics.REASON_EXCEPTION,
        )
        == 1
    )


# ---------------------------------------------------------------------------
# metrics=None path: existing tests must stay green
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handler_runs_without_metrics(mock_runtime):
    """Handler default (metrics=None) must still complete cleanly —
    the metrics kwarg is additive, not required."""
    handler = ScaleRequestHandler(
        runtime=mock_runtime,
        managed_namespaces=["app-ns"],
        k8s_namespace="default",
        no_operation=True,
    )
    assert handler._metrics is None
    results = []
    async for response in handler.scale_request(_make_request().model_dump()):
        results.append(response)
    assert results[0]["status"] == "success"
