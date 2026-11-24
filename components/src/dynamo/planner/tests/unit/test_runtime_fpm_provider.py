# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FPM subscribers follow the endpoint the role's WorkerInfo names."""

from unittest.mock import Mock

import pytest

import dynamo.llm
from dynamo.planner.environment.metrics_provider.runtime_provider import (
    RuntimeFpmProvider,
)
from dynamo.planner.environment.state import DeploymentState
from dynamo.planner.monitoring.worker_info import WorkerInfo

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


class _FakeSubscriber:
    def __init__(self, endpoint):
        self.endpoint = endpoint
        self.closed = False

    def start_tracking(self, **_kwargs):
        pass

    def shutdown(self):
        self.closed = True


def _provider(state: DeploymentState, role: str) -> RuntimeFpmProvider:
    state_source = Mock()
    state_source.deployment_state.return_value = state
    namespace = Mock()
    namespace.runtime_namespace.return_value = "ns"
    runtime = Mock()
    runtime.endpoint.side_effect = lambda path: path
    return RuntimeFpmProvider(
        require_prefill=role == "prefill",
        require_decode=role == "decode",
        backend="vllm",
        model_name="test-model",
        runtime=runtime,
        state_source=state_source,
        namespace_source=namespace,
    )


def _subscriber(provider: RuntimeFpmProvider, role: str):
    return getattr(provider, f"_{role}_fpm_sub")


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["prefill", "decode"])
async def test_refresh_binds_once_worker_info_appears(role, monkeypatch):
    monkeypatch.setattr(dynamo.llm, "FpmEventSubscriber", _FakeSubscriber)
    state = DeploymentState()
    provider = _provider(state, role)

    await provider.async_init()
    assert _subscriber(provider, role) is None

    getattr(state, role).info = WorkerInfo(component_name="custom", endpoint="generate")
    await provider.refresh(state)

    assert _subscriber(provider, role).endpoint == "ns.custom.generate"


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["prefill", "decode"])
async def test_refresh_rebinds_only_when_the_worker_endpoint_changes(role, monkeypatch):
    monkeypatch.setattr(dynamo.llm, "FpmEventSubscriber", _FakeSubscriber)
    state = DeploymentState()
    getattr(state, role).info = WorkerInfo(component_name="custom", endpoint="generate")
    provider = _provider(state, role)

    await provider.async_init()
    first = _subscriber(provider, role)
    await provider.refresh(state)
    assert _subscriber(provider, role) is first

    provider._unjoined_workers[role] = ("worker-0",)
    getattr(state, role).info.endpoint = "infer"
    await provider.refresh(state)

    assert first.closed
    assert _subscriber(provider, role).endpoint == "ns.custom.infer"
    assert role not in provider._unjoined_workers


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["prefill", "decode"])
async def test_failed_rebind_does_not_keep_the_shut_down_subscriber(role, monkeypatch):
    monkeypatch.setattr(dynamo.llm, "FpmEventSubscriber", _FakeSubscriber)
    state = DeploymentState()
    getattr(state, role).info = WorkerInfo(component_name="custom", endpoint="generate")
    provider = _provider(state, role)
    await provider.async_init()
    provider.runtime.endpoint.side_effect = ValueError("invalid endpoint")
    getattr(state, role).info.endpoint = "infer"

    with pytest.raises(ValueError):
        await provider.refresh(state)

    assert _subscriber(provider, role) is None
