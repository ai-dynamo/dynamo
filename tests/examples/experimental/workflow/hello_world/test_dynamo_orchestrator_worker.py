# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncIterator, Mapping
from typing import Any

import pytest

from dynamo.experimental.workflow import RemoteStageServer, WorkflowEndpointHandler
from examples.experimental.workflow.hello_world.orchestrator_worker import (
    ENDPOINTS,
    build_orchestrator,
)
from examples.experimental.workflow.hello_world.stages import STAGES

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]


class _RequestContext:
    def __init__(self) -> None:
        self._stopped = asyncio.Event()

    def id(self) -> str:
        return "orchestrator-test"

    def detached(self, request_id: str) -> "_RequestContext":
        del request_id
        return self

    def is_stopped(self) -> bool:
        return self._stopped.is_set()

    def is_killed(self) -> bool:
        return False

    def stop_generating(self) -> None:
        self._stopped.set()

    async def async_killed_or_stopped(self) -> None:
        await self._stopped.wait()


class _LoopbackClient:
    def __init__(self, server: RemoteStageServer) -> None:
        self._server = server

    async def wait_for_instances(self) -> None:
        return None

    async def round_robin(
        self,
        request: Mapping[str, Any],
        *,
        annotated: bool,
        context: Any = None,
    ) -> AsyncIterator[dict[str, Any]]:
        assert annotated is False
        return self._server.generate(request, context)


class _Endpoint:
    def __init__(self, client: _LoopbackClient) -> None:
        self._client = client

    async def client(self) -> _LoopbackClient:
        return self._client


class _Runtime:
    def __init__(self, clients: Mapping[str, _LoopbackClient]) -> None:
        self._clients = clients
        self.endpoint_ids: list[str] = []

    def endpoint(self, endpoint_id: str) -> _Endpoint:
        self.endpoint_ids.append(endpoint_id)
        return _Endpoint(self._clients[endpoint_id])


async def test_orchestrator_worker_calls_three_remote_stages() -> None:
    clients = {
        endpoint_id: _LoopbackClient(RemoteStageServer(stage_id, STAGES[stage_id]()))
        for stage_id, endpoint_id in ENDPOINTS.items()
    }
    runtime = _Runtime(clients)
    orchestrator = await build_orchestrator(runtime)

    chunks = [
        chunk
        async for chunk in WorkflowEndpointHandler(orchestrator).generate(
            {"token_ids": [1]},
            _RequestContext(),
        )
    ]

    assert chunks == [
        {
            "token_ids": [],
            "text": "Hello, World!",
            "index": 0,
            "finish_reason": "stop",
        }
    ]
    assert runtime.endpoint_ids == sorted(ENDPOINTS.values())
