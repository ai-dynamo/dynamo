# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dynamo.llm.exceptions import HttpError
from dynamo.vllm.decision_dispatch import DECISION_DISPATCH_METADATA_KEY
from dynamo.vllm.decision_native_errors import NativeExecutionError
from dynamo.vllm.decision_surface import NativeDecisionSurface
from dynamo.vllm.pooling_handlers import ClassifyWorkerHandler

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
]


def context(stopped=False, killed=False):
    return SimpleNamespace(
        metadata={DECISION_DISPATCH_METADATA_KEY: "head"},
        id=lambda: "head-request",
        is_stopped=lambda: stopped,
        is_killed=lambda: killed,
        trace_headers=lambda: {"traceparent": "test-parent"},
    )


def surface(adapter):
    return NativeDecisionSurface("head", adapter)


@pytest.mark.asyncio
async def test_head_dispatch_precedes_ordinary_pooling_validation():
    response = {"protocol_version": 1, "outcomes": [], "usage": {}}
    adapter = SimpleNamespace(kind="head", execute=AsyncMock(return_value=response))
    handler = object.__new__(ClassifyWorkerHandler)
    handler.decision = surface(adapter)
    wire = {"decision_head": {"protocol_version": 1}}
    request_context = context()

    returned = [item async for item in handler.generate(wire, request_context)]

    assert returned == [{**response, "finish_reason": "stop"}]
    assert "finish_reason" not in response
    assert adapter.execute.await_args.args == (wire, request_context)
    assert adapter.execute.await_args.kwargs["request_id"] == "head-request"
    assert adapter.execute.await_args.kwargs["trace_headers"] == {
        "traceparent": "test-parent"
    }
    assert adapter.execute.await_args.kwargs["cancelled"]() is False


@pytest.mark.asyncio
async def test_unregistered_head_request_does_not_fall_through_to_pooling():
    handler = object.__new__(ClassifyWorkerHandler)
    handler.decision = NativeDecisionSurface("head")
    with pytest.raises(HttpError, match="not available") as failure:
        _ = [item async for item in handler.generate({"decision_head": {}}, context())]
    assert failure.value.code == 503


@pytest.mark.asyncio
@pytest.mark.parametrize(("stopped", "killed"), [(True, False), (False, True)])
async def test_head_dispatch_preserves_context_cancellation(stopped, killed):
    adapter = SimpleNamespace(kind="head", execute=AsyncMock(return_value={}))
    handler = object.__new__(ClassifyWorkerHandler)
    handler.decision = surface(adapter)
    _ = [
        item
        async for item in handler.generate(
            {"decision_head": {}}, context(stopped=stopped, killed=killed)
        )
    ]
    assert adapter.execute.await_args.kwargs["cancelled"]() is True


@pytest.mark.asyncio
async def test_head_dispatch_propagates_execution_failure():
    adapter = SimpleNamespace(
        kind="head", execute=AsyncMock(side_effect=NativeExecutionError("bad read"))
    )
    handler = object.__new__(ClassifyWorkerHandler)
    handler.decision = surface(adapter)
    with pytest.raises(NativeExecutionError, match="bad read"):
        _ = [item async for item in handler.generate({"decision_head": {}}, context())]
