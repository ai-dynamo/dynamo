# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for frontend routed-engine topology adapters."""

import asyncio
from typing import Any

import pytest

from dynamo.common.global_router_protocol import (
    GLOBAL_ROUTER_RETRY_ATTEMPT_KEY,
    make_global_router_retry_control,
    make_global_router_retry_exhausted_control,
)
from dynamo.frontend.routed_engine_adapters import (
    GlobalRouterRoutedEngineAdapter,
    wrap_routed_engine,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.asyncio,
]


class FakeRoutedItem:
    def __init__(self, data: dict[str, Any]):
        self._data = data

    def data(self) -> dict[str, Any]:
        return self._data


class SequenceRoutedEngine:
    def __init__(self, streams: list[Any]):
        self._streams = list(streams)
        self.requests: list[dict[str, Any]] = []
        self.kwargs: list[dict[str, Any]] = []

    async def generate(self, request: dict[str, Any], **kwargs: Any):
        self.requests.append(request)
        self.kwargs.append(kwargs)
        items = self._streams.pop(0)
        if isinstance(items, DelayedGenerate):
            await asyncio.sleep(items.delay_s)
            items = items.items
        if isinstance(items, Exception):
            raise items

        async def stream():
            for item in items:
                if isinstance(item, Exception):
                    raise item
                yield item

        return stream()


class DelayedGenerate:
    def __init__(self, delay_s: float, items: list[FakeRoutedItem]):
        self.delay_s = delay_s
        self.items = items


class FakeConfig:
    def __init__(
        self,
        routed_engine_adapter: str = "global-router",
        global_router_response_prologue_timeout_s: float | None = 30.0,
    ):
        self.routed_engine_adapter = routed_engine_adapter
        self.global_router_response_prologue_timeout_s = (
            global_router_response_prologue_timeout_s
        )


async def _collect(stream) -> list[dict[str, Any]]:
    return [item.data() async for item in stream]


async def test_global_router_adapter_reissues_request_on_retry_control():
    control = make_global_router_retry_control(
        request_type="prefill",
        retry_attempt=0,
        next_retry_attempt=1,
        failed_pool=2,
        failed_namespace="prefill-slow",
        next_pool=1,
        next_namespace="prefill-mid",
        error="dispatch failed",
    )
    routed_engine = SequenceRoutedEngine(
        streams=[
            [FakeRoutedItem(control)],
            [FakeRoutedItem({"token_ids": [101], "index": 0})],
        ]
    )
    adapter = GlobalRouterRoutedEngineAdapter(routed_engine)
    request = {"token_ids": [1, 2, 3], "routing": {"priority": 7}}

    stream = await adapter.generate(request, request_id="request-123")
    outputs = await _collect(stream)

    assert outputs == [{"token_ids": [101], "index": 0}]
    assert routed_engine.requests[0]["routing"] == {
        "priority": 7,
        GLOBAL_ROUTER_RETRY_ATTEMPT_KEY: 0,
    }
    assert routed_engine.requests[1]["routing"] == {
        "priority": 7,
        GLOBAL_ROUTER_RETRY_ATTEMPT_KEY: 1,
    }
    assert request["routing"] == {"priority": 7}
    assert routed_engine.kwargs == [
        {"request_id": "request-123"},
        {"request_id": "request-123"},
    ]


async def test_global_router_adapter_does_not_yield_retry_control():
    control = make_global_router_retry_control(
        request_type="agg",
        retry_attempt=0,
        next_retry_attempt=1,
        failed_pool=0,
        failed_namespace="agg-slow",
        next_pool=1,
        next_namespace="agg-fast",
        error="dispatch failed",
    )
    routed_engine = SequenceRoutedEngine(
        streams=[
            [FakeRoutedItem(control)],
            [FakeRoutedItem({"done": True})],
        ]
    )
    adapter = GlobalRouterRoutedEngineAdapter(routed_engine)

    stream = await adapter.generate({"token_ids": [1]})
    outputs = await _collect(stream)

    assert outputs == [{"done": True}]


async def test_global_router_adapter_rejects_retry_control_after_output():
    control = make_global_router_retry_control(
        request_type="prefill",
        retry_attempt=0,
        next_retry_attempt=1,
        failed_pool=2,
        failed_namespace="prefill-slow",
        next_pool=1,
        next_namespace="prefill-mid",
        error="dispatch failed",
    )
    routed_engine = SequenceRoutedEngine(
        streams=[
            [
                FakeRoutedItem({"token_ids": [101], "index": 0}),
                FakeRoutedItem(control),
            ],
        ]
    )
    adapter = GlobalRouterRoutedEngineAdapter(routed_engine)

    stream = await adapter.generate({"token_ids": [1]})
    with pytest.raises(RuntimeError, match="after response streaming started"):
        await _collect(stream)


async def test_global_router_adapter_does_not_retry_generate_failure_before_output():
    routed_engine = SequenceRoutedEngine(
        streams=[
            RuntimeError("response prologue failed"),
            [FakeRoutedItem({"done": True})],
        ]
    )
    adapter = GlobalRouterRoutedEngineAdapter(routed_engine)

    stream = await adapter.generate({"token_ids": [1]})
    with pytest.raises(RuntimeError, match="response prologue failed"):
        await _collect(stream)

    assert [
        request["routing"][GLOBAL_ROUTER_RETRY_ATTEMPT_KEY]
        for request in routed_engine.requests
    ] == [0]


async def test_global_router_adapter_retries_generate_timeout_before_output():
    routed_engine = SequenceRoutedEngine(
        streams=[
            DelayedGenerate(0.05, [FakeRoutedItem({"late": True})]),
            [FakeRoutedItem({"done": True})],
        ]
    )
    adapter = GlobalRouterRoutedEngineAdapter(
        routed_engine,
        response_prologue_timeout_s=0.001,
    )

    stream = await adapter.generate({"token_ids": [1]})
    outputs = await _collect(stream)

    assert outputs == [{"done": True}]
    assert [
        request["routing"][GLOBAL_ROUTER_RETRY_ATTEMPT_KEY]
        for request in routed_engine.requests
    ] == [0, 1]


async def test_wrap_global_router_adapter_passes_prologue_timeout():
    routed_engine = SequenceRoutedEngine(streams=[])

    adapter = wrap_routed_engine(
        FakeConfig(global_router_response_prologue_timeout_s=12.5),
        routed_engine,
    )

    assert isinstance(adapter, GlobalRouterRoutedEngineAdapter)
    assert adapter._response_prologue_timeout_s == 12.5


async def test_global_router_adapter_retries_stream_failure_before_output():
    routed_engine = SequenceRoutedEngine(
        streams=[
            [RuntimeError("stream failed before output")],
            [FakeRoutedItem({"done": True})],
        ]
    )
    adapter = GlobalRouterRoutedEngineAdapter(routed_engine)

    stream = await adapter.generate({"token_ids": [1]})
    outputs = await _collect(stream)

    assert outputs == [{"done": True}]
    assert [
        request["routing"][GLOBAL_ROUTER_RETRY_ATTEMPT_KEY]
        for request in routed_engine.requests
    ] == [0, 1]


async def test_global_router_adapter_does_not_retry_stream_failure_after_output():
    routed_engine = SequenceRoutedEngine(
        streams=[
            [
                FakeRoutedItem({"token_ids": [101], "index": 0}),
                RuntimeError("stream failed after output"),
            ],
            [FakeRoutedItem({"done": True})],
        ]
    )
    adapter = GlobalRouterRoutedEngineAdapter(routed_engine)

    stream = await adapter.generate({"token_ids": [1]})
    with pytest.raises(RuntimeError, match="stream failed after output"):
        await _collect(stream)

    assert len(routed_engine.requests) == 1


async def test_global_router_adapter_raises_retry_exhausted_control():
    control = make_global_router_retry_exhausted_control(
        request_type="prefill",
        retry_attempt=3,
        failed_pool=0,
        failed_namespace="prefill-fast",
        error="no priority retry pools remain",
    )
    routed_engine = SequenceRoutedEngine(streams=[[FakeRoutedItem(control)]])
    adapter = GlobalRouterRoutedEngineAdapter(routed_engine)

    stream = await adapter.generate({"token_ids": [1]})
    with pytest.raises(RuntimeError, match="no priority retry pools remain"):
        await _collect(stream)

    assert len(routed_engine.requests) == 1
