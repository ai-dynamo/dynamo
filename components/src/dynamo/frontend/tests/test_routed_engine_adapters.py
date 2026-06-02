# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for frontend routed-engine topology adapters."""

from typing import Any

import pytest

from dynamo.common.global_router_protocol import (
    GLOBAL_ROUTER_RETRY_ATTEMPT_KEY,
    make_global_router_retry_control,
)
from dynamo.frontend.routed_engine_adapters import GlobalRouterRoutedEngineAdapter

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


class FakeRoutedItem:
    def __init__(self, data: dict[str, Any]):
        self._data = data

    def data(self) -> dict[str, Any]:
        return self._data


class SequenceRoutedEngine:
    def __init__(self, streams: list[list[FakeRoutedItem]]):
        self._streams = list(streams)
        self.requests: list[dict[str, Any]] = []
        self.kwargs: list[dict[str, Any]] = []

    async def generate(self, request: dict[str, Any], **kwargs: Any):
        self.requests.append(request)
        self.kwargs.append(kwargs)
        items = self._streams.pop(0)

        async def stream():
            for item in items:
                yield item

        return stream()


async def _collect(stream) -> list[dict[str, Any]]:
    return [item.data() async for item in stream]


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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
