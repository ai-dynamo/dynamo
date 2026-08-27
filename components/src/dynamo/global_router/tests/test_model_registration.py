#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Tests for GlobalRouter model registration."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dynamo.global_router import __main__ as global_router_main
from dynamo.llm import WorkerType

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.parallel,
    pytest.mark.unit,
]


class FakeEndpoint:
    async def serve_endpoint(self, handler, **kwargs):
        return None


class FakeRuntime:
    def endpoint(self, endpoint_name):
        return FakeEndpoint()


@pytest.mark.asyncio
async def test_disagg_registration_skips_model_weights(monkeypatch):
    register_model = AsyncMock()
    monkeypatch.setattr(global_router_main, "register_model", register_model)
    config = SimpleNamespace(
        model_name="test-model",
        namespace="test-namespace",
        component_name="global-router",
    )
    handler = SimpleNamespace(
        handle_prefill=AsyncMock(),
        handle_decode=AsyncMock(),
    )

    await global_router_main._serve_disagg(FakeRuntime(), config, handler)

    assert register_model.await_count == 2
    prefill_call, decode_call = register_model.await_args_list
    assert prefill_call.kwargs["worker_type"] == WorkerType.Prefill
    assert prefill_call.kwargs["ignore_weights"] is True
    assert decode_call.kwargs["worker_type"] == WorkerType.Decode
    assert decode_call.kwargs["ignore_weights"] is True


@pytest.mark.asyncio
async def test_agg_registration_skips_model_weights(monkeypatch):
    register_model = AsyncMock()
    monkeypatch.setattr(global_router_main, "register_model", register_model)
    config = SimpleNamespace(
        model_name="test-model",
        namespace="test-namespace",
        component_name="global-router",
    )
    handler = SimpleNamespace(handle_generate=AsyncMock())

    await global_router_main._serve_agg(FakeRuntime(), config, handler)

    register_model.assert_awaited_once()
    assert register_model.await_args.kwargs["worker_type"] == WorkerType.Aggregated
    assert register_model.await_args.kwargs["ignore_weights"] is True
