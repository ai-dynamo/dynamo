# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm.config")

from dynamo.vllm.handlers import BaseWorkerHandler  # noqa: E402

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class _TestWorkerHandler(BaseWorkerHandler):
    async def generate(self, request, context):
        yield {}


def _make_handler(tensor_parallel_size: int) -> _TestWorkerHandler:
    handler = _TestWorkerHandler.__new__(_TestWorkerHandler)
    handler.engine_client = SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(
                tensor_parallel_size=tensor_parallel_size,
            ),
        ),
        scale_elastic_ep=AsyncMock(),
    )
    handler._scale_ep_lock = asyncio.Lock()
    handler._scale_ep_in_progress = False
    return handler


@pytest.fixture
def stub_ray(monkeypatch):
    """Stand in for ray, which the scale path imports lazily and CI lacks."""
    state = SimpleNamespace(list_nodes=lambda **kwargs: [])
    util = SimpleNamespace(state=state)
    monkeypatch.setitem(
        sys.modules, "ray", SimpleNamespace(nodes=lambda: [], util=util)
    )
    monkeypatch.setitem(sys.modules, "ray.util", util)
    monkeypatch.setitem(sys.modules, "ray.util.state", state)
    return state


@pytest.mark.asyncio
async def test_missing_size_is_rejected():
    handler = _make_handler(tensor_parallel_size=4)

    result = await handler.scale_elastic_ep({})

    assert result["status"] == "error"
    assert "new_data_parallel_size" in result["message"]
    handler.engine_client.scale_elastic_ep.assert_not_awaited()


@pytest.mark.parametrize("size", [0, -1])
@pytest.mark.asyncio
async def test_sizes_below_one_are_rejected(size):
    handler = _make_handler(tensor_parallel_size=4)

    result = await handler.scale_elastic_ep({"new_data_parallel_size": size})

    assert result["status"] == "error"
    assert "must be >= 1" in result["message"]
    handler.engine_client.scale_elastic_ep.assert_not_awaited()


@pytest.mark.asyncio
async def test_single_rank_expert_group_is_rejected():
    """At TP=1 a target of dp=1 leaves one EP rank, which EPLB does not allow."""
    handler = _make_handler(tensor_parallel_size=1)

    result = await handler.scale_elastic_ep({"new_data_parallel_size": 1})

    assert result["status"] == "error"
    assert "must be > 1" in result["message"]
    handler.engine_client.scale_elastic_ep.assert_not_awaited()


@pytest.mark.asyncio
async def test_dp_one_allowed_when_tensor_parallelism_widens_the_group(stub_ray):
    """One pod per DP rank: TP=4 still leaves four EP ranks at dp=1."""
    handler = _make_handler(tensor_parallel_size=4)

    result = await handler.scale_elastic_ep({"new_data_parallel_size": 1})

    assert result["status"] == "ok"
    assert result["new_data_parallel_size"] == 1
    handler.engine_client.scale_elastic_ep.assert_awaited_once_with(1)


@pytest.mark.asyncio
async def test_dp_two_allowed_at_tp_one(stub_ray):
    handler = _make_handler(tensor_parallel_size=1)

    result = await handler.scale_elastic_ep({"new_data_parallel_size": 2})

    assert result["status"] == "ok"
    handler.engine_client.scale_elastic_ep.assert_awaited_once_with(2)
