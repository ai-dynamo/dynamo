# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from dynamo.llm.exceptions import InvalidArgument
from dynamo.sglang.request_handlers.llm.diffusion_handler import DiffusionWorkerHandler

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", [0, 16])
async def test_diffusion_rejects_thinking_budget_before_engine_generation(budget):
    handler = object.__new__(DiffusionWorkerHandler)
    handler.engine = SimpleNamespace(async_generate=AsyncMock())
    request = {
        "stop_conditions": {"max_thinking_tokens": budget},
        "require_reasoning": True,
    }

    with pytest.raises(InvalidArgument, match="diffusion language model"):
        await anext(handler.generate(request, Mock()))

    handler.engine.async_generate.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "salt_fields",
    [
        {},
        {"routing": {"cache_salt": "tenant-a"}},
        {"extra_args": {"nvext": {"cache_salt": "tenant-a"}}},
    ],
    ids=["unsalted", "routing", "extra_args"],
)
async def test_diffusion_without_budget_keeps_existing_generation(salt_fields):
    handler = object.__new__(DiffusionWorkerHandler)
    handler.engine = SimpleNamespace(async_generate=AsyncMock(return_value=object()))
    handler.enable_trace = False
    handler.use_sglang_tokenizer = False
    handler._get_input_param = Mock(return_value={"input_ids": [1]})
    handler._build_sampling_params = Mock(return_value={"max_new_tokens": 8})

    async def stream(generator, context):
        yield {"token_ids": [2]}

    handler._process_token_stream = stream
    request = {"token_ids": [1], **salt_fields}
    output = [chunk async for chunk in handler.generate(request, Mock())]

    assert output == [{"token_ids": [2]}]
    handler.engine.async_generate.assert_awaited_once()
    kwargs = handler.engine.async_generate.await_args.kwargs
    if salt_fields:
        assert kwargs["cache_salt"] == "tenant-a"
    else:
        assert "cache_salt" not in kwargs
