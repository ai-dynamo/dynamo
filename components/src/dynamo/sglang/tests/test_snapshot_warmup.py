# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dynamo.sglang.snapshot import warmup_engine

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _server_args(**overrides):
    values = {
        "skip_server_warmup": False,
        "skip_tokenizer_init": True,
        "dp_size": 1,
        "tp_size": 1,
        "debug_tensor_dump_input_file": None,
        "disaggregation_mode": "null",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.asyncio
async def test_snapshot_warmup_runs_generation_path():
    engine = SimpleNamespace(
        tokenizer_manager=SimpleNamespace(is_generation=True),
        async_generate=AsyncMock(return_value={}),
    )

    await warmup_engine(engine, _server_args())

    engine.async_generate.assert_awaited_once_with(
        input_ids=[10, 11, 12],
        sampling_params={"temperature": 0, "max_new_tokens": 8},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("skip_server_warmup", "is_generation"),
    [(True, True), (False, False)],
)
async def test_snapshot_warmup_skips_disabled_or_non_generation_models(
    skip_server_warmup, is_generation
):
    engine = SimpleNamespace(
        tokenizer_manager=SimpleNamespace(is_generation=is_generation),
        async_generate=AsyncMock(return_value={}),
    )

    await warmup_engine(
        engine,
        _server_args(skip_server_warmup=skip_server_warmup),
    )

    engine.async_generate.assert_not_awaited()
