# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from pathlib import Path

import pytest

from dynamo.llm import EngineType, EntrypointArgs, PythonAsyncEngine

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.core,
]


async def _generate(request, context):
    yield {"token_ids": [1], "index": 0, "finish_reason": "stop"}


async def test_in_process_tokens_requires_typed_python_engine() -> None:
    engine = PythonAsyncEngine(_generate, asyncio.get_running_loop())

    EntrypointArgs(
        EngineType.InProcessTokens,
        model_path="org/model",
        custom_template_path=Path("templates/vision.jinja"),
        in_process_token_engine=engine,
    )

    with pytest.raises(ValueError, match="requires in_process_token_engine"):
        EntrypointArgs(EngineType.InProcessTokens, model_path="org/model")
    with pytest.raises(ValueError, match="requires EngineType.InProcessTokens"):
        EntrypointArgs(
            EngineType.Dynamic,
            model_path="org/model",
            in_process_token_engine=engine,
        )
