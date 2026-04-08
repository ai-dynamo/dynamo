# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncGenerator

from dynamo._core import Context
from dynamo.common.engine_utils import build_completion_usage

from .engine import DynamoEngine, EngineConfig


class SampleDynamoEngine(DynamoEngine):
    """Reference DynamoEngine implementation.

    Generates rotating token IDs with configurable per-token latency.
    Useful for testing the DynamoPythonBackendModel lifecycle end-to-end
    and as a template for engine leads implementing real backends.
    """

    def __init__(
        self,
        model_name: str = "sample-model",
        max_tokens: int = 16,
        delay: float = 0.01,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.delay = delay

    async def init(self) -> EngineConfig:
        return EngineConfig(
            model=self.model_name,
            served_model_name=self.model_name,
            context_length=2048,
            kv_cache_block_size=16,
            total_kv_blocks=1000,
            max_num_seqs=64,
            max_num_batched_tokens=2048,
        )

    async def generate(
        self, request: dict, context: Context
    ) -> AsyncGenerator[dict, None]:
        token_ids = request.get("token_ids", [])
        prompt_len = len(token_ids)
        stop_conditions = request.get("stop_conditions", {})
        max_new = stop_conditions.get("max_tokens") or self.max_tokens

        for i in range(max_new):
            if context.is_stopped():
                break
            await asyncio.sleep(self.delay)
            token_id = (i + 1) % 32000
            out: dict = {"token_ids": [token_id]}
            if i == max_new - 1:
                out["finish_reason"] = "length"
                out["completion_usage"] = build_completion_usage(prompt_len, max_new)
            yield out

    async def cleanup(self) -> None:
        pass
