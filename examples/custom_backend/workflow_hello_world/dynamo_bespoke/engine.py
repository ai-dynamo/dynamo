# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User-owned orchestration for the aggregated worker."""

import asyncio
from collections.abc import AsyncIterator, Mapping
from typing import Any

from examples.custom_backend.workflow_hello_world.common.stages import (
    HelloStage,
    MergeStage,
    WorldStage,
)


class AggregatedHelloWorldEngine:
    """Own the workflow control flow inside one model worker."""

    async def generate(
        self, request: Mapping[str, Any], context: Any
    ) -> AsyncIterator[dict[str, Any]]:
        del context
        hello, world = await asyncio.gather(
            HelloStage().run(request),
            WorldStage().run(request),
        )
        text = await MergeStage().run(hello, world)
        yield {
            "token_ids": [],
            "text": text,
            "index": 0,
            "finish_reason": "stop",
        }
