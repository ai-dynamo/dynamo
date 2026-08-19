# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from examples.custom_backend.workflow_hello_world.dynamo_bespoke.engine import (
    AggregatedHelloWorldEngine,
)

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]


async def test_aggregated_worker_runs_three_stages_inline() -> None:
    request: dict[str, Any] = {"token_ids": [1]}
    chunks = [
        chunk
        async for chunk in AggregatedHelloWorldEngine().generate(request, object())
    ]

    assert chunks == [
        {
            "token_ids": [],
            "text": "Hello, World!",
            "index": 0,
            "finish_reason": "stop",
        }
    ]
