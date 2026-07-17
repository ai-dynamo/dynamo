# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dynamo.planner import __main__ as planner_main
from dynamo.planner.config.planner_config import PlannerConfig

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


@pytest.mark.asyncio
async def test_start_planner_constructs_and_initializes_planner():
    runtime = MagicMock()
    runtime.endpoint.return_value.serve_endpoint = AsyncMock()
    config = PlannerConfig(namespace="test-ns")

    planner = MagicMock()
    planner._async_init = AsyncMock()
    planner.run = AsyncMock()

    with patch.object(
        planner_main, "construct_planner", return_value=planner
    ) as mock_construct_planner:
        await planner_main.start_planner(runtime, config)

    mock_construct_planner.assert_called_once_with(runtime=runtime, config=config)
    planner._async_init.assert_awaited_once_with()
    planner.run.assert_awaited_once_with()
    runtime.endpoint.assert_called_once_with("test-ns.Planner.generate")
    runtime.endpoint.return_value.serve_endpoint.assert_awaited_once()
