# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Handler cleanup stops monitoring without taking ownership of AsyncLLM."""

from unittest.mock import MagicMock

import pytest

from dynamo.vllm.engine_monitor import VllmEngineMonitor
from dynamo.vllm.handlers import (
    DecodeWorkerHandler,
    EmbeddingWorkerHandler,
    PrefillWorkerHandler,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.fault_tolerance,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]


@pytest.mark.parametrize(
    "handler_type", [DecodeWorkerHandler, PrefillWorkerHandler, EmbeddingWorkerHandler]
)
def test_cleanup_stops_monitor_without_shutting_down_engine(handler_type):
    handler = object.__new__(handler_type)
    handler.engine_monitor = MagicMock(spec=VllmEngineMonitor)
    handler.engine_client = MagicMock()
    handler.runtime = MagicMock()
    handler._ep_capacity_executor = None
    handler._custom_encoder = None
    handler.temp_dirs = [MagicMock()]

    handler.cleanup()
    handler.engine_monitor.stop.assert_called_once_with()
    if handler_type is not EmbeddingWorkerHandler:
        handler.temp_dirs[0].cleanup.assert_called_once_with()
    handler.cleanup()

    handler.engine_client.shutdown.assert_not_called()
    handler.runtime.shutdown.assert_not_called()
