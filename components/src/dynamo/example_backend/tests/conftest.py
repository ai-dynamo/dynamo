# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conftest that patches missing dynamo._core symbols before test collection.

The installed dynamo._core binary may not export all symbols needed by the
source-tree modules (e.g. Context, ModelType, parser name helpers).  We inject
lightweight stubs so the handler modules can be imported without rebuilding
the full Rust runtime.
"""

import asyncio
from enum import IntFlag
from unittest.mock import MagicMock

import dynamo._core as _core

# --- Context stub --------------------------------------------------------
if not hasattr(_core, "Context"):

    class _StubContext:
        """Minimal stub matching the Context interface for unit tests."""

        def __init__(self, id=None):
            self._id = id
            self._stopped = False
            self._killed = False
            self.trace_id = None
            self.span_id = None
            self.parent_span_id = None

        def id(self):
            return self._id

        def is_stopped(self):
            return self._stopped

        def is_killed(self):
            return self._killed

        def stop_generating(self):
            self._stopped = True

        async def async_killed_or_stopped(self):
            while not (self._stopped or self._killed):
                await asyncio.sleep(0.01)
            return True

    _core.Context = _StubContext

# --- ModelType stub (IntFlag used in endpoint_types) ---------------------
if not hasattr(_core, "ModelType"):

    class _ModelType(IntFlag):
        Chat = 1
        Completions = 2

    _core.ModelType = _ModelType

# --- Callable stubs ------------------------------------------------------
if not hasattr(_core, "get_reasoning_parser_names"):
    _core.get_reasoning_parser_names = lambda: []

if not hasattr(_core, "get_tool_parser_names"):
    _core.get_tool_parser_names = lambda: []

# --- Remaining symbols used by dynamo.llm and dynamo.backend.base --------
_MOCK_NAMES = [
    "ModelInput",
    "ModelRuntimeConfig",
    "register_llm",
    "unregister_llm",
    "EngineType",
    "EntrypointArgs",
    "KserveGrpcService",
    "KvPushRouter",
    "KvRouterConfig",
    "LoRADownloader",
    "MediaDecoder",
    "MediaFetcher",
    "ModelCardInstanceId",
    "RadixTree",
    "RouterConfig",
    "RouterMode",
    "WorkerMetricsPublisher",
    "ZmqKvEventListener",
    "ZmqKvEventPublisherConfig",
    "compute_block_hash_for_seq_py",
    "fetch_llm",
    "lora_name_to_id",
    "make_engine",
    "run_input",
]
for _name in _MOCK_NAMES:
    if not hasattr(_core, _name):
        setattr(_core, _name, MagicMock())
