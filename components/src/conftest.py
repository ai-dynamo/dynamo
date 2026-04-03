# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Root conftest: patches missing dynamo._core symbols before package imports.

This conftest lives ABOVE the dynamo namespace package so it runs before
any dynamo.* __init__.py is loaded.  Without this, packages like
dynamo.backend (which eagerly imports from dynamo.common → dynamo._core)
fail during collection when the installed _core binary is out of date.
"""

import asyncio
import importlib
import sys
from enum import IntFlag
from types import ModuleType
from unittest.mock import MagicMock

# --- Pre-import: ensure dynamo.prometheus_names exists --------------------
# dynamo.common.utils.prometheus imports from dynamo.prometheus_names which
# is a generated module only present in the built package.  If missing,
# create a stub from the source tree.
if "dynamo.prometheus_names" not in sys.modules:
    try:
        importlib.import_module("dynamo.prometheus_names")
    except (ImportError, ModuleNotFoundError):
        # Try loading from lib/bindings source tree
        import pathlib

        _prom_src = (
            pathlib.Path(__file__).resolve().parents[2]
            / "lib"
            / "bindings"
            / "python"
            / "src"
            / "dynamo"
            / "prometheus_names.py"
        )
        if _prom_src.exists():
            import importlib.util

            _spec = importlib.util.spec_from_file_location(
                "dynamo.prometheus_names", str(_prom_src)
            )
            _mod = importlib.util.module_from_spec(_spec)
            sys.modules["dynamo.prometheus_names"] = _mod
            _spec.loader.exec_module(_mod)
        else:
            # Last resort: empty stub
            _mod = ModuleType("dynamo.prometheus_names")
            for _cls_name in [
                "kvstats",
                "labels",
                "model_info",
                "name_prefix",
                "frontend_service",
                "work_handler",
            ]:
                setattr(_mod, _cls_name, MagicMock())
            sys.modules["dynamo.prometheus_names"] = _mod

import dynamo._core as _core
import dynamo.llm as _llm

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
    # NOTE: ModelType in dynamo._core conflates endpoint types (Chat,
    # Completions), worker roles (Prefill), capability flags (Embedding),
    # and output modalities (Images, Videos, Audios) into a single IntFlag.
    # These should ideally be separate enums. The stub here mirrors
    # what core expects at import time (e.g. output_modalities.py).
    class _ModelType(IntFlag):
        Chat = 1
        Completions = 2
        Prefill = 4
        Embedding = 8
        Images = 16
        Videos = 32
        Audios = 64

    _core.ModelType = _ModelType
    _llm.ModelType = _ModelType

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

# --- Patch dynamo.llm re-exports ------------------------------------------
# dynamo.llm.__init__ re-exports from _core, but the installed package may
# not include newer symbols.  Mirror the patched _core symbols into llm.
_LLM_REEXPORTS = [
    "ModelInput",
    "ModelRuntimeConfig",
    "ModelType",
    "register_llm",
    "unregister_llm",
]
for _name in _LLM_REEXPORTS:
    if not hasattr(_llm, _name) and hasattr(_core, _name):
        setattr(_llm, _name, getattr(_core, _name))
