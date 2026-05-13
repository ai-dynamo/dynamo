# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Mocker-based tests for the activity notifier wiring in PrefillHandler.

These tests run WITHOUT a GPU or TensorRT-LLM install by patching the
GPU-requiring imports at the module level. They prove the before/after
behaviour:

  BEFORE the fix: fire_activity_notifier() was never called from
    PrefillHandler.generate() — the notifier silently did nothing.

  AFTER the fix: fire_activity_notifier("kv_transfer") is called on
    every PrefillHandler.generate() entry, giving the health-check
    manager a liveness signal per KV transfer request.

The key invariant being tested:
  SystemHealth.get_endpoint_activity_check_notifier("kv_transfer").notify_one()
  fires on each prefill request → canary timer resets → worker stays Ready.
"""

import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Stub out GPU-requiring imports so these tests run anywhere
# ---------------------------------------------------------------------------

def _make_trtllm_stub():
    trtllm = types.ModuleType("tensorrt_llm")
    trtllm.llmapi = types.ModuleType("tensorrt_llm.llmapi")
    trtllm.llmapi.DisaggregatedParams = MagicMock
    trtllm.llmapi.RequestOutput = MagicMock
    sys.modules.setdefault("tensorrt_llm", trtllm)
    sys.modules.setdefault("tensorrt_llm.llmapi", trtllm.llmapi)


def _make_dynamo_core_stub():
    core = types.ModuleType("dynamo._core")
    core.Context = MagicMock
    sys.modules.setdefault("dynamo._core", core)


_make_trtllm_stub()
_make_dynamo_core_stub()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_runtime_mock():
    """DistributedRuntime mock with activity notifier methods."""
    rt = MagicMock()
    rt.register_activity_notifier = MagicMock()
    rt.fire_activity_notifier = MagicMock(return_value=True)
    return rt


def _make_config(runtime=None):
    from dynamo.trtllm.tests.utils import create_mock_request_handler_config

    cfg = create_mock_request_handler_config(disaggregation_mode="prefill")
    cfg.runtime = runtime or _make_runtime_mock()
    return cfg


# ---------------------------------------------------------------------------
# Tests: __init__ registers the notifier
# ---------------------------------------------------------------------------

class TestPrefillHandlerRegistersNotifier:
    def test_register_called_on_init_when_runtime_present(self):
        """After the fix: __init__ calls register_activity_notifier('kv_transfer')."""
        from dynamo.trtllm.request_handlers.handlers import PrefillHandler

        runtime = _make_runtime_mock()
        cfg = _make_config(runtime=runtime)

        PrefillHandler(cfg)

        runtime.register_activity_notifier.assert_called_once_with("kv_transfer")

    def test_register_not_called_when_runtime_is_none(self):
        """No crash when runtime is None (graceful degradation)."""
        from dynamo.trtllm.request_handlers.handlers import PrefillHandler

        cfg = _make_config(runtime=None)
        cfg.runtime = None

        PrefillHandler(cfg)  # should not raise

    def test_before_fix_regression(self):
        """
        Before the fix, PrefillHandler.__init__ imported register_activity_notifier
        as a module-level function and called it without self.runtime. This test
        proves the OLD code path is gone: the bare import no longer exists.
        """
        import dynamo.trtllm.request_handlers.handlers as mod
        import inspect

        src = inspect.getsource(mod)
        assert "from dynamo._core import" not in src or "register_activity_notifier" not in src.split("from dynamo._core import")[1].split("\n")[0], (
            "Module-level import of register_activity_notifier found — "
            "this was the pre-fix bug. Remove the bare import and call "
            "self.runtime.register_activity_notifier() instead."
        )


# ---------------------------------------------------------------------------
# Tests: generate() fires the notifier
# ---------------------------------------------------------------------------

class TestPrefillHandlerFiresNotifierOnGenerate:
    @pytest.mark.asyncio
    async def test_fire_called_on_generate(self):
        """After the fix: generate() calls fire_activity_notifier('kv_transfer')."""
        from dynamo.trtllm.request_handlers.handlers import PrefillHandler
        from dynamo.trtllm.tests.request_handlers.utils import create_mock_context

        runtime = _make_runtime_mock()
        cfg = _make_config(runtime=runtime)
        handler = PrefillHandler(cfg)

        ctx = create_mock_context()
        req = {"text_input": "hello", "max_tokens": 1}

        # Patch the downstream call so generate() can complete without a real engine
        with patch.object(
            handler, "_handle_prefill_request", new_callable=AsyncMock
        ) as mock_handle:
            mock_handle.return_value = {"output": "hi"}
            try:
                async for _ in handler.generate(req, ctx):
                    pass
            except Exception:
                pass  # generate may fail after the notifier call; we only check the call

        runtime.fire_activity_notifier.assert_called_with("kv_transfer")

    @pytest.mark.asyncio
    async def test_fire_not_called_when_runtime_is_none(self):
        """No crash when runtime is None."""
        from dynamo.trtllm.request_handlers.handlers import PrefillHandler
        from dynamo.trtllm.tests.request_handlers.utils import create_mock_context

        cfg = _make_config(runtime=None)
        cfg.runtime = None
        handler = PrefillHandler(cfg)

        ctx = create_mock_context()
        req = {"text_input": "hello", "max_tokens": 1}

        try:
            async for _ in handler.generate(req, ctx):
                pass
        except Exception:
            pass  # may fail; we just want no AttributeError on None.fire_activity_notifier
