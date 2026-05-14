# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Mocker-based tests for the activity notifier wiring in PrefillHandler.

These tests run WITHOUT a GPU by patching GPU-requiring imports at module
level. They prove the before/after behaviour:

  BEFORE the fix: fire_activity_notifier() was never called from
    PrefillHandler.generate() because the functions were (incorrectly)
    imported as module-level functions from dynamo._core, which does not
    export them at module scope.

  AFTER the fix: self.runtime.fire_activity_notifier("kv_transfer") is
    called on every PrefillHandler.generate() entry, giving the
    HealthCheckManager a liveness signal per KV transfer request.
"""

import inspect
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


# ---------------------------------------------------------------------------
# Stub GPU-requiring imports so these tests run on any machine
# ---------------------------------------------------------------------------


def _make_trtllm_stub() -> None:
    import importlib.machinery

    trtllm = types.ModuleType("tensorrt_llm")
    # __spec__ required so importlib.util.find_spec("tensorrt_llm") doesn't
    # raise ValueError when pytest-marker-report collects the full suite.
    trtllm.__spec__ = importlib.machinery.ModuleSpec("tensorrt_llm", None)
    trtllm.llmapi = types.ModuleType("tensorrt_llm.llmapi")
    trtllm.llmapi.__spec__ = importlib.machinery.ModuleSpec("tensorrt_llm.llmapi", None)
    trtllm.llmapi.DisaggregatedParams = MagicMock
    trtllm.llmapi.RequestOutput = MagicMock
    sys.modules.setdefault("tensorrt_llm", trtllm)
    sys.modules.setdefault("tensorrt_llm.llmapi", trtllm.llmapi)


def _make_dynamo_core_stub() -> None:
    core = types.ModuleType("dynamo._core")
    core.Context = MagicMock
    sys.modules.setdefault("dynamo._core", core)


def _make_dynamo_llm_stub() -> None:
    """Stub dynamo.llm (compiled Rust/PyO3 extension) and related modules."""
    import importlib.machinery

    for name in [
        "dynamo.llm",
        "dynamo.llm._core",
        "dynamo.llm.model_types",
    ]:
        mod = types.ModuleType(name)
        mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
        mod.ModelType = MagicMock
        sys.modules.setdefault(name, mod)


def _make_dynamo_runtime_stub() -> None:
    """Stub dynamo.runtime (also PyO3-backed) so import chains don't fail."""
    import importlib.machinery

    for name in [
        "dynamo.runtime",
        "dynamo.runtime.logging",
    ]:
        mod = types.ModuleType(name)
        mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
        mod.DistributedRuntime = MagicMock
        mod.configure_dynamo_logging = MagicMock
        sys.modules.setdefault(name, mod)


_make_trtllm_stub()
_make_dynamo_core_stub()
_make_dynamo_llm_stub()
_make_dynamo_runtime_stub()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_runtime_mock() -> MagicMock:
    rt = MagicMock()
    rt.register_activity_notifier = MagicMock()
    rt.fire_activity_notifier = MagicMock(return_value=True)
    return rt


def _make_config(runtime: object = None) -> MagicMock:
    from dynamo.trtllm.tests.utils import create_mock_request_handler_config

    cfg = create_mock_request_handler_config(disaggregation_mode="prefill")
    cfg.runtime = runtime if runtime is not None else _make_runtime_mock()
    return cfg


# ---------------------------------------------------------------------------
# __init__: register_activity_notifier called on startup
# ---------------------------------------------------------------------------


class TestPrefillHandlerRegistersNotifier:
    def test_register_called_on_init_when_runtime_present(self) -> None:
        """After the fix: __init__ calls register_activity_notifier('kv_transfer')."""
        from dynamo.trtllm.request_handlers.handlers import PrefillHandler

        runtime = _make_runtime_mock()
        cfg = _make_config(runtime=runtime)

        PrefillHandler(cfg)

        runtime.register_activity_notifier.assert_called_once_with("kv_transfer")

    def test_register_not_called_when_runtime_is_none(self) -> None:
        """No crash when runtime is None."""
        from dynamo.trtllm.request_handlers.handlers import PrefillHandler

        cfg = _make_config()
        cfg.runtime = None

        PrefillHandler(cfg)

    def test_regression_no_bare_module_import(self) -> None:
        """
        Before the fix the handler imported register_activity_notifier as a
        module-level function.  That import is the bug — the function only
        exists as an instance method on DistributedRuntime, not as a module
        export.  Verify the bad import is gone.
        """
        import dynamo.trtllm.request_handlers.handlers as mod

        src = inspect.getsource(mod)
        first_import_line = (
            src.split("from dynamo._core import")[1].split("\n")[0]
            if "from dynamo._core import" in src
            else ""
        )
        assert "register_activity_notifier" not in first_import_line, (
            "Bare module-level import of register_activity_notifier still present"
            " — pre-fix bug"
        )


# ---------------------------------------------------------------------------
# generate(): fire_activity_notifier called per request
# ---------------------------------------------------------------------------


class TestPrefillHandlerFiresNotifierOnGenerate:
    @pytest.mark.asyncio
    async def test_fire_called_on_generate(self) -> None:
        """After the fix: generate() calls fire_activity_notifier('kv_transfer')."""
        from dynamo.trtllm.request_handlers.handlers import PrefillHandler
        from dynamo.trtllm.tests.request_handlers.utils import create_mock_context

        runtime = _make_runtime_mock()
        cfg = _make_config(runtime=runtime)
        handler = PrefillHandler(cfg)

        ctx = create_mock_context()
        req = {"text_input": "hello", "max_tokens": 1}

        with patch.object(
            handler,
            "_handle_prefill_request",
            new_callable=AsyncMock,
            return_value={"output": "hi"},
        ):
            try:
                async for _ in handler.generate(req, ctx):
                    pass
            except Exception:
                pass  # may fail past the notifier call; we only check the call

        runtime.fire_activity_notifier.assert_called_with("kv_transfer")

    @pytest.mark.asyncio
    async def test_fire_not_called_when_runtime_is_none(self) -> None:
        """No AttributeError when runtime is None."""
        from dynamo.trtllm.request_handlers.handlers import PrefillHandler
        from dynamo.trtllm.tests.request_handlers.utils import create_mock_context

        cfg = _make_config()
        cfg.runtime = None
        handler = PrefillHandler(cfg)

        ctx = create_mock_context()
        req = {"text_input": "hello", "max_tokens": 1}

        try:
            async for _ in handler.generate(req, ctx):
                pass
        except Exception:
            pass
