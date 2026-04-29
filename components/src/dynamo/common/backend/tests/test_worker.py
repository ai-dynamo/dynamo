# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Worker._cleanup_once.

Locks down the invariant that engine.cleanup() runs exactly once even when
both shutdown paths fire (graceful-shutdown signal handler and run()'s
finally block).

These tests stub the dynamo native imports so they work without GPU or the
PyO3 extension installed.
"""

import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


# ---------------------------------------------------------------------------
# Module loading: import worker.py without triggering the full dynamo package
# (which requires dynamo._core, dynamo.llm, CUDA, etc.).
# ---------------------------------------------------------------------------

_BACKEND_DIR = Path(__file__).parent.parent
_WORKER_PATH = _BACKEND_DIR / "worker.py"
_ENGINE_PATH = _BACKEND_DIR / "engine.py"
_GRACEFUL_SHUTDOWN_PATH = _BACKEND_DIR.parent / "utils" / "graceful_shutdown.py"


def _install_stubs() -> None:
    dynamo = types.ModuleType("dynamo")
    sys.modules.setdefault("dynamo", dynamo)

    core = types.ModuleType("dynamo._core")
    core.Context = object
    core.DistributedRuntime = object
    sys.modules.setdefault("dynamo._core", core)

    common = types.ModuleType("dynamo.common")
    sys.modules.setdefault("dynamo.common", common)
    common_utils = types.ModuleType("dynamo.common.utils")
    sys.modules.setdefault("dynamo.common.utils", common_utils)

    endpoint_types = types.ModuleType("dynamo.common.utils.endpoint_types")
    endpoint_types.parse_endpoint_types = lambda _s: []
    sys.modules.setdefault("dynamo.common.utils.endpoint_types", endpoint_types)

    runtime_mod = types.ModuleType("dynamo.common.utils.runtime")
    runtime_mod.create_runtime = lambda **_kw: (None, None)
    sys.modules.setdefault("dynamo.common.utils.runtime", runtime_mod)

    llm = types.ModuleType("dynamo.llm")
    llm.ModelInput = types.SimpleNamespace(Tokens=object())
    llm.ModelRuntimeConfig = type("ModelRuntimeConfig", (), {})
    llm.register_model = AsyncMock()
    sys.modules.setdefault("dynamo.llm", llm)

    llm_exceptions = types.ModuleType("dynamo.llm.exceptions")

    class _DynamoException(Exception):
        pass

    llm_exceptions.DynamoException = _DynamoException
    llm_exceptions.CannotConnect = type("CannotConnect", (_DynamoException,), {})
    llm_exceptions.EngineShutdown = type("EngineShutdown", (_DynamoException,), {})
    llm_exceptions.Unknown = type("Unknown", (_DynamoException,), {})
    sys.modules.setdefault("dynamo.llm.exceptions", llm_exceptions)

    runtime_pkg = types.ModuleType("dynamo.runtime")
    sys.modules.setdefault("dynamo.runtime", runtime_pkg)
    runtime_logging = types.ModuleType("dynamo.runtime.logging")
    runtime_logging.configure_dynamo_logging = lambda: None
    sys.modules.setdefault("dynamo.runtime.logging", runtime_logging)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_install_stubs()
# Load real graceful_shutdown so install_signal_handlers is the real function;
# Worker._cleanup_once doesn't use it but worker.py imports it at module load.
_load_module("dynamo.common.utils.graceful_shutdown", _GRACEFUL_SHUTDOWN_PATH)
# engine.py is imported by worker.py via relative import; load it under the
# package path Worker expects.
_engine_pkg = types.ModuleType("dynamo.common.backend")
_engine_pkg.__path__ = [str(_BACKEND_DIR)]
sys.modules.setdefault("dynamo.common.backend", _engine_pkg)
_load_module("dynamo.common.backend.engine", _ENGINE_PATH)
_worker = _load_module("dynamo.common.backend.worker", _WORKER_PATH)

Worker = _worker.Worker
WorkerConfig = _worker.WorkerConfig


def _make_worker() -> tuple:
    engine = AsyncMock()
    engine.cleanup = AsyncMock()
    config = WorkerConfig(namespace="test")
    return Worker(engine=engine, config=config), engine


def test_cleanup_once_calls_engine_cleanup_exactly_once():
    """Both shutdown paths racing must not double-cleanup the engine.

    The signal-handler path and run()'s finally block both call _cleanup_once.
    Engines like vLLM and TRT-LLM tear down NCCL process groups in cleanup();
    calling it twice can hang or raise.
    """
    worker, engine = _make_worker()

    async def _run():
        await worker._cleanup_once()
        await worker._cleanup_once()
        await worker._cleanup_once()

    asyncio.run(_run())
    assert engine.cleanup.await_count == 1


def test_cleanup_once_concurrent_invocations_only_run_once():
    """Concurrent _cleanup_once invocations must coalesce AND serialize.

    Mirrors the real race: the signal-handler task awaits _cleanup_once while
    run()'s finally block awaits it. The second caller must wait for the
    first's engine.cleanup() to finish before returning — otherwise the
    signal handler can call runtime.shutdown() while the finally-block's
    cleanup is still mid-flight, collapsing the loop under it.
    """
    worker, engine = _make_worker()

    async def _run():
        started = asyncio.Event()
        release = asyncio.Event()

        async def slow_cleanup():
            started.set()
            await release.wait()

        engine.cleanup.side_effect = slow_cleanup

        first = asyncio.create_task(worker._cleanup_once())
        await started.wait()
        # First caller is now suspended inside engine.cleanup(). A second
        # caller arriving here must NOT return until the first finishes.
        second = asyncio.create_task(worker._cleanup_once())
        # Yield enough times that a flag-only short-circuit would let `second`
        # complete while `first` is still inside engine.cleanup().
        for _ in range(10):
            await asyncio.sleep(0)
        assert not second.done(), (
            "second _cleanup_once returned while the first was still inside "
            "engine.cleanup() — late callers must wait for the in-flight cleanup"
        )

        release.set()
        await asyncio.gather(first, second)

    asyncio.run(_run())
    assert engine.cleanup.await_count == 1


def test_cleanup_once_propagates_exception_but_marks_done():
    """If engine.cleanup() raises, _cleanup_done is still set so a follow-up
    invocation from the other shutdown path is a no-op rather than a retry.
    """
    worker, engine = _make_worker()
    engine.cleanup.side_effect = RuntimeError("boom")

    async def _run():
        with pytest.raises(RuntimeError, match="boom"):
            await worker._cleanup_once()
        # Second call must not re-invoke cleanup, even though the first raised.
        await worker._cleanup_once()

    asyncio.run(_run())
    assert engine.cleanup.await_count == 1
