# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``Worker.generate`` and ``WorkerConfig.from_runtime_config``.

Only tests that exercise actual logic — chunk forwarding, cancellation
monitor, exception wrapping, monitor-task cleanup, getattr-fallback logic in
``from_runtime_config``.  Pure dataclass defaults are not tested.
"""

import asyncio
from dataclasses import dataclass
from typing import Optional, cast

import pytest

from dynamo._core import Context
from dynamo.common.backend import LLMEngine, Worker, WorkerConfig
from dynamo.llm.exceptions import DynamoException, Unknown

# Framework-agnostic; routed to sample-unified-test via `pre_merge and gpu_0
# and unified`.  See test_engine.py for the rationale.
pytestmark = [
    pytest.mark.unified,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class FakeContext:
    """Duck-typed stand-in for ``dynamo._core.Context``.

    Worker only calls three methods on the context — ``id()``,
    ``is_stopped()``, and ``async_killed_or_stopped()`` — so a plain Python
    class with the same surface works at runtime.  Keeps these tests off the
    Rust binding so they stay fast and deterministic.
    """

    def __init__(self, request_id: str = "req-1") -> None:
        self._id = request_id
        self._stopped = asyncio.Event()
        self._killed_or_stopped = asyncio.Event()

    @property
    def trace_id(self) -> Optional[str]:
        return self._id

    def id(self) -> str:
        return self._id

    def is_stopped(self) -> bool:
        return self._stopped.is_set()

    def mark_stopped(self) -> None:
        self._stopped.set()
        self._killed_or_stopped.set()

    def mark_killed(self) -> None:
        self._killed_or_stopped.set()

    async def async_killed_or_stopped(self) -> None:
        await self._killed_or_stopped.wait()


def _ctx(fake: FakeContext) -> Context:
    """Cast the duck-typed FakeContext to Context for the type checker.  At
    runtime Worker only calls methods on the context; it does not type-check
    it."""
    return cast(Context, fake)


class _RecordingEngine(LLMEngine):
    """Programmable LLMEngine for Worker tests.  ``chunks_factory`` supplies
    the per-test generate body inline so each test describes behavior without
    a new subclass."""

    def __init__(self, chunks_factory, abort_raises: Optional[Exception] = None):
        self._chunks_factory = chunks_factory
        self._abort_raises = abort_raises
        self.abort_calls: list = []

    @classmethod
    async def from_args(cls, argv=None):  # type: ignore[override]
        raise NotImplementedError  # not exercised

    async def start(self):
        raise NotImplementedError  # not exercised

    async def generate(self, request, context):  # type: ignore[override]
        async for chunk in self._chunks_factory(request, context):
            yield chunk

    async def abort(self, context) -> None:
        self.abort_calls.append(context)
        if self._abort_raises is not None:
            raise self._abort_raises

    async def cleanup(self) -> None:
        pass


def _make_worker(engine: LLMEngine) -> Worker:
    return Worker(engine, WorkerConfig(namespace="test-ns"))


async def _collect(async_gen):
    return [chunk async for chunk in async_gen]


# ---------------------------------------------------------------------------
# Worker.generate logic
# ---------------------------------------------------------------------------


async def test_worker_forwards_chunks_and_skips_abort_on_clean_completion():
    """Happy path: every chunk from the engine is yielded verbatim, and the
    cancellation monitor never fires ``engine.abort``."""

    async def chunks(_request, _context):
        yield {"token_ids": [1]}
        yield {"token_ids": [2]}
        yield {
            "token_ids": [3],
            "finish_reason": "stop",
            "completion_usage": {
                "prompt_tokens": 5,
                "completion_tokens": 3,
                "total_tokens": 8,
            },
        }

    engine = _RecordingEngine(chunks)
    worker = _make_worker(engine)

    out = await _collect(worker.generate({"token_ids": []}, _ctx(FakeContext())))
    assert [c["token_ids"] for c in out] == [[1], [2], [3]]
    assert engine.abort_calls == []


async def test_worker_breaks_when_context_stopped_mid_stream():
    """Once ``context.is_stopped()`` flips mid-stream, remaining engine
    chunks are not yielded to the caller."""

    async def chunks(_request, context):
        yield {"token_ids": [10]}
        context.mark_stopped()
        yield {"token_ids": [11]}  # must not reach caller
        yield {"token_ids": [12]}

    worker = _make_worker(_RecordingEngine(chunks))
    out = await _collect(worker.generate({"token_ids": []}, _ctx(FakeContext())))
    assert [c["token_ids"] for c in out] == [[10]]


async def test_worker_passes_through_dynamo_exception():
    class _MyDynamoErr(DynamoException):
        pass

    async def chunks(_request, _context):
        yield {"token_ids": [1]}
        raise _MyDynamoErr("deliberate")

    worker = _make_worker(_RecordingEngine(chunks))
    with pytest.raises(_MyDynamoErr):
        await _collect(worker.generate({"token_ids": []}, _ctx(FakeContext())))


async def test_worker_wraps_non_dynamo_exception_as_unknown():
    async def chunks(_request, _context):
        yield {"token_ids": [1]}
        raise ValueError("oops")

    worker = _make_worker(_RecordingEngine(chunks))
    with pytest.raises(Unknown) as excinfo:
        await _collect(worker.generate({"token_ids": []}, _ctx(FakeContext())))
    assert isinstance(excinfo.value.__cause__, ValueError)


async def test_worker_cancel_monitor_calls_engine_abort_on_kill():
    first_chunk = asyncio.Event()

    async def chunks(_request, _context):
        yield {"token_ids": [1]}
        first_chunk.set()
        await asyncio.sleep(10)  # stall until the test cancels the driver
        yield {"token_ids": [2]}

    engine = _RecordingEngine(chunks)
    worker = _make_worker(engine)
    ctx = FakeContext()

    driver = asyncio.create_task(
        _collect(worker.generate({"token_ids": []}, _ctx(ctx)))
    )
    await first_chunk.wait()
    ctx.mark_killed()
    await asyncio.sleep(0.05)  # let the monitor run

    assert len(engine.abort_calls) == 1
    assert engine.abort_calls[0] is ctx

    driver.cancel()
    try:
        await driver
    except asyncio.CancelledError:
        pass


async def test_worker_swallows_abort_errors():
    """If ``engine.abort`` raises, the monitor must not propagate it into
    the generate path."""

    async def chunks(_request, _context):
        yield {"token_ids": [1]}

    engine = _RecordingEngine(chunks, abort_raises=RuntimeError("abort failed"))
    worker = _make_worker(engine)
    ctx = FakeContext()
    ctx.mark_killed()  # monitor fires immediately

    out = await _collect(worker.generate({"token_ids": []}, _ctx(ctx)))
    assert [c["token_ids"] for c in out] == [[1]]


async def test_worker_does_not_leak_monitor_task_on_success():
    """The background monitor task must be cancelled/awaited when the
    stream completes cleanly — otherwise long-lived workers leak tasks."""

    async def chunks(_request, _context):
        yield {"token_ids": [1], "finish_reason": "stop"}

    worker = _make_worker(_RecordingEngine(chunks))
    before = {t for t in asyncio.all_tasks()}
    await _collect(worker.generate({"token_ids": []}, _ctx(FakeContext())))
    await asyncio.sleep(0)  # let the cancelled monitor finish teardown
    leaked = [t for t in asyncio.all_tasks() - before if not t.done()]
    assert leaked == []


# ---------------------------------------------------------------------------
# WorkerConfig.from_runtime_config logic
# Only exercised paths: the `getattr(..., default)` / `or` fallback chain and
# the **overrides merge.  Pure dataclass defaults are not tested.
# ---------------------------------------------------------------------------


@dataclass
class _BareRuntime:
    """Runtime object with only the fields ``from_runtime_config`` treats as
    required (no fallbacks).  Every other field relies on getattr defaults."""

    namespace: str = "ns"
    discovery_backend: str = "etcd"
    request_plane: str = "tcp"
    event_plane: str = "nats"


def test_from_runtime_config_applies_defaults_when_fields_absent():
    """Exercises the ``getattr(..., default)`` fallback chain for ``component``,
    ``endpoint``, ``endpoint_types``, ``use_kv_events``, ``custom_jinja_template``."""
    cfg = WorkerConfig.from_runtime_config(_BareRuntime(), model_name="m")
    assert cfg.component == "backend"
    assert cfg.endpoint == "generate"
    assert cfg.endpoint_types == "chat,completions"
    assert cfg.use_kv_events is False
    assert cfg.custom_jinja_template is None


def test_from_runtime_config_overrides_win_over_runtime_values():
    """``**overrides`` is applied after the runtime-config mapping, so callers
    can force-override any field."""

    @dataclass
    class _WithComponent:
        namespace: str = "ns"
        component: str = "from-runtime"
        discovery_backend: str = "etcd"
        request_plane: str = "tcp"
        event_plane: str = "nats"

    cfg = WorkerConfig.from_runtime_config(
        _WithComponent(), model_name="m", component="from-override"
    )
    assert cfg.component == "from-override"
