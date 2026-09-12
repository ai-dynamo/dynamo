# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for BaseWorkerHandler.scale_elastic_ep.

Two concerns: input validation (the TP-derived EP-size floor) and fail-fast
handling of a failed grow. vLLM does no rollback on a failed scale, so the
handler restarts the worker instead of recovering in process; the fail-fast
tests assert the restart is actually triggered (via ``_WorkerShutdown``), not
just that a value was returned.

Fail-fast is stubbed at ``_fail_fast`` rather than at ``_shutdown_worker``:
``_fail_fast`` is the seam that guarantees the exit even when the shutdown call
underneath it raises, so a test that replaced only the shutdown helpers would
not exercise it.

``ray`` is stubbed via the ``stub_ray`` fixture (the scale path imports it
lazily and CI lacks it).
"""

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm.config")

from vllm.v1.engine.exceptions import EngineDeadError  # noqa: E402

from dynamo.vllm import handlers as handlers_mod  # noqa: E402
from dynamo.vllm.handlers import BaseWorkerHandler  # noqa: E402

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class _TestWorkerHandler(BaseWorkerHandler):
    async def generate(self, request, context):
        yield {}


def _make_handler(
    tensor_parallel_size: int,
    prefill_context_parallel_size: int = 1,
    data_parallel_size: int = 1,
) -> _TestWorkerHandler:
    handler = _TestWorkerHandler.__new__(_TestWorkerHandler)
    handler.engine_client = SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(
                tensor_parallel_size=tensor_parallel_size,
                prefill_context_parallel_size=prefill_context_parallel_size,
                data_parallel_size=data_parallel_size,
                # Elastic EP enabled + Ray backend so the capability gate passes.
                enable_elastic_ep=True,
                data_parallel_backend="ray",
            ),
        ),
        scale_elastic_ep=AsyncMock(),
    )
    handler._scale_ep_lock = asyncio.Lock()
    handler._scale_ep_in_progress = False
    handler._scale_ep_cancelled = False
    return handler


@pytest.fixture
def short_deadline(monkeypatch):
    """Shrink the scale deadline for the hang tests.

    The deadline is resolved once at import (a per-request read would let a
    typo'd env var restart the worker on every scale), so tests set the module
    constant rather than the environment -- which also makes them hermetic
    against a developer or CI shell that exports DYN_SCALE_EP_TIMEOUT_S.
    """

    def _set(timeout_s: float, grace_s: float = 0.05) -> None:
        monkeypatch.setattr(handlers_mod, "_SCALE_EP_TIMEOUT_S", timeout_s)
        monkeypatch.setattr(handlers_mod, "_SCALE_EP_KILL_GRACE_S", grace_s)

    return _set


@pytest.fixture
def stub_ray(monkeypatch):
    """Stand in for ray, which the scale path imports lazily and CI lacks."""
    state = SimpleNamespace(list_nodes=lambda **kwargs: [])
    util = SimpleNamespace(state=state)
    monkeypatch.setitem(
        sys.modules, "ray", SimpleNamespace(nodes=lambda: [], util=util)
    )
    monkeypatch.setitem(sys.modules, "ray.util", util)
    monkeypatch.setitem(sys.modules, "ray.util.state", state)
    return state


# --------------------------------------------------------------------------- #
# Input validation and the TP-derived EP-size floor.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_missing_size_is_rejected():
    handler = _make_handler(tensor_parallel_size=4)

    result = await handler.scale_elastic_ep({})

    assert result["status"] == "error"
    assert "new_data_parallel_size" in result["message"]
    handler.engine_client.scale_elastic_ep.assert_not_awaited()


@pytest.mark.parametrize("size", [0, -1])
@pytest.mark.asyncio
async def test_sizes_below_one_are_rejected(size):
    handler = _make_handler(tensor_parallel_size=4)

    result = await handler.scale_elastic_ep({"new_data_parallel_size": size})

    assert result["status"] == "error"
    assert "must be >= 1" in result["message"]
    handler.engine_client.scale_elastic_ep.assert_not_awaited()


@pytest.mark.parametrize("size", [1.5, "1.5", "abc", [2], True])
@pytest.mark.asyncio
async def test_non_integer_sizes_are_rejected(size):
    """A bare int() would truncate 1.5 -> 1 and coerce True -> 1; reject instead."""
    handler = _make_handler(tensor_parallel_size=4)

    result = await handler.scale_elastic_ep({"new_data_parallel_size": size})

    assert result["status"] == "error"
    assert "must be an integer" in result["message"]
    handler.engine_client.scale_elastic_ep.assert_not_awaited()


@pytest.mark.parametrize("size", [2.0, "2"])
@pytest.mark.asyncio
async def test_integer_valued_sizes_are_accepted(size, stub_ray):
    """Integer-valued floats and decimal-free strings coerce to the exact int."""
    handler = _make_handler(tensor_parallel_size=1)

    result = await handler.scale_elastic_ep({"new_data_parallel_size": size})

    assert result["status"] == "ok"
    handler.engine_client.scale_elastic_ep.assert_awaited_once_with(2)


@pytest.mark.asyncio
async def test_single_rank_expert_group_is_rejected():
    """At TP=1 a target of dp=1 leaves one EP rank, which EPLB does not allow."""
    handler = _make_handler(tensor_parallel_size=1)

    result = await handler.scale_elastic_ep({"new_data_parallel_size": 1})

    assert result["status"] == "error"
    assert "must be > 1" in result["message"]
    handler.engine_client.scale_elastic_ep.assert_not_awaited()


@pytest.mark.asyncio
async def test_dp_one_allowed_when_tensor_parallelism_widens_the_group(stub_ray):
    """One pod per DP rank: TP=4 still leaves four EP ranks at dp=1."""
    handler = _make_handler(tensor_parallel_size=4)

    result = await handler.scale_elastic_ep({"new_data_parallel_size": 1})

    assert result["status"] == "ok"
    assert result["new_data_parallel_size"] == 1
    handler.engine_client.scale_elastic_ep.assert_awaited_once_with(1)


@pytest.mark.asyncio
async def test_dp_two_allowed_at_tp_one(stub_ray):
    handler = _make_handler(tensor_parallel_size=1)

    result = await handler.scale_elastic_ep({"new_data_parallel_size": 2})

    assert result["status"] == "ok"
    handler.engine_client.scale_elastic_ep.assert_awaited_once_with(2)


@pytest.mark.parametrize("size", [1, 2])
@pytest.mark.asyncio
async def test_prefill_context_parallelism_is_rejected(size):
    """vLLM's elastic EP sizes the EP world as data_parallel_size * tensor_parallel_size
    (excluding PCP) and forbids PCP>1 with DP>1, so a PCP>1 engine cannot be scaled --
    reject instead of admitting a topology elastic EP does not model."""
    handler = _make_handler(tensor_parallel_size=1, prefill_context_parallel_size=2)

    result = await handler.scale_elastic_ep({"new_data_parallel_size": size})

    assert result["status"] == "error"
    assert "prefill_context_parallel_size" in result["message"]
    handler.engine_client.scale_elastic_ep.assert_not_awaited()


# --------------------------------------------------------------------------- #
# Fail-fast handling of a failed grow.
# --------------------------------------------------------------------------- #


class _WorkerShutdown(BaseException):
    """Stand-in for the NoReturn _fail_fast, which ends in os._exit(1).

    BaseException (not Exception) so the handler's broad ``except Exception``
    can't swallow it -- the test sees the restart as production does: control
    leaves scale_elastic_ep without reporting success. Production gets that
    same guarantee from _fail_fast calling os._exit unconditionally, which
    ``test_fail_fast_exits_even_if_shutdown_raises`` covers directly.
    """


_RESTARTED = object()  # sentinel: handler restarted the worker instead of returning


class _FakeVllmEngine:
    """Stand-in engine client: scaling to a ``fail_sizes`` size raises
    ``RuntimeError`` and to a ``dead_sizes`` size raises ``EngineDeadError``.
    ``tensor_parallel_size`` is set so the TP floor check passes and the grow is
    actually attempted.
    """

    def __init__(
        self,
        prev_dp,
        fail_sizes=(),
        dead_sizes=(),
        tensor_parallel_size=1,
        enable_elastic_ep=True,
        data_parallel_backend="ray",
        hang_sizes=(),
    ):
        self.vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(
                data_parallel_size=prev_dp,
                tensor_parallel_size=tensor_parallel_size,
                enable_elastic_ep=enable_elastic_ep,
                data_parallel_backend=data_parallel_backend,
            )
        )
        self._fail_sizes = list(fail_sizes)
        self._dead_sizes = list(dead_sizes)
        self._hang_sizes = list(hang_sizes)
        self.calls: list[int] = []

    async def scale_elastic_ep(self, size: int) -> None:
        self.calls.append(size)
        if size in self._hang_sizes:
            await asyncio.sleep(3600)
        if size in self._dead_sizes:
            raise EngineDeadError()
        if size in self._fail_sizes:
            raise RuntimeError(f"scale to {size} failed")
        self.vllm_config.parallel_config.data_parallel_size = size


def _make_self(engine: _FakeVllmEngine, shutdown_log: list) -> SimpleNamespace:
    def _shutdown_worker():
        shutdown_log.append("worker")
        raise _WorkerShutdown()

    def _shutdown_on_engine_dead(err):
        shutdown_log.append("engine_dead")
        raise _WorkerShutdown()

    fake_self = SimpleNamespace(
        _scale_ep_lock=asyncio.Lock(),
        _scale_ep_in_progress=False,
        _scale_ep_cancelled=False,
        engine_client=engine,
        _shutdown_worker=_shutdown_worker,
        _shutdown_on_engine_dead=_shutdown_on_engine_dead,
    )
    # Stub the seam the handler actually calls. In production _fail_fast ends in
    # os._exit(1) whatever the shutdown callable does; here it just lets the
    # callable raise, so the test sees the same "never returns a result".
    fake_self._fail_fast = lambda shutdown, reason: shutdown()
    return fake_self


def _run(engine: _FakeVllmEngine, body: dict, fake_self=None):
    """Drive scale_elastic_ep. Returns ``(result, shutdown_log)``; ``result`` is
    ``_RESTARTED`` when the handler restarted the worker instead of returning."""
    shutdown_log: list[str] = []
    target = fake_self if fake_self is not None else _make_self(engine, shutdown_log)

    async def _coro():
        result = await BaseWorkerHandler.scale_elastic_ep(target, body)
        # Whenever the handler returns at all, the endpoint must be reusable:
        # a stuck in-progress flag is what wedges it for every later request.
        assert target._scale_ep_in_progress is False
        return result

    try:
        result = asyncio.run(_coro())
    except _WorkerShutdown:
        result = _RESTARTED
    return result, shutdown_log


def test_scale_success(stub_ray):
    engine = _FakeVllmEngine(prev_dp=2)

    result, shutdown = _run(engine, {"new_data_parallel_size": 3})

    assert shutdown == []
    assert result["status"] == "ok"
    assert result["new_data_parallel_size"] == 3
    assert engine.calls == [3]


def test_validation_error_does_not_restart_the_worker(stub_ray):
    # Fail-fast is scoped to real scale failures: a request rejected up front
    # (dp=1 at TP=1 collapses the EP world) returns an error and must NOT restart
    # the worker or touch the engine.
    engine = _FakeVllmEngine(prev_dp=2, tensor_parallel_size=1)

    result, shutdown = _run(engine, {"new_data_parallel_size": 1})

    assert shutdown == []
    assert result["status"] == "error"
    assert engine.calls == []


def test_unsupported_config_is_rejected_without_restart(stub_ray):
    # control/scale_elastic_ep is registered on every worker, but a worker
    # without elastic EP / the Ray DP backend must get a nonfatal error, not a
    # fail-fast restart -- vLLM would raise NotImplementedError before any scale
    # state is mutated.
    engine = _FakeVllmEngine(
        prev_dp=2, enable_elastic_ep=False, data_parallel_backend="mp"
    )

    result, shutdown = _run(engine, {"new_data_parallel_size": 3})

    assert shutdown == []
    assert result["status"] == "error"
    assert "not enabled" in result["message"]
    assert engine.calls == []


def test_failed_grow_restarts_the_worker(stub_ray):
    # vLLM does not roll back a failed scale, so a failed grow must fail fast:
    # restart the worker rather than report a recovery that no caller acts on.
    engine = _FakeVllmEngine(prev_dp=2, fail_sizes=[3])

    result, shutdown = _run(engine, {"new_data_parallel_size": 3})

    assert result is _RESTARTED
    assert shutdown == ["worker"]
    assert engine.calls == [3]


@pytest.mark.timeout(15)
def test_hung_grow_times_out_and_restarts(stub_ray, short_deadline):
    # A scale that hangs past the deadline is a wedged engine: fail fast and
    # restart, rather than leave the endpoint's in-progress flag stuck forever.
    # This hang answers cancellation, so the cooperative wait_for handles it.
    short_deadline(0.05)
    engine = _FakeVllmEngine(prev_dp=2, hang_sizes=[3])

    result, shutdown = _run(engine, {"new_data_parallel_size": 3})

    assert result is _RESTARTED
    assert shutdown == ["worker"]
    assert engine.calls == [3]


def test_watchdog_bounds_a_hang_that_ignores_cancellation(
    stub_ray, short_deadline, monkeypatch
):
    """The hang that actually happens in production.

    asyncio.wait_for cancels the inner coroutine and then *awaits* the
    cancellation, so a scale wedged in a step that does not answer cancellation
    runs straight past the deadline and the in-progress flag never clears. The
    watchdog is the only thing that bounds that, so assert it is armed across the
    whole critical section and that its callback is a hard exit.

    The timer itself is stubbed: a real one would call os._exit from another
    thread, and the pytest process is not the thing under test.
    """
    short_deadline(0.05, grace_s=0.05)
    armed: dict = {}

    class _CapturingTimer:
        def __init__(self, interval, function):
            armed["interval"] = interval
            armed["function"] = function
            self.daemon = False

        def start(self):
            armed["started"] = True

        def cancel(self):
            armed["cancelled"] = True

    monkeypatch.setattr(handlers_mod.threading, "Timer", _CapturingTimer)
    engine = _FakeVllmEngine(prev_dp=2)

    result, _ = _run(engine, {"new_data_parallel_size": 3})

    assert result["status"] == "ok"
    assert armed["started"] and armed["cancelled"]  # armed for the scale, then off
    assert armed["interval"] == pytest.approx(0.10)  # deadline + grace

    # The armed callback exits the process. Nothing else can end a hang that
    # swallows cancellation, because wait_for waits for the cancellation.
    exits: list[int] = []
    monkeypatch.setattr(handlers_mod.os, "_exit", lambda code: exits.append(code))
    monkeypatch.setattr(handlers_mod.logging, "shutdown", lambda: None)
    armed["function"]()
    assert exits == [1]


def test_fail_fast_exits_even_if_shutdown_raises(monkeypatch):
    """_shutdown_worker calls runtime.shutdown() before os._exit(1). On the
    degraded worker that got us here that call can raise, and the exception would
    unwind into scale_elastic_ep's broad ``except Exception`` -- turning a wedged
    engine into an ordinary error response while the worker stays registered and
    keeps taking traffic. _fail_fast must exit regardless."""
    exits: list[int] = []
    monkeypatch.setattr(handlers_mod.os, "_exit", lambda code: exits.append(code))
    monkeypatch.setattr(handlers_mod.logging, "shutdown", lambda: None)

    def _shutdown_that_raises():
        raise RuntimeError("runtime.shutdown() failed on a degraded worker")

    handler = _TestWorkerHandler.__new__(_TestWorkerHandler)
    BaseWorkerHandler._fail_fast(handler, _shutdown_that_raises, "wedged engine")

    assert exits == [1]


@pytest.mark.timeout(15)
def test_cancelled_scale_latches_the_endpoint_closed(stub_ray, short_deadline):
    """An externally-cancelled scale must not restart the worker, but vLLM's grow
    is not cancelled with us -- so the next scale must be refused rather than
    raced against the orphan."""
    short_deadline(30.0)
    engine = _FakeVllmEngine(prev_dp=2, hang_sizes=[3])
    fake_self = _make_self(engine, [])

    async def _cancel_mid_scale():
        task = asyncio.ensure_future(
            BaseWorkerHandler.scale_elastic_ep(fake_self, {"new_data_parallel_size": 3})
        )
        await asyncio.sleep(0.05)  # let it reach the engine
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # A healthy worker is not killed...
        assert fake_self._scale_ep_cancelled is True
        # ...but it will not accept another scale that would race the orphan.
        return await BaseWorkerHandler.scale_elastic_ep(
            fake_self, {"new_data_parallel_size": 4}
        )

    result = asyncio.run(_cancel_mid_scale())

    assert result["status"] == "error"
    assert "must restart" in result["message"]
    assert engine.calls == [3]  # the second request never reached the engine


def test_list_nodes_patch_is_restored_and_passes_through_kwargs(stub_ray):
    """The patch is a process-global. It must be restored exactly, and it must
    not answer callers it was not meant to intercept: it honours no filters and
    carries only node_id/node_ip."""
    original = stub_ray.list_nodes
    seen: list[dict] = []
    engine = _FakeVllmEngine(prev_dp=2)

    async def _scale_and_probe(size):
        # Runs while the patch is installed.
        seen.append({"no_args": stub_ray.list_nodes()})
        seen.append({"with_kwargs": stub_ray.list_nodes(filters=[("x", "y")])})
        engine.calls.append(size)

    engine.scale_elastic_ep = _scale_and_probe

    result, _ = _run(engine, {"new_data_parallel_size": 3})

    assert result["status"] == "ok"
    assert stub_ray.list_nodes is original  # restored exactly
    assert seen[0]["no_args"] == []  # our GCS stand-in (no live ray nodes)
    assert seen[1]["with_kwargs"] == []  # delegated to the real list_nodes


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, handlers_mod._SCALE_EP_TIMEOUT_DEFAULT_S),
        ("120", 120.0),
        ("0", handlers_mod._SCALE_EP_TIMEOUT_DEFAULT_S),  # would restart every scale
        ("-1", handlers_mod._SCALE_EP_TIMEOUT_DEFAULT_S),
        ("600s", handlers_mod._SCALE_EP_TIMEOUT_DEFAULT_S),  # unparseable
        ("", handlers_mod._SCALE_EP_TIMEOUT_DEFAULT_S),
    ],
)
def test_scale_deadline_rejects_unusable_env_values(raw, expected, monkeypatch):
    """A non-positive deadline makes every healthy scale time out, and a timed-out
    scale restarts the worker -- so a typo'd env var would be a crashloop."""
    if raw is None:
        monkeypatch.delenv(handlers_mod._SCALE_EP_TIMEOUT_ENV, raising=False)
    else:
        monkeypatch.setenv(handlers_mod._SCALE_EP_TIMEOUT_ENV, raw)

    assert handlers_mod._read_scale_ep_timeout_s() == expected


def test_engine_dead_restarts_the_worker(stub_ray):
    # A dead engine routes through the same _shutdown_on_engine_dead path the rest
    # of the handler uses for engine death.
    engine = _FakeVllmEngine(prev_dp=2, dead_sizes=[3])

    result, shutdown = _run(engine, {"new_data_parallel_size": 3})

    assert result is _RESTARTED
    assert shutdown == ["engine_dead"]
    assert engine.calls == [3]
