# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for BaseWorkerHandler.scale_elastic_ep.

Covers two independent concerns of the same method:

* input validation and the TP-derived EP-size floor, and
* fail-fast handling of a failed grow.

vLLM performs no rollback or cleanup on a failed elastic-EP scale, so the handler
does not try to recover in process: on any failed grow it restarts the worker
(``_shutdown_worker`` / ``_shutdown_on_engine_dead``) so it comes back clean. The
fail-fast tests verify that shutdown is actually triggered (not just that a
boolean was returned) via the ``_WorkerShutdown`` stand-in.

``ray`` is imported lazily inside the scale path and is absent in CI, so it is
stubbed in ``sys.modules`` via the ``stub_ray`` fixture.
"""

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm.config")

from vllm.v1.engine.exceptions import EngineDeadError  # noqa: E402

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
                # Captured up front by the rollback path before the grow; real
                # vLLM engines always set it.
                data_parallel_size=data_parallel_size,
            ),
        ),
        scale_elastic_ep=AsyncMock(),
    )
    handler._scale_ep_lock = asyncio.Lock()
    handler._scale_ep_in_progress = False
    return handler


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
    """Stand-in for the real _shutdown_worker / _shutdown_on_engine_dead, which
    never return (they call runtime.shutdown() + os._exit). Subclassing
    BaseException (not Exception) means the handler's broad ``except Exception``
    cannot swallow it, so a test observes the restart exactly as production does:
    control leaves scale_elastic_ep and it never reports success.
    """


_RESTARTED = object()  # sentinel: handler restarted the worker instead of returning


class _FakeVllmEngine:
    """Stand-in engine client. Scaling to a size in ``fail_sizes`` raises
    ``RuntimeError`` and to a size in ``dead_sizes`` raises ``EngineDeadError``;
    ``tensor_parallel_size`` is exposed so the handler's TP-derived floor check
    passes and the grow is actually attempted.
    """

    def __init__(self, prev_dp, fail_sizes=(), dead_sizes=(), tensor_parallel_size=1):
        self.vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(
                data_parallel_size=prev_dp,
                tensor_parallel_size=tensor_parallel_size,
            )
        )
        self._fail_sizes = list(fail_sizes)
        self._dead_sizes = list(dead_sizes)
        self.calls: list[int] = []  # every requested size, in order

    async def scale_elastic_ep(self, size: int) -> None:
        self.calls.append(size)
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

    return SimpleNamespace(
        _scale_ep_lock=asyncio.Lock(),
        _scale_ep_in_progress=False,
        engine_client=engine,
        _shutdown_worker=_shutdown_worker,
        _shutdown_on_engine_dead=_shutdown_on_engine_dead,
    )


def _run(engine: _FakeVllmEngine, body: dict):
    """Drive scale_elastic_ep. Returns ``(result, shutdown_log)``; ``result`` is
    ``_RESTARTED`` when the handler restarted the worker instead of returning."""
    shutdown_log: list[str] = []

    async def _coro():
        fake_self = _make_self(engine, shutdown_log)
        return await BaseWorkerHandler.scale_elastic_ep(fake_self, body)

    try:
        result = asyncio.run(_coro())
    except _WorkerShutdown:
        result = _RESTARTED
    return result, shutdown_log


def test_scale_success(stub_ray):
    engine = _FakeVllmEngine(prev_dp=2)

    result, shutdown = _run(engine, {"new_data_parallel_size": 3})

    assert shutdown == []  # no restart on success
    assert result["status"] == "ok"
    assert result["new_data_parallel_size"] == 3
    assert engine.calls == [3]


def test_failed_grow_restarts_the_worker(stub_ray):
    # vLLM does not roll back a failed scale, so a failed grow must fail fast:
    # restart the worker rather than report a recovery that no caller acts on.
    engine = _FakeVllmEngine(prev_dp=2, fail_sizes=[3])

    result, shutdown = _run(engine, {"new_data_parallel_size": 3})

    assert result is _RESTARTED  # never returned a success/recovery dict
    assert shutdown == ["worker"]  # worker restart was triggered
    assert engine.calls == [3]  # grow attempted once; no rollback call


def test_engine_dead_restarts_the_worker(stub_ray):
    # A dead engine routes through the same _shutdown_on_engine_dead path the rest
    # of the handler uses for engine death.
    engine = _FakeVllmEngine(prev_dp=2, dead_sizes=[3])

    result, shutdown = _run(engine, {"new_data_parallel_size": 3})

    assert result is _RESTARTED
    assert shutdown == ["engine_dead"]
    assert engine.calls == [3]  # no rollback call
