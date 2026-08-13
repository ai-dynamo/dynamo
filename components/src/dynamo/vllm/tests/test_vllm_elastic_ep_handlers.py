# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for BaseWorkerHandler.scale_elastic_ep.

Covers two independent concerns of the same method:

* input validation and the TP-derived EP-size floor, and
* the failed-grow rollback and engine-death handling.

``ray`` is imported lazily inside the scale path and is absent in CI, so it is
stubbed in ``sys.modules`` via the ``stub_ray`` fixture. ``_FakeVllmEngine``
mirrors two subtle semantics of vLLM v0.26's ``AsyncLLM.scale_elastic_ep`` that
a plain mock would paper over: it records ``parallel_config.data_parallel_size``
only *after* a reconfigure succeeds, and it short-circuits to a no-op when asked
to scale to the size it already records. Together those mean a naive rollback to
``prev_dp`` after a failed grow would be silently skipped; the handler advances
the recorded size before rolling back so the rollback drives a *real* reconfigure,
which the rollback tests assert via ``real_reconfigures``.
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
) -> _TestWorkerHandler:
    handler = _TestWorkerHandler.__new__(_TestWorkerHandler)
    handler.engine_client = SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(
                tensor_parallel_size=tensor_parallel_size,
                prefill_context_parallel_size=prefill_context_parallel_size,
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
# Failed-grow rollback and engine-death handling.
# --------------------------------------------------------------------------- #


class _FakeVllmEngine:
    """Stand-in engine client that models vLLM v0.26 scale semantics.

    Mirrors ``AsyncLLM.scale_elastic_ep``: a request to the currently-recorded
    size is a no-op (vLLM's "already at this size, skipping scale" guard), and the
    recorded size is only advanced *after* a reconfigure succeeds. Sizes in
    ``fail_sizes`` raise ``RuntimeError``; sizes in ``dead_sizes`` raise
    ``EngineDeadError``. ``tensor_parallel_size`` is exposed so the handler's
    TP-derived floor check passes and the rollback path is reached.
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
        self.real_reconfigures: list[int] = []  # sizes that did real work

    async def scale_elastic_ep(self, size: int) -> None:
        self.calls.append(size)
        if self.vllm_config.parallel_config.data_parallel_size == size:
            # vLLM guard: no reconfigure, recorded size unchanged.
            return
        self.real_reconfigures.append(size)
        if size in self._dead_sizes:
            raise EngineDeadError()
        if size in self._fail_sizes:
            raise RuntimeError(f"reconfigure to {size} failed")
        # Only advance the recorded size on success, exactly like vLLM.
        self.vllm_config.parallel_config.data_parallel_size = size


def _make_self(engine: _FakeVllmEngine) -> SimpleNamespace:
    return SimpleNamespace(
        _scale_ep_lock=asyncio.Lock(),
        _scale_ep_in_progress=False,
        engine_client=engine,
    )


def _run(engine: _FakeVllmEngine, body: dict) -> dict:
    async def _coro():
        fake_self = _make_self(engine)
        result = await BaseWorkerHandler.scale_elastic_ep(fake_self, body)
        # The in-progress flag must always be cleared for the next request.
        assert fake_self._scale_ep_in_progress is False
        return result

    return asyncio.run(_coro())


def _recorded_dp(engine: _FakeVllmEngine) -> int:
    return engine.vllm_config.parallel_config.data_parallel_size


def test_scale_success(stub_ray):
    engine = _FakeVllmEngine(prev_dp=2)

    result = _run(engine, {"new_data_parallel_size": 3})

    assert result["status"] == "ok"
    assert result["new_data_parallel_size"] == 3
    # grow only, no rollback; a real reconfigure happened and stuck.
    assert engine.calls == [3]
    assert engine.real_reconfigures == [3]
    assert _recorded_dp(engine) == 3


def test_failed_grow_rolls_back_with_a_real_reconfigure(stub_ray):
    # The grow to 3 fails; the rollback to 2 must be a *real* reconfigure, not a
    # no-op. vLLM records dp only after success, so unless the handler advances
    # the recorded size before rolling back, vLLM's guard silently skips it.
    engine = _FakeVllmEngine(prev_dp=2, fail_sizes=[3])

    result = _run(engine, {"new_data_parallel_size": 3})

    assert result["status"] == "error"
    assert result["recoverable"] is True
    assert result["data_parallel_size"] == 2
    assert "rolled back to dp=2" in result["message"]
    # grow(3) then rollback(2)...
    assert engine.calls == [3, 2]
    # ...and both were *real* reconfigures -- the rollback was NOT swallowed by
    # vLLM's "already at this size" guard. If the guard-bypass in the handler is
    # dropped, this drops to [3] and the test fails.
    assert engine.real_reconfigures == [3, 2]
    # Engine ends back at the last good size.
    assert _recorded_dp(engine) == 2


def test_failed_rollback_reports_unrecoverable(stub_ray):
    # Both the grow and the (real) rollback fail -> unrecoverable, needs restart.
    engine = _FakeVllmEngine(prev_dp=2, fail_sizes=[3, 2])

    result = _run(engine, {"new_data_parallel_size": 3})

    assert result["status"] == "error"
    assert result["recoverable"] is False
    assert "must be restarted" in result["message"]
    # The rollback was actually attempted (a real reconfigure), not skipped.
    assert engine.calls == [3, 2]
    assert engine.real_reconfigures == [3, 2]


def test_engine_dead_error_is_fatal_and_skips_rollback(stub_ray):
    # vLLM raises EngineDeadError when the engine core is gone. A dead engine
    # cannot be reconfigured, so the handler must report unrecoverable and must
    # NOT attempt a rollback (which would only mask the dead engine).
    engine = _FakeVllmEngine(prev_dp=2, dead_sizes=[3])

    result = _run(engine, {"new_data_parallel_size": 3})

    assert result["status"] == "error"
    assert result["recoverable"] is False
    assert "restart" in result["message"].lower()
    # only the failed grow reached the engine -- no rollback call
    assert engine.calls == [3]
