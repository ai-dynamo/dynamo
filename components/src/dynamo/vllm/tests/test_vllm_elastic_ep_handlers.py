# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for BaseWorkerHandler.scale_elastic_ep rollback behavior.

A failed data-parallel grow must not wedge the engine: on failure the handler
rolls back to the previous dp so the engine keeps serving ("either it grows, or
nothing changes"). If the rollback itself fails, the engine is unrecoverable in
process and the handler must say so (recoverable=False) rather than report a
false recovery.

The handler only touches ``self._scale_ep_lock``, ``self._scale_ep_in_progress``
and ``self.engine_client``, so a SimpleNamespace stands in for ``self`` -- no
need to build a real (abstract) BaseWorkerHandler. ``ray`` is imported lazily
inside the method, so it is stubbed in ``sys.modules``.

``_FakeVllmEngine`` deliberately mirrors two subtle semantics of vLLM v0.26's
``AsyncLLM.scale_elastic_ep`` that a plain mock would paper over:

* it records ``parallel_config.data_parallel_size`` only *after* a reconfigure
  succeeds (a failed grow leaves the recorded size unchanged), and
* it short-circuits to a no-op when asked to scale to the size it already
  records ("Data parallel size is already N, skipping scale").

Together those mean a naive rollback to ``prev_dp`` after a failed grow would be
silently skipped. The handler advances the recorded size to the attempted target
before rolling back so the rollback drives a *real* reconfigure; the tests below
assert that (``real_reconfigures``) so the guard-bypass cannot be dropped without
a failure here.
"""

import asyncio
import sys
from types import ModuleType, SimpleNamespace

import pytest

from dynamo.vllm.handlers import BaseWorkerHandler

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.xpu_1,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.timeout(180),  # 0-GiB unit tests, floor 180s
    pytest.mark.pre_merge,
]


class _FakeVllmEngine:
    """Stand-in for the vLLM engine client that models v0.26 scale semantics.

    Mirrors ``AsyncLLM.scale_elastic_ep``: a request to the currently-recorded
    size is a no-op (vLLM's "already at this size, skipping scale" guard), and
    the recorded size is only advanced *after* a reconfigure succeeds. Sizes in
    ``fail_sizes`` raise instead of completing.
    """

    def __init__(self, prev_dp: int, fail_sizes=()):
        self.vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_size=prev_dp)
        )
        self._fail_sizes = list(fail_sizes)
        self.calls: list[int] = []  # every requested size, in order
        self.real_reconfigures: list[int] = []  # sizes that did real work

    async def scale_elastic_ep(self, size: int) -> None:
        self.calls.append(size)
        if self.vllm_config.parallel_config.data_parallel_size == size:
            # vLLM guard: no reconfigure, recorded size unchanged.
            return
        self.real_reconfigures.append(size)
        if size in self._fail_sizes:
            raise RuntimeError(f"reconfigure to {size} failed")
        # Only advance the recorded size on success, exactly like vLLM.
        self.vllm_config.parallel_config.data_parallel_size = size


def _install_ray_stub(monkeypatch):
    """Stub the ``ray`` / ``ray.util.state`` the handler imports lazily.

    The handler reads and temporarily overwrites ``ray.util.state.list_nodes``
    and (only if vLLM's reconfigure calls it) ``ray.nodes()``. With the engine
    stubbed, list_nodes is never invoked, but the imports and the read/restore of
    ``list_nodes`` still run.
    """
    ray_mod = ModuleType("ray")
    ray_mod.nodes = lambda: [
        {"NodeManagerAddress": "127.0.0.1", "NodeID": "node0", "Alive": True}
    ]
    util_mod = ModuleType("ray.util")
    state_mod = ModuleType("ray.util.state")
    state_mod.list_nodes = lambda **kw: []
    util_mod.state = state_mod
    ray_mod.util = util_mod
    monkeypatch.setitem(sys.modules, "ray", ray_mod)
    monkeypatch.setitem(sys.modules, "ray.util", util_mod)
    monkeypatch.setitem(sys.modules, "ray.util.state", state_mod)


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


def test_scale_success(monkeypatch):
    _install_ray_stub(monkeypatch)
    engine = _FakeVllmEngine(prev_dp=2)

    result = _run(engine, {"new_data_parallel_size": 3})

    assert result["status"] == "ok"
    assert result["new_data_parallel_size"] == 3
    # grow only, no rollback; a real reconfigure happened and stuck.
    assert engine.calls == [3]
    assert engine.real_reconfigures == [3]
    assert _recorded_dp(engine) == 3


def test_failed_grow_rolls_back_with_a_real_reconfigure(monkeypatch):
    _install_ray_stub(monkeypatch)
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


def test_failed_rollback_reports_unrecoverable(monkeypatch):
    _install_ray_stub(monkeypatch)
    # Both the grow and the (real) rollback fail -> unrecoverable, needs restart.
    engine = _FakeVllmEngine(prev_dp=2, fail_sizes=[3, 2])

    result = _run(engine, {"new_data_parallel_size": 3})

    assert result["status"] == "error"
    assert result["recoverable"] is False
    assert "must be restarted" in result["message"]
    # The rollback was actually attempted (a real reconfigure), not skipped.
    assert engine.calls == [3, 2]
    assert engine.real_reconfigures == [3, 2]


def test_missing_field_is_rejected(monkeypatch):
    _install_ray_stub(monkeypatch)
    engine = _FakeVllmEngine(prev_dp=2)

    result = _run(engine, {})

    assert result["status"] == "error"
    assert "new_data_parallel_size" in result["message"]
    # never reached the engine
    assert engine.calls == []


def test_below_floor_is_rejected(monkeypatch):
    _install_ray_stub(monkeypatch)
    engine = _FakeVllmEngine(prev_dp=2)

    result = _run(engine, {"new_data_parallel_size": 1})

    assert result["status"] == "error"
    assert engine.calls == []
