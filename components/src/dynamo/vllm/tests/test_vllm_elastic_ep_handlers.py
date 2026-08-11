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
"""

import asyncio
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

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


def _install_ray_stub(monkeypatch):
    """Stub the ``ray`` / ``ray.util.state`` the handler imports lazily.

    The handler reads and temporarily overwrites ``ray.util.state.list_nodes``
    and (only if vLLM's reconfigure calls it) ``ray.nodes()``. With the engine
    mocked, list_nodes is never invoked, but the imports and the read/restore of
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


def _make_self(scale_mock: AsyncMock, prev_dp: int = 2) -> SimpleNamespace:
    engine_client = SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_size=prev_dp)
        ),
        scale_elastic_ep=scale_mock,
    )
    return SimpleNamespace(
        _scale_ep_lock=asyncio.Lock(),
        _scale_ep_in_progress=False,
        engine_client=engine_client,
    )


def _run(scale_mock: AsyncMock, body: dict, prev_dp: int = 2) -> dict:
    async def _coro():
        fake_self = _make_self(scale_mock, prev_dp=prev_dp)
        result = await BaseWorkerHandler.scale_elastic_ep(fake_self, body)
        # The in-progress flag must always be cleared for the next request.
        assert fake_self._scale_ep_in_progress is False
        return result

    return asyncio.run(_coro())


def _sizes(scale_mock: AsyncMock) -> list:
    return [call.args[0] for call in scale_mock.await_args_list]


def test_scale_success(monkeypatch):
    _install_ray_stub(monkeypatch)
    scale = AsyncMock(return_value=None)

    result = _run(scale, {"new_data_parallel_size": 3}, prev_dp=2)

    assert result["status"] == "ok"
    assert result["new_data_parallel_size"] == 3
    # grow only, no rollback
    assert _sizes(scale) == [3]


def test_failed_grow_rolls_back_to_prev_dp(monkeypatch):
    _install_ray_stub(monkeypatch)
    # First call (grow to 3) fails; second call (rollback to 2) succeeds.
    scale = AsyncMock(side_effect=[RuntimeError("grow failed"), None])

    result = _run(scale, {"new_data_parallel_size": 3}, prev_dp=2)

    assert result["status"] == "error"
    assert result["recoverable"] is True
    assert result["data_parallel_size"] == 2
    assert "rolled back to dp=2" in result["message"]
    # grow(3) then rollback(2)
    assert _sizes(scale) == [3, 2]


def test_failed_rollback_reports_unrecoverable(monkeypatch):
    _install_ray_stub(monkeypatch)
    # Both the grow and the rollback fail -> unrecoverable, needs restart.
    scale = AsyncMock(
        side_effect=[RuntimeError("grow failed"), RuntimeError("rollback failed")]
    )

    result = _run(scale, {"new_data_parallel_size": 3}, prev_dp=2)

    assert result["status"] == "error"
    assert result["recoverable"] is False
    assert "must be restarted" in result["message"]
    assert _sizes(scale) == [3, 2]


def test_missing_field_is_rejected(monkeypatch):
    _install_ray_stub(monkeypatch)
    scale = AsyncMock(return_value=None)

    result = _run(scale, {}, prev_dp=2)

    assert result["status"] == "error"
    assert "new_data_parallel_size" in result["message"]
    # never reached the engine
    assert _sizes(scale) == []


def test_below_floor_is_rejected(monkeypatch):
    _install_ray_stub(monkeypatch)
    scale = AsyncMock(return_value=None)

    result = _run(scale, {"new_data_parallel_size": 1}, prev_dp=2)

    assert result["status"] == "error"
    assert _sizes(scale) == []
