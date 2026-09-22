# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for BaseWorkerHandler.get_ep_capacity (Phase 5 read-only capacity endpoint).

The handler only touches ``self.engine_client.vllm_config.parallel_config`` and (for the
Ray DP backend) a lazily-imported ``ray``, so a SimpleNamespace stands in for ``self``
and the ``ray`` package is stubbed in ``sys.modules``.
"""

import asyncio
import sys
import threading
import time
from types import ModuleType, SimpleNamespace

import pytest

from dynamo.vllm import handlers as vllm_handlers
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


class _StubRayError(Exception):
    """Stands in for ray.exceptions.RayError, the only failure handled in band."""


def _install_ray_stub(
    monkeypatch,
    nodes=(),
    idle_by_node_id=None,
    raises=None,
    delay=0.0,
    thread_log=None,
):
    """Register a fake ``ray`` package tree covering everything the handler imports.

    ``raises``   -- exception instance raised by every query, so a test can prove a
                    path never touches Ray, or assert how a failure is reported.
    ``delay``    -- seconds each query blocks for, to model a slow GCS.
    ``thread_log`` -- list that each query appends its thread ident to.
    """
    idle = dict(idle_by_node_id or {})

    def _enter():
        if thread_log is not None:
            thread_log.append(threading.get_ident())
        if delay:
            time.sleep(delay)
        if raises is not None:
            raise raises

    def _nodes():
        _enter()
        return list(nodes)

    def _available_resources():
        _enter()
        return {"GPU": sum(r.get("GPU", 0.0) for r in idle.values())}

    def _available_resources_per_node():
        _enter()
        return dict(idle)

    ray_mod = ModuleType("ray")
    private_mod = ModuleType("ray._private")
    state_mod = ModuleType("ray._private.state")
    exceptions_mod = ModuleType("ray.exceptions")

    exceptions_mod.RayError = _StubRayError
    ray_mod.nodes = _nodes
    ray_mod.available_resources = _available_resources
    state_mod.available_resources_per_node = _available_resources_per_node

    private_mod.state = state_mod
    ray_mod._private = private_mod
    ray_mod.exceptions = exceptions_mod
    for name, mod in (
        ("ray", ray_mod),
        ("ray._private", private_mod),
        ("ray._private.state", state_mod),
        ("ray.exceptions", exceptions_mod),
    ):
        monkeypatch.setitem(sys.modules, name, mod)


def _node(node_id, ip, total_gpus, alive=True):
    return {
        "NodeID": node_id,
        "NodeManagerAddress": ip,
        "Alive": alive,
        "Resources": {"GPU": total_gpus},
    }


def _make_self(dp=2, tp=1, backend="ray", external_lb=False) -> SimpleNamespace:
    parallel_config = SimpleNamespace(
        data_parallel_size=dp,
        tensor_parallel_size=tp,
        data_parallel_backend=backend,
        data_parallel_external_lb=external_lb,
    )
    engine_client = SimpleNamespace(
        vllm_config=SimpleNamespace(parallel_config=parallel_config)
    )
    # Mirrors the real handler's single-flight state (BaseWorkerHandler.__init__).
    return SimpleNamespace(
        engine_client=engine_client,
        _ep_capacity_inflight=None,
        _ep_capacity_executor=None,
    )


def _shutdown(fake_self) -> None:
    """Drop the dedicated executor a test may have caused the handler to create."""
    if fake_self._ep_capacity_executor is not None:
        fake_self._ep_capacity_executor.shutdown(wait=False, cancel_futures=True)
        fake_self._ep_capacity_executor = None


def _run(fake_self) -> dict:
    try:
        return asyncio.run(BaseWorkerHandler.get_ep_capacity(fake_self, {}))
    finally:
        _shutdown(fake_self)


def test_ray_backend_reports_per_node_gpu_capacity(monkeypatch):
    nodes = [
        _node("n1", "10.0.0.1", 8.0),
        _node("n2", "10.0.0.2", 8.0),
        # dead node must be excluded from totals:
        _node("n9", "10.0.0.9", 8.0, alive=False),
    ]
    # 6 idle GPUs cluster-wide, but split 4 + 2 across two nodes.
    idle = {"n1": {"GPU": 4.0}, "n2": {"GPU": 2.0}, "n9": {"GPU": 8.0}}
    _install_ray_stub(monkeypatch, nodes=nodes, idle_by_node_id=idle)

    r = _run(_make_self(dp=2, tp=4, backend="ray"))

    assert r["status"] == "ok"
    assert r["data_parallel_size"] == 2
    assert r["tensor_parallel_size"] == 4
    assert r["data_parallel_backend"] == "ray"
    assert r["data_parallel_external_lb"] is False
    assert r["total_gpus"] == 16.0  # 2 alive x 8 GPUs; dead node excluded
    assert r["used_gpus"] == 16.0 - r["available_gpus"]
    assert [n["available_gpus"] for n in r["nodes"]] == [4.0, 2.0]
    # Capacity only -- node identity is not part of this payload.
    assert all(set(n) == {"total_gpus", "available_gpus"} for n in r["nodes"])
    # The point of the per-node numbers: only one node can take another tp=4 rank,
    # even though the cluster-wide idle count would suggest room for more.
    placeable = sum(int(n["available_gpus"]) // 4 for n in r["nodes"])
    assert placeable == 1


def test_fully_consumed_node_reports_zero_available(monkeypatch):
    # Ray drops a resource from the availability map once it is fully consumed.
    nodes = [_node("n1", "10.0.0.1", 2.0)]
    _install_ray_stub(monkeypatch, nodes=nodes, idle_by_node_id={"n1": {"CPU": 30.0}})

    r = _run(_make_self(dp=2, tp=1, backend="ray"))

    assert r["status"] == "ok"
    assert r["nodes"] == [{"total_gpus": 2.0, "available_gpus": 0.0}]
    assert r["used_gpus"] == 2.0


def test_ray_queries_run_off_the_event_loop(monkeypatch):
    """A slow GCS must not stall the worker: the queries belong in a thread.

    The reconciler polls this endpoint, so blocking the loop here would stall token
    generation and every other control route on this worker.
    """
    nodes = [_node("n1", "10.0.0.1", 4.0)]
    call_threads: list[int] = []
    _install_ray_stub(
        monkeypatch,
        nodes=nodes,
        idle_by_node_id={"n1": {"GPU": 4.0}},
        delay=0.1,
        thread_log=call_threads,
    )

    async def _drive():
        ticks = 0

        async def _ticker():
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0.005)

        task = asyncio.create_task(_ticker())
        await asyncio.sleep(0)  # let the ticker reach its first await
        res = await BaseWorkerHandler.get_ep_capacity(_make_self(backend="ray"), {})
        task.cancel()
        return res, ticks

    r, ticks = asyncio.run(_drive())

    assert r["status"] == "ok"
    # Three blocking calls at 0.1s each; the loop stayed free to run the ticker.
    assert ticks > 5, f"event loop appears blocked (only {ticks} ticks)"
    assert call_threads, "ray was never queried"
    loop_thread = threading.get_ident()
    assert all(t != loop_thread for t in call_threads)


def test_slow_ray_times_out_and_still_reports_dp_tp(monkeypatch):
    monkeypatch.setattr(vllm_handlers, "_EP_CAPACITY_RAY_TIMEOUT_S", 0.05)
    _install_ray_stub(
        monkeypatch,
        nodes=[_node("n1", "10.0.0.1", 4.0)],
        idle_by_node_id={"n1": {"GPU": 4.0}},
        delay=0.2,
    )

    async def _timed():
        started = time.monotonic()
        res = await BaseWorkerHandler.get_ep_capacity(
            _make_self(dp=3, tp=2, backend="ray"), {}
        )
        return res, time.monotonic() - started

    r, elapsed = asyncio.run(_timed())

    assert r["status"] == "error"
    assert "timed out" in r["message"].lower()
    # The caller is released at the timeout instead of waiting out the GCS stall.
    # Timed around the await, not around asyncio.run: the timeout frees the caller,
    # not the thread, so loop shutdown still joins the orphaned executor thread.
    assert elapsed < 0.15, f"caller waited {elapsed:.2f}s, expected to bail at 0.05s"
    # dp/tp still reported even though the GPU query timed out.
    assert r["data_parallel_size"] == 3
    assert r["tensor_parallel_size"] == 2
    assert r["total_gpus"] is None
    assert r["nodes"] is None


def test_repeated_timeouts_do_not_pile_up_gcs_queries(monkeypatch):
    """A degraded GCS must not strand one blocked thread per reconciler poll.

    The timeout releases the caller but cannot interrupt the blocking call, so
    without single-flight each poll would start another query and a slow GCS would
    accumulate threads until the pool starved.
    """
    monkeypatch.setattr(vllm_handlers, "_EP_CAPACITY_RAY_TIMEOUT_S", 0.02)
    started: list[int] = []
    _install_ray_stub(
        monkeypatch,
        nodes=[_node("n1", "10.0.0.1", 4.0)],
        idle_by_node_id={"n1": {"GPU": 4.0}},
        delay=0.5,
        thread_log=started,
    )
    handler = _make_self(dp=2, tp=1, backend="ray")

    async def _poll_repeatedly():
        out = []
        for _ in range(6):
            out.append(await BaseWorkerHandler.get_ep_capacity(handler, {}))
        return out

    try:
        results = asyncio.run(_poll_repeatedly())
    finally:
        _shutdown(handler)

    assert all(r["status"] == "error" for r in results)
    assert all("timed out" in r["message"].lower() for r in results)
    # Six polls, one outstanding GCS query: later polls joined the in-flight
    # snapshot instead of launching their own.
    assert len(started) == 1, f"expected 1 in-flight query, got {len(started)}"
    # And it never touched asyncio's shared default executor.
    assert started[0] != threading.get_ident()


def test_concurrent_callers_share_one_snapshot(monkeypatch):
    nodes = [_node("n1", "10.0.0.1", 4.0)]
    started: list[int] = []
    _install_ray_stub(
        monkeypatch,
        nodes=nodes,
        idle_by_node_id={"n1": {"GPU": 3.0}},
        delay=0.05,
        thread_log=started,
    )
    handler = _make_self(dp=2, tp=1, backend="ray")

    async def _gather_four():
        return await asyncio.gather(
            *(BaseWorkerHandler.get_ep_capacity(handler, {}) for _ in range(4))
        )

    try:
        results = asyncio.run(_gather_four())
    finally:
        _shutdown(handler)

    assert all(r["status"] == "ok" for r in results)
    assert all(
        r["nodes"] == [{"total_gpus": 4.0, "available_gpus": 3.0}] for r in results
    )
    # One snapshot is three ray calls; four concurrent callers must not make twelve.
    assert len(started) == 3, f"expected one snapshot (3 calls), got {len(started)}"


def test_mp_backend_is_reported_and_skips_ray(monkeypatch):
    # A ray stub that explodes if queried, proving the mp path never touches it.
    _install_ray_stub(monkeypatch, raises=AssertionError("ray must not be queried"))

    r = _run(_make_self(dp=2, tp=1, backend="mp"))

    assert r["status"] == "ok"
    assert r["data_parallel_backend"] == "mp"
    assert r["data_parallel_size"] == 2
    assert r["tensor_parallel_size"] == 1
    assert r["total_gpus"] is None
    assert r["available_gpus"] is None
    assert r["used_gpus"] is None
    assert r["nodes"] is None


def test_external_lb_is_reported_alongside_backend(monkeypatch):
    _install_ray_stub(monkeypatch, raises=AssertionError("ray must not be queried"))

    r = _run(_make_self(dp=2, tp=1, backend="mp", external_lb=True))

    assert r["status"] == "ok"
    assert r["data_parallel_backend"] == "mp"
    assert r["data_parallel_external_lb"] is True
    assert r["nodes"] is None


def test_ray_error_reports_error_but_keeps_dp_tp(monkeypatch):
    _install_ray_stub(monkeypatch, raises=_StubRayError("GCS unreachable"))

    r = _run(_make_self(dp=3, tp=1, backend="ray"))

    assert r["status"] == "error"
    assert "capacity query failed" in r["message"].lower()
    # dp/tp still reported even though the GPU query failed
    assert r["data_parallel_size"] == 3
    assert r["tensor_parallel_size"] == 1
    assert r["total_gpus"] is None


def test_unexpected_error_propagates(monkeypatch):
    # Only Ray failures are handled in band; a programming error must not be
    # laundered into a capacity "error" response.
    _install_ray_stub(monkeypatch, raises=TypeError("bad schema"))

    with pytest.raises(TypeError):
        _run(_make_self(dp=2, tp=1, backend="ray"))
