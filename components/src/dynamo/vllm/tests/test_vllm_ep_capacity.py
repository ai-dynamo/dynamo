# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for BaseWorkerHandler.get_ep_capacity (Phase 5 read-only capacity endpoint).

The handler only touches ``self.engine_client.vllm_config.parallel_config`` and (for the
Ray DP backend) a lazily-imported ``ray``, so a SimpleNamespace stands in for ``self``
and the ``ray`` package is stubbed in ``sys.modules``.
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


class _StubRayError(Exception):
    """Stands in for ray.exceptions.RayError, the only failure handled in band."""


def _install_ray_stub(monkeypatch, nodes=(), idle_by_node_id=None, raises=None):
    """Register a fake ``ray`` package tree covering everything the handler imports.

    ``raises``, when set to an exception instance, is raised by every query so a test
    can prove either that a path never touches Ray or how a failure is reported.
    """
    ray_mod = ModuleType("ray")
    private_mod = ModuleType("ray._private")
    state_mod = ModuleType("ray._private.state")
    exceptions_mod = ModuleType("ray.exceptions")
    exceptions_mod.RayError = _StubRayError

    if raises is not None:

        def _boom(*_a, **_k):
            raise raises

        ray_mod.nodes = _boom
        ray_mod.available_resources = _boom
        state_mod.available_resources_per_node = _boom
    else:
        ray_mod.nodes = lambda: list(nodes)
        ray_mod.available_resources = lambda: {
            "GPU": sum(r.get("GPU", 0.0) for r in (idle_by_node_id or {}).values())
        }
        state_mod.available_resources_per_node = lambda: dict(idle_by_node_id or {})

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
    return SimpleNamespace(engine_client=engine_client)


def _run(fake_self) -> dict:
    return asyncio.run(BaseWorkerHandler.get_ep_capacity(fake_self, {}))


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
    assert [n["node_ip"] for n in r["nodes"]] == ["10.0.0.1", "10.0.0.2"]
    assert [n["available_gpus"] for n in r["nodes"]] == [4.0, 2.0]
    # The point of the per-node numbers: only node 1 can take another tp=4 rank,
    # even though the cluster-wide idle count would suggest room for more.
    placeable = sum(int(n["available_gpus"]) // 4 for n in r["nodes"])
    assert placeable == 1


def test_fully_consumed_node_reports_zero_available(monkeypatch):
    # Ray drops a resource from the availability map once it is fully consumed.
    nodes = [_node("n1", "10.0.0.1", 2.0)]
    _install_ray_stub(monkeypatch, nodes=nodes, idle_by_node_id={"n1": {"CPU": 30.0}})

    r = _run(_make_self(dp=2, tp=1, backend="ray"))

    assert r["status"] == "ok"
    assert r["nodes"] == [
        {"node_ip": "10.0.0.1", "total_gpus": 2.0, "available_gpus": 0.0}
    ]
    assert r["used_gpus"] == 2.0


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
