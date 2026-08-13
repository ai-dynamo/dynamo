# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for BaseWorkerHandler.get_ep_capacity (Phase 5 read-only capacity endpoint).

The handler only touches ``self.engine_client.vllm_config.parallel_config`` and (for the
Ray backend) a lazily-imported ``ray``, so a SimpleNamespace stands in for ``self`` and
``ray`` is stubbed in ``sys.modules``.
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


def _install_ray_stub(monkeypatch, nodes, available_gpu, raises=False):
    ray_mod = ModuleType("ray")
    if raises:

        def _boom(*_a, **_k):
            raise RuntimeError("ray down")

        ray_mod.nodes = _boom
        ray_mod.available_resources = _boom
    else:
        ray_mod.nodes = lambda: nodes
        ray_mod.available_resources = lambda: {"GPU": available_gpu}
    monkeypatch.setitem(sys.modules, "ray", ray_mod)


def _make_self(dp=2, tp=1, external_lb=False) -> SimpleNamespace:
    parallel_config = SimpleNamespace(
        data_parallel_size=dp,
        tensor_parallel_size=tp,
        data_parallel_external_lb=external_lb,
    )
    engine_client = SimpleNamespace(
        vllm_config=SimpleNamespace(parallel_config=parallel_config)
    )
    return SimpleNamespace(engine_client=engine_client)


def _run(fake_self) -> dict:
    return asyncio.run(BaseWorkerHandler.get_ep_capacity(fake_self, {}))


def test_ray_backend_reports_gpu_capacity(monkeypatch):
    nodes = [
        {"NodeManagerAddress": "10.0.0.1", "Alive": True, "Resources": {"GPU": 1.0}},
        {"NodeManagerAddress": "10.0.0.2", "Alive": True, "Resources": {"GPU": 1.0}},
        {"NodeManagerAddress": "10.0.0.3", "Alive": True, "Resources": {"GPU": 1.0}},
        # dead node must be excluded from totals:
        {"NodeManagerAddress": "10.0.0.9", "Alive": False, "Resources": {"GPU": 1.0}},
    ]
    _install_ray_stub(monkeypatch, nodes, available_gpu=1.0)

    r = _run(_make_self(dp=2, tp=1, external_lb=False))

    assert r["status"] == "ok"
    assert r["data_parallel_size"] == 2
    assert r["tensor_parallel_size"] == 1
    assert r["data_parallel_backend"] == "ray"
    assert r["total_gpus"] == 3.0  # 3 alive x 1 GPU; dead node excluded
    assert r["available_gpus"] == 1.0
    assert r["used_gpus"] == 2.0
    assert len(r["nodes"]) == 3


def test_external_lb_backend_skips_ray(monkeypatch):
    # A ray stub that explodes if queried, proving the external-lb path never touches it.
    _install_ray_stub(monkeypatch, nodes=[], available_gpu=0, raises=True)

    r = _run(_make_self(dp=2, tp=1, external_lb=True))

    assert r["status"] == "ok"
    assert r["data_parallel_backend"] == "external_lb"
    assert r["data_parallel_size"] == 2
    assert r["total_gpus"] is None
    assert r["available_gpus"] is None
    assert r["nodes"] is None


def test_ray_query_failure_reports_error_but_keeps_dp_tp(monkeypatch):
    _install_ray_stub(monkeypatch, nodes=[], available_gpu=0, raises=True)

    r = _run(_make_self(dp=3, tp=1, external_lb=False))

    assert r["status"] == "error"
    assert "capacity query failed" in r["message"].lower()
    # dp/tp still reported even though the GPU query failed
    assert r["data_parallel_size"] == 3
    assert r["tensor_parallel_size"] == 1
    assert r["total_gpus"] is None
