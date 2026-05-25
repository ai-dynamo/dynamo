# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""POC tests (Phase-3 M1): per-pool curves -> grid -> multi-pool DGD. Uses
injected synthetic curves so the AIC dependency isn't needed on the host."""

import pytest

from dynamo.global_router.pool_grid_builder import (
    POOL_LABEL,
    PoolSpec,
    build_multipool_dgd,
    build_pool_grid,
    pool_selector_params,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.parallel,
    pytest.mark.unit,
]

# Synthetic prefill curves (ms) mimicking the real AIC shape: TTFT = base + slope*ISL,
# bigger pools faster. TP1 cheap-but-slow-on-long, TP4 fast.
_CURVES = [
    lambda isl: 50 + 0.20 * isl,  # tp1
    lambda isl: 40 + 0.10 * isl,  # tp2
    lambda isl: 30 + 0.04 * isl,  # tp4
]
_TPS = [1, 2, 4]
_BINS = dict(
    isl_max=4000, ttft_min_ms=0, ttft_max_ms=300, isl_resolution=2, ttft_resolution=3
)


def test_grid_is_cost_efficient():
    grid = build_pool_grid(_TPS, _CURVES, **_BINS)
    assert len(grid) == 2 and all(len(r) == 3 for r in grid)
    # Short ISL band (mid 1000): tp1 TTFT=250 — meets the loose 300ms target →
    # cheapest tp1 in the loosest column; tighter columns escalate.
    assert grid[0][-1] == 0  # loosest TTFT, short ISL -> TP1 (index 0)
    # Long ISL band (mid 3000): tp1 TTFT=650 (misses all), tp4=150 → tightest
    # column must be the fastest pool (tp4 = index 2).
    assert grid[1][0] == 2


def test_selector_params_shape():
    grid = build_pool_grid(_TPS, _CURVES, **_BINS)
    p = pool_selector_params(_TPS, grid, **_BINS)
    assert p["poolLabel"] == POOL_LABEL
    assert p["poolLabels"] == ["tp1", "tp2", "tp4"]
    assert p["mapping"] == grid
    assert p["sizeResolution"] == 2 and p["latencyResolution"] == 3


def test_multipool_dgd_structure():
    grid = build_pool_grid(_TPS, _CURVES, **_BINS)
    params = pool_selector_params(_TPS, grid, **_BINS)
    dgd = build_multipool_dgd(
        name="q32",
        model="Qwen/Qwen3-32B",
        runtime_image="img/vllm:tag",
        epp_image="img/epp:tag",
        pools=[PoolSpec(tp=1, replicas=2), PoolSpec(tp=2), PoolSpec(tp=4)],
        selector_params=params,
        gateway_name="inference-gateway",
        gateway_namespace="gw-ns",
    )
    svcs = dgd["spec"]["services"]
    # one EPP + one decode worker per pool
    assert set(svcs) == {
        "Epp",
        "VllmDecodeWorkerTp1",
        "VllmDecodeWorkerTp2",
        "VllmDecodeWorkerTp4",
    }

    # EPP carries the pool-selector with our grid, first in the decode profile.
    plugins = svcs["Epp"]["eppConfig"]["config"]["plugins"]
    ps = next(p for p in plugins if p.get("name") == "pool-selector")
    assert ps["parameters"]["mapping"] == grid
    profile = svcs["Epp"]["eppConfig"]["config"]["schedulingProfiles"][0]["plugins"]
    assert profile[0]["pluginRef"] == "pool-selector"

    # each worker: correct pool label, TP size, GPU count, replicas.
    w1 = svcs["VllmDecodeWorkerTp1"]
    assert w1["extraPodMetadata"]["labels"][POOL_LABEL] == "tp1"
    assert w1["replicas"] == 2
    assert w1["resources"]["limits"]["gpu"] == "1"
    assert "--tensor-parallel-size 1" in w1["extraPodSpec"]["mainContainer"]["args"][0]
    w4 = svcs["VllmDecodeWorkerTp4"]
    assert w4["resources"]["limits"]["gpu"] == "4"
    assert "--tensor-parallel-size 4" in w4["extraPodSpec"]["mainContainer"]["args"][0]

    # gateway annotation handoff set.
    ann = dgd["metadata"]["annotations"]
    assert ann["nvidia.com/inference-gateway-name"] == "inference-gateway"
    assert ann["nvidia.com/inference-gateway-namespace"] == "gw-ns"
