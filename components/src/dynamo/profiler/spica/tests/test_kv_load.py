# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import dynamo.profiler.spica.kv_estimate as kv_estimate_mod
from dynamo._internal.aic import AicMemoryEstimatorUnavailableError
from dynamo.profiler.spica.config import Workload
from dynamo.profiler.spica.kv_load import InfeasibleKVCapacity, resolve_kv_load
from dynamo.profiler.spica.parallel_enum import (
    DisaggParallelConfig,
    ParallelShape,
    ReplicaParallelConfig,
)


def _sample(mode: str) -> dict:
    sample = {
        "deployment_mode": mode,
        "model_name": "m",
        "hardware_sku": "h",
        "backend": "vllm",
        "aic_nextn": None,
    }
    roles = ("agg",) if mode == "agg" else ("prefill", "decode")
    for role in roles:
        sample.update(
            {
                f"{role}_block_size": 64,
                f"{role}_max_num_batched_tokens": 8192,
                f"{role}_max_num_seqs": 256,
                f"{role}_gpu_memory_utilization": 0.9,
            }
        )
    return sample


def test_agg_capacity_scales_by_attention_dp_and_replicas(monkeypatch):
    shape = ParallelShape(tp=1, dp=2, moe_tp=1, moe_ep=2)
    config = ReplicaParallelConfig(shape=shape, replicas=3)
    monkeypatch.setattr(
        "dynamo.profiler.spica.kv_load._per_rank_capacity_tokens",
        lambda *args, **kwargs: 10_000,
    )

    resolution = resolve_kv_load(
        _sample("agg"),
        workload=Workload(isl=100, osl=100, kv_load_ratio=0.5, num_request_ratio=10),
        parallel_config=config,
        ratio=0.5,
        backend_version="v",
    )

    # floor(10000 / 64) * 64 per rank, then x2 attention-DP ranks x3 replicas.
    assert resolution.role_capacity_tokens == {"agg": 59_904}
    assert resolution.concurrency_capacity == 399  # 59904 / (100 + 100/2)
    assert resolution.concurrency == 199


def test_disagg_load_uses_decode_capacity_but_validates_prefill(monkeypatch):
    prefill = ReplicaParallelConfig(
        ParallelShape(tp=2, dp=1, moe_tp=1, moe_ep=2), replicas=1
    )
    decode = ReplicaParallelConfig(
        ParallelShape(tp=1, dp=4, moe_tp=1, moe_ep=4), replicas=2
    )
    config = DisaggParallelConfig(prefill=prefill, decode=decode)
    seen = []

    def fake_role(sample, *, role, config, backend_version):
        seen.append(role)
        return {"prefill": 100_000, "decode": 300_000}[role]

    monkeypatch.setattr(
        "dynamo.profiler.spica.kv_load._role_capacity_tokens", fake_role
    )
    resolution = resolve_kv_load(
        _sample("disagg"),
        workload=Workload(isl=1000, osl=1000, kv_load_ratio=1.0, num_request_ratio=10),
        parallel_config=config,
        ratio=1.0,
        backend_version="v",
    )

    assert seen == ["prefill", "decode"]
    assert resolution.concurrency_capacity == 200  # decode 300k / (1000 + 500)
    assert resolution.concurrency == 200


def test_zero_ratio_maps_to_one_request(monkeypatch):
    config = ReplicaParallelConfig(
        ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1), replicas=1
    )
    monkeypatch.setattr(
        "dynamo.profiler.spica.kv_load._role_capacity_tokens",
        lambda *args, **kwargs: 10_000,
    )

    resolution = resolve_kv_load(
        _sample("agg"),
        workload=Workload(isl=100, osl=100, kv_load_ratio=0.0, num_request_ratio=10),
        parallel_config=config,
        ratio=0.0,
        backend_version="v",
    )

    assert resolution.concurrency == 1


def test_capacity_smaller_than_one_average_request_is_infeasible(monkeypatch):
    config = ReplicaParallelConfig(
        ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1), replicas=1
    )
    monkeypatch.setattr(
        "dynamo.profiler.spica.kv_load._role_capacity_tokens",
        lambda *args, **kwargs: 100,
    )

    with pytest.raises(InfeasibleKVCapacity, match="cannot hold"):
        resolve_kv_load(
            _sample("agg"),
            workload=Workload(
                isl=100, osl=100, kv_load_ratio=1.0, num_request_ratio=10
            ),
            parallel_config=config,
            ratio=1.0,
            backend_version="v",
        )


def test_kv_load_fails_closed_without_memory_estimator(monkeypatch):
    config = ReplicaParallelConfig(
        ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1), replicas=1
    )

    def missing_memory(module_name):
        raise ModuleNotFoundError(name=module_name)

    monkeypatch.setattr(kv_estimate_mod.importlib, "import_module", missing_memory)

    with pytest.raises(
        AicMemoryEstimatorUnavailableError,
        match=r"compatible estimator.*AIC 0\.10",
    ):
        resolve_kv_load(
            _sample("agg"),
            workload=Workload(
                isl=100, osl=100, kv_load_ratio=1.0, num_request_ratio=10
            ),
            parallel_config=config,
            ratio=1.0,
            backend_version="v",
        )
