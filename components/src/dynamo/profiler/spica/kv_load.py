# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve candidate-relative synthetic load from AIC KV-cache capacity."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Any

from .config import Workload
from .kv_estimate import estimate_kv_tokens
from .parallel_enum import DisaggParallelConfig, ParallelShape, ReplicaParallelConfig


class InfeasibleKVCapacity(ValueError):
    """A candidate's selected shape and batching leave no usable KV capacity."""


@dataclass(frozen=True)
class KVLoadResolution:
    """The requested normalized load and the concrete closed-loop concurrency."""

    ratio: float
    concurrency: int
    concurrency_capacity: int
    role_capacity_tokens: dict[str, int]


@cache
def _per_rank_capacity_tokens(
    shape: ParallelShape,
    *,
    model_name: str,
    hardware_sku: str,
    backend: str,
    backend_version: str,
    max_num_tokens: int,
    max_batch_size: int,
    memory_fraction: float,
    nextn: int,
) -> int:
    tokens = estimate_kv_tokens(
        shape,
        model_name=model_name,
        hardware_sku=hardware_sku,
        backend=backend,
        backend_version=backend_version,
        max_num_tokens=max_num_tokens,
        max_batch_size=max_batch_size,
        memory_fraction=memory_fraction,
        nextn=nextn,
    )
    if tokens is None:
        raise InfeasibleKVCapacity(
            f"no KV budget for backend={backend}, shape={shape}, "
            f"max_num_batched_tokens={max_num_tokens}, max_num_seqs={max_batch_size}"
        )
    return tokens


def _role_capacity_tokens(
    sample: dict[str, Any],
    *,
    role: str,
    config: ReplicaParallelConfig,
    backend_version: str,
) -> int:
    """Aggregate scheduler-visible KV tokens across attention-DP ranks and replicas."""
    block_size = int(sample[f"{role}_block_size"])
    per_rank_tokens = _per_rank_capacity_tokens(
        config.shape,
        model_name=str(sample["model_name"]),
        hardware_sku=str(sample["hardware_sku"]),
        backend=str(sample["backend"]),
        backend_version=backend_version,
        max_num_tokens=int(sample[f"{role}_max_num_batched_tokens"]),
        max_batch_size=int(sample[f"{role}_max_num_seqs"]),
        memory_fraction=float(sample[f"{role}_gpu_memory_utilization"]),
        nextn=int(sample.get("aic_nextn") or 0),
    )
    # Dynamo's AIC estimator returns per-rank blocks. Offline replay models one
    # engine-wide KV pool, so attention-DP ranks contribute independent capacity;
    # tensor/expert parallel ranks shard the same sequences and are not multipliers.
    per_rank_usable_tokens = (per_rank_tokens // block_size) * block_size
    return per_rank_usable_tokens * config.shape.dp * config.replicas


def resolve_kv_load(
    sample: dict[str, Any],
    *,
    workload: Workload,
    parallel_config: ReplicaParallelConfig | DisaggParallelConfig,
    ratio: float,
    backend_version: str,
) -> KVLoadResolution:
    """Map a normalized KV load to candidate-specific closed-loop concurrency.

    ``ratio=1`` is the estimated steady-state KV occupancy where each in-flight
    request holds ``isl + floor(osl / 2)`` tokens on average. ``ratio=0`` is the
    minimum useful replay load and therefore maps to one request.
    """
    if workload.isl is None or workload.osl is None:
        raise ValueError("kv_load_ratio requires a synthetic workload with isl and osl")

    if isinstance(parallel_config, DisaggParallelConfig):
        role_configs = {
            "prefill": parallel_config.prefill,
            "decode": parallel_config.decode,
        }
        load_role = "decode"
    elif isinstance(parallel_config, ReplicaParallelConfig):
        role_configs = {"agg": parallel_config}
        load_role = "agg"
    else:
        raise TypeError(
            f"unsupported parallel config for KV load: {type(parallel_config).__name__}"
        )

    capacities = {
        role: _role_capacity_tokens(
            sample, role=role, config=config, backend_version=backend_version
        )
        for role, config in role_configs.items()
    }
    expected_tokens_per_request = int(workload.isl) + int(workload.osl) // 2
    concurrency_capacity = capacities[load_role] // expected_tokens_per_request
    if concurrency_capacity < 1:
        raise InfeasibleKVCapacity(
            f"{load_role} KV capacity {capacities[load_role]} tokens cannot hold the "
            f"estimated {expected_tokens_per_request} tokens per in-flight request"
        )
    concurrency = max(1, int(float(ratio) * concurrency_capacity))
    return KVLoadResolution(
        ratio=float(ratio),
        concurrency=concurrency,
        concurrency_capacity=concurrency_capacity,
        role_capacity_tokens=capacities,
    )
