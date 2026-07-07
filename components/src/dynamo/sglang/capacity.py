# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dynamo.common.native_offloading import native_offloading_capacity


@dataclass(frozen=True)
class RuntimeCapacity:
    total_kv_blocks: int | None
    max_num_seqs: int | None
    max_num_batched_tokens: int | None
    data_parallel_start_rank: int
    data_parallel_size: int


def local_dp_rank_bounds(server_args: Any) -> tuple[int, int]:
    dp_size = getattr(server_args, "dp_size", 1) or 1
    enable_dp_attention = getattr(server_args, "enable_dp_attention", False)
    nnodes = getattr(server_args, "nnodes", 1) or 1
    node_rank = getattr(server_args, "node_rank", 0) or 0

    if enable_dp_attention and dp_size > 1:
        local_dp_size = dp_size // nnodes if nnodes > 0 else dp_size
        start_dp_rank = node_rank * local_dp_size
        return start_dp_rank, start_dp_rank + local_dp_size

    return 0, 1


def model_card_dp_rank_bounds(server_args: Any) -> tuple[int, int]:
    dp_size = getattr(server_args, "dp_size", 1) or 1
    return 0, dp_size


def per_rank_max_running_requests(server_args: Any) -> int | None:
    max_running_requests = getattr(server_args, "max_running_requests", None)
    if max_running_requests is None:
        return None

    dp_size = getattr(server_args, "dp_size", 1) or 1
    if dp_size <= 1:
        return max_running_requests

    return max_running_requests // dp_size


def tokens_to_kv_blocks(tokens: int, page_size: int | None) -> int:
    if not page_size or page_size <= 1:
        return tokens

    return (tokens + page_size - 1) // page_size


def runtime_capacity(
    server_args: Any, scheduler_info: dict[str, Any]
) -> RuntimeCapacity:
    max_total_tokens = scheduler_info.get("max_total_num_tokens")
    page_size = getattr(server_args, "page_size", None)
    total_kv_blocks = (
        tokens_to_kv_blocks(max_total_tokens, page_size)
        if max_total_tokens and page_size
        else None
    )

    dp_start, dp_end = local_dp_rank_bounds(server_args)
    return RuntimeCapacity(
        total_kv_blocks=total_kv_blocks,
        max_num_seqs=per_rank_max_running_requests(server_args),
        max_num_batched_tokens=(
            getattr(server_args, "max_prefill_tokens", None) or max_total_tokens
        ),
        data_parallel_start_rank=dp_start,
        data_parallel_size=dp_end - dp_start,
    )


def kv_metrics_block_values(kv_metrics: Any, page_size: int | None) -> tuple[int, int]:
    return (
        tokens_to_kv_blocks(kv_metrics.kv_active_blocks, page_size),
        tokens_to_kv_blocks(kv_metrics.kv_total_blocks, page_size),
    )


def get_spec_decode_runtime_data(server_args: Any) -> dict[str, Any] | None:
    try:
        nextn = int(getattr(server_args, "speculative_num_steps", 0) or 0)
    except (TypeError, ValueError):
        return None
    if nextn <= 0:
        return None

    data: dict[str, Any] = {"nextn": nextn, "source": "backend_config"}
    method = getattr(server_args, "speculative_algorithm", None)
    if method:
        data["method"] = str(method)
    return data


def get_hicache_native_offloading_capacity(
    server_args: Any, scheduler_info: dict[str, Any]
) -> dict[str, int] | None:
    """Return HiCache capacity that is unique beyond the device KV pool."""
    device_capacity = native_offloading_capacity(
        scheduler_info.get("max_total_num_tokens")
    )
    if device_capacity is None:
        return None

    policy = getattr(server_args, "hicache_write_policy", None)
    if policy == "write_through_selective":
        return None

    host_capacity = None
    if "hicache_host_total_tokens" in scheduler_info:
        host_capacity = native_offloading_capacity(
            scheduler_info["hicache_host_total_tokens"]
        )
        if host_capacity is None:
            return None

    if host_capacity is None:
        if (
            not getattr(server_args, "enable_hierarchical_cache", False)
            or (getattr(server_args, "hicache_size", 0) or 0) > 0
            or policy not in ("write_back", "write_through")
            or (getattr(server_args, "dcp_size", 1) or 1) > 1
            or getattr(
                getattr(server_args, "model_config", None),
                "is_deepseek_v4_arch",
                False,
            )
        ):
            return None

        page_size = getattr(server_args, "page_size", None)
        ratio = getattr(server_args, "hicache_ratio", None)
        if not page_size:
            return None

        try:
            host_tokens = int(device_capacity["total_tokens"] * ratio)
        except (TypeError, ValueError, OverflowError):
            return None

        host_capacity = native_offloading_capacity(
            (host_tokens // page_size + 1) * page_size
        )
        if host_capacity is None:
            return None

    host_tokens = host_capacity["total_tokens"]
    if policy == "write_back":
        return host_capacity
    if policy == "write_through":
        return native_offloading_capacity(host_tokens - device_capacity["total_tokens"])
    return None
