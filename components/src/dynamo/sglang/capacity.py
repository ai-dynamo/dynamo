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


def _positive_int(value: Any) -> int:
    # Older SGLang exposes attn_dp_size as a derived-field descriptor, not an int.
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return 1


def sglang_dp_layout(server_args: Any) -> tuple[int, bool]:
    """Return ``(dp_size, enable_dp_attention)`` in the meaning Dynamo routes by.

    ``dp_size`` is the number of SGLang DP ranks (one scheduler and one KV cache
    each), and ``enable_dp_attention`` says whether they are attention-DP groups
    inside one TP group rather than independent replicas.

    Before SGLang #41818 (2026-09-30), ``--dp-size N --enable-dp-attention`` left
    those two fields as given. Since then the pair is deprecated and resolves to
    ``attn_dp_size=N`` with ``enable_dp_attention=False``; ``dp_size`` counts
    replicas only, and SGLang has ``num_dp_ranks = dp_size * attn_dp_size``.
    Reading only the old fields makes an attention-DP worker look like one rank,
    so the router subscribes to rank 0's KV events and sends it every request.
    """
    dp_size = _positive_int(getattr(server_args, "dp_size", 1))
    if getattr(server_args, "enable_dp_attention", False):
        return dp_size, True
    attn_dp_size = _positive_int(getattr(server_args, "attn_dp_size", 1))
    if attn_dp_size > 1:
        return dp_size * attn_dp_size, True
    return dp_size, False


def local_dp_rank_bounds(server_args: Any) -> tuple[int, int]:
    dp_size, enable_dp_attention = sglang_dp_layout(server_args)
    nnodes = getattr(server_args, "nnodes", 1) or 1
    node_rank = getattr(server_args, "node_rank", 0) or 0

    if not enable_dp_attention:
        return 0, dp_size

    if dp_size > 1:
        local_dp_size = dp_size // nnodes if nnodes > 0 else dp_size
        start_dp_rank = node_rank * local_dp_size
        return start_dp_rank, start_dp_rank + local_dp_size

    return 0, 1


def publishes_kv_events(server_args: Any) -> bool:
    """Whether this node should advertise a KV-event source.

    The router keys KV sources by ``(worker_id, dp_rank)``, and non-leader nodes
    publish under the leader's worker ID so the router-visible trees stay keyed
    to one logical worker. That only yields a unique key per node while DP
    attention gives each node a distinct rank slice.

    Pure DP is single-node in SGLang, so its leader publishes every replica's
    distinct rank. Only the leader owns the single logical rank in multinode
    TP-only mode.
    """
    dp_size, enable_dp_attention = sglang_dp_layout(server_args)
    nnodes = getattr(server_args, "nnodes", 1) or 1
    node_rank = getattr(server_args, "node_rank", 0) or 0

    if enable_dp_attention and dp_size > 1:
        return True

    return not (nnodes > 1 and node_rank > 0)


def model_card_dp_rank_bounds(server_args: Any) -> tuple[int, int]:
    dp_size, _ = sglang_dp_layout(server_args)
    return 0, dp_size


def per_rank_max_running_requests(server_args: Any) -> int | None:
    max_running_requests = getattr(server_args, "max_running_requests", None)
    if max_running_requests is None:
        return None

    dp_size, enable_dp_attention = sglang_dp_layout(server_args)
    if dp_size <= 1 or not enable_dp_attention:
        return max_running_requests

    return max_running_requests // dp_size


def tokens_to_kv_blocks(tokens: int, page_size: int | None) -> int:
    if not page_size or page_size <= 1:
        return tokens

    return (tokens + page_size - 1) // page_size


def kv_event_block_size(server_args: Any) -> int:
    """Return the router-facing block size for SGLang's paged KV allocator.

    Under DCP, SGLang widens the allocator and radix-tree page size to
    ``page_size * dcp_size``; the tree emits KV events at that granularity.
    """
    page_size = int(server_args.page_size)
    # Older SGLang configs and non-LLM argument stubs may omit dcp_size.
    dcp_size = int(getattr(server_args, "dcp_size", 1) or 1)
    return page_size * dcp_size


def runtime_capacity(
    server_args: Any, scheduler_info: dict[str, Any]
) -> RuntimeCapacity:
    max_total_tokens = scheduler_info.get("max_total_num_tokens")
    page_size = getattr(server_args, "page_size", None)
    # SGLang reports tokens per DCP rank here, so division by the physical
    # page size already produces the widened logical-block count.
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

    host_tokens = scheduler_info.get("hicache_host_total_tokens")
    policy = getattr(server_args, "hicache_write_policy", None)
    if "hicache_host_total_tokens" not in scheduler_info:
        model_config = getattr(server_args, "model_config", None)
        if (
            not getattr(server_args, "enable_hierarchical_cache", False)
            or getattr(server_args, "hicache_size", None) != 0
            or policy not in ("write_back", "write_through")
            or (getattr(server_args, "dcp_size", 1) or 1) > 1
            or getattr(model_config, "is_deepseek_v4_arch", False)
        ):
            return None
        page_size = getattr(server_args, "page_size", None)
        ratio = getattr(server_args, "hicache_ratio", None)
        if not page_size or ratio is None:
            return None
        try:
            host_tokens = int(device_capacity["total_tokens"] * ratio)
            # Match SGLang HostKVCache's realized ratio-based allocation.
            host_tokens = (host_tokens // page_size + 1) * page_size
        except (TypeError, ValueError, OverflowError):
            return None

    host_capacity = native_offloading_capacity(host_tokens)
    if host_capacity is None:
        return None

    host_tokens = host_capacity["total_tokens"]
    # The router already counts device capacity: write-back adds the disjoint host
    # pool, write-through subtracts its device mirror, and selective overlap is dynamic.
    if policy == "write_back":
        return host_capacity
    if policy == "write_through":
        return native_offloading_capacity(host_tokens - device_capacity["total_tokens"])
    return None
