# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def engine_max_num_seqs(max_num_seqs: int | None, local_dp_size: int) -> int | None:
    """Normalize vLLM's per-rank scheduler limit to this worker's DP scope."""
    if (
        not isinstance(max_num_seqs, int)
        or isinstance(max_num_seqs, bool)
        or max_num_seqs <= 0
    ):
        return None
    return max_num_seqs * max(local_dp_size, 1)


def per_rank_kv_blocks(
    total_kv_blocks: int | None, data_parallel_size: int
) -> int | None:
    if total_kv_blocks is None:
        return None

    if data_parallel_size <= 1 or total_kv_blocks <= 0:
        return total_kv_blocks

    per_rank = total_kv_blocks // data_parallel_size
    if per_rank == 0:
        logger.warning(
            "vLLM reported fewer total KV blocks (%s) than DP ranks (%s); "
            "publishing 1 block per rank",
            total_kv_blocks,
            data_parallel_size,
        )
        return 1

    remainder = total_kv_blocks % data_parallel_size
    if remainder:
        logger.warning(
            "vLLM reported aggregate KV blocks (%s) not divisible by DP ranks (%s); "
            "publishing floor per-rank capacity %s",
            total_kv_blocks,
            data_parallel_size,
            per_rank,
        )

    return per_rank


def get_metrics_model_name(config: Any) -> str:
    return str(
        getattr(config, "served_model_name", None) or getattr(config, "model", "")
    )


def _as_mapping(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else None
    if hasattr(value, "__dict__"):
        return vars(value)
    return None


def get_spec_decode_runtime_data(
    config: Any, vllm_config: Any
) -> dict[str, Any] | None:
    spec_config = getattr(vllm_config, "speculative_config", None)
    if spec_config is None:
        engine_args = getattr(config, "engine_args", None)
        spec_config = getattr(engine_args, "speculative_config", None)
    spec = _as_mapping(spec_config)
    if not spec:
        return None

    try:
        nextn = int(spec.get("num_speculative_tokens") or 0)
    except (TypeError, ValueError):
        return None
    if nextn <= 0:
        return None

    data: dict[str, Any] = {"nextn": nextn, "source": "backend_config"}
    method = spec.get("method")
    if method:
        data["method"] = str(method)
    return data
