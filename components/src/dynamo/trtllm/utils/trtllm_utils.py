# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for the TRT-LLM backend."""

import logging
from collections.abc import Mapping
from typing import Any

from dynamo.trtllm.constants import DisaggregationMode


def deep_update(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    """Recursively update nested dictionaries.

    Args:
        target: Dictionary to update.
        source: Dictionary with new values.
    """
    for key, value in source.items():
        if isinstance(value, dict) and key in target and isinstance(target[key], dict):
            deep_update(target[key], value)
        else:
            target[key] = value


def warn_override_collisions(
    target: Mapping[str, Any], source: Mapping[str, Any], path: str = ""
) -> None:
    """Log warnings for keys in *source* that will overwrite existing values in *target*."""
    for key, new_val in source.items():
        full_key = f"{path}.{key}" if path else key
        if key in target:
            old_val = target[key]
            if isinstance(new_val, dict) and isinstance(old_val, dict):
                warn_override_collisions(old_val, new_val, full_key)
            elif old_val != new_val:
                logging.warning(
                    "override_engine_args will replace %s: %r -> %r",
                    full_key,
                    old_val,
                    new_val,
                )


def _as_mapping(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else None
    if hasattr(value, "__dict__"):
        return vars(value)
    return None


def get_spec_decode_runtime_data(engine_args: Any) -> dict[str, Any] | None:
    args = _as_mapping(engine_args)
    if not args:
        return None
    spec = _as_mapping(args.get("speculative_config"))
    if not spec:
        return None

    raw_nextn = spec.get("max_draft_len")
    if raw_nextn is None:
        raw_nextn = spec.get("num_nextn_predict_layers")
    try:
        nextn = int(raw_nextn or 0)
    except (TypeError, ValueError):
        return None
    if nextn <= 0:
        return None

    data: dict[str, Any] = {"nextn": nextn, "source": "backend_config"}
    method = spec.get("decoding_type")
    if method:
        data["method"] = str(method)
    return data


def per_dp_rank_max_num_seqs(engine: Any, data_parallel_size: int) -> int | None:
    """Return TRT-LLM's post-initialization request capacity per attention-DP rank."""
    if (
        getattr(engine, "disaggregation_mode", None) == DisaggregationMode.ENCODE
        and not engine.encoder_available
    ):
        return None

    value = engine.llm.args.max_batch_size
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        return None
    if (
        not isinstance(data_parallel_size, int)
        or isinstance(data_parallel_size, bool)
        or data_parallel_size <= 0
    ):
        return None

    per_rank = value // data_parallel_size
    if per_rank <= 0:
        return None
    if value % data_parallel_size:
        logging.warning(
            "TRT-LLM max_batch_size=%d is not divisible by attention DP size=%d; "
            "advertising conservative per-rank max_num_seqs=%d",
            value,
            data_parallel_size,
            per_rank,
        )
    return per_rank
