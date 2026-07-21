# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for the TRT-LLM backend."""

import dataclasses
import logging
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any


def deep_update(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    """Recursively update nested dictionaries and model-like config values.

    Args:
        target: Dictionary to update.
        source: Dictionary with new values.
    """
    for key, value in source.items():
        if isinstance(value, Mapping) and key in target:
            old_val = target[key]
            if isinstance(old_val, dict):
                deep_update(old_val, value)
                continue

            old_mapping = _as_mapping(old_val)
            if old_mapping is not None:
                merged = dict(old_mapping)
                deep_update(merged, value)
                target[key] = merged
                continue

        target[key] = value


def warn_override_collisions(
    target: Mapping[str, Any], source: Mapping[str, Any], path: str = ""
) -> None:
    """Log warnings for keys in *source* that will overwrite existing values in *target*."""
    for key, new_val in source.items():
        full_key = f"{path}.{key}" if path else key
        if key in target:
            old_val = target[key]
            old_mapping = _as_mapping(old_val)
            new_mapping = _as_mapping(new_val)
            if old_mapping is not None and new_mapping is not None:
                warn_override_collisions(old_mapping, new_mapping, full_key)
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
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, SimpleNamespace):
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
