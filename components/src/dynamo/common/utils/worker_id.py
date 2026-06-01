# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stable worker identity helpers shared by backend integrations."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

WORKER_ID_PLACEMENT_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "ZE_AFFINITY_MASK",
    "HABANA_VISIBLE_DEVICES",
    "ASCEND_RT_VISIBLE_DEVICES",
)

WORKER_ID_CONFIG_KEYS = (
    "namespace",
    "component",
    "endpoint",
    "model",
    "model_path",
    "served_model_name",
    "disaggregation_mode",
    "tensor_parallel_size",
    "tp_size",
    "data_parallel_size",
    "dp_size",
    "nnodes",
    "node_rank",
)


def _stable_snapshot(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _stable_snapshot(asdict(value))
    if isinstance(value, dict):
        return {
            str(key): _stable_snapshot(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_stable_snapshot(item) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return _stable_snapshot(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return {
            key: _stable_snapshot(item)
            for key, item in sorted(vars(value).items())
            if not key.startswith("_") and not callable(item)
        }
    return str(value)


def _stable_worker_identity(config: Any) -> dict[str, Any]:
    return {
        key: _stable_snapshot(value)
        for key in WORKER_ID_CONFIG_KEYS
        if hasattr(config, key) and (value := getattr(config, key)) is not None
    }


def make_fpm_worker_id(config: Any) -> str:
    """Return a deterministic numeric FPM worker id for a backend config.

    The id must stay stable across process restarts for the same logical worker
    because vLLM includes additional_config in its config hash; changing this
    runtime-only value can prevent compiled graph reuse. Hash only stable worker
    identity fields plus placement so transient config internals do not perturb
    the value, while different workers still get distinct FPM identities.
    """
    placement = {
        key: value
        for key in WORKER_ID_PLACEMENT_ENV_KEYS
        if (value := os.environ.get(key))
    }
    payload = {
        "config": _stable_worker_identity(config),
        "placement": placement,
    }
    logger.debug(f"construct fpm_worker_id with payload: {payload}")
    digest = hashlib.blake2b(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(),
        digest_size=8,
    ).hexdigest()
    return str(int(digest, 16))