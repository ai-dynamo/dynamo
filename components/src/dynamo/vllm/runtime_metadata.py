# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any, Optional


def get_metrics_model_name(config: Any) -> str:
    return str(getattr(config, "served_model_name", None) or getattr(config, "model", ""))


def _as_mapping(value: Any) -> Optional[dict[str, Any]]:
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
) -> Optional[dict[str, Any]]:
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
