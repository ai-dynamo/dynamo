# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional


def _as_mapping(value: Any) -> Optional[dict[str, Any]]:
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


def get_spec_decode_runtime_data(engine_args: Any) -> Optional[dict[str, Any]]:
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
