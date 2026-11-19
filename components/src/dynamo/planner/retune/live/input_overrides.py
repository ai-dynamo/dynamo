# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalize DGDR input override aliases before interpreting their merge patch."""

import json
from collections.abc import Mapping
from typing import Any


def expand_input_override_nulls(value: Any) -> Any:
    """Validate object-field deletion markers without mutating the stored JSON."""

    if value is not None and not isinstance(value, Mapping):
        raise ValueError("spec.aisim.inputOverrides: must be a JSON object")
    return _expand_null_sentinels(value, "", False)


def _expand_null_sentinels(value: Any, path: str, in_array: bool) -> Any:
    if isinstance(value, Mapping):
        if "$null" in value:
            location = f"spec.aisim.inputOverrides{path}"
            if not path:
                raise ValueError(
                    f"{location}: $null is not allowed at the document root"
                )
            if in_array:
                raise ValueError(f"{location}: $null is not allowed inside arrays")
            if len(value) != 1 or value["$null"] is not True:
                raise ValueError(
                    f"{location}: $null must be the boolean true and the only object field"
                )
            return None
        return {
            key: _expand_null_sentinels(
                value[key], f"{path}[{json.dumps(key)}]", in_array
            )
            for key in sorted(value)
        }
    if isinstance(value, list):
        return [
            _expand_null_sentinels(member, f"{path}[{i}]", True)
            for i, member in enumerate(value)
        ]
    return value
