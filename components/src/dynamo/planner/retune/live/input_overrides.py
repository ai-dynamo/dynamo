# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalize DGDR input override aliases before interpreting their merge patch."""

from collections.abc import Mapping
from typing import Any


def expand_input_override_nulls(value: Any) -> Any:
    """Expand exact ``{"$null": true}`` objects without mutating the stored JSON."""

    if isinstance(value, Mapping):
        if len(value) == 1 and value.get("$null") is True:
            return None
        return {
            key: expand_input_override_nulls(member) for key, member in value.items()
        }
    if isinstance(value, list):
        return [expand_input_override_nulls(member) for member in value]
    return value
