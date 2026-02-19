# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for the TRT-LLM backend."""

from collections.abc import Mapping
from typing import Any


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
