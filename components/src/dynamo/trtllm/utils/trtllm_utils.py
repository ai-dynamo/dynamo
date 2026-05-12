# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for the TRT-LLM backend."""

import logging
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
