# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic dictionary utilities for Dynamo.

This module provides common dictionary manipulation functions used across
multiple backends.
"""


def deep_update(target: dict, source: dict) -> None:
    """Recursively update nested dictionaries.

    Args:
        target: Dictionary to update (modified in place).
        source: Dictionary with new values.
    """
    for key, value in source.items():
        if isinstance(value, dict) and key in target and isinstance(target[key], dict):
            deep_update(target[key], value)
        else:
            target[key] = value
