# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for parsing environment variables."""

import os

_TRUTHY = ("true", "1", "yes")


def env_bool(name: str, *, default: bool = False) -> bool:
    """Return True if env var `name` is set to a truthy value.

    Truthy values (case-insensitive): "true", "1", "yes". Any other non-empty
    value is treated as False. When the var is unset or empty, returns `default`.
    """
    raw = os.environ.get(name)
    if not raw:
        return default
    return raw.lower() in _TRUTHY
