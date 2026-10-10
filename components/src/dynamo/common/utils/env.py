# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for parsing environment variables."""

import logging
import os

logger = logging.getLogger(__name__)

_TRUTHY = ("true", "1", "yes")


def env_bool(name: str, *, default: bool = False) -> bool:
    """Return True if env var `name` is set to a truthy value.

    Truthy values (case-insensitive): "true", "1", "yes". Any other
    non-empty value is treated as False. When the var is unset or empty,
    returns `default`.
    """
    raw = os.environ.get(name)
    if not raw:
        return default
    return raw.strip().lower() in _TRUTHY


def env_set_unless_false(name: str) -> bool:
    """True when `name` is set to anything that is not explicitly falsey.

    Use this only for guards that refuse a banned or removed option. There the
    permissive reading is the safe one: `env_bool` would let a typo such as
    `FOO=enabled` slip past the guard, while this reports it as set.
    """
    raw = os.environ.get(name)
    if raw is None:
        return False
    return raw.strip().lower() not in ("", "0", "false", "no", "off")


def env_int(name: str, default: int) -> int:
    """Return env var `name` as an int, or `default` when unset or invalid."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %d", name, raw, default)
        return default


def env_float(name: str, default: float) -> float:
    """Return env var `name` as a float, or `default` when unset or invalid."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %s", name, raw, default)
        return default
