# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for parsing environment variables."""

import os

# Same spellings as the Rust runtime's `lib/truthy` `is_truthy`, so a variable read by
# both sides of one worker process (for example DYN_LORA_ENABLED) means the same thing.
_TRUTHY = ("1", "true", "on", "yes")


def env_bool(name: str, *, default: bool = False) -> bool:
    """Return True if env var `name` is set to a truthy value.

    Truthy values (case-insensitive, surrounding whitespace ignored): "1",
    "true", "on", "yes". Any other non-empty value is treated as False. When
    the var is unset or blank, returns `default`.
    """
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    return raw.lower() in _TRUTHY
