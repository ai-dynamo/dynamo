# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for parsing dynamic ``--trtllm.*`` CLI flags into nested dicts."""

import logging
import sys
from typing import Any, Dict, List

DYNAMIC_FLAG_PREFIX = "--trtllm."


def infer_type(value: str) -> Any:
    """Infer the Python type of a CLI value string.

    Tries int, float, bool, then falls back to string.
    """
    # int
    try:
        return int(value)
    except ValueError:
        pass
    # float
    try:
        return float(value)
    except ValueError:
        pass
    # bool
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    # string
    return value


def set_nested(d: Dict[str, Any], keys: List[str], value: Any) -> None:
    """Set a value in a nested dict, creating intermediate dicts as needed."""
    current: Dict[str, Any] = d
    for key in keys[:-1]:
        existing = current.get(key)
        if existing is None:
            current[key] = {}
            current = current[key]
        elif not isinstance(existing, dict):
            raise ValueError(
                f"Conflicting dynamic flag path: key '{key}' is already set "
                f"to a {type(existing).__name__} value"
            )
        else:
            current = existing
    current[keys[-1]] = value


def parse_dynamic_flags(remaining: List[str]) -> dict:
    """Parse ``--trtllm.a.b.c value`` flags into a nested dict.

    Returns the nested dict built from all ``--trtllm.*`` flags.
    Raises ``SystemExit`` if a flag has no value or if unknown flags remain.
    """
    result: Dict[str, Any] = {}
    i = 0
    while i < len(remaining):
        arg = remaining[i]
        if not arg.startswith(DYNAMIC_FLAG_PREFIX):
            logging.error("Unrecognized argument: %s", arg)
            sys.exit(1)

        dotted_key = arg[len(DYNAMIC_FLAG_PREFIX) :]
        keys = dotted_key.split(".")
        if not all(keys):
            logging.error("Invalid dynamic flag (empty key segment): %s", arg)
            sys.exit(1)

        i += 1
        if i >= len(remaining) or remaining[i].startswith("--"):
            logging.error("Dynamic flag %s requires a value", arg)
            sys.exit(1)

        value = infer_type(remaining[i])
        try:
            set_nested(result, keys, value)
        except ValueError as e:
            logging.error("%s", e)
            sys.exit(1)
        i += 1

    return result
