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


def set_nested(d: dict, keys: List[str], value: Any) -> None:
    """Set a value in a nested dict, creating intermediate dicts as needed."""
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


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
        set_nested(result, keys, value)
        i += 1

    return result
