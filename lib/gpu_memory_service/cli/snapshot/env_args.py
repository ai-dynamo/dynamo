# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Argparse helpers for checkpoint CLIs that keep env fallback compatibility."""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any, Callable

logger = logging.getLogger(__name__)


def arg_or_env(
    parser: argparse.ArgumentParser,
    value: Any,
    env_name: str,
    *,
    coerce: Callable[[Any], Any] | None = None,
    default: Any = None,
    required: bool = False,
    required_flag: str = "",
) -> Any:
    if value is not None:
        return _coerce_arg(parser, value, env_name, coerce)
    env_value = os.environ.get(env_name)
    if env_value is not None:
        resolved = _coerce_arg(parser, env_value, env_name, coerce)
        logger.info("Using %s from environment", env_name)
        return resolved
    if required:
        parser.error(f"{required_flag} is required when {env_name} is not set")
    return _coerce_arg(parser, default, env_name, coerce)


def _coerce_arg(
    parser: argparse.ArgumentParser,
    value: Any,
    env_name: str,
    coerce: Callable[[Any], Any] | None,
) -> Any:
    if coerce is None:
        return value
    try:
        return coerce(value)
    except (TypeError, ValueError) as exc:
        parser.error(f"invalid value for {env_name}: {value!r} ({exc})")
