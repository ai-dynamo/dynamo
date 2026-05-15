# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Argparse helpers for checkpoint CLIs that keep env fallback compatibility."""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def arg_or_env(
    parser: argparse.ArgumentParser,
    value: Any,
    env_name: str,
    *,
    default: Any = None,
    required: bool = False,
    required_flag: str = "",
) -> Any:
    if value is not None:
        return value
    env_value = os.environ.get(env_name)
    if env_value is not None:
        logger.info("Using %s from environment", env_name)
        return env_value
    if required:
        parser.error(f"{required_flag} is required when {env_name} is not set")
    return default
