# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Environment variable utilities for Dynamo components.

This module provides common utilities for parsing and handling environment variables.

Note: The function names in this module match their Rust counterparts in lib/runtime/src/config.rs
to maintain consistency across the codebase.
"""

import os


def env_is_truthy(name: str) -> bool:
    """
    Check if an environment variable is truthy.

    This matches the Rust function `env_is_truthy` in lib/runtime/src/config.rs.

    Args:
        name: Environment variable name

    Returns:
        True if the environment variable is set to a truthy value (1, t, true, T, True, TRUE).
        False if not set, set to a falsy value (0, f, false, F, False, FALSE), or unrecognized.

    Example:
        export DYN_ENGINE_METRICS_ENABLED=1    # True
        export DYN_ENGINE_METRICS_ENABLED=True # True
        export DYN_ENGINE_METRICS_ENABLED=t    # True
        export DYN_ENGINE_METRICS_ENABLED=0    # False
        export DYN_ENGINE_METRICS_ENABLED=false # False
    """
    value = os.getenv(name)
    if value is None:
        return False

    value = value.lower().strip()

    truthy_values = ("true", "1", "t")

    if value in truthy_values:
        return True
    else:
        # Treat anything else (including falsy values and unrecognized) as False
        return False
