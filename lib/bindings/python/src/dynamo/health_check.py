#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""
Health check utilities for Dynamo backends.

This module provides a base class for backend-specific health check payloads.
Each backend should extend HealthCheckPayload and define its default payload.
"""

import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def load_health_check_from_env(
    env_var: str = "DYN_HEALTH_CHECK_PAYLOAD",
) -> Optional[Dict[str, Any]]:
    """
    Load health check payload from environment variable.

    Supports two formats:
    1. JSON string: export DYN_HEALTH_CHECK_PAYLOAD='{"prompt": "test", "max_tokens": 1}'
    2. File path: export DYN_HEALTH_CHECK_PAYLOAD='@/path/to/health_check.json'

    Args:
        env_var: Name of the environment variable to check (default: DYN_HEALTH_CHECK_PAYLOAD)

    Returns:
        Dict containing the health check payload, or None if not set.
    """
    env_value = os.environ.get(env_var)
    if not env_value:
        return None

    try:
        if env_value.startswith("@"):
            # Load from file
            file_path = env_value[1:]
            with open(file_path, "r") as f:
                parsed = json.load(f)
        else:
            # Parse as JSON
            parsed = json.loads(env_value)
        if not isinstance(parsed, dict):
            logger.warning(
                "%s must be a JSON object (dict). Got: %s",
                env_var,
                type(parsed).__name__,
            )
            return None
        return parsed
    except (json.JSONDecodeError, FileNotFoundError, OSError) as e:
        logger.warning("Failed to parse %s: %s", env_var, e)
        return None


class HealthCheckPayload:
    """
    Base class for managing health check payloads.

    Each backend should extend this class and set self.default_payload
    in their __init__ method.

    Environment variable DYN_HEALTH_CHECK_PAYLOAD can override the default.
    """

    default_payload: Dict[str, Any]  # Type hint for mypy - set by subclasses

    def __init__(self):
        """
        Initialize health check payload.

        Subclasses should call super().__init__() after setting self.default_payload.
        """
        if not hasattr(self, "default_payload"):
            raise NotImplementedError(
                "Subclass must set self.default_payload before calling super().__init__()"
            )

        self._payload = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Get the health check payload as a dictionary.

        Returns the environment override if DYN_HEALTH_CHECK_PAYLOAD is set,
        otherwise returns the default payload.
        """
        if self._payload is None:
            # Check for environment override
            self._payload = load_health_check_from_env() or self.default_payload
        return self._payload
