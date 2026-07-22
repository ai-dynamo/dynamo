# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared engine health monitor configuration."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

from dynamo.common.configuration.utils import env_or_default

logger = logging.getLogger(__name__)

ENGINE_HEALTH_CHECK_INTERVAL = 2.0
ENGINE_HEALTH_CHECK_TIMEOUT = 30.0
ENGINE_HEALTH_SHUTDOWN_TIMEOUT = 60.0
ENGINE_HEALTH_CHECK_INTERVAL_ENV = "DYN_ENGINE_HEALTH_CHECK_INTERVAL"
ENGINE_HEALTH_CHECK_TIMEOUT_ENV = "DYN_ENGINE_HEALTH_CHECK_TIMEOUT"
ENGINE_HEALTH_SHUTDOWN_TIMEOUT_ENV = "DYN_ENGINE_HEALTH_SHUTDOWN_TIMEOUT"


def _env_float(name: str, default: float) -> float:
    try:
        parsed = env_or_default(name, default, value_type=float)
    except ValueError:
        logger.warning("Invalid %s; using default %.1f", name, default)
        return default
    if not math.isfinite(parsed):
        logger.warning("Non-finite %s=%r; using default %.1f", name, parsed, default)
        return default
    if parsed < 0:
        logger.warning("Negative %s=%r; using 0", name, parsed)
        return 0.0
    return parsed


@dataclass(frozen=True)
class EngineHealthMonitorConfig:
    interval: float
    check_timeout: float
    shutdown_timeout: float

    @classmethod
    def from_env(
        cls,
        *,
        interval: Optional[float] = None,
        check_timeout: Optional[float] = None,
        shutdown_timeout: Optional[float] = None,
    ) -> "EngineHealthMonitorConfig":
        return cls(
            interval=(
                _env_float(
                    ENGINE_HEALTH_CHECK_INTERVAL_ENV,
                    ENGINE_HEALTH_CHECK_INTERVAL,
                )
                if interval is None
                else interval
            ),
            check_timeout=(
                _env_float(
                    ENGINE_HEALTH_CHECK_TIMEOUT_ENV,
                    ENGINE_HEALTH_CHECK_TIMEOUT,
                )
                if check_timeout is None
                else check_timeout
            ),
            shutdown_timeout=(
                _env_float(
                    ENGINE_HEALTH_SHUTDOWN_TIMEOUT_ENV,
                    ENGINE_HEALTH_SHUTDOWN_TIMEOUT,
                )
                if shutdown_timeout is None
                else shutdown_timeout
            ),
        )
