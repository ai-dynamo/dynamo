# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Basic validation check functions for fault tolerance testing.

This module provides atomic validation primitives:
- Success rate validation
- Recovery time validation
- Latency SLA validation
- Kubernetes health checks
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def check_success_rate(metrics: Dict[str, Any], min_threshold: float = 0.80) -> None:
    """Check that success rate meets minimum threshold.

    Args:
        metrics: Parsed metrics dictionary with request counts
        min_threshold: Minimum acceptable success rate (0.0 to 1.0)

    Raises:
        AssertionError: If success rate is below threshold
    """
    total_requests = metrics.get("total_requests", 0)
    successful_requests = metrics.get("successful_requests", 0)

    if total_requests == 0:
        logger.warning("No requests found in metrics")
        raise AssertionError("Validation failed: No requests were executed")

    success_rate = successful_requests / total_requests
    logger.info(
        f"Success rate: {success_rate:.2%} "
        f"({successful_requests}/{total_requests} requests)"
    )

    if success_rate < min_threshold:
        raise AssertionError(
            f"Success rate {success_rate:.2%} is below threshold {min_threshold:.2%}"
        )


def check_recovery_time(
    recovery_time: Optional[float], max_seconds: Optional[float] = None
) -> None:
    """Check that recovery time is within acceptable bounds.

    Args:
        recovery_time: Recovery time in seconds (can be None for no-failure scenarios)
        max_seconds: Maximum acceptable recovery time (None = no check)

    Raises:
        AssertionError: If recovery time exceeds maximum
    """
    if recovery_time is None:
        logger.info("No recovery time measured (expected for no-failure scenarios)")
        return

    logger.info(f"Recovery time: {recovery_time:.2f} seconds")

    if max_seconds is not None and recovery_time > max_seconds:
        raise AssertionError(
            f"Recovery time {recovery_time:.2f}s exceeds maximum {max_seconds}s"
        )


def check_no_failures(metrics: Dict[str, Any]) -> None:
    """Check that there were no failed requests.

    Args:
        metrics: Parsed metrics dictionary

    Raises:
        AssertionError: If any requests failed
    """
    failed_requests = metrics.get("failed_requests", 0)
    total_requests = metrics.get("total_requests", 0)

    if failed_requests > 0:
        raise AssertionError(
            f"Expected no failures, but {failed_requests}/{total_requests} requests failed"
        )

    logger.info(f"All {total_requests} requests succeeded")
