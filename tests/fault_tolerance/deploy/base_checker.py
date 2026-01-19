# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Base checker class and validation context for fault tolerance testing.

This module provides:
1. ValidationContext - standardized input for all checkers
2. BaseChecker - abstract base class for all validation checks
3. Common interface for scenario and results validation
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class ValidationContext:
    """Standardized context passed to all checkers.

    This ensures all checkers receive the same input structure.

    Attributes:
        scenario: Scenario object being tested
        log_dir: Test log directory
        metrics: Parsed metrics from results (success rate, latencies, etc.)
        deployment: ManagedDeployment instance (optional)
        namespace: Kubernetes namespace (optional)
        recovery_time: Recovery time in seconds (optional)
        affected_pods: Dict mapping failure key to affected pod names (optional)
    """

    scenario: Any  # Scenario type to avoid circular import
    log_dir: str
    metrics: Dict[str, Any]
    deployment: Optional[Any] = None  # ManagedDeployment
    namespace: Optional[str] = None
    recovery_time: Optional[float] = None
    affected_pods: Optional[Dict[str, list]] = None


class BaseChecker(ABC):
    """Abstract base class for all validation checkers.

    All checkers must:
    1. Implement check() method
    2. Accept ValidationContext as input
    3. Raise AssertionError on validation failure
    4. Log validation progress and results

    Usage:
        class MyChecker(BaseChecker):
            def check(self, context: ValidationContext) -> None:
                if not validate_something(context.metrics):
                    raise AssertionError("Validation failed")
    """

    def __init__(self, name: Optional[str] = None):
        """Initialize checker with optional name.

        Args:
            name: Human-readable name for the checker (defaults to class name)
        """
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(f"{__name__}.{self.name}")

    @abstractmethod
    def check(self, context: ValidationContext) -> None:
        """Perform validation check.

        Args:
            context: ValidationContext with all necessary data

        Raises:
            AssertionError: If validation fails
        """
        pass

    def __call__(self, context: ValidationContext) -> None:
        """Allow checker to be called directly.

        Args:
            context: ValidationContext with all necessary data
        """
        self.logger.info(f"Running checker: {self.name}")
        self.check(context)
        self.logger.debug(f"✓ {self.name} passed")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
