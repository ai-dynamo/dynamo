# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Scenario reports for fault tolerance testing.

Reports have:
- generate(ctx): Create the report after checks pass
- description: Human-readable description

Reports are run after all checks pass and can generate
additional artifacts (HTML reports, metrics summaries, etc.)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tests.fault_tolerance.deploy.scenario import ScenarioContext


# =============================================================================
# Report Base Class
# =============================================================================


@dataclass
class Report(ABC):
    """Base class for report generation.

    Reports are run after checks pass and can generate
    additional artifacts (HTML reports, metrics summaries, etc.)
    """

    @abstractmethod
    def generate(self, ctx: "ScenarioContext") -> None:
        """Generate the report."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the report."""
        pass
