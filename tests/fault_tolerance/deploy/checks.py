# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Scenario checks for fault tolerance testing.

Checks have:
- validate(ctx): Assert conditions, raises AssertionError on failure
- description: Human-readable description
- get_load(ctx, name): Helper to find StartLoad events by name
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tests.fault_tolerance.deploy.events import StartLoad
    from tests.fault_tolerance.deploy.scenario import ScenarioContext


# =============================================================================
# Check Base Class
# =============================================================================


@dataclass
class Check(ABC):
    """Base class for result validation.

    Checks receive ScenarioContext and can access:
    - self.get_load(ctx, name) to find StartLoad and get results
    - ctx.deployment.collect_service_logs() for service logs
    """

    @abstractmethod
    def validate(self, ctx: "ScenarioContext") -> None:
        """Validate results. Raises AssertionError on failure."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the check."""
        pass

    def get_load(self, ctx: "ScenarioContext", name: str) -> "StartLoad | None":
        """Find StartLoad event by name."""
        from tests.fault_tolerance.deploy.events import StartLoad

        for event in ctx.events:
            if isinstance(event, StartLoad) and event.name == name:
                return event
        return None


# =============================================================================
# Check Implementations
# =============================================================================


@dataclass
class ZeroErrors(Check):
    """Assert zero errors in load results."""

    name: str = "default"

    def validate(self, ctx: "ScenarioContext") -> None:
        load = self.get_load(ctx, self.name)
        assert load and load.results, f"No results for load '{self.name}'"

        error_result = load.results.get("error_request_count")
        error_count = error_result.get("avg", 0) if error_result else 0

        ctx.logger.info(f"ZeroErrors: error_count = {error_count}")
        assert error_count == 0, f"Expected 0 errors, got {error_count}"

    @property
    def description(self) -> str:
        return f"Zero errors ('{self.name}')"


@dataclass
class MaxErrors(Check):
    """Assert errors below a threshold."""

    max_errors: int
    name: str = "default"

    def validate(self, ctx: "ScenarioContext") -> None:
        load = self.get_load(ctx, self.name)
        assert load and load.results, f"No results for load '{self.name}'"

        error_result = load.results.get("error_request_count")
        error_count = error_result.get("avg", 0) if error_result else 0

        ctx.logger.info(
            f"MaxErrors: error_count = {error_count}, max = {self.max_errors}"
        )
        assert (
            error_count <= self.max_errors
        ), f"Expected at most {self.max_errors} errors, got {error_count}"

    @property
    def description(self) -> str:
        return f"Max {self.max_errors} errors ('{self.name}')"


@dataclass
class MinRequests(Check):
    """Assert minimum number of successful requests."""

    min_count: int
    name: str = "default"

    def validate(self, ctx: "ScenarioContext") -> None:
        load = self.get_load(ctx, self.name)
        assert load and load.results, f"No results for load '{self.name}'"

        request_count = load.results.get("request_count", {}).get("avg", 0)

        ctx.logger.info(
            f"MinRequests: request_count = {request_count}, min = {self.min_count}"
        )
        assert (
            request_count >= self.min_count
        ), f"Expected at least {self.min_count} requests, got {request_count}"

    @property
    def description(self) -> str:
        return f"Min {self.min_count} requests ('{self.name}')"


@dataclass
class WasCancelled(Check):
    """Assert was_cancelled flag matches expected value."""

    expected: bool = True
    name: str = "default"

    def validate(self, ctx: "ScenarioContext") -> None:
        load = self.get_load(ctx, self.name)
        assert load and load.results, f"No results for load '{self.name}'"

        was_cancelled = load.results.get("was_cancelled", False)

        ctx.logger.info(
            f"WasCancelled: was_cancelled = {was_cancelled}, expected = {self.expected}"
        )
        assert (
            was_cancelled == self.expected
        ), f"Expected was_cancelled={self.expected}, got {was_cancelled}"

    @property
    def description(self) -> str:
        return f"was_cancelled={self.expected} ('{self.name}')"


@dataclass
class ServiceLogContains(Check):
    """Assert a service log contains a pattern."""

    service: str
    pattern: str

    def validate(self, ctx: "ScenarioContext") -> None:
        logs = ctx.deployment.collect_service_logs()
        log = logs.get(self.service, "")
        ctx.logger.info(
            f"ServiceLogContains: checking '{self.pattern}' in {self.service}"
        )
        assert self.pattern in log, (
            f"Pattern '{self.pattern}' not found in {self.service} logs "
            f"(log length: {len(log)} chars)"
        )

    @property
    def description(self) -> str:
        return f"Service '{self.service}' logs contain '{self.pattern}'"


@dataclass
class ServiceLogNotContains(Check):
    """Assert a service log does NOT contain a pattern."""

    service: str
    pattern: str

    def validate(self, ctx: "ScenarioContext") -> None:
        logs = ctx.deployment.collect_service_logs()
        log = logs.get(self.service, "")
        ctx.logger.info(
            f"ServiceLogNotContains: checking '{self.pattern}' NOT in {self.service}"
        )
        assert (
            self.pattern not in log
        ), f"Pattern '{self.pattern}' should NOT be in {self.service} logs but was found"

    @property
    def description(self) -> str:
        return f"Service '{self.service}' logs do NOT contain '{self.pattern}'"
