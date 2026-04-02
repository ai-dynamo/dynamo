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

"""Concrete checker implementations for fault tolerance testing.

This module provides specific checker implementations:
- Scenario checkers: Verify test execution (K8s events, pod states)
- Results checkers: Verify system behavior (success rate, recovery time)
"""

import logging
from typing import Optional

from tests.fault_tolerance.deploy.base_checker import BaseChecker, ValidationContext
from tests.fault_tolerance.deploy.k8s_utils import (
    check_container_restart_events,
    get_k8s_events_for_pod,
    get_pod_restart_count,
)
from tests.fault_tolerance.deploy.validation_checks import (
    check_no_failures,
    check_recovery_time,
    check_success_rate,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Scenario Checkers (Stage 1: Verify test execution)
# ============================================================================


class PodDeletionChecker(BaseChecker):
    """Verify that pod deletion scenario was executed correctly.

    Checks:
    - Specific pods were deleted (via K8s events)
    - Pod lifecycle events (deletion → recreation)
    """

    def __init__(self):
        super().__init__(name="PodDeletionChecker")

    def check(self, context: ValidationContext) -> None:
        """Verify pod deletion via K8s events."""
        self.logger.info("╔" + "═" * 78 + "╗")
        self.logger.info(
            "║" + " " * 20 + "STAGE 1: SCENARIO VERIFICATION" + " " * 28 + "║"
        )
        self.logger.info(
            "║"
            + " " * 15
            + "(Verify test scenario executed correctly)"
            + " " * 20
            + "║"
        )
        self.logger.info("╚" + "═" * 78 + "╝")
        self.logger.info("")

        scenario_verified = False

        if context.affected_pods and context.namespace and context.deployment:
            self.logger.info("─" * 80)
            self.logger.info("1.1 Verifying Specific Pod Deletion via K8s Events")
            self.logger.info("─" * 80)

            # Find all deleted pods from affected_pods
            deleted_pod_names = []
            for failure_key, pod_list in context.affected_pods.items():
                if "delete_pod" in failure_key:
                    deleted_pod_names.extend(pod_list)
                    self.logger.info(f"Target pod(s) for deletion: {pod_list}")

            if deleted_pod_names:
                # Verify each deleted pod in K8s events
                for pod_name in deleted_pod_names:
                    self.logger.info(f"\nChecking K8s events for: {pod_name}")
                    events = get_k8s_events_for_pod(
                        context.deployment, pod_name, context.namespace
                    )

                    if not events:
                        self.logger.warning(
                            f"No K8s events found for {pod_name} (events may have expired)"
                        )
                    else:
                        # Look for deletion events
                        deletion_found = False
                        for event in events:
                            reason_lower = event["reason"].lower()
                            if any(
                                x in reason_lower
                                for x in ["killing", "deleted", "terminating"]
                            ):
                                deletion_found = True
                                self.logger.info(
                                    f"✓ DELETION CONFIRMED: [{event['type']}] {event['reason']} - {event['message']}"
                                )

                        if deletion_found:
                            self.logger.info(
                                f"✓ Pod {pod_name} deletion verified via K8s events"
                            )
                            scenario_verified = True
                        else:
                            self.logger.warning(
                                f"⚠ No deletion events found for {pod_name}. "
                                f"Events may have expired or pod wasn't deleted."
                            )

                if scenario_verified:
                    self.logger.info(
                        "\n✓ STAGE 1.1 PASSED: Pod deletion confirmed via K8s events"
                    )
                else:
                    self.logger.warning(
                        "\n⚠ STAGE 1.1 WARNING: Could not confirm pod deletion via K8s events"
                    )
            else:
                self.logger.warning("No delete_pod failures found in affected_pods")
        else:
            self.logger.info(
                "Skipping pod deletion verification (missing required info)"
            )


class ProcessTerminationChecker(BaseChecker):
    """Verify that process termination scenario was executed correctly.

    Checks for process termination by looking at:
    1. Container restart count (most reliable)
    2. Container restart events in K8s
    """

    def __init__(self):
        super().__init__(name="ProcessTerminationChecker")

    def check(self, context: ValidationContext) -> None:
        """Verify process termination via container restarts."""
        self.logger.info("╔" + "═" * 78 + "╗")
        self.logger.info(
            "║" + " " * 20 + "STAGE 1: SCENARIO VERIFICATION" + " " * 28 + "║"
        )
        self.logger.info(
            "║"
            + " " * 10
            + "(Verify process was terminated and restarted)"
            + " " * 19
            + "║"
        )
        self.logger.info("╚" + "═" * 78 + "╝")
        self.logger.info("")

        scenario_verified = False

        if context.affected_pods and context.namespace and context.deployment:
            self.logger.info("─" * 80)
            self.logger.info("1.1 Verifying Process Termination via Container Restart")
            self.logger.info("─" * 80)

            # Find all process termination failures (not pod deletions)
            terminated_pod_names = []
            for failure_key, pod_list in context.affected_pods.items():
                # Skip pod deletions - only check process terminations
                if "delete_pod" not in failure_key:
                    terminated_pod_names.extend(pod_list)
                    self.logger.info(
                        f"Target pod(s) for process termination: {pod_list}"
                    )
                    self.logger.info(f"Process killed: {failure_key}")

            if terminated_pod_names:
                # Method 1: Check container restart count
                for pod_name in terminated_pod_names:
                    self.logger.info(f"\nChecking container restarts for: {pod_name}")
                    restart_counts = get_pod_restart_count(
                        context.deployment, pod_name, context.namespace
                    )

                    if restart_counts:
                        total_restarts = sum(restart_counts.values())
                        if total_restarts > 0:
                            self.logger.info(
                                f"✓ PROCESS TERMINATION CONFIRMED: Container(s) restarted {total_restarts} time(s)"
                            )
                            for container_name, count in restart_counts.items():
                                if count > 0:
                                    self.logger.info(
                                        f"  - Container '{container_name}': {count} restart(s)"
                                    )
                            scenario_verified = True
                        else:
                            self.logger.warning(
                                f"⚠ No container restarts detected for {pod_name}. "
                                f"Process may not have been killed or restart was too fast."
                            )
                    else:
                        self.logger.warning(
                            f"Could not get restart count for {pod_name}"
                        )

                    # Method 2: Check for restart events
                    self.logger.info("\nChecking K8s events for container restarts...")
                    found_events = check_container_restart_events(
                        context.deployment, pod_name, context.namespace
                    )
                    if found_events:
                        self.logger.info(
                            f"✓ Container restart events found for {pod_name}"
                        )
                        if not scenario_verified:
                            scenario_verified = True

                if scenario_verified:
                    self.logger.info(
                        "\n✓ STAGE 1.1 PASSED: Process termination confirmed"
                    )
                else:
                    self.logger.warning(
                        "\n⚠ STAGE 1.1 WARNING: Could not fully confirm process termination. "
                        "This may be OK if container restart was very fast."
                    )
            else:
                self.logger.warning(
                    "No process termination failures found in affected_pods"
                )
        else:
            self.logger.info(
                "Skipping process termination verification (missing required info)"
            )

        self.logger.info("")


class NoFailureChecker(BaseChecker):
    """Verify that no-failure baseline scenario was executed correctly."""

    def __init__(self):
        super().__init__(name="NoFailureChecker")

    def check(self, context: ValidationContext) -> None:
        """Verify that no failures were injected (baseline scenario)."""
        self.logger.info("╔" + "═" * 78 + "╗")
        self.logger.info(
            "║" + " " * 20 + "STAGE 1: SCENARIO VERIFICATION" + " " * 28 + "║"
        )
        self.logger.info(
            "║" + " " * 15 + "(No failures - baseline scenario)" + " " * 26 + "║"
        )
        self.logger.info("╚" + "═" * 78 + "╝")
        self.logger.info("")

        # Verify no failures were injected
        self.logger.info("─" * 80)
        self.logger.info("1.1 Verifying No Failures Were Injected")
        self.logger.info("─" * 80)

        if context.affected_pods:
            # If affected_pods is not None/empty, failures were injected
            affected_count = sum(len(pods) for pods in context.affected_pods.values())
            self.logger.error(
                f"✗ BASELINE SCENARIO VIOLATED: Found {affected_count} affected pod(s)"
            )
            for failure_key, pod_list in context.affected_pods.items():
                self.logger.error(f"  - {failure_key}: {pod_list}")
            raise AssertionError(
                f"Baseline scenario failed: {affected_count} pod(s) were affected by failures. "
                "Expected no failures for baseline test."
            )

        self.logger.info("✓ Verified: No pods were affected by failures")
        self.logger.info(
            "✓ STAGE 1 COMPLETE: Baseline scenario verified (no failures injected)\n"
        )


# ============================================================================
# Results Checkers (Stage 2: Verify system behavior)
# ============================================================================


class SuccessRateChecker(BaseChecker):
    """Check that success rate meets minimum threshold."""

    def __init__(self, min_threshold: float = 0.80, stage_label: str = ""):
        """Initialize success rate checker.

        Args:
            min_threshold: Minimum acceptable success rate (0.0 to 1.0)
            stage_label: Optional label for logging (e.g., "High Availability")
        """
        super().__init__(name="SuccessRateChecker")
        self.min_threshold = min_threshold
        self.stage_label = stage_label

    def check(self, context: ValidationContext) -> None:
        """Verify success rate meets threshold."""
        self.logger.info("\n" + "─" * 80)
        if self.stage_label:
            self.logger.info(f"2.1 Success Rate Validation ({self.stage_label})")
        else:
            self.logger.info("2.1 Success Rate Validation")
        self.logger.info("─" * 80)

        try:
            check_success_rate(context.metrics, min_threshold=self.min_threshold)
            self.logger.info(
                f"✓ STAGE 2.1 PASSED: Success rate meets threshold ({self.min_threshold:.0%})"
            )
        except AssertionError as e:
            self.logger.error(f"✗ STAGE 2.1 FAILED: {e}")
            raise


class RecoveryTimeChecker(BaseChecker):
    """Check that recovery time is within acceptable bounds."""

    def __init__(self, max_seconds: Optional[float] = None):
        """Initialize recovery time checker.

        Args:
            max_seconds: Maximum acceptable recovery time (None = no check)
        """
        super().__init__(name="RecoveryTimeChecker")
        self.max_seconds = max_seconds

    def check(self, context: ValidationContext) -> None:
        """Verify recovery time is within bounds."""
        self.logger.info("\n" + "─" * 80)
        self.logger.info("2.2 Recovery Time Validation")
        self.logger.info("─" * 80)

        try:
            check_recovery_time(context.recovery_time, max_seconds=self.max_seconds)
            if self.max_seconds:
                self.logger.info(
                    f"✓ STAGE 2.2 PASSED: Recovery time within acceptable range ({self.max_seconds}s max)"
                )
            else:
                self.logger.info("✓ STAGE 2.2 PASSED: Recovery time recorded")
        except AssertionError as e:
            self.logger.error(f"✗ STAGE 2.2 FAILED: {e}")
            raise


class NoFailuresChecker(BaseChecker):
    """Check that there were no failed requests (for baseline tests)."""

    def __init__(self):
        super().__init__(name="NoFailuresChecker")

    def check(self, context: ValidationContext) -> None:
        """Verify no requests failed."""
        self.logger.info("─" * 80)
        self.logger.info("2.1 Baseline Validation")
        self.logger.info("─" * 80)

        try:
            check_no_failures(context.metrics)
            check_success_rate(context.metrics, min_threshold=1.0)
            self.logger.info(
                "✓ STAGE 2.1 PASSED: All requests succeeded (100% success rate)"
            )
        except AssertionError as e:
            self.logger.error(f"✗ STAGE 2.1 FAILED: {e}")
            raise


class BasicRecoveryChecker(BaseChecker):
    """Check that system recovered (at least some requests succeeded)."""

    def __init__(self):
        super().__init__(name="BasicRecoveryChecker")

    def check(self, context: ValidationContext) -> None:
        """Verify system recovered."""
        self.logger.info("\n" + "─" * 80)
        self.logger.info("2.1 Basic Recovery Check")
        self.logger.info("─" * 80)

        successful_requests = context.metrics.get("successful_requests", 0)
        if successful_requests == 0:
            raise AssertionError(
                "✗ STAGE 2.1 FAILED: No requests succeeded - system did not recover"
            )
        self.logger.info(
            f"✓ System recovered: {successful_requests} requests succeeded"
        )


# ============================================================================
# Composite Checkers (Stage wrappers)
# ============================================================================


class HighAvailabilityResultsChecker(BaseChecker):
    """Composite checker for high availability scenarios.

    Validates:
    - High success rate (≥90%)
    - Fast recovery time (≤60s)
    - System handled failures gracefully
    """

    def __init__(
        self,
        min_success_rate: float = 0.90,
        max_recovery_time: float = 60,
    ):
        super().__init__(name="HighAvailabilityResultsChecker")
        self.min_success_rate = min_success_rate
        self.max_recovery_time = max_recovery_time

    def check(self, context: ValidationContext) -> None:
        """Run all high availability checks."""
        self.logger.info("╔" + "═" * 78 + "╗")
        self.logger.info(
            "║" + " " * 20 + "STAGE 2: RESULTS VERIFICATION" + " " * 29 + "║"
        )
        self.logger.info(
            "║" + " " * 17 + "(High availability - with redundancy)" + " " * 22 + "║"
        )
        self.logger.info("╚" + "═" * 78 + "╝")
        self.logger.info("")

        # Run individual checks
        BasicRecoveryChecker().check(context)
        SuccessRateChecker(
            min_threshold=self.min_success_rate, stage_label="High Availability"
        ).check(context)
        RecoveryTimeChecker(max_seconds=self.max_recovery_time).check(context)

        self.logger.info("\n")
        self.logger.info("╔" + "═" * 78 + "╗")
        self.logger.info(
            "║"
            + "VALIDATION STAGE 2 PASSED: Results verification passed"
            + " " * 30
            + "║"
        )
        self.logger.info("╚" + "═" * 78 + "╝")


class SingleWorkerResultsChecker(BaseChecker):
    """Composite checker for single worker scenarios.

    Validates:
    - Acceptable success rate (≥10%)
    - Reasonable recovery time (≤180s)
    - System eventually recovered
    """

    def __init__(
        self,
        min_success_rate: float = 0.10,
        max_recovery_time: float = 180,
    ):
        super().__init__(name="SingleWorkerResultsChecker")
        self.min_success_rate = min_success_rate
        self.max_recovery_time = max_recovery_time

    def check(self, context: ValidationContext) -> None:
        """Run all single worker checks."""
        self.logger.info("╔" + "═" * 78 + "╗")
        self.logger.info(
            "║" + " " * 20 + "STAGE 2: RESULTS VERIFICATION" + " " * 29 + "║"
        )
        self.logger.info(
            "║" + " " * 17 + "(Single worker - no redundancy)" + " " * 28 + "║"
        )
        self.logger.info("╚" + "═" * 78 + "╝")
        self.logger.info("")

        # Run individual checks
        BasicRecoveryChecker().check(context)
        SuccessRateChecker(
            min_threshold=self.min_success_rate, stage_label="Single Worker"
        ).check(context)
        RecoveryTimeChecker(max_seconds=self.max_recovery_time).check(context)

        self.logger.info("\n")
        self.logger.info("╔" + "═" * 78 + "╗")
        self.logger.info(
            "║"
            + "VALIDATION STAGE 2 PASSED: Results verification passed"
            + " " * 30
            + "║"
        )
        self.logger.info("╚" + "═" * 78 + "╝")


class BaselineResultsChecker(BaseChecker):
    """Composite checker for no-failure baseline scenarios."""

    def __init__(self):
        super().__init__(name="BaselineResultsChecker")

    def check(self, context: ValidationContext) -> None:
        """Run baseline checks."""
        self.logger.info("╔" + "═" * 78 + "╗")
        self.logger.info(
            "║" + " " * 20 + "STAGE 2: RESULTS VERIFICATION" + " " * 29 + "║"
        )
        self.logger.info("║" + " " * 17 + "(No failures - baseline)" + " " * 34 + "║")
        self.logger.info("╚" + "═" * 78 + "╝")
        self.logger.info("")

        NoFailuresChecker().check(context)

        self.logger.info("\n")
        self.logger.info("╔" + "═" * 78 + "╗")
        self.logger.info(
            "║"
            + "VALIDATION STAGE 2 PASSED: Results verification passed"
            + " " * 30
            + "║"
        )
        self.logger.info("╚" + "═" * 78 + "╝")
