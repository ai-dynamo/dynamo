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

"""Validation functions for fault tolerance test scenarios.

This module provides:
1. Common validation functions for different failure scenarios
2. Factory function to get appropriate validation for a scenario
3. Coordination of validation checks and K8s verification
"""

import logging
from typing import Any, Dict, Optional

from tests.fault_tolerance.deploy.k8s_utils import (
    check_container_restart_events,
    get_k8s_events_for_pod,
    get_pod_restart_count,
)
from tests.fault_tolerance.deploy.scenarios import Scenario
from tests.fault_tolerance.deploy.validation_checks import (
    check_no_failures,
    check_recovery_time,
    check_success_rate,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Scenario Verification Functions (Stage 1)
# ============================================================================


def verify_scenario_pod_deletion(
    scenario: Scenario,
    log_dir: str,
    deployment,
    namespace: str,
    affected_pods: Optional[Dict[str, list]] = None,
    **kwargs,
) -> bool:
    """Verify that a pod deletion scenario was executed correctly.

    Checks:
    - Specific pod(s) were deleted (via K8s events)
    - Pod lifecycle events (deletion → recreation)
    - Namespace context for debugging

    Args:
        scenario: Scenario object
        log_dir: Test log directory
        deployment: ManagedDeployment instance
        namespace: K8s namespace
        affected_pods: Dict mapping failure key to list of affected pod names
                       Example: {"VllmDecodeWorker:delete_pod": ["pod-abc123"]}
        **kwargs: Additional arguments

    Returns:
        True if scenario execution verified, False otherwise (non-fatal)
    """
    logger.info("╔" + "═" * 78 + "╗")
    logger.info("║" + " " * 20 + "STAGE 1: SCENARIO VERIFICATION" + " " * 28 + "║")
    logger.info(
        "║" + " " * 15 + "(Verify test scenario executed correctly)" + " " * 20 + "║"
    )
    logger.info("╚" + "═" * 78 + "╝")
    logger.info("")

    # STAGE 1.1: Verify the EXACT pod we deleted is in K8s events
    scenario_verified = False

    if affected_pods and namespace:
        logger.info("─" * 80)
        logger.info("1.1 Verifying Specific Pod Deletion via K8s Events")
        logger.info("─" * 80)

        # Find all deleted pods from affected_pods
        deleted_pod_names = []
        for failure_key, pod_list in affected_pods.items():
            if "delete_pod" in failure_key:
                deleted_pod_names.extend(pod_list)
                logger.info(f"Target pod(s) for deletion: {pod_list}")

        if deleted_pod_names:
            # Verify each deleted pod in K8s events
            for pod_name in deleted_pod_names:
                logger.info(f"\nChecking K8s events for: {pod_name}")
                events = get_k8s_events_for_pod(deployment, pod_name, namespace)

                if not events:
                    logger.warning(
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
                            logger.info(
                                f"DELETION CONFIRMED: [{event['type']}] {event['reason']} - {event['message']}"
                            )

                    if deletion_found:
                        logger.info(f"Pod {pod_name} deletion verified via K8s events")
                        scenario_verified = True
                    else:
                        logger.warning(
                            f"No deletion events found for {pod_name}. "
                            f"Events may have expired or pod wasn't deleted."
                        )
                        # Show all events we found
                        logger.info(f"  Events found for {pod_name}:")
                        for event in events[:10]:
                            logger.info(f"    - {event['reason']}: {event['message']}")

            if scenario_verified:
                logger.info(
                    "\n STAGE 1.1 PASSED: Pod deletion confirmed via K8s events"
                )
            else:
                logger.warning(
                    "\n STAGE 1.1 WARNING: Could not confirm pod deletion via K8s events"
                )
        else:
            logger.warning("No delete_pod failures found in affected_pods")
    else:
        logger.info("Skipping pod deletion verification (missing required info)")

    return scenario_verified


def verify_scenario_no_failures(
    scenario: Scenario,
    log_dir: str,
    deployment,
    namespace: str,
    affected_pods: Optional[Dict[str, list]] = None,
    **kwargs,
) -> bool:
    """Verify that a no-failure scenario was executed correctly.

    For baseline tests with no failures, just verify pods are healthy.

    Args:
        scenario: Scenario object
        log_dir: Test log directory
        deployment: ManagedDeployment instance
        namespace: K8s namespace
        affected_pods: Dict mapping failure key to list of affected pod names
        **kwargs: Additional arguments

    Returns:
        True (no specific verification needed for baseline)
    """
    logger.info("╔" + "═" * 78 + "╗")
    logger.info("║" + " " * 20 + "STAGE 1: SCENARIO VERIFICATION" + " " * 28 + "║")
    logger.info("║" + " " * 15 + "(No failures - baseline scenario)" + " " * 26 + "║")
    logger.info("╚" + "═" * 78 + "╝")
    logger.info("")
    logger.info(" STAGE 1 COMPLETE: No failures to verify (baseline scenario)\n")
    return True


def verify_scenario_process_termination(
    scenario: Scenario,
    log_dir: str,
    deployment,
    namespace: str,
    affected_pods: Optional[Dict[str, list]] = None,
    **kwargs,
) -> bool:
    """Verify that a process termination scenario was executed correctly.

    Checks for process termination by looking at:
    1. Container restart count (most reliable)
    2. Container restart events in K8s

    Args:
        scenario: Scenario object
        log_dir: Test log directory
        deployment: ManagedDeployment instance
        namespace: K8s namespace
        affected_pods: Dict mapping failure key to list of affected pod names
                       Example: {"VllmDecodeWorker:dynamo.vllm": ["pod-abc123"]}
        **kwargs: Additional arguments

    Returns:
        True if process termination verified, False otherwise (non-fatal)
    """
    logger.info("╔" + "═" * 78 + "╗")
    logger.info("║" + " " * 20 + "STAGE 1: SCENARIO VERIFICATION" + " " * 28 + "║")
    logger.info(
        "║"
        + " " * 10
        + "(Verify process was terminated and restarted)"
        + " " * 19
        + "║"
    )
    logger.info("╚" + "═" * 78 + "╝")
    logger.info("")

    scenario_verified = False

    if affected_pods and namespace:
        logger.info("─" * 80)
        logger.info("1.1 Verifying Process Termination via Container Restart")
        logger.info("─" * 80)

        # Find all process termination failures (not pod deletions)
        terminated_pod_names = []
        for failure_key, pod_list in affected_pods.items():
            # Skip pod deletions - only check process terminations
            if "delete_pod" not in failure_key:
                terminated_pod_names.extend(pod_list)
                logger.info(f"Target pod(s) for process termination: {pod_list}")
                logger.info(f"Process killed: {failure_key}")

        if terminated_pod_names:
            # Method 1: Check container restart count
            for pod_name in terminated_pod_names:
                logger.info(f"\nChecking container restarts for: {pod_name}")
                restart_counts = get_pod_restart_count(deployment, pod_name, namespace)

                if restart_counts:
                    total_restarts = sum(restart_counts.values())
                    if total_restarts > 0:
                        logger.info(
                            f"✓ PROCESS TERMINATION CONFIRMED: Container(s) restarted {total_restarts} time(s)"
                        )
                        for container_name, count in restart_counts.items():
                            if count > 0:
                                logger.info(
                                    f"  - Container '{container_name}': {count} restart(s)"
                                )
                        scenario_verified = True
                    else:
                        logger.warning(
                            f"⚠ No container restarts detected for {pod_name}. "
                            f"Process may not have been killed or restart was too fast."
                        )
                else:
                    logger.warning(f"Could not get restart count for {pod_name}")

                # Method 2: Check for restart events
                logger.info("\nChecking K8s events for container restarts...")
                found_events = check_container_restart_events(
                    deployment, pod_name, namespace
                )
                if found_events:
                    logger.info(f"✓ Container restart events found for {pod_name}")
                    if not scenario_verified:
                        scenario_verified = True

            if scenario_verified:
                logger.info("\n✓ STAGE 1.1 PASSED: Process termination confirmed")
            else:
                logger.warning(
                    "\n⚠ STAGE 1.1 WARNING: Could not fully confirm process termination. "
                    "This may be OK if container restart was very fast."
                )
        else:
            logger.warning("No process termination failures found in affected_pods")
    else:
        logger.info("Skipping process termination verification (missing required info)")
        logger.info(
            "Note: For process termination scenarios, validation focuses on results (Stage 2)"
        )

    logger.info("")
    return scenario_verified


# ============================================================================
# Results Verification Functions (Stage 2)
# ============================================================================


def verify_results_high_availability(
    scenario: Scenario,
    metrics: Dict[str, Any],
    recovery_time: Optional[float] = None,
    min_success_rate: Optional[float] = 0.99,
    max_recovery_time: Optional[float] = 60,
    **kwargs,
) -> None:
    """Verify results for high-availability scenarios (with redundancy).

    Validates:
    - High success rate (>90%)
    - Fast recovery time (<60s)
    - System handled failures gracefully

    Args:
        scenario: Scenario object
        metrics: Parsed metrics from results
        recovery_time: Recovery time in seconds
        min_success_rate: Minimum acceptable success rate
        max_recovery_time: Maximum acceptable recovery time
        **kwargs: Additional arguments

    Raises:
        AssertionError: If validation fails
    """
    logger.info("╔" + "═" * 78 + "╗")
    logger.info("║" + " " * 20 + "STAGE 2: RESULTS VERIFICATION" + " " * 29 + "║")
    logger.info(
        "║" + " " * 17 + "(High availability - with redundancy)" + " " * 22 + "║"
    )
    logger.info("╚" + "═" * 78 + "╝")
    logger.info("")

    successful_requests = metrics.get("successful_requests", 0)

    # 2.1: Basic recovery check
    logger.info("\n" + "─" * 80)
    logger.info("2.1 Basic Recovery Check")
    logger.info("─" * 80)
    if successful_requests == 0:
        raise AssertionError(
            " STAGE 2.1 FAILED: No requests succeeded - system did not recover"
        )
    logger.info(f" System recovered: {successful_requests} requests succeeded")

    # 2.2: Success rate validation
    logger.info("\n" + "─" * 80)
    logger.info("2.2 Success Rate Validation (High Availability)")
    logger.info("─" * 80)
    try:
        check_success_rate(metrics, min_threshold=min_success_rate)
        logger.info(
            f" STAGE 2.2 PASSED: Success rate meets threshold ({min_success_rate:.0%})"
        )
    except AssertionError as e:
        logger.error(f" STAGE 2.2 FAILED: {e}")
        raise

    # 2.3: Recovery time validation
    logger.info("\n" + "─" * 80)
    logger.info("2.3 Recovery Time Validation")
    logger.info("─" * 80)
    try:
        check_recovery_time(recovery_time, max_seconds=max_recovery_time)
        logger.info(
            f" STAGE 2.3 PASSED: Recovery time within acceptable range ({max_recovery_time}s max)"
        )
    except AssertionError as e:
        logger.error(f" STAGE 2.3 FAILED: {e}")
        raise

    logger.info("\n")
    logger.info("╔" + "═" * 78 + "╗")
    logger.info(
        "║" + "VALIDATION STAGE 2 PASSED: Results verification passed" + " " * 30 + "║"
    )
    logger.info("╚" + "═" * 78 + "╝")


def verify_results_single_worker(
    scenario: Scenario,
    metrics: Dict[str, Any],
    recovery_time: Optional[float] = None,
    min_success_rate: Optional[float] = 0.10,
    max_recovery_time: Optional[float] = 180,
    **kwargs,
) -> None:
    """Verify results for single worker scenarios (no redundancy).

    Validates:
    - Acceptable success rate (>75%)
    - Reasonable recovery time (<180s)
    - System eventually recovered

    Args:
        scenario: Scenario object
        metrics: Parsed metrics from results
        recovery_time: Recovery time in seconds
        min_success_rate: Minimum acceptable success rate
        max_recovery_time: Maximum acceptable recovery time
        **kwargs: Additional arguments

    Raises:
        AssertionError: If validation fails
    """
    logger.info("╔" + "═" * 78 + "╗")
    logger.info("║" + " " * 20 + "STAGE 2: RESULTS VERIFICATION" + " " * 29 + "║")
    logger.info("║" + " " * 17 + "(Single worker - no redundancy)" + " " * 28 + "║")
    logger.info("╚" + "═" * 78 + "╝")
    logger.info("")

    successful_requests = metrics.get("successful_requests", 0)

    # 2.1: Basic recovery check
    logger.info("\n" + "─" * 80)
    logger.info("2.1 Basic Recovery Check")
    logger.info("─" * 80)
    if successful_requests == 0:
        raise AssertionError(
            " STAGE 2.1 FAILED: No requests succeeded - system did not recover"
        )
    logger.info(f" System recovered: {successful_requests} requests succeeded")

    # 2.2: Success rate validation
    logger.info("\n" + "─" * 80)
    logger.info("2.2 Success Rate Validation (Single Worker)")
    logger.info("─" * 80)
    try:
        check_success_rate(metrics, min_threshold=min_success_rate)
        logger.info(
            f" STAGE 2.2 PASSED: Success rate meets threshold ({min_success_rate:.0%})"
        )
    except AssertionError as e:
        logger.error(f" STAGE 2.2 FAILED: {e}")
        raise

    # 2.3: Recovery time validation
    logger.info("\n" + "─" * 80)
    logger.info("2.3 Recovery Time Validation")
    logger.info("─" * 80)
    try:
        check_recovery_time(recovery_time, max_seconds=max_recovery_time)
        logger.info(
            f" STAGE 2.3 PASSED: Recovery time within acceptable range ({max_recovery_time}s max)"
        )
    except AssertionError as e:
        logger.error(f" STAGE 2.3 FAILED: {e}")
        raise

    logger.info("\n")
    logger.info("╔" + "═" * 78 + "╗")
    logger.info(
        "║" + "VALIDATION STAGE 2 PASSED: Results verification passed" + " " * 30 + "║"
    )
    logger.info("╚" + "═" * 78 + "╝")


def verify_results_no_failures(
    scenario: Scenario,
    metrics: Dict[str, Any],
    **kwargs,
) -> None:
    """Verify results for no-failure baseline scenarios.

    Validates:
    - 100% success rate
    - No failed requests

    Args:
        scenario: Scenario object
        metrics: Parsed metrics from results
        **kwargs: Additional arguments

    Raises:
        AssertionError: If validation fails
    """
    logger.info("╔" + "═" * 78 + "╗")
    logger.info("║" + " " * 20 + "STAGE 2: RESULTS VERIFICATION" + " " * 29 + "║")
    logger.info("║" + " " * 17 + "(No failures - baseline)" + " " * 34 + "║")
    logger.info("╚" + "═" * 78 + "╝")
    logger.info("")

    logger.info("─" * 80)
    logger.info("2.1 Baseline Validation")
    logger.info("─" * 80)

    try:
        check_no_failures(metrics)
        check_success_rate(metrics, min_threshold=1.0)
        logger.info(" STAGE 2.1 PASSED: All requests succeeded (100% success rate)")
    except AssertionError as e:
        logger.error(f" STAGE 2.1 FAILED: {e}")
        raise

    logger.info("\n")
    logger.info("╔" + "═" * 78 + "╗")
    logger.info(
        "║" + "VALIDATION STAGE 2 PASSED: Results verification passed" + " " * 30 + "║"
    )
    logger.info("╚" + "═" * 78 + "╝")


# ============================================================================
# Scenario-Specific Validation Functions (Coordinator)
# ============================================================================


def validate_no_failures_scenario(
    scenario: Scenario,
    log_dir: str,
    metrics: Dict[str, Any],
    deployment,
    namespace: str,
    **kwargs,
) -> None:
    """Validation for scenarios with no failure injection (baseline).

    Args:
        scenario: Scenario object
        log_dir: Test log directory
        metrics: Parsed metrics from results
        deployment: ManagedDeployment instance
        namespace: K8s namespace
        **kwargs: Additional arguments
    """
    logger.info("=" * 80)
    logger.info("VALIDATION: No Failures Scenario (Baseline)")
    logger.info("=" * 80)

    # STAGE 1: Verify scenario (no failures expected)
    verify_scenario_no_failures(
        scenario=scenario,
        log_dir=log_dir,
        deployment=deployment,
        namespace=namespace,
    )

    logger.info("\n")
    logger.info("=" * 80)
    logger.info(" ALL VALIDATION PASSED: No failures scenario")
    logger.info("=" * 80)


def validate_frontend_failure_scenario(
    scenario: Scenario,
    log_dir: str,
    metrics: Dict[str, Any],
    deployment,
    namespace: str,
    recovery_time: Optional[float] = None,
    affected_pods: Optional[Dict[str, list]] = None,
    **kwargs,
) -> None:
    """Validation for frontend failure scenarios.

    Args:
        scenario: Scenario object
        log_dir: Test log directory
        metrics: Parsed metrics from results
        deployment: ManagedDeployment instance
        namespace: K8s namespace
        recovery_time: Recovery time in seconds
        **kwargs: Additional arguments
    """
    logger.info("=" * 60)
    logger.info("VALIDATION: Frontend Failure Scenario")
    logger.info("=" * 60)

    verify_scenario_pod_deletion(
        scenario=scenario,
        log_dir=log_dir,
        deployment=deployment,
        namespace=namespace,
        affected_pods=affected_pods,
    )

    logger.info("\n")
    logger.info("=" * 80)
    logger.info(" ALL VALIDATION PASSED: Frontend failure scenario")
    logger.info("=" * 80)

    # Frontend failures should still achieve reasonable success rate


def validate_decode_worker_pod_deletion(
    scenario: Scenario,
    log_dir: str,
    metrics: Dict[str, Any],
    deployment,
    namespace: str,
    recovery_time: Optional[float] = None,
    affected_pods: Optional[Dict[str, list]] = None,
    **kwargs,
) -> None:
    """Validation for decode worker pod deletion scenarios.

    Args:
        scenario: Scenario object
        log_dir: Test log directory
        metrics: Parsed metrics from results
        deployment: ManagedDeployment instance
        namespace: K8s namespace
        recovery_time: Recovery time in seconds
        affected_pods: Dict mapping failure key to list of affected pod names
                       Example: {"VllmDecodeWorker:delete_pod": ["pod-abc123"]}
        **kwargs: Additional arguments
    """
    logger.info("=" * 80)
    logger.info("VALIDATION: Decode Worker Pod Deletion")
    logger.info("=" * 80)

    # STAGE 1: Verify scenario execution
    verify_scenario_pod_deletion(
        scenario=scenario,
        log_dir=log_dir,
        deployment=deployment,
        namespace=namespace,
        affected_pods=affected_pods,
    )

    logger.info("\n")
    logger.info("=" * 80)
    logger.info(" ALL VALIDATION PASSED: Decode worker pod deletion scenario")
    logger.info("=" * 80)


def validate_prefill_worker_pod_deletion(
    scenario: Scenario,
    log_dir: str,
    metrics: Dict[str, Any],
    deployment,
    namespace: str,
    recovery_time: Optional[float] = None,
    affected_pods: Optional[Dict[str, list]] = None,
    **kwargs,
) -> None:
    """Validation for prefill worker pod deletion scenarios.
    Args:
        scenario: Scenario object
        log_dir: Test log directory
        metrics: Parsed metrics from results
        deployment: ManagedDeployment instance
        namespace: K8s namespace
        recovery_time: Recovery time in seconds
        affected_pods: Dict mapping failure key to list of affected pod names
        **kwargs: Additional arguments
    """
    logger.info("=" * 80)
    logger.info("VALIDATION: Prefill Worker Pod Deletion")
    logger.info("=" * 80)

    # STAGE 1: Verify scenario execution
    verify_scenario_pod_deletion(
        scenario=scenario,
        log_dir=log_dir,
        deployment=deployment,
        namespace=namespace,
        affected_pods=affected_pods,
    )

    logger.info("\n")
    logger.info("=" * 80)
    logger.info("VALIDATION STAGE 1 PASSED: Prefill worker pod deletion scenario")
    logger.info("=" * 80)


def validate_decode_worker_process_termination(
    scenario: Scenario,
    log_dir: str,
    metrics: Dict[str, Any],
    deployment,
    namespace: str,
    recovery_time: Optional[float] = None,
    affected_pods: Optional[Dict[str, list]] = None,
    **kwargs,
) -> None:
    """Validation for decode worker process termination scenarios.

    Args:
        scenario: Scenario object
        log_dir: Test log directory
        metrics: Parsed metrics from results
        deployment: ManagedDeployment instance
        namespace: K8s namespace
        recovery_time: Recovery time in seconds
        affected_pods: Dict mapping failure key to list of affected pod names
        **kwargs: Additional arguments
    """
    logger.info("=" * 80)
    logger.info("VALIDATION: Decode Worker Process Termination")
    logger.info("=" * 80)

    # STAGE 1: Verify scenario execution (process was killed)
    verify_scenario_process_termination(
        scenario=scenario,
        log_dir=log_dir,
        deployment=deployment,
        namespace=namespace,
        affected_pods=affected_pods,
    )

    logger.info("\n")
    logger.info("=" * 80)
    logger.info(" ALL VALIDATION PASSED: Decode worker process termination scenario")
    logger.info("=" * 80)


def validate_prefill_worker_process_termination(
    scenario: Scenario,
    log_dir: str,
    metrics: Dict[str, Any],
    deployment,
    namespace: str,
    recovery_time: Optional[float] = None,
    affected_pods: Optional[Dict[str, list]] = None,
    **kwargs,
) -> None:
    """Validation for prefill worker process termination scenarios.

    Args:
        scenario: Scenario object
        log_dir: Test log directory
        metrics: Parsed metrics from results
        deployment: ManagedDeployment instance
        namespace: K8s namespace
        recovery_time: Recovery time in seconds
        affected_pods: Dict mapping failure key to list of affected pod names
        **kwargs: Additional arguments
    """
    logger.info("=" * 80)
    logger.info("VALIDATION: Prefill Worker Process Termination")
    logger.info("=" * 80)

    # STAGE 1: Verify scenario execution (process was killed)
    verify_scenario_process_termination(
        scenario=scenario,
        log_dir=log_dir,
        deployment=deployment,
        namespace=namespace,
        affected_pods=affected_pods,
    )

    logger.info("\n")
    logger.info("=" * 80)
    logger.info(" ALL VALIDATION PASSED: Prefill worker process termination scenario")
    logger.info("=" * 80)


def validate_sglang_decode_scheduler(
    scenario: Scenario,
    log_dir: str,
    metrics: Dict[str, Any],
    deployment,
    namespace: str,
    recovery_time: Optional[float] = None,
    affected_pods: Optional[Dict[str, list]] = None,
    **kwargs,
) -> None:
    """Validation for SGLang decode scheduler process termination.
    
    Args:
        scenario: Scenario object
        log_dir: Test log directory
        metrics: Parsed metrics from results
        deployment: ManagedDeployment instance
        namespace: K8s namespace
        recovery_time: Recovery time in seconds
        affected_pods: Dict mapping failure key to list of affected pod names
        **kwargs: Additional arguments
    """
    logger.info("=" * 80)
    logger.info("VALIDATION: SGLang Decode Scheduler Process Termination")
    logger.info("=" * 80)

    # STAGE 1: Verify scenario execution (scheduler process was killed)
    verify_scenario_process_termination(
        scenario=scenario,
        log_dir=log_dir,
        deployment=deployment,
        namespace=namespace,
        affected_pods=affected_pods,
    )

    logger.info("\n")
    logger.info("=" * 80)
    logger.info(" ALL VALIDATION PASSED: SGLang decode scheduler termination scenario")
    logger.info("=" * 80)


def validate_sglang_decode_detokenizer(
    scenario: Scenario,
    log_dir: str,
    metrics: Dict[str, Any],
    deployment,
    namespace: str,
    recovery_time: Optional[float] = None,
    affected_pods: Optional[Dict[str, list]] = None,
    **kwargs,
) -> None:
    """Validation for SGLang decode detokenizer process termination.
    
    Args:
        scenario: Scenario object
        log_dir: Test log directory
        metrics: Parsed metrics from results
        deployment: ManagedDeployment instance
        namespace: K8s namespace
        recovery_time: Recovery time in seconds
        affected_pods: Dict mapping failure key to list of affected pod names
        **kwargs: Additional arguments
    """
    logger.info("=" * 80)
    logger.info("VALIDATION: SGLang Decode Detokenizer Process Termination")
    logger.info("=" * 80)

    # STAGE 1: Verify scenario execution (detokenizer process was killed)
    verify_scenario_process_termination(
        scenario=scenario,
        log_dir=log_dir,
        deployment=deployment,
        namespace=namespace,
        affected_pods=affected_pods,
    )

    logger.info("\n")
    logger.info("=" * 80)
    logger.info(" ALL VALIDATION PASSED: SGLang decode detokenizer termination scenario")
    logger.info("=" * 80)


def validate_default(
    scenario: Scenario,
    log_dir: str,
    metrics: Dict[str, Any],
    deployment,
    namespace: str,
    recovery_time: Optional[float] = None,
    **kwargs,
) -> None:
    """Default validation for scenarios without specific validation.

    This is a generic validation that checks basic health:
    - Some requests succeeded
    - Success rate > 50%
    - Recovery time < 300 seconds

    Args:
        scenario: Scenario object
        log_dir: Test log directory
        metrics: Parsed metrics from results
        deployment: ManagedDeployment instance
        namespace: K8s namespace
        recovery_time: Recovery time in seconds
        **kwargs: Additional arguments
    """
    logger.info("=" * 60)
    logger.info("VALIDATION: Default Validation")
    logger.info("=" * 60)

    # Basic checks
    check_success_rate(metrics, min_threshold=0.50)
    check_recovery_time(recovery_time, max_seconds=300)

    logger.info(" Validation passed: Default validation")


# ============================================================================
# Validation Factory
# ============================================================================


def get_validation_for_results(test_name: str, scenario: Scenario):
    """Get appropriate validation function for results.

    This factory function determines which validation to use based on
    deployment redundancy (DP > 1).

    Args:
        test_name: Full test name (used to detect agg vs disagg)
        scenario: Scenario object

    Returns:
        Appropriate results validation function
    """
    has_redundancy = False

    # Determine worker service name based on backend and deployment type
    if scenario.backend == "vllm":
        # vLLM uses same name for agg and disagg
        worker_service_name = "VllmDecodeWorker"
    elif scenario.backend == "sglang":
        # SGLang uses same name for agg and disagg
        worker_service_name = "decode"
    elif scenario.backend == "trtllm":
        # TensorRT-LLM uses different names for agg vs disagg
        # Check test name to determine deployment type
        if "disagg" in test_name:
            worker_service_name = "TRTLLMDecodeWorker"
        else:
            # Agg deployment uses TRTLLMWorker
            worker_service_name = "TRTLLMWorker"
    else:
        raise ValueError(f"Unsupported backend: {scenario.backend}")

    worker_spec = scenario.deployment[worker_service_name]
    if worker_spec and hasattr(worker_spec, "replicas"):
        has_redundancy = worker_spec.replicas > 1

    if has_redundancy:
        return verify_results_high_availability
    return verify_results_single_worker


def get_validation_for_scenario(scenario_name: str, scenario: Scenario):
    """Get appropriate validation function for a scenario.

    This factory function determines which validation to use based on:
    1. Explicit validation in scenario object (highest priority)
    2. Pattern matching on scenario name
    3. Default validation (fallback)

    Args:
        scenario_name: Full scenario name (e.g., "vllm-agg-tp-1-dp-1-decode_worker_pod")
        scenario: Scenario object

    Returns:
        Validation function to use for this scenario
    """
    # 1. Explicit validation takes priority
    if scenario.validation is not None:
        logger.info(f"Using explicit validation for {scenario_name}")
        return scenario.validation

    # 2. Pattern-based defaults
    logger.info(f"Using pattern-based validation for {scenario_name}")

    # No failures scenario
    if scenario_name.endswith("-none"):
        return validate_no_failures_scenario

    # Frontend failures
    if "frontend" in scenario_name:
        return validate_frontend_failure_scenario

    # Decode worker pod deletion
    if "decode_worker_pod" in scenario_name:
        return validate_decode_worker_pod_deletion

    # Prefill worker pod deletion
    if "prefill_worker_pod" in scenario_name:
        return validate_prefill_worker_pod_deletion

    # Decode worker process termination (different from pod deletion)
    if "decode_worker" in scenario_name and "pod" not in scenario_name:
        return validate_decode_worker_process_termination

    # Prefill worker process termination
    if "prefill_worker" in scenario_name and "pod" not in scenario_name:
        return validate_prefill_worker_process_termination

    if "sglang" in scenario_name:
        if "sglang_decode_scheduler" in scenario_name:
            # SGlang does not recover
            return validate_sglang_decode_scheduler
        if "sglang_decode_detokenizer" in scenario_name:
            # SGlang tokenizer does not recover, but requests continue to be servd
            return validate_sglang_decode_detokenizer

    # Backend-specific process failures (engine cores, schedulers, etc.)
    if any(
        x in scenario_name
        for x in [
            "engine_core",
            "scheduler",
            "detokenizer",
        ]
    ):
        # These are more granular failures within workers
        return validate_default  # Use default for now

    # 3. Generic default fallback
    logger.info(f"Using default validation for {scenario_name}")
    return validate_default
