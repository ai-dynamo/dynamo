# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import multiprocessing
import os
import re
import signal
from contextlib import contextmanager
from multiprocessing.context import SpawnProcess
from typing import Any, Optional

import pytest

from tests.fault_tolerance.deploy.base_checker import ValidationContext
from tests.fault_tolerance.deploy.client_factory import get_client_function
from tests.fault_tolerance.deploy.parse_factory import parse_test_results
from tests.fault_tolerance.deploy.parse_results import process_overflow_recovery_test
from tests.fault_tolerance.deploy.scenarios import (
    OVERFLOW_SUFFIX,
    RECOVERY_SUFFIX,
    Failure,
    Load,
    Scenario,
    scenarios,
)
from tests.utils.managed_deployment import DeploymentSpec, ManagedDeployment


@pytest.fixture
def scenario(scenario_name, client_type):
    """Get scenario and optionally override client type from command line.

    If --client-type is specified, it overrides the scenario's default client type.
    """
    scenario_obj = scenarios[scenario_name]

    # Override client type if specified on command line
    if client_type is not None:
        # Create a copy of the load config with overridden client type
        import copy

        scenario_obj = copy.deepcopy(scenario_obj)
        scenario_obj.load.client_type = client_type

        # Adjust retry settings based on client type
        if client_type == "legacy":
            # Legacy uses per-request retries
            if scenario_obj.load.max_retries > 1:
                scenario_obj.load.max_retries = 1
        elif client_type == "aiperf":
            # AI-Perf uses full test retries
            if scenario_obj.load.max_retries < 3:
                scenario_obj.load.max_retries = 3

    return scenario_obj


@contextmanager
def _clients(
    logger: logging.Logger,
    log_dir: str,
    deployment_spec: DeploymentSpec,
    namespace: str,
    model: str,
    load_config: Load,
):
    """Start client processes using factory pattern for client selection.

    Args:
        logger: Logger instance
        log_dir: Log directory for output logs and client logs/artifacts
        deployment_spec: Deployment specification
        namespace: Kubernetes namespace
        model: Model name to test
        load_config: Load configuration object containing client settings
    """
    # Get appropriate client function based on configuration
    client_func = get_client_function(load_config.client_type)

    logger.info(
        f"Starting {load_config.clients} clients using '{load_config.client_type}' client"
    )

    procs: list[SpawnProcess] = []
    ctx = multiprocessing.get_context("spawn")

    # Determine retry_delay_or_rate based on client type
    if load_config.client_type == "legacy":
        # Legacy client uses max_request_rate for rate limiting
        retry_delay_or_rate = load_config.max_request_rate
    else:
        # AI-Perf client uses retry_delay between attempts (default 5s)
        retry_delay_or_rate = 5

    # Check if this is a continuous load test (rolling upgrade scenarios)
    continuous_load = getattr(load_config, "continuous_load", False)

    # Check if this is a mixed token test (overflow + recovery)
    # If mixed_token_test is True, run two phases; otherwise run normally
    if hasattr(load_config, "mixed_token_test") and load_config.mixed_token_test:
        logger.info(
            f"Mixed token test: {load_config.overflow_request_count} overflow requests "
            f"({load_config.overflow_token_length} tokens) + "
            f"{load_config.normal_request_count} normal requests "
            f"({load_config.input_token_length} tokens)"
        )

        # First phase: Send overflow requests
        for i in range(load_config.clients):
            proc_overflow = ctx.Process(
                target=client_func,
                args=(
                    deployment_spec,
                    namespace,
                    model,
                    f"{log_dir}{OVERFLOW_SUFFIX}",
                    i,
                    load_config.overflow_request_count,  # 15 overflow requests
                    load_config.overflow_token_length,  # 2x max_seq_len tokens
                    load_config.output_token_length,
                    load_config.max_retries,
                    retry_delay_or_rate,
                    continuous_load,
                ),
            )
            proc_overflow.start()
            procs.append(proc_overflow)
            logger.debug(f"Started overflow client {i} (PID: {proc_overflow.pid})")

        # Wait for overflow requests to complete
        for proc in procs:
            proc.join()

        logger.info("Overflow requests completed. Starting recovery phase...")

        # Second phase: Send normal requests to test recovery
        procs_recovery: list[SpawnProcess] = []
        for i in range(load_config.clients):
            proc_normal = ctx.Process(
                target=client_func,
                args=(
                    deployment_spec,
                    namespace,
                    model,
                    f"{log_dir}{RECOVERY_SUFFIX}",
                    i,
                    load_config.normal_request_count,  # 15 normal requests
                    load_config.input_token_length,  # Normal token count
                    load_config.output_token_length,
                    load_config.max_retries,
                    retry_delay_or_rate,
                ),
            )
            proc_normal.start()
            procs_recovery.append(proc_normal)
            logger.debug(f"Started recovery client {i} (PID: {proc_normal.pid})")

        # Add recovery processes to main list
        procs.extend(procs_recovery)
    else:
        # Normal test - single phase
        for i in range(load_config.clients):
            procs.append(
                ctx.Process(
                    target=client_func,
                    args=(
                        deployment_spec,
                        namespace,
                        model,
                        log_dir,
                        i,
                        load_config.requests_per_client,
                        load_config.input_token_length,
                        load_config.output_token_length,
                        load_config.max_retries,
                        retry_delay_or_rate,
                        continuous_load,  # Pass continuous_load flag
                    ),
                )
            )
            procs[-1].start()
            logger.debug(f"Started client {i} (PID: {procs[-1].pid})")

    yield procs

    for proc in procs:
        logger.debug(f"{proc} waiting for join")
        proc.join()
        logger.debug(f"{proc} joined")


def _terminate_client_processes(
    client_procs: list[SpawnProcess],
    logger: logging.Logger,
):
    """
    Terminate client processes.
    """
    # Send SIGINT to client processes to stop continuous load
    if client_procs:
        logger.info(f"Sending SIGINT to {len(client_procs)} client processes...")
        for proc in client_procs:
            if proc.is_alive():
                try:
                    if proc.pid is not None:
                        logger.debug(f"Sending SIGINT to client process {proc.pid}")
                        os.kill(proc.pid, signal.SIGINT)
                    else:
                        raise ValueError(f"Process {proc} has no PID")
                except ProcessLookupError:
                    logger.debug(f"Process {proc.pid} already terminated")
                except Exception as e:
                    logger.warning(f"Failed to send SIGINT to process {proc.pid}: {e}")
        logger.info(
            "SIGINT sent to all client processes, waiting for graceful shutdown..."
        )
    else:
        logger.warning("No client processes provided to terminate")


async def _inject_failures(
    failures: list[Failure],
    logger: logging.Logger,
    deployment: ManagedDeployment,
) -> dict[str, list]:  # noqa: F811
    affected_pods: dict[str, list] = {}

    for failure in failures:
        await asyncio.sleep(failure.time)

        logger.info(f"Injecting failure for: {failure}")

        affected_pods[failure.get_failure_key()] = await failure.execute(
            deployment, logger
        )

    return affected_pods


global_result_list = []
# Global storage for test results (used by validation fixture)
test_results_cache = {}


@pytest.fixture(autouse=True)
def validation_context(request, scenario):  # noqa: F811
    """Provides shared context between test execution and validation.

    This fixture creates a shared dictionary that the test populates during
    execution (deployment, namespace, affected_pods), then uses that data
    in teardown to parse results and run checkers.

    Automatically detects result type (AI-Perf or legacy) and uses
    the appropriate parser. After parsing, immediately runs validation checkers.
    """
    # Shared context that test will populate during execution
    context: dict[str, Any] = {
        "deployment": None,
        "namespace": None,
        "affected_pods": {},
    }

    yield context  # Test receives this and populates it

    # Determine log paths based on whether this is a mixed token test
    log_paths = []
    test_name = request.node.name
    logger = logging.getLogger(test_name)

    if hasattr(scenario.load, "mixed_token_test") and scenario.load.mixed_token_test:
        # For mixed token tests, we have separate overflow and recovery directories
        overflow_dir = f"{request.node.name}{OVERFLOW_SUFFIX}"
        recovery_dir = f"{request.node.name}{RECOVERY_SUFFIX}"
        log_paths = [overflow_dir, recovery_dir]

        logging.info("Mixed token test detected. Looking for results in:")
        logging.info(f"  - Overflow phase: {overflow_dir}")
        logging.info(f"  - Recovery phase: {recovery_dir}")
    else:
        # Standard test with single directory
        log_paths = [request.node.name]

    # Use factory to auto-detect and parse results
    try:
        results = parse_test_results(
            log_dir=None,
            log_paths=log_paths,
            tablefmt="fancy_grid",
            sla=scenario.load.sla,
            success_threshold=scenario.load.success_threshold,
            print_output=True,
            # force_parser can be set based on client_type if needed
            # force_parser=scenario.load.client_type,
        )
        # Store results for reference
        if results:
            logging.info(f"Results parsed: {type(results)}")
            test_results_cache[test_name] = results

            # IMMEDIATELY run validation now that we have results
            try:
                logger.info("\n" + "=" * 60)
                logger.info("Running validation checks...")
                logger.info("=" * 60)

                # Extract metrics and recovery time from parsed results
                if isinstance(results, list) and len(results) > 0:
                    result = results[0]
                elif isinstance(results, dict):
                    result = results
                else:
                    logger.warning(f"Unexpected result format: {type(results)}")
                    result = None

                if result:
                    metrics = result.get("metrics", {})
                    recovery_time = result.get("recovery_time")

                    # Create ValidationContext for all checkers
                    validation_ctx = ValidationContext(
                        scenario=scenario,
                        log_dir=test_name,
                        metrics=metrics,
                        deployment=context.get("deployment"),
                        namespace=context.get("namespace"),
                        recovery_time=recovery_time,
                        affected_pods=context.get("affected_pods", {}),
                    )

                    # Use pre-generated checkers from scenario
                    # Checkers were already determined during scenario creation
                    checkers = scenario.checkers or []

                    # Run all checkers
                    for checker in checkers:
                        logger.info(f"\nRunning checker: {checker.name}")
                        checker.check(validation_ctx)

                    logger.info("=" * 60)
                    logger.info("✓ All validation checks passed")
                    logger.info("=" * 60 + "\n")

            except AssertionError as e:
                logger.error("=" * 60)
                logger.error(f"✗ Validation failed: {e}")
                logger.error("=" * 60 + "\n")
                # Re-raise to fail the test
                raise
            except Exception as e:
                logger.error(f"Validation error: {e}")
                # Don't fail test on validation errors (non-assertion exceptions)
                logger.warning("Skipping validation due to error")

    except Exception:
        logging.exception("Failed to parse results for %s", test_name)

    # Add all directories to global list for session summary
    global_result_list.extend(log_paths)


@pytest.fixture(autouse=True, scope="session")
def results_summary():
    """
    Session summary that processes all tests but only prints paired tests.
    """
    yield

    if not global_result_list:
        return

    # Step 1: Group directories
    test_groups: dict[str, dict[str, str]] = {}

    for log_path in global_result_list:
        if log_path.endswith(OVERFLOW_SUFFIX):
            base_name = log_path[: -len(OVERFLOW_SUFFIX)]
            if base_name not in test_groups:
                test_groups[base_name] = {}
            test_groups[base_name]["overflow"] = log_path
        elif log_path.endswith(RECOVERY_SUFFIX):
            base_name = log_path[: -len(RECOVERY_SUFFIX)]
            if base_name not in test_groups:
                test_groups[base_name] = {}
            test_groups[base_name]["recovery"] = log_path

    # Step 2: Process all tests (get results) but only print paired ones
    try:
        # First, silently parse all tests to get results (for any downstream processing)
        parse_test_results(
            log_dir=None,
            log_paths=global_result_list,
            tablefmt="fancy_grid",
            print_output=False,  # Don't print anything
        )

        for base_name, paths in test_groups.items():
            if "overflow" in paths and "recovery" in paths:
                # Extract scenario from test name to pass configs
                scenario_obj = None
                match = re.search(r"\[(.*)\]", base_name)
                if match:
                    scenario_name = match.group(1)
                    if scenario_name in scenarios:
                        scenario_obj = scenarios[scenario_name]
                        logging.info(
                            f"Found scenario '{scenario_name}' for combined results."
                        )

                if not scenario_obj:
                    logging.warning(
                        f"Could not find scenario for '{base_name}'. Using default thresholds."
                    )

                success_threshold = (
                    scenario_obj.load.success_threshold if scenario_obj else 90.0
                )
                logging.info(
                    f"Using success_threshold: {success_threshold} for combined summary of '{base_name}'"
                )

                # This function will print the combined summary
                process_overflow_recovery_test(
                    overflow_path=paths["overflow"],
                    recovery_path=paths["recovery"],
                    tablefmt="fancy_grid",
                    sla=scenario_obj.load.sla if scenario_obj else None,
                    success_threshold=success_threshold,
                )

    except Exception as e:
        logging.error(f"Failed to parse combined results: {e}")


@pytest.mark.k8s
@pytest.mark.fault_tolerance
@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_fault_scenario(
    scenario: Scenario,  # noqa: F811
    request,
    image: str,
    namespace: str,
    validation_context,  # noqa: F811  # Shared context for passing data to validation
    skip_service_restart: bool,
):
    """
    Test dynamo serve deployments with injected failures

    Flow:
    1. validation_context fixture creates empty dict: {"deployment": None, "namespace": None, "affected_pods": {}}
    2. This test populates it: validation_context["deployment"] = deployment, etc.
    3. After test completes, fixture reads validation_context and runs validation checkers
    4. Checkers use the populated ValidationContext to verify test results and K8s events
    """

    logger = logging.getLogger(request.node.name)

    scenario.deployment.name = "fault-tolerance-test"

    if image:
        scenario.deployment.set_image(image)

    model: Optional[str] = None
    if scenario.model:
        scenario.deployment.set_model(scenario.model)
        model = scenario.model
    else:
        # Get model from the appropriate worker based on backend
        try:
            if scenario.backend == "vllm":
                model = scenario.deployment["VllmDecodeWorker"].model
            elif scenario.backend == "sglang":
                model = scenario.deployment["decode"].model
            elif scenario.backend == "trtllm":
                # Determine deployment type from scenario deployment name
                if (
                    "agg" in scenario.deployment.name
                    and "disagg" not in scenario.deployment.name
                ):
                    model = scenario.deployment["TRTLLMWorker"].model
                else:
                    model = scenario.deployment["TRTLLMDecodeWorker"].model
            else:
                model = None
        except (KeyError, AttributeError):
            model = None
    # Fallback to default if still None
    model = model or "Qwen/Qwen3-0.6B"

    scenario.deployment.set_logging(True, "info")

    async with ManagedDeployment(
        namespace=namespace,
        log_dir=request.node.name,
        deployment_spec=scenario.deployment,
        skip_service_restart=skip_service_restart,
    ) as deployment:
        # Populate shared context for validation
        validation_context["deployment"] = deployment
        validation_context["namespace"] = namespace

        with _clients(
            logger,
            request.node.name,
            scenario.deployment,
            namespace,
            model,
            scenario.load,  # Pass entire Load config object
        ) as client_procs:
            # Inject failures and capture which pods were affected
            affected_pods = await _inject_failures(
                scenario.failures, logger, deployment
            )
            logger.info(f"Affected pods during test: {affected_pods}")

            if scenario.load.continuous_load:
                _terminate_client_processes(client_procs, logger)
