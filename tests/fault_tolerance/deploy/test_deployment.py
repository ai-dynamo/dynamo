# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import multiprocessing
import re
import time
from contextlib import contextmanager

import pytest

from tests.fault_tolerance.deploy.client_factory import get_client_function
from tests.fault_tolerance.deploy.parse_factory import parse_test_results
from tests.fault_tolerance.deploy.parse_results import process_overflow_recovery_test
from tests.fault_tolerance.deploy.scenarios import (
    OVERFLOW_SUFFIX,
    RECOVERY_SUFFIX,
    Load,
    TokenOverflowFailure,
    scenarios,
)
from tests.utils.managed_deployment import ManagedDeployment


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
    logger,
    request,
    deployment_spec,
    namespace,
    model,
    load_config: Load,
):
    """Start client processes using factory pattern for client selection.

    Args:
        logger: Logger instance
        request: Pytest request fixture
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

    procs = []
    ctx = multiprocessing.get_context("spawn")

    # Determine retry_delay_or_rate based on client type
    if load_config.client_type == "legacy":
        # Legacy client uses max_request_rate for rate limiting
        retry_delay_or_rate = load_config.max_request_rate
    else:
        # AI-Perf client uses retry_delay between attempts (default 5s)
        retry_delay_or_rate = 5

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
                    request.node.name + OVERFLOW_SUFFIX,
                    i,
                    load_config.overflow_request_count,  # 15 overflow requests
                    load_config.overflow_token_length,  # 2x max_seq_len tokens
                    load_config.output_token_length,
                    load_config.max_retries,
                    retry_delay_or_rate,
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
        procs_recovery = []
        for i in range(load_config.clients):
            proc_normal = ctx.Process(
                target=client_func,
                args=(
                    deployment_spec,
                    namespace,
                    model,
                    request.node.name + RECOVERY_SUFFIX,
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
                        request.node.name,
                        i,
                        load_config.requests_per_client,
                        load_config.input_token_length,
                        load_config.output_token_length,
                        load_config.max_retries,
                        retry_delay_or_rate,
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


def _inject_failures(failures, logger, deployment: ManagedDeployment):  # noqa: F811
    for failure in failures:
        time.sleep(failure.time)

        # Handle TokenOverflowFailure differently - it's a client-side injection
        if isinstance(failure, TokenOverflowFailure):
            # The actual overflow is handled by the client configuration
            # which uses the input_token_length from the Load config
            # This is just logging for visibility
            continue

        pods = deployment.get_pods(failure.pod_name)[failure.pod_name]

        num_pods = len(pods)

        if not pods:
            continue

        replicas = failure.replicas

        if not replicas:
            replicas = num_pods

        logger.info(f"Injecting failure for: {failure}")

        for x in range(replicas):
            pod = pods[x % num_pods]

            if failure.command == "delete_pod":
                deployment.get_pod_logs(failure.pod_name, pod, ".before_delete")
                pod.delete(force=True)
            else:
                processes = deployment.get_processes(pod)
                for process in processes:
                    if failure.command in process.command:
                        logger.info(
                            f"Terminating {failure.pod_name} Pid {process.pid} Command {process.command}"
                        )
                        process.kill(failure.signal)


global_result_list = []


@pytest.fixture(autouse=True)
def results_table(request, scenario):  # noqa: F811
    """Parse and display results for individual test using factory pattern.

    Automatically detects result type (AI-Perf or legacy) and uses
    the appropriate parser.
    """
    yield

    # Determine log paths based on whether this is a mixed token test
    log_paths = []
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
        parse_test_results(
            log_dir=None,
            log_paths=log_paths,
            tablefmt="fancy_grid",
            sla=scenario.load.sla,
            success_threshold=scenario.load.success_threshold,
            print_output=True,
            # force_parser can be set based on client_type if needed
            # force_parser=scenario.load.client_type,
        )
    except Exception:
        logging.exception("Failed to parse results for %s", request.node.name)

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


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_fault_scenario(
    scenario,  # noqa: F811
    request,
    image,
    namespace,
):
    """
    Test dynamo serve deployments with injected failures
    """

    logger = logging.getLogger(request.node.name)

    scenario.deployment.name = "fault-tolerance-test"

    if image:
        scenario.deployment.set_image(image)

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
    ) as deployment:
        with _clients(
            logger,
            request,
            scenario.deployment,
            namespace,
            model,
            scenario.load,  # Pass entire Load config object
        ):
            _inject_failures(scenario.failures, logger, deployment)
