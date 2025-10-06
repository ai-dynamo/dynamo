# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import multiprocessing
import time
from contextlib import contextmanager

import pytest

from tests.fault_tolerance.deploy.client_factory import get_client_function
from tests.fault_tolerance.deploy.parse_factory import parse_test_results
from tests.fault_tolerance.deploy.scenarios import Load, scenarios
from tests.utils.managed_deployment import ManagedDeployment


@pytest.fixture(params=scenarios.keys())
def scenario(request, client_type):
    """Get scenario and optionally override client type from command line.

    If --client-type is specified, it overrides the scenario's default client type.
    """
    scenario_obj = scenarios[request.param]

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

    # Use factory to auto-detect and parse results
    try:
        parse_test_results(
            log_dir=None,
            log_paths=[request.node.name],
            tablefmt="fancy_grid",
            sla=scenario.load.sla,
            # force_parser can be set based on client_type if needed
            # force_parser=scenario.load.client_type,
        )
    except Exception as e:
        logging.error(f"Failed to parse results for {request.node.name}: {e}")

    global_result_list.append(request.node.name)


@pytest.fixture(autouse=True, scope="session")
def results_summary():
    """Parse and display combined results for all tests in session.

    Automatically detects result types and uses appropriate parsers.
    """
    yield

    if not global_result_list:
        logging.info("No test results to summarize")
        return

    # Use factory to auto-detect and parse combined results
    try:
        parse_test_results(
            log_dir=None,
            log_paths=global_result_list,
            tablefmt="fancy_grid",
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

    scenario.deployment.disable_grove()

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
