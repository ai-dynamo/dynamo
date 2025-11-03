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
from tests.fault_tolerance.deploy.validations import get_validation_for_scenario, get_validation_for_results
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
    """Inject failures and return info about affected pods.
    
    Returns:
        Dict mapping failure info to list of affected pod names
        Example: {"VllmDecodeWorker:delete_pod": ["pod-abc123", "pod-xyz789"]}
    """
    affected_pods = {}
    
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
        
        # Track which pods were affected by this failure
        failure_key = f"{failure.pod_name}:{failure.command}"
        if failure_key not in affected_pods:
            affected_pods[failure_key] = []

        for x in range(replicas):
            pod = pods[x % num_pods]
            
            # Capture the exact pod name before we kill it
            pod_name = pod.name
            affected_pods[failure_key].append(pod_name)
            
            logger.info(f"Target pod for failure: {pod_name}")

            if failure.command == "delete_pod":
                deployment.get_pod_logs(failure.pod_name, pod, ".before_delete")
                logger.info(f"Deleting pod: {pod_name}")
                pod.delete(force=True)
            else:
                processes = deployment.get_processes(pod)
                for process in processes:
                    if failure.command in process.command:
                        logger.info(
                            f"Terminating {failure.pod_name} Pid {process.pid} Command {process.command} in pod {pod_name}"
                        )
                        process.kill(failure.signal)
    
    return affected_pods


global_result_list = []
# Global storage for test results (used by validation fixture)
test_results_cache = {}


@pytest.fixture(autouse=True)
def test_context(request, scenario):  # noqa: F811
    """Provides shared context between test execution and validation.
    
    This fixture creates a shared dictionary that the test populates during
    execution (deployment, namespace, affected_pods), then uses that data
    in teardown to parse results and run validation.

    Automatically detects result type (AI-Perf or legacy) and uses
    the appropriate parser. After parsing, immediately runs validation.
    """
    # Shared context that test will populate during execution
    context = {"deployment": None, "namespace": None, "affected_pods": {}}
    
    yield context  # Test receives this and populates it

    logger = logging.getLogger(request.node.name)
    test_name = request.node.name

    # Use factory to auto-detect and parse results
    try:
        results = parse_test_results(
            log_dir=None,
            log_paths=[test_name],
            tablefmt="fancy_grid",
            sla=scenario.load.sla,
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
                
                # Get validation function for this scenario
                result_validation_func = get_validation_for_results(test_name, scenario)
                
                scenario_validation_func = get_validation_for_scenario(test_name, scenario)
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
                    
                    # Run validation with context populated by test
                    scenario_validation_func(
                        scenario=scenario,
                        log_dir=test_name,
                        metrics=metrics,
                        deployment=context.get("deployment"),
                        namespace=context.get("namespace"),
                        recovery_time=recovery_time,
                        affected_pods=context.get("affected_pods", {}),
                    )
                    
                    result_validation_func(
                        scenario=scenario,
                        metrics=metrics,
                        recovery_time=recovery_time,
                    )
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
                
    except Exception as e:
        logging.error(f"Failed to parse results for {test_name}: {e}")

    global_result_list.append(test_name)


# NOTE: validation fixture removed - validation now runs directly in results_table
# after results are parsed, ensuring proper execution order


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
    test_context,  # noqa: F811  # Shared context for passing data to validation
):
    """
    Test dynamo serve deployments with injected failures
    
    Flow:
    1. test_context fixture creates empty dict: {"deployment": None, "namespace": None, "affected_pods": {}}
    2. This test populates it: test_context["deployment"] = deployment, etc.
    3. After test completes, fixture reads test_context and runs validation
    4. Validation uses the populated values to verify test results and K8s events
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
        # Populate shared context for validation
        test_context["deployment"] = deployment
        test_context["namespace"] = namespace

        with _clients(
            logger,
            request,
            scenario.deployment,
            namespace,
            model,
            scenario.load,  # Pass entire Load config object
        ):
            # Inject failures and capture which pods were affected
            affected_pods = _inject_failures(scenario.failures, logger, deployment)
            test_context["affected_pods"] = affected_pods
            
            logger.info(f"Affected pods during test: {affected_pods}")
