# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
import os
import time

import pytest

from tests.fault_tolerance.deploy.scenarios import DeletePodFailure
from tests.utils.managed_aiperf_deployment import LoadConfig, ManagedAIPerfDeployment
from tests.utils.managed_deployment import DeploymentSpec, ManagedDeployment


@pytest.mark.k8s
@pytest.mark.fault_tolerance
@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.weekly
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_rolling_restart(
    request,
    image: str,
    namespace: str,
    skip_service_restart: bool,
):
    logger = logging.getLogger(request.node.name)

    deployment_spec = DeploymentSpec(
        "/workspace/examples/backends/trtllm/deploy/disagg.yaml"
    )

    if image:
        deployment_spec.set_image(image)

    _model = deployment_spec.get_model()  # noqa: F841

    deployment_spec.set_logging(True, "info")

    # Enable PVC-based log collection with RWX storage
    # Uses nebius-shared-fs storage class by default
    deployment_spec.enable_log_collection(
        pvc_size="500Mi", container_log_dir="/tmp/service_logs"
    )

    # Debug: Show what command wrapping did
    logger.info("=== COMMAND WRAPPING DEBUG ===")
    for service_name in ["Frontend", "TRTLLMPrefillWorker", "TRTLLMDecodeWorker"]:
        if service_name in deployment_spec._deployment_spec["spec"]["services"]:
            service = deployment_spec._deployment_spec["spec"]["services"][service_name]
            main_container = service.get("extraPodSpec", {}).get("mainContainer", {})
            command = main_container.get("command", [])
            logger.info(
                f"{service_name} command: {command[:2] if len(command) >= 2 else command}"
            )
            volumes = service.get("extraPodSpec", {}).get("volumes", [])
            logger.info(f"{service_name} volumes: {volumes}")
    logger.info("=== END DEBUG ===")

    endpoint_url = f"http://{deployment_spec.name.lower()}-{deployment_spec.frontend_service.name.lower()}.{namespace.lower()}.svc.cluster.local:{deployment_spec.port}"

    async with ManagedDeployment(
        namespace=namespace,
        log_dir=request.node.name,
        deployment_spec=deployment_spec,
        skip_service_restart=skip_service_restart,
        enable_volume_log_collection=True,  # Enable volume-based log collection
    ) as deployment:
        #       time.sleep(10)

        #        <k8s-ns>-<lowercase(dgd-name)>.<k8s-ns>.svc.cluster.local

        endpoint_url = f"http://{deployment_spec.name.lower()}-{deployment_spec.frontend_service.name.lower()}.{namespace.lower()}.svc.cluster.local:{deployment_spec.port}"
        print(endpoint_url)

        async with ManagedAIPerfDeployment(
            namespace=namespace,
            load_config=LoadConfig(endpoint_url=endpoint_url, duration_minutes=100),
            log_dir=request.node.name,
        ) as load:
            _results = await load.run(wait_for_completion=False)  # noqa: F841

            await load._wait_for_started()

            time.sleep(30)

            await DeletePodFailure(0, ["frontend"]).execute(deployment, logger)

            await deployment.wait_for_unready(timeout=60, log_interval=10)
            await deployment._wait_for_ready(timeout=1800)

            time.sleep(30)

            await load.terminate()

            # print(results)


#        print(model)


# Populate shared context for validation
#        validation_context["deployment"] = deployment
#       validation_context["namespace"] = namespace

#        with _clients(
#           logger,
#          request.node.name,
#         deployment_spec,
#        namespace,
#       model,
#      scenario.load,  # Pass entire Load config object
#  ) as client_procs:
#     time.sleep(100)
#    # Inject failures and capture which pods were affected
#   affected_pods = await _inject_failures(
#      scenario.failures, logger, deployment
#  )
#  logger.info(f"Affected pods during test: {affected_pods}")

#  if scenario.load.continuous_load:
#     _terminate_client_processes(client_procs, logger)


@pytest.mark.k8s
@pytest.mark.e2e
@pytest.mark.weekly
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_1000_requests_success(
    request,
    image: str,
    namespace: str,
    skip_service_restart: bool,
):
    """
    Test that 1000 requests complete successfully with no errors.

    This is a basic smoke test to verify the deployment is working correctly
    before running more complex fault tolerance tests.
    """
    logger = logging.getLogger(request.node.name)

    deployment_spec = DeploymentSpec(
        "/workspace/examples/backends/trtllm/deploy/disagg.yaml"
    )

    if image:
        deployment_spec.set_image(image)

    model = deployment_spec.get_model()
    logger.info(f"Testing with model: {model}")

    deployment_spec.set_logging(True, "info")

    # Enable PVC-based log collection
    deployment_spec.enable_log_collection(
        pvc_size="500Mi", container_log_dir="/tmp/service_logs"
    )

    endpoint_url = f"http://{deployment_spec.name.lower()}-{deployment_spec.frontend_service.name.lower()}.{namespace.lower()}.svc.cluster.local:{deployment_spec.port}"

    async with ManagedDeployment(
        namespace=namespace,
        log_dir=request.node.name,
        deployment_spec=deployment_spec,
        skip_service_restart=skip_service_restart,
        enable_volume_log_collection=True,
    ):
        logger.info(f"Deployment ready, endpoint: {endpoint_url}")

        # Configure load test for 1000 requests
        load_config = LoadConfig(
            endpoint_url=endpoint_url,
            request_count=1000,
            concurrency=10,  # Moderate concurrency for throughput
            input_tokens_mean=128,  # Shorter prompts for faster test
            output_tokens_mean=32,  # Shorter outputs for faster test
            warmup_requests=10,
        )

        async with ManagedAIPerfDeployment(
            namespace=namespace,
            load_config=load_config,
            log_dir=request.node.name,
        ) as load:
            # Run the load test and wait for completion
            logger.info("Starting load test with 1000 requests...")
            await load.run(wait_for_completion=True)

            # Get results
            results = await load.get_results()

            # Save results for debugging
            results_file = os.path.join(request.node.name, "test_results.json")
            os.makedirs(os.path.dirname(results_file), exist_ok=True)
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {results_file}")

            # Extract key metrics (error_request_count can be null)
            request_count = results.get("request_count", {}).get("avg", 0)
            error_result = results.get("error_request_count")
            error_count = error_result.get("avg", 0) if error_result else 0
            throughput = results.get("request_throughput", {}).get("avg", 0)
            avg_latency = results.get("request_latency", {}).get("avg", 0)

            logger.info("=== Test Results ===")
            logger.info(f"Successful requests: {request_count}")
            logger.info(f"Error requests: {error_count}")
            logger.info(f"Throughput: {throughput:.2f} req/s")
            logger.info(f"Average latency: {avg_latency:.2f} ms")

            # Assertions
            assert (
                request_count >= 1000
            ), f"Expected at least 1000 successful requests, got {request_count}"
            assert error_count == 0, f"Expected 0 errors, got {error_count}"

            logger.info(
                "Test PASSED: 1000 requests completed successfully with no errors"
            )


@pytest.mark.k8s
@pytest.mark.fault_tolerance
@pytest.mark.e2e
@pytest.mark.weekly
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize(
    "upgrade_order",
    [
        pytest.param(
            ["TRTLLMDecodeWorker", "TRTLLMPrefillWorker"], id="decode_then_prefill"
        ),
        pytest.param(
            ["TRTLLMPrefillWorker", "TRTLLMDecodeWorker"], id="prefill_then_decode"
        ),
    ],
)
async def test_rolling_restart_no_errors(
    request,
    image: str,
    namespace: str,
    skip_service_restart: bool,
    upgrade_order: list[str],
):
    """
    DEP-709: Test that rolling restart of disagg deployment has 0 errors.

    Repro steps:
    1. Deploy disagg with 2 prefill + 2 decode replicas
    2. Start continuous load with aiperf
    3. Wait for load to stabilize (~30s)
    4. Trigger rolling upgrade on services in order (parameterized)
    5. Wait for each service to restart
    6. Assert 0 errors throughout
    """
    logger = logging.getLogger(request.node.name)

    deployment_spec = DeploymentSpec(
        "/workspace/examples/backends/trtllm/deploy/disagg.yaml"
    )

    if image:
        deployment_spec.set_image(image)

    model = deployment_spec.get_model()
    logger.info(f"Testing with model: {model}")

    deployment_spec.set_logging(True, "debug")

    # Set 2 replicas for prefill and decode (required for rolling restart)
    deployment_spec.set_service_replicas("TRTLLMPrefillWorker", 2)
    deployment_spec.set_service_replicas("TRTLLMDecodeWorker", 2)

    # Increase readiness probe period and termination grace period
    # This reduces probe frequency during rolling restart and gives pods more time to terminate
    # for service in ["TRTLLMPrefillWorker", "TRTLLMDecodeWorker"]:
    #    deployment_spec.set_service_readiness_probe(service, period_seconds=60)
    #   deployment_spec.set_service_termination_grace_period(service, seconds=120)

    # Enable PVC-based log collection
    deployment_spec.enable_log_collection(
        pvc_size="500Mi", container_log_dir="/tmp/service_logs"
    )

    endpoint_url = f"http://{deployment_spec.name.lower()}-{deployment_spec.frontend_service.name.lower()}.{namespace.lower()}.svc.cluster.local:{deployment_spec.port}"

    async with ManagedDeployment(
        namespace=namespace,
        log_dir=request.node.name,
        deployment_spec=deployment_spec,
        skip_service_restart=skip_service_restart,
        enable_volume_log_collection=True,
    ) as deployment:
        logger.info(
            f"Deployment ready with 2 prefill + 2 decode replicas, endpoint: {endpoint_url}"
        )

        # Configure continuous load for duration of rolling restarts
        load_config = LoadConfig(
            endpoint_url=endpoint_url,
            duration_minutes=30,  # Long enough to cover both restarts
            concurrency=8,
        )

        async with ManagedAIPerfDeployment(
            namespace=namespace,
            load_config=load_config,
            log_dir=request.node.name,
        ) as load:
            # Start continuous load
            logger.info("Starting continuous load...")
            await load.run(wait_for_completion=False)
            await load._wait_for_started()

            # Wait for load to stabilize
            logger.info("Waiting 30s for load to stabilize...")
            await asyncio.sleep(60)

            # Trigger rolling upgrades in the parameterized order
            # Use CR status (wait_for_unready + _wait_for_ready) like RollingUpgradeFailure
            logger.info(f"Upgrade order: {upgrade_order}")
            for i, service in enumerate(upgrade_order):
                logger.info(f"Triggering rolling upgrade on {service}...")

                await deployment.trigger_rolling_upgrade([service])

                # Wait for CR status to cycle: unready -> ready
                await deployment.wait_for_unready(timeout=60, log_interval=10)
                await deployment._wait_for_ready(timeout=1800)
                logger.info(f"{service} restarted successfully")

                # Wait between restarts (but not after the last one)
                if i < len(upgrade_order) - 1:
                    logger.info("Waiting 30s between restarts...")
                    await asyncio.sleep(30)

            # Let load continue for a bit after restarts
            logger.info("Waiting 30s after restarts...")
            await asyncio.sleep(60)

            # Terminate load and get results
            logger.info("Terminating load and collecting results...")
            # await load._wait_for_completion()
            await load.terminate()
            results = await load.get_results()

            # Save results for debugging
            results_file = os.path.join(
                request.node.name, "rolling_restart_results.json"
            )
            os.makedirs(os.path.dirname(results_file), exist_ok=True)
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {results_file}")

            # Extract key metrics (error_request_count can be null)
            request_count = results.get("request_count", {}).get("avg", 0)
            error_result = results.get("error_request_count")
            error_count = error_result.get("avg", 0) if error_result else 0
            throughput = results.get("request_throughput", {}).get("avg", 0)

            logger.info("=== Rolling Restart Test Results ===")
            logger.info(f"Successful requests: {request_count}")
            logger.info(f"Error requests: {error_count}")
            logger.info(f"Throughput: {throughput:.2f} req/s")

            # Assert 0 errors - this is the key DEP-709 requirement
            assert (
                error_count == 0
            ), f"DEP-709: Expected 0 errors during rolling restart, got {error_count}"
            assert (
                request_count > 0
            ), "Expected some successful requests during rolling restart"

            logger.info("Test PASSED: Rolling restart completed with 0 errors")
