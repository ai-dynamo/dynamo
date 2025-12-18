# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# TODO: Update to use dynamic port allocation (allocate_free_port) for parallel execution
# Currently uses hardcoded ports: FRONTEND_PORT (8000), system ports (8081, 8082)
# See tests/fault_tolerance/migration/test_sglang.py for dynamic port pattern

import logging
import os
import shutil
import time

import pytest

from tests.conftest import NatsServer
from tests.fault_tolerance.etcd_ha.utils import (
    DynamoFrontendProcess,
    EtcdCluster,
    send_inference_request,
    wait_for_processes_to_terminate,
)
from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME
from tests.utils.engine_process import FRONTEND_PORT
from tests.utils.managed_process import ManagedProcess
from tests.utils.payloads import check_health_generate, check_models_api

logger = logging.getLogger(__name__)


class DynamoWorkerProcess(ManagedProcess):
    """Process manager for Dynamo worker with SGLang backend and ETCD HA support"""

    def __init__(self, request, etcd_endpoints: list, mode: str = "agg"):
        """
        Initialize SGLang worker process with ETCD HA support.

        Args:
            request: pytest request object
            etcd_endpoints: List of ETCD endpoints
            mode: One of "agg", "prefill", "decode"
        """
        command = [
            "python3",
            "-m",
            "dynamo.sglang",
            "--model-path",
            FAULT_TOLERANCE_MODEL_NAME,
            "--served-model-name",
            FAULT_TOLERANCE_MODEL_NAME,
            "--page-size",
            "16",
            "--tp",
            "1",
            "--trust-remote-code",
        ]

        # Add mode-specific arguments
        if mode == "agg":
            # Aggregated mode - add skip-tokenizer-init
            command.append("--skip-tokenizer-init")
        else:
            # Disaggregated mode - add disaggregation arguments
            command.extend(
                [
                    "--disaggregation-mode",
                    mode,
                    "--disaggregation-bootstrap-port",
                    "12345",
                    "--host",
                    "0.0.0.0",
                    "--disaggregation-transfer-backend",
                    "nixl",
                ]
            )

        health_check_urls = [
            (f"http://localhost:{FRONTEND_PORT}/v1/models", check_models_api),
            (f"http://localhost:{FRONTEND_PORT}/health", check_health_generate),
        ]

        # Set port based on worker type
        if mode == "prefill":
            port = "8082"
            health_check_urls = [(f"http://localhost:{port}/health", self.is_ready)]
        elif mode == "decode":
            port = "8081"
            health_check_urls = [(f"http://localhost:{port}/health", self.is_ready)]
        else:  # agg (aggregated mode)
            port = "8081"

        # Set debug logging and ETCD endpoints
        env = os.environ.copy()
        env["DYN_LOG"] = "debug"
        env["ETCD_ENDPOINTS"] = ",".join(etcd_endpoints)
        env["DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS"] = '["generate"]'
        env["DYN_SYSTEM_PORT"] = port

        # Set GPU assignment for disaggregated mode
        if mode == "decode":
            env["CUDA_VISIBLE_DEVICES"] = "1"  # Use GPU 1 for decode worker
        elif mode == "prefill":
            env["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0 for prefill worker
        # For agg (aggregated) mode, use default GPU assignment

        # Set log directory based on worker type
        log_dir = f"{request.node.name}_{mode}_worker"

        # Clean up any existing log directory from previous runs
        try:
            shutil.rmtree(log_dir)
            logger.info(f"Cleaned up existing log directory: {log_dir}")
        except FileNotFoundError:
            pass

        super().__init__(
            command=command,
            env=env,
            health_check_urls=health_check_urls,
            timeout=300,
            display_output=True,
            terminate_existing=False,
            # Ensure any orphaned SGLang engine cores or child helpers are cleaned up
            stragglers=[
                "SGLANG:EngineCore",
            ],
            straggler_commands=[
                "-m dynamo.sglang",
            ],
            log_dir=log_dir,
        )

        self.mode = mode

    def is_ready(self, response) -> bool:
        """Check the health of the worker process"""
        try:
            data = response.json()
            if data.get("status") == "ready":
                logger.info(f"{self.mode.capitalize()} worker status is ready")
                return True
            logger.warning(
                f"{self.mode.capitalize()} worker status is not ready: {data.get('status')}"
            )
        except ValueError:
            logger.warning(
                f"{self.mode.capitalize()} worker health response is not valid JSON"
            )
        return False


@pytest.mark.sglang
@pytest.mark.gpu_1
@pytest.mark.e2e
@pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME)
@pytest.mark.timeout(600)
def test_etcd_ha_failover_sglang_aggregated(request, predownload_models):
    """
    Test ETCD High Availability with repeated node failures and recoveries using SGLang.

    This test:
    1. Starts a 3-node ETCD cluster
    2. Starts NATS, frontend, and an SGLang worker
    3. Cycles through each of the 3 replicas:
       - Terminate the replica by index
       - Send inference request to verify system still works
       - Restart the terminated node

    This ensures testing of:
    - ETCD leader termination
    - Frontend/worker disconnection from their connected ETCD replica
    """
    # Step 1: Start NATS server
    with NatsServer(request):
        logger.info("NATS server started successfully")

        # Step 2: Start 3-node ETCD cluster
        num_replicas = 3
        with EtcdCluster(request, num_replicas=num_replicas) as etcd_cluster:
            logger.info("3-node ETCD cluster started successfully")

            # Get the endpoints for all ETCD nodes
            etcd_endpoints = etcd_cluster.get_client_endpoints()
            logger.info(f"ETCD endpoints: {etcd_endpoints}")

            # Step 3: Start the frontend with ETCD endpoints
            with DynamoFrontendProcess(request, etcd_endpoints):
                logger.info("Frontend started successfully")

                # Step 4: Start an SGLang worker
                with DynamoWorkerProcess(request, etcd_endpoints, mode="agg"):
                    logger.info("SGLang worker started successfully")
                    # Small wait to ensure worker is fully ready
                    time.sleep(2)

                    # Step 5: Send initial inference request to verify system is working
                    logger.info("Sending initial inference request")
                    result = send_inference_request("What is 2+2? The answer is")
                    assert (
                        "4" in result.lower() or "four" in result.lower()
                    ), f"Expected '4' or 'four' in response, got: '{result}'"

                    # Step 6: Cycle through each replica to terminate/verify/restart
                    for i in range(num_replicas):
                        # Terminate a replica
                        logger.info(f"Iteration {i}: Terminating replica etcd-{i}")
                        etcd_cluster.terminate_replica(i)

                        # Send inference request to verify system still works
                        logger.info(
                            f"Iteration {i}: Sending inference request after termination"
                        )
                        result = send_inference_request(
                            "The capital of France is", max_tokens=20
                        )
                        assert (
                            "paris" in result.lower()
                        ), f"Iteration {i}: Expected 'Paris' in response, got: '{result}'"

                        # Restart the terminated replica
                        logger.info(f"Iteration {i}: Restarting replica etcd-{i}")
                        etcd_cluster.restart_replica(i)


@pytest.mark.sglang
@pytest.mark.gpu_2
@pytest.mark.e2e
@pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME)
@pytest.mark.timeout(600)
def test_etcd_ha_failover_sglang_disaggregated(
    request, predownload_models, set_ucx_tls_no_mm
):
    """
    Test ETCD High Availability with repeated node failures and recoveries in disaggregated mode using SGLang.

    This test:
    1. Starts a 3-node ETCD cluster
    2. Starts NATS, frontend, and both prefill and decode SGLang workers
    3. Cycles through each of the 3 replicas:
       - Terminate the replica by index
       - Send inference request to verify system still works
       - Restart the terminated node

    This ensures testing of:
    - ETCD leader termination
    - Frontend/worker disconnection from their connected ETCD replica

    Note: This test requires 2 GPUs to run decode and prefill workers on separate GPUs.
    """
    # Step 1: Start NATS server
    with NatsServer(request):
        logger.info("NATS server started successfully")

        # Step 2: Start 3-node ETCD cluster
        num_replicas = 3
        with EtcdCluster(request, num_replicas=num_replicas) as etcd_cluster:
            logger.info("3-node ETCD cluster started successfully")

            # Get the endpoints for all ETCD nodes
            etcd_endpoints = etcd_cluster.get_client_endpoints()
            logger.info(f"ETCD endpoints: {etcd_endpoints}")

            # Step 3: Start the frontend with ETCD endpoints
            with DynamoFrontendProcess(request, etcd_endpoints):
                logger.info("Frontend started successfully")

                # Step 4: Start the decode worker
                with DynamoWorkerProcess(request, etcd_endpoints, mode="decode"):
                    logger.info("Decode worker started successfully")

                    # Step 5: Start the prefill worker
                    with DynamoWorkerProcess(request, etcd_endpoints, mode="prefill"):
                        logger.info("Prefill worker started successfully")
                        # Small wait to ensure workers are fully ready
                        time.sleep(2)

                        # Step 6: Send initial inference request to verify system is working
                        logger.info("Sending initial inference request")
                        result = send_inference_request("What is 2+2? The answer is")
                        assert (
                            "4" in result.lower() or "four" in result.lower()
                        ), f"Expected '4' or 'four' in response, got: '{result}'"

                        # Step 7: Cycle through each replica to terminate/verify/restart
                        for i in range(num_replicas):
                            # Terminate a replica
                            logger.info(f"Iteration {i}: Terminating replica etcd-{i}")
                            etcd_cluster.terminate_replica(i)

                            # Send inference request to verify system still works
                            logger.info(
                                f"Iteration {i}: Sending inference request after termination"
                            )
                            result = send_inference_request(
                                "The capital of France is", max_tokens=20
                            )
                            assert (
                                "paris" in result.lower()
                            ), f"Iteration {i}: Expected 'Paris' in response, got: '{result}'"

                            # Restart the terminated replica
                            logger.info(f"Iteration {i}: Restarting replica etcd-{i}")
                            etcd_cluster.restart_replica(i)


@pytest.mark.sglang
@pytest.mark.gpu_1
@pytest.mark.e2e
@pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME)
@pytest.mark.timeout(600)
def test_etcd_non_ha_shutdown_sglang_aggregated(request, predownload_models):
    """
    Test that frontend and worker shut down when single ETCD node is terminated using SGLang.

    This test:
    1. Starts a single ETCD node (no cluster)
    2. Starts NATS, frontend, and an SGLang worker
    3. Sends an inference request to verify the system works
    4. Terminates the single ETCD node
    5. Verifies that frontend and worker shut down gracefully
    """
    # Step 1: Start NATS server
    with NatsServer(request):
        logger.info("NATS server started successfully")

        # Step 2: Start single ETCD node using EtcdCluster with num_replicas=1
        with EtcdCluster(request, num_replicas=1) as etcd_cluster:
            logger.info("Single ETCD node started successfully")

            # Get the endpoint for the single ETCD node
            etcd_endpoints = etcd_cluster.get_client_endpoints()
            logger.info(f"ETCD endpoint: {etcd_endpoints}")

            # Step 3: Start the frontend with ETCD endpoint
            with DynamoFrontendProcess(request, etcd_endpoints) as frontend:
                logger.info("Frontend started successfully")

                # Step 4: Start an SGLang worker
                with DynamoWorkerProcess(request, etcd_endpoints, mode="agg") as worker:
                    logger.info("SGLang worker started successfully")
                    # Small wait to ensure worker is fully ready
                    time.sleep(2)

                    # Step 5: Send inference request to verify system is working
                    logger.info("Sending inference request")
                    result = send_inference_request("What is 2+2? The answer is")
                    assert (
                        "4" in result.lower() or "four" in result.lower()
                    ), f"Expected '4' or 'four' in response, got: '{result}'"

                    logger.info("System is working correctly with single ETCD node")

                    # Step 6: Terminate the ETCD node
                    logger.info("Terminating single ETCD node")
                    etcd_cluster.stop()

                    # Step 7: Wait and verify frontend and worker detect the loss
                    wait_for_processes_to_terminate(
                        {"Worker": worker, "Frontend": frontend}
                    )


@pytest.mark.sglang
@pytest.mark.gpu_2
@pytest.mark.e2e
@pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME)
@pytest.mark.timeout(600)
def test_etcd_non_ha_shutdown_sglang_disaggregated(
    request, predownload_models, set_ucx_tls_no_mm
):
    """
    Test that frontend and workers shut down when single ETCD node is terminated in disaggregated mode using SGLang.

    This test:
    1. Starts a single ETCD node (no cluster)
    2. Starts NATS, frontend, and both prefill and decode SGLang workers
    3. Sends an inference request to verify the system works
    4. Terminates the single ETCD node
    5. Verifies that frontend and both workers shut down gracefully

    Note: This test requires 2 GPUs to run decode and prefill workers on separate GPUs.
    """
    # Step 1: Start NATS server
    with NatsServer(request):
        logger.info("NATS server started successfully")

        # Step 2: Start single ETCD node using EtcdCluster with num_replicas=1
        with EtcdCluster(request, num_replicas=1) as etcd_cluster:
            logger.info("Single ETCD node started successfully")

            # Get the endpoint for the single ETCD node
            etcd_endpoints = etcd_cluster.get_client_endpoints()
            logger.info(f"ETCD endpoint: {etcd_endpoints}")

            # Step 3: Start the frontend with ETCD endpoint
            with DynamoFrontendProcess(request, etcd_endpoints) as frontend:
                logger.info("Frontend started successfully")

                # Step 4: Start the decode worker
                with DynamoWorkerProcess(
                    request, etcd_endpoints, mode="decode"
                ) as decode_worker:
                    logger.info("Decode worker started successfully")

                    # Step 5: Start the prefill worker
                    with DynamoWorkerProcess(
                        request, etcd_endpoints, mode="prefill"
                    ) as prefill_worker:
                        logger.info("Prefill worker started successfully")
                        # Small wait to ensure workers are fully ready
                        time.sleep(2)

                        # Step 6: Send inference request to verify system is working
                        logger.info("Sending inference request")
                        result = send_inference_request("What is 2+2? The answer is")
                        assert (
                            "4" in result.lower() or "four" in result.lower()
                        ), f"Expected '4' or 'four' in response, got: '{result}'"

                        logger.info(
                            "System is working correctly with single ETCD node in disaggregated mode"
                        )

                        # Step 7: Terminate the ETCD node
                        logger.info("Terminating single ETCD node")
                        etcd_cluster.stop()

                        # Step 8: Wait and verify frontend and both workers detect the loss
                        wait_for_processes_to_terminate(
                            {
                                "Decode Worker": decode_worker,
                                "Prefill Worker": prefill_worker,
                                "Frontend": frontend,
                            }
                        )
