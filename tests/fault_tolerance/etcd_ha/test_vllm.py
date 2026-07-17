# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import shutil
import threading
import time
from enum import Enum

import psutil
import pytest
import requests

from tests.conftest import NatsServer
from tests.fault_tolerance.etcd_ha.utils import (
    DynamoFrontendProcess,
    EtcdCluster,
    send_inference_request,
    wait_for_processes_to_terminate,
)
from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME, DynamoPortRange
from tests.utils.device import (
    build_nixl_kv_transfer_config,
    get_default_vllm_block_size,
)
from tests.utils.engine_process import FRONTEND_PORT
from tests.utils.managed_process import ManagedProcess
from tests.utils.payloads import check_health_generate, check_models_api
from tests.utils.port_utils import allocate_port, deallocate_port

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.fault_tolerance,
    pytest.mark.vllm,
]


class WorkerMode(Enum):
    AGGREGATED = "aggregated"
    PREFILL = "prefill"
    DECODE = "decode"


class DynamoWorkerProcess(ManagedProcess):
    """Process manager for Dynamo worker with vLLM backend and ETCD HA support"""

    def __init__(
        self,
        request,
        etcd_endpoints: list,
        mode: WorkerMode = WorkerMode.AGGREGATED,
    ):
        self.system_port = allocate_port(DynamoPortRange.SERVE.value)
        self._ports_cleanup_done = False
        request.addfinalizer(self._release_worker_ports)

        command = [
            "python3",
            "-m",
            "dynamo.vllm",
            "--model",
            FAULT_TOLERANCE_MODEL_NAME,
            "--enforce-eager",
            "--gpu-memory-utilization",
            "0.45",
            "--max-model-len",
            "8192",
            "--block-size",
            str(get_default_vllm_block_size()),
        ]

        port = str(self.system_port)

        if mode == WorkerMode.PREFILL:
            command.extend(["--disaggregation-mode", "prefill"])
            health_check_urls = [(f"http://localhost:{port}/health", self.is_ready)]
        else:
            if mode == WorkerMode.DECODE:
                command.extend(["--disaggregation-mode", "decode"])
            health_check_urls = [
                (f"http://localhost:{port}/health", self.is_ready),
                (f"http://localhost:{FRONTEND_PORT}/v1/models", check_models_api),
                (f"http://localhost:{FRONTEND_PORT}/health", check_health_generate),
            ]

        env = os.environ.copy()
        env["DYN_LOG"] = "debug"
        env["ETCD_ENDPOINTS"] = ",".join(etcd_endpoints)
        env["DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS"] = '["generate"]'
        env["DYN_SYSTEM_PORT"] = port

        if mode != WorkerMode.AGGREGATED:
            command.extend(
                [
                    "--kv-transfer-config",
                    json.dumps(build_nixl_kv_transfer_config()),
                ]
            )
            self.fpm_port = allocate_port(DynamoPortRange.FPM.value)
            env["DYN_FORWARDPASS_METRIC_PORT"] = str(self.fpm_port)

        if mode == WorkerMode.PREFILL:
            self.kv_event_port = allocate_port(DynamoPortRange.SERVE.value)
            self.nixl_side_channel_port = allocate_port(DynamoPortRange.NIXL.value)
            command.extend(
                [
                    "--kv-events-config",
                    json.dumps(
                        {
                            "publisher": "zmq",
                            "topic": "kv-events",
                            "endpoint": f"tcp://*:{self.kv_event_port}",
                            "enable_kv_cache_events": True,
                        }
                    ),
                ]
            )
            env["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(self.nixl_side_channel_port)

        worker_type = "prefill_worker" if mode == WorkerMode.PREFILL else "worker"
        log_dir = f"{request.node.name}_{worker_type}"

        try:
            shutil.rmtree(log_dir)
            logger.info(f"Cleaned up existing log directory: {log_dir}")
        except FileNotFoundError:
            pass

        super().__init__(
            command=command,
            env=env,
            health_check_urls=health_check_urls,
            timeout=120,
            display_output=True,
            terminate_all_matching_process_names=False,
            stragglers=["VLLM::EngineCore"],
            straggler_commands=["-m dynamo.vllm"],
            log_dir=log_dir,
        )

        self.mode = mode

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release allocated ports when worker exits."""
        super_error = None
        try:
            result = super().__exit__(exc_type, exc_val, exc_tb)
        except Exception as exc:
            super_error = exc
            result = None

        try:
            self._release_worker_ports()
        except Exception as exc:
            if super_error is not None:
                raise exc from super_error
            raise

        if super_error is not None:
            raise super_error

        return result

    def _release_worker_ports(self):
        if self._ports_cleanup_done:
            return

        cleanup_errors = []
        for port_attr in (
            "system_port",
            "fpm_port",
            "kv_event_port",
            "nixl_side_channel_port",
        ):
            port = getattr(self, port_attr, None)
            if port is None:
                continue

            try:
                deallocate_port(port)
            except Exception as exc:
                logger.exception("Failed to release %s=%s", port_attr, port)
                cleanup_errors.append(exc)
            else:
                setattr(self, port_attr, None)

        if cleanup_errors:
            raise cleanup_errors[0]

        self._ports_cleanup_done = True

    def is_ready(self, response) -> bool:
        """Check the health of the worker process"""
        worker_type = "Prefill worker" if self.mode == WorkerMode.PREFILL else "Worker"
        try:
            data = response.json()
            if data.get("status") == "ready":
                logger.info(f"{worker_type} status is ready")
                return True
            logger.warning(f"{worker_type} status is not ready: {data.get('status')}")
        except ValueError:
            logger.warning(f"{worker_type} health response is not valid JSON")
        return False


@pytest.mark.gpu_1
@pytest.mark.xpu_1
@pytest.mark.e2e
@pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME)
@pytest.mark.nightly
@pytest.mark.timeout(600)
def test_etcd_ha_failover_vllm_aggregated(request, predownload_models):
    """
    Test ETCD High Availability with repeated node failures and recoveries.

    This test:
    1. Starts a 3-node ETCD cluster
    2. Starts NATS, frontend, and a vLLM worker
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

                # Step 4: Start a vLLM worker
                with DynamoWorkerProcess(request, etcd_endpoints):
                    logger.info("Worker started successfully")

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


@pytest.mark.gpu_1
@pytest.mark.xpu_1
@pytest.mark.e2e
@pytest.mark.nightly
@pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME)
@pytest.mark.timeout(600)
def test_etcd_ha_failover_vllm_disaggregated(
    request, predownload_models, set_ucx_tls_no_mm
):
    """
    Test ETCD High Availability with repeated node failures and recoveries in disaggregated mode.

    This test:
    1. Starts a 3-node ETCD cluster
    2. Starts NATS, frontend, and both prefill and decode vLLM workers
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

                # Step 4: Start the prefill worker
                with DynamoWorkerProcess(
                    request,
                    etcd_endpoints,
                    mode=WorkerMode.PREFILL,
                ):
                    logger.info("Prefill worker started successfully")

                    # Step 5: Start the decode worker
                    with DynamoWorkerProcess(
                        request,
                        etcd_endpoints,
                        mode=WorkerMode.DECODE,
                    ):
                        logger.info("Decode worker started successfully")

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


@pytest.mark.gpu_1
@pytest.mark.xpu_1
@pytest.mark.e2e
@pytest.mark.nightly
@pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME)
@pytest.mark.timeout(600)
def test_etcd_non_ha_shutdown_vllm_aggregated(request, predownload_models):
    """
    Test that frontend and worker shut down when single ETCD node is terminated.

    This test:
    1. Starts a single ETCD node (no cluster)
    2. Starts NATS, frontend, and a vLLM worker
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

                # Step 4: Start a vLLM worker
                with DynamoWorkerProcess(request, etcd_endpoints) as worker:
                    logger.info("Worker started successfully")

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


@pytest.mark.gpu_1
@pytest.mark.xpu_1
@pytest.mark.e2e
@pytest.mark.nightly
@pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME)
@pytest.mark.timeout(600)
def test_etcd_non_ha_shutdown_vllm_disaggregated(
    request, predownload_models, set_ucx_tls_no_mm
):
    """
    Test that frontend and workers shut down when single ETCD node is terminated in disaggregated mode.

    This test:
    1. Starts a single ETCD node (no cluster)
    2. Starts NATS, frontend, and both prefill and decode vLLM workers
    3. Sends an inference request to verify the system works
    4. Terminates the single ETCD node
    5. Verifies that frontend and both workers shut down gracefully
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

                # Step 4: Start the prefill worker
                with DynamoWorkerProcess(
                    request,
                    etcd_endpoints,
                    mode=WorkerMode.PREFILL,
                ) as prefill_worker:
                    logger.info("Prefill worker started successfully")

                    # Step 5: Start the decode worker
                    with DynamoWorkerProcess(
                        request,
                        etcd_endpoints,
                        mode=WorkerMode.DECODE,
                    ) as decode_worker:
                        logger.info("Decode worker started successfully")

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


# Lease-loss zombie: a non-cancellable in-flight request (modeled by
# SIGSTOP-ing the vLLM engine) must not wedge worker teardown.
ZOMBIE_GRACEFUL_SHUTDOWN_TIMEOUT_SECS = 10  # worker-side drain bound
# Hold the frontend's drain open so it does not abort the request and mask the
# worker-side behavior under test.
ZOMBIE_FRONTEND_DRAIN_TIMEOUT_SECS = 300
ZOMBIE_WORKER_EXIT_DEADLINE_SECS = ZOMBIE_GRACEFUL_SHUTDOWN_TIMEOUT_SECS + 50
ZOMBIE_INFLIGHT_MAX_TOKENS = 8000


def _zombie_verify_serving():
    r = requests.post(
        f"http://localhost:{FRONTEND_PORT}/v1/completions",
        json={
            "model": FAULT_TOLERANCE_MODEL_NAME,
            "prompt": "The capital of France is",
            "max_tokens": 5,
            "temperature": 0.0,
        },
        timeout=120,
    )
    assert (
        r.status_code == 200
    ), f"pre-fault completion failed: {r.status_code} {r.text}"


def _zombie_start_inflight_request():
    """Fire a long completion in the background so a request is in flight.

    Non-streaming on purpose: the worker's handle_payload runs the full
    generation before returning, holding the push-endpoint inflight counter.
    """

    errors: list = []

    def _run():
        try:
            requests.post(
                f"http://localhost:{FRONTEND_PORT}/v1/completions",
                json={
                    "model": FAULT_TOLERANCE_MODEL_NAME,
                    "prompt": "Tell me a very long story.",
                    "max_tokens": ZOMBIE_INFLIGHT_MAX_TOKENS,
                    "temperature": 0.0,
                },
                timeout=600,
            )
        except requests.RequestException as exc:
            errors.append(exc)
            logger.info("in-flight request ended: %s", exc)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t, errors


def _freeze_vllm_engine_descendants(worker_pid: int) -> list:
    """SIGSTOP the vLLM engine (rank) subprocess(es) under this worker.

    Scoped to the worker's process tree so concurrent tests/workers on the same
    host are untouched. A frozen engine cannot finish or abort the in-flight
    request, so it stays pinned in the endpoint inflight counter -- a
    non-cancellable inflight.
    """
    frozen = []
    for child in psutil.Process(worker_pid).children(recursive=True):
        try:
            if "EngineCore" in child.name() or "EngineCore" in " ".join(
                child.cmdline()
            ):
                child.suspend()
                frozen.append(child.pid)
                logger.info("SIGSTOP vLLM engine pid=%s", child.pid)
        except psutil.Error:
            continue
    assert frozen, "no vLLM EngineCore descendant of the worker to freeze"
    return frozen


def _resume_kill(pids: list) -> None:
    """Resume then kill frozen engines so they can't hang teardown or hold a GPU."""
    for pid in pids:
        try:
            proc = psutil.Process(pid)
            proc.resume()
            proc.kill()
        except psutil.NoSuchProcess:
            pass


@pytest.mark.gpu_1
@pytest.mark.xpu_1
@pytest.mark.e2e
@pytest.mark.nightly
@pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME)
@pytest.mark.timeout(420)
@pytest.mark.parametrize("request_plane", ["tcp", "nats"])
def test_etcd_lease_loss_zombie_vllm_frozen_engine(
    request, monkeypatch, request_plane, predownload_models
):
    """A worker that loses its etcd lease with a non-cancellable in-flight
    request must still exit.

    Repro: freeze the vLLM engine mid-generation so the request can neither
    complete nor be aborted, then kill etcd. With the bounded endpoint drain the
    worker times out the drain and exits; the unbounded drain wedges (zombie).

    Parametrized over both request planes: the bounded drain must cover the
    default ``tcp`` plane (SharedTcpServer) as well as ``nats`` (PushEndpoint).
    """
    # Hold the frontend's drain open so only the worker behavior is measured.
    monkeypatch.setenv("DYN_REQUEST_PLANE", request_plane)
    monkeypatch.setenv(
        "DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS",
        str(ZOMBIE_FRONTEND_DRAIN_TIMEOUT_SECS),
    )

    with NatsServer(request):
        with EtcdCluster(request, num_replicas=1) as etcd_cluster:
            etcd_endpoints = etcd_cluster.get_client_endpoints()
            with DynamoFrontendProcess(request, etcd_endpoints):
                worker = DynamoWorkerProcess(
                    request, etcd_endpoints, mode=WorkerMode.AGGREGATED
                )
                worker.env["DYN_RUNTIME_GRACEFUL_SHUTDOWN_TIMEOUT_SECS"] = str(
                    ZOMBIE_GRACEFUL_SHUTDOWN_TIMEOUT_SECS
                )
                with worker:
                    _zombie_verify_serving()

                    logger.info("Starting long in-flight request")
                    (
                        inflight_thread,
                        inflight_errors,
                    ) = _zombie_start_inflight_request()
                    time.sleep(5)  # let it reach decode
                    assert inflight_thread.is_alive(), (
                        "in-flight request ended before engine freeze: "
                        f"{inflight_errors!r}"
                    )

                    # Freeze the engine: the in-flight request is now
                    # non-cancellable (cannot complete or be aborted). Resume/kill
                    # in finally before the worker context unwinds, so a SIGSTOP-ed
                    # engine can't hang teardown or strand a GPU-holding process.
                    frozen_pids = _freeze_vllm_engine_descendants(worker.proc.pid)
                    try:
                        time.sleep(1)

                        logger.info("Terminating ETCD to induce lease loss")
                        etcd_cluster.stop()

                        # Bounded drain -> worker exits; unbounded -> zombie.
                        wait_for_processes_to_terminate(
                            {"Worker": worker},
                            timeout=ZOMBIE_WORKER_EXIT_DEADLINE_SECS,
                        )
                    finally:
                        _resume_kill(frozen_pids)
