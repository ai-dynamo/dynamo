# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for frontend tests.

Handles conditional test collection to prevent import errors when required
dependencies are not installed in the current environment.
"""

import importlib.util
import logging
import os
import shutil
import time

import pytest
import requests

from tests.utils.constants import QWEN
from tests.utils.managed_process import DynamoFrontendProcess, ManagedProcess
from tests.utils.port_utils import allocate_port, deallocate_port

logger = logging.getLogger(__name__)


def pytest_ignore_collect(collection_path, config):
    """Skip collecting test files if required dependencies aren't installed."""
    filename = collection_path.name

    # Skip prompt_embeds tests if openai or torch aren't available
    if filename == "test_prompt_embeds.py":
        if importlib.util.find_spec("openai") is None:
            return True  # openai not available, skip this file
        if importlib.util.find_spec("torch") is None:
            return True  # torch not available, skip this file

    # In-proc OTLP test needs grpc + opentelemetry-proto. Skip when
    # missing (e.g. pre-commit env that only installs lint deps).
    # find_spec on a dotted name imports the parent and raises ModuleNotFoundError
    # if it doesn't exist, so guard the opentelemetry.proto lookup.
    if filename == "test_unified_worker_otlp_export.py":
        if importlib.util.find_spec("grpc") is None:
            return True
        try:
            if importlib.util.find_spec("opentelemetry.proto") is None:
                return True
        except ModuleNotFoundError:
            return True

    return None


@pytest.fixture(scope="function")
def start_services_with_http(
    request, runtime_services_dynamic_ports, dynamo_dynamic_ports
):
    """Start HTTP frontend with dynamic ports.

    Function-scoped to allow parallel test execution.
    Each test gets its own HTTP frontend on a unique port.
    Uses runtime_services_dynamic_ports for truly dynamic NATS/Etcd ports.

    Individual test files should start their specific worker processes.

    Yields:
        Tuple of (frontend_port, system_port) for use by worker processes
    """
    ports = dynamo_dynamic_ports
    # In xdist/parallel runs, never kill other workers' frontends.
    with DynamoFrontendProcess(
        request,
        frontend_port=ports.frontend_port,
        terminate_all_matching_process_names=False,
    ):
        logger.info(f"HTTP Frontend started on port {ports.frontend_port}")
        yield ports.frontend_port, ports.system_ports[0]


def check_grpc_server_live(
    port: int, max_attempts: int = 30, retry_delay: float = 0.5
) -> bool:
    """Check if the gRPC server process is up and accepting connections.

    Uses KServe `server_live` rather than `server_ready` because this fixture
    runs before any worker is launched — `server_ready` only flips true once
    a model has a WorkerSet ready to serve, which won't happen until callers
    of this fixture start their worker processes.

    Args:
        port: gRPC server port
        max_attempts: Maximum number of connection attempts
        retry_delay: Delay between retry attempts in seconds

    Returns:
        True if server is live

    Raises:
        Exception: If server is not live after max_attempts
    """
    import tritonclient.grpc as grpcclient

    for attempt in range(max_attempts):
        try:
            client = grpcclient.InferenceServerClient(f"localhost:{port}")
            if client.is_server_live():
                logger.info(
                    f"gRPC server is live on port {port} (attempt {attempt + 1}/{max_attempts})"
                )
                # Add delay after liveness check to ensure server is fully stable for parallel tests
                # Retry the check once more to confirm stability
                time.sleep(0.5)
                if client.is_server_live():
                    logger.info(f"gRPC server confirmed stable on port {port}")
                    return True
                else:
                    logger.warning(
                        f"gRPC server became unstable on port {port}, retrying..."
                    )
                    continue
        except Exception as e:
            if attempt < max_attempts - 1:
                logger.debug(f"gRPC server not live on attempt {attempt + 1}: {e}")
                time.sleep(retry_delay)
            else:
                logger.error(f"gRPC server not live after {max_attempts} attempts: {e}")
                raise
    return False


def wait_for_http_completions_ready(
    *,
    frontend_port: int,
    model: str,
    max_attempts: int = 30,
    retry_delay: float = 0.25,
) -> None:
    """Wait until the HTTP completions route can actually serve the given model.

    Why this exists:
    - `/v1/models` can list a model slightly before the HTTP completions route is
      ready to route requests to it (under xdist parallel startup).
    - If we start sending requests immediately, we can intermittently get 404
      "Model not found" even though the model shows up in `/v1/models`.
    """

    payload = {"model": model, "prompt": "ping", "max_tokens": 1}
    last_status: int | None = None
    last_body: str = ""

    for attempt in range(max_attempts):
        try:
            resp = requests.post(
                f"http://localhost:{frontend_port}/v1/completions",
                json=payload,
                timeout=10,
            )
            last_status = resp.status_code
            last_body = resp.text

            if resp.status_code == 200:
                return

            # Common transient during startup: model is discovered but not routable yet.
            if resp.status_code == 404 and "Model not found" in resp.text:
                time.sleep(retry_delay)
                continue

            # Any other error is likely real (e.g. schema validation changed).
            time.sleep(retry_delay)
        except requests.RequestException as e:
            last_body = str(e)
            time.sleep(retry_delay)

    raise RuntimeError(
        "HTTP completions route did not become ready "
        f"(frontend_port={frontend_port}, model={model}, "
        f"last_status={last_status}, last_body={last_body})"
    )


@pytest.fixture(scope="function")
def start_services_with_grpc(
    request, runtime_services_dynamic_ports, dynamo_dynamic_ports
):
    """Start gRPC frontend with dynamic ports.

    Function-scoped to allow parallel test execution.
    Each test gets its own gRPC frontend on a unique port.
    Uses runtime_services_dynamic_ports which provides isolated NATS/Etcd per test,
    so no namespace conflicts - each test has its own Etcd/NATS instance!

    Allocates an additional port for HTTP metrics server (used by gRPC service internally)
    to enable parallel test execution without port 8788 conflicts.

    Individual test files should start their specific worker processes.

    Yields:
        Tuple of (frontend_port, system_port) for use by worker processes
    """
    ports = dynamo_dynamic_ports
    # Allocate additional port for HTTP metrics server (gRPC service requirement)
    grpc_metrics_port = allocate_port(8788)

    try:
        with DynamoFrontendProcess(
            request,
            frontend_port=ports.frontend_port,
            terminate_all_matching_process_names=False,
            extra_args=[
                "--kserve-grpc-server",
                "--grpc-metrics-port",
                str(grpc_metrics_port),
            ],
        ):
            logger.info(
                f"gRPC Frontend starting on port {ports.frontend_port} "
                f"(metrics on {grpc_metrics_port})"
            )
            check_grpc_server_live(ports.frontend_port)
            yield ports.frontend_port, ports.system_ports[0]
    finally:
        deallocate_port(grpc_metrics_port)


########################################################
# Shared Worker Classes
########################################################


class MockerWorkerProcess(ManagedProcess):
    """Shared mocker worker process for frontend tests.

    Uses dynamo.mocker with configurable model and speedup ratio.
    Can be used by any frontend test that needs a fast mock backend.
    """

    def __init__(
        self,
        request,
        model: str,
        frontend_port: int,
        system_port: int,
        speedup_ratio: int = 100,
        worker_id: str = "mocker-worker",
        extra_args: list = None,
        extra_env: dict = None,
    ):
        self.worker_id = worker_id
        self.frontend_port = frontend_port
        self.system_port = system_port

        command = [
            "python3",
            "-m",
            "dynamo.mocker",
            "--model-path",
            model,
            "--speedup-ratio",
            str(speedup_ratio),
        ]

        env = os.environ.copy()
        env["DYN_LOG"] = "debug"
        env["DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS"] = '["generate"]'
        env["DYN_SYSTEM_PORT"] = str(system_port)
        if extra_env:
            env.update(extra_env)
        if extra_args:
            command.extend(extra_args)

        log_dir = f"{request.node.name}_{worker_id}"

        try:
            shutil.rmtree(log_dir)
        except FileNotFoundError:
            pass

        super().__init__(
            command=command,
            env=env,
            health_check_urls=[
                (f"http://localhost:{frontend_port}/v1/models", self._check_models_api),
                (f"http://localhost:{system_port}/health", self.is_ready),
            ],
            timeout=300,
            display_output=True,
            terminate_all_matching_process_names=False,
            stragglers=["VLLM::EngineCore"],
            straggler_commands=["-m dynamo.mocker"],
            log_dir=log_dir,
        )

    def _check_models_api(self, response):
        """Check if models API is ready"""
        try:
            if response.status_code != 200:
                return False
            data = response.json()
            models = data.get("data", [])
            return len(models) > 0
        except Exception:
            return False

    def is_ready(self, response) -> bool:
        try:
            status = (response.json() or {}).get("status")
        except ValueError:
            logger.warning("%s health response is not valid JSON", self.worker_id)
            return False

        is_ready = status == "ready"
        if is_ready:
            logger.info("%s status is ready", self.worker_id)
        else:
            logger.warning("%s status is not ready: %s", self.worker_id, status)
        return is_ready


@pytest.fixture(scope="function")
def start_services_with_mocker(
    request, start_services_with_http, predownload_tokenizers
):
    """Start mocker worker with the shared HTTP frontend.

    Function-scoped to allow parallel test execution.
    Each test gets its own frontend + mocker worker on unique ports.

    Yields:
        frontend_port: Port where frontend is running
    """
    frontend_port, system_port = start_services_with_http
    # Default to QWEN for compatibility; per-test model selection not yet implemented.
    model = QWEN

    with MockerWorkerProcess(request, model, frontend_port, system_port):
        wait_for_http_completions_ready(frontend_port=frontend_port, model=model)
        logger.info(f"Mocker Worker started for test on port {frontend_port}")
        yield frontend_port


class SampleUnifiedWorkerProcess(ManagedProcess):
    """Unified-backend sample worker (`dynamo.common.backend.sample_main`).

    CPU-only Python reference engine that exercises the unified backend's
    `Worker.run()` path — the same code path real backends (vllm/trtllm/
    sglang) go through. Useful for tests that need to validate the unified
    Worker/EngineAdapter pipeline without a GPU.

    Mirrors `MockerWorkerProcess` but uses the unified entry point. Accepts
    `extra_env` for tests that need to inject telemetry / tracing env vars
    (e.g. `OTEL_EXPORT_ENABLED=1`, `DYN_LOGGING_JSONL=1`).
    """

    def __init__(
        self,
        request,
        frontend_port: int,
        system_port: int,
        model_name: str = QWEN,
        component: str = "sample",
        disaggregation_mode: str = "agg",
        worker_id: str = "sample-worker",
        extra_args: list | None = None,
        extra_env: dict | None = None,
    ):
        self.worker_id = worker_id
        self.frontend_port = frontend_port
        self.system_port = system_port

        command = [
            "python3",
            "-m",
            "dynamo.common.backend.sample_main",
            "--model-name",
            model_name,
            "--component",
            component,
            "--disaggregation-mode",
            disaggregation_mode,
        ]
        if extra_args:
            command.extend(extra_args)

        env = os.environ.copy()
        env["DYN_LOG"] = "info"
        env["DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS"] = '["generate"]'
        env["DYN_SYSTEM_PORT"] = str(system_port)
        if extra_env:
            env.update(extra_env)

        log_dir = f"{request.node.name}_{worker_id}"

        try:
            shutil.rmtree(log_dir)
        except FileNotFoundError:
            pass

        super().__init__(
            command=command,
            env=env,
            health_check_urls=[
                (f"http://localhost:{frontend_port}/v1/models", self._check_models_api),
                (f"http://localhost:{system_port}/health", self.is_ready),
            ],
            timeout=120,
            display_output=True,
            terminate_all_matching_process_names=False,
            straggler_commands=["-m dynamo.common.backend.sample_main"],
            log_dir=log_dir,
        )

    def _check_models_api(self, response):
        try:
            if response.status_code != 200:
                return False
            data = response.json()
            return len(data.get("data", [])) > 0
        except Exception:
            return False

    def is_ready(self, response) -> bool:
        try:
            status = (response.json() or {}).get("status")
        except ValueError:
            logger.warning("%s health response is not valid JSON", self.worker_id)
            return False
        is_ready = status == "ready"
        if is_ready:
            logger.info("%s status is ready", self.worker_id)
        else:
            logger.warning("%s status is not ready: %s", self.worker_id, status)
        return is_ready
