# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
GPU Memory Service Shadow Engine Failover Tests (SGLang).

This test validates the GPU Memory Service shadow engine architecture for fault tolerance:
1. Start GPU Memory Service servers for each GPU device
2. Start a shadow engine and put it to sleep
3. Start a primary engine and run inference
4. Kill the primary engine (simulating failure)
5. Start a new shadow engine
6. Wake the original shadow engine and verify it can handle inference

Based on:
- components/src/dynamo/sglang/gpu_memory_service_adapters/TESTING.md

Test Execution Notes:
- Requires 2+ GPUs (for TP=2 model configurations)
- Uses Qwen/Qwen3-0.6B by default for faster testing (can use Qwen/Qwen3-14B for full validation)
- GPU Memory Service enables VA-stable sleep/wake for weights
"""

import importlib.util
import json
import logging
import os
import shutil
import time
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import pynvml
import pytest
import requests

from tests.utils.constants import QWEN
from tests.utils.managed_process import DynamoFrontendProcess, ManagedProcess
from tests.utils.port_utils import allocate_port, deallocate_port, deallocate_ports

logger = logging.getLogger(__name__)


# =============================================================================
# Backend availability detection
# =============================================================================


def _check_backend_available(module_name: str) -> bool:
    """Check if a backend module is available and importable."""
    if importlib.util.find_spec(module_name) is None:
        return False
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


HAS_SGLANG = _check_backend_available("sglang")


# =============================================================================
# Helper functions
# =============================================================================


def get_gpu_memory_usage(device: int = 0) -> Tuple[int, int, int]:
    """Get GPU memory usage for a device.

    Args:
        device: GPU device index

    Returns:
        Tuple of (used_bytes, free_bytes, total_bytes)
    """
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used, info.free, info.total
    finally:
        pynvml.nvmlShutdown()


def bytes_to_mb(bytes_val: int) -> float:
    """Convert bytes to megabytes."""
    return bytes_val / (1024 * 1024)


def bytes_to_gb(bytes_val: int) -> float:
    """Convert bytes to gigabytes."""
    return bytes_val / (1024 * 1024 * 1024)


# Default model for testing - use smaller model for faster test execution
GPU_MEMORY_SERVICE_TEST_MODEL = os.environ.get("GPU_MEMORY_SERVICE_TEST_MODEL", QWEN)

# Default tensor parallelism - TP=1 for single GPU testing
GPU_MEMORY_SERVICE_TP = int(os.environ.get("GPU_MEMORY_SERVICE_TP", "1"))


pytestmark = [
    pytest.mark.gpu_2,  # Requires 2 GPUs for TP=2 or to run multiple engines
    pytest.mark.e2e,
    pytest.mark.model(GPU_MEMORY_SERVICE_TEST_MODEL),
    pytest.mark.nightly,  # Resource-intensive test, run nightly
    pytest.mark.fault_tolerance,
    pytest.mark.sglang,
]


# =============================================================================
# Process managers
# =============================================================================


class GPUMemoryServiceProcess(ManagedProcess):
    """Process manager for GPU Memory Service allocation server.

    The GPU Memory Service is responsible for managing GPU memory allocations
    with connection-based RW/RO locking. Each GPU device has its own service.
    """

    def __init__(
        self,
        request,
        device: int,
        socket_path: Optional[str] = None,
        timeout: int = 60,
    ):
        self.device = device
        self.socket_path = socket_path or f"/tmp/gpu_memory_service_{device}.sock"

        command = [
            "python3",
            "-m",
            "dynamo.gpu_memory_service",
            "--device",
            str(device),
            "--socket-path",
            self.socket_path,
            "--verbose",
        ]

        env = os.environ.copy()
        env["DYN_LOG"] = "debug"

        log_dir = f"/tmp/{request.node.name}_gpu_memory_service_{device}"

        # Clean up any existing log directory
        try:
            shutil.rmtree(log_dir)
        except FileNotFoundError:
            pass

        # Clean up any existing socket file
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        super().__init__(
            command=command,
            env=env,
            timeout=timeout,
            display_output=True,
            terminate_existing=False,
            log_dir=log_dir,
            # Health check: verify socket file exists
            health_check_funcs=[self._check_socket_exists],
        )

    def _check_socket_exists(self, timeout: float = 30) -> bool:
        """Check if the Unix socket file exists (indicates server is ready)."""
        start = time.time()
        while time.time() - start < timeout:
            if os.path.exists(self.socket_path):
                logger.info(f"GPU Memory Service socket ready: {self.socket_path}")
                return True
            time.sleep(0.1)
        return False


class EngineWithGPUMemoryServiceProcess(ManagedProcess, ABC):
    """Abstract base class for engine processes with GPU Memory Service integration."""

    def __init__(
        self,
        request,
        engine_id: str,
        system_port: int,
        timeout: int,
        command: list,
        env: dict,
    ):
        self.engine_id = engine_id
        self.system_port = system_port

        log_dir = f"/tmp/{request.node.name}_{engine_id}"

        # Clean up any existing log directory
        try:
            shutil.rmtree(log_dir)
        except FileNotFoundError:
            pass

        super().__init__(
            command=command,
            env=env,
            # Only check system port health - /v1/models is served by frontend
            health_check_urls=[
                (f"http://localhost:{system_port}/health", self._is_ready),
            ],
            timeout=timeout,
            display_output=True,
            terminate_existing=False,
            # Don't use straggler cleanup - this test runs multiple engines with same command pattern
            # and we don't want killing one to kill the others
            stragglers=[],
            straggler_commands=[],
            log_dir=log_dir,
        )

    def _is_ready(self, response) -> bool:
        """Check if the engine is ready to serve requests."""
        try:
            data = response.json()
            if data.get("status") == "ready":
                logger.info(f"{self.engine_id} status is ready")
                return True
            logger.warning(
                f"{self.engine_id} status is not ready: {data.get('status')}"
            )
        except ValueError:
            logger.warning(f"{self.engine_id} health response is not valid JSON")
        return False

    @abstractmethod
    def sleep(self, *, timeout: int = 30) -> dict:
        """Put the engine to sleep via HTTP API."""
        pass

    @abstractmethod
    def wake(self, *, timeout: int = 30) -> dict:
        """Wake the engine from sleep via HTTP API."""
        pass


class SGLangWithGPUMemoryServiceProcess(EngineWithGPUMemoryServiceProcess):
    """Process manager for SGLang engine with GPU Memory Service integration.

    This starts an SGLang engine configured to use GPU Memory Service for weight loading,
    enabling VA-stable sleep/wake functionality.

    Note: This worker registers with the dynamo runtime for discovery by the frontend.
    The /v1/models and /v1/completions endpoints are served by DynamoFrontendProcess.
    """

    def __init__(
        self,
        request,
        engine_id: str,
        socket_path_template: str,
        system_port: int,
        model: str = GPU_MEMORY_SERVICE_TEST_MODEL,
        tp: int = GPU_MEMORY_SERVICE_TP,
        sglang_port: int = 30000,
        disaggregation_bootstrap_port: int = 8998,
        timeout: int = 300,
    ):
        self.sglang_port = sglang_port
        self.disaggregation_bootstrap_port = disaggregation_bootstrap_port

        # Socket path template uses {device} placeholder that gets substituted per GPU
        socket_path_config = socket_path_template

        # Build model_loader_extra_config with socket path
        extra_config = {"gpu_memory_service_socket_path": socket_path_config}
        extra_config_str = json.dumps(extra_config)

        command = [
            "python3",
            "-m",
            "dynamo.sglang",
            "--model-path",
            model,
            "--tensor-parallel-size",
            str(tp),
            "--load-format",
            "gpu_memory_service",
            "--enable-memory-saver",
            "--mem-fraction-static",
            "0.8",  # Test multiple engine sleep/wake
            "--port",
            str(sglang_port),
            "--disaggregation-bootstrap-port",
            str(disaggregation_bootstrap_port),
            "--model-loader-extra-config",
            extra_config_str,
        ]

        env = os.environ.copy()
        env["DYN_LOG"] = "debug"
        env["DYN_SYSTEM_PORT"] = str(system_port)

        super().__init__(
            request=request,
            engine_id=engine_id,
            system_port=system_port,
            timeout=timeout,
            command=command,
            env=env,
        )

    def sleep(self, *, timeout: int = 30) -> dict:
        """Put the engine to sleep via HTTP API (release memory occupation).

        SGLang uses /engine/release_memory_occupation instead of /engine/sleep.
        By default, releases all memory types (kv_cache, weights, cuda_graph).

        Args:
            timeout: Request timeout in seconds

        Returns:
            Response JSON from the release_memory_occupation endpoint
        """
        url = f"http://localhost:{self.system_port}/engine/release_memory_occupation"
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json={},  # Default releases all memory types
            timeout=timeout,
        )
        response.raise_for_status()
        result = response.json()
        logger.info(f"{self.engine_id} release_memory_occupation response: {result}")
        return result

    def wake(self, *, timeout: int = 30) -> dict:
        """Wake the engine from sleep via HTTP API (resume memory occupation).

        SGLang uses /engine/resume_memory_occupation instead of /engine/wake.

        Args:
            timeout: Request timeout in seconds

        Returns:
            Response JSON from the resume_memory_occupation endpoint
        """
        url = f"http://localhost:{self.system_port}/engine/resume_memory_occupation"
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json={},
            timeout=timeout,
        )
        response.raise_for_status()
        result = response.json()
        logger.info(f"{self.engine_id} resume_memory_occupation response: {result}")
        return result


# =============================================================================
# Helper functions for tests
# =============================================================================


def send_completion_request(
    frontend_port: int,
    prompt: str = "What is 2 + 2?",
    max_tokens: int = 50,
    timeout: int = 120,
) -> dict:
    """Send a completion request to the frontend."""
    url = f"http://localhost:{frontend_port}/v1/completions"
    payload = {
        "model": GPU_MEMORY_SERVICE_TEST_MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
    }

    logger.info(f"Sending completion request to {url}")
    response = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    result = response.json()

    assert "choices" in result, "Missing 'choices' in response"
    assert len(result["choices"]) > 0, "Empty choices in response"
    logger.info(f"Completion response: {result['choices'][0]}")

    return result


def create_engine_process(
    request,
    engine_id: str,
    socket_path_template: str,
    system_port: int,
    ports: dict,
    is_primary: bool,
    tp: int = GPU_MEMORY_SERVICE_TP,
) -> SGLangWithGPUMemoryServiceProcess:
    """Factory function to create SGLang engine process."""
    kwargs = {
        "request": request,
        "engine_id": engine_id,
        "socket_path_template": socket_path_template,
        "system_port": system_port,
        "tp": tp,
    }
    if is_primary:
        kwargs["sglang_port"] = ports["primary_sglang_port"]
        kwargs["disaggregation_bootstrap_port"] = ports["primary_bootstrap_port"]
    else:
        kwargs["sglang_port"] = ports["shadow_sglang_port"]
        kwargs["disaggregation_bootstrap_port"] = ports["shadow_bootstrap_port"]
    return SGLangWithGPUMemoryServiceProcess(**kwargs)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def gpu_memory_service_ports(request):
    """Allocate ports for GPU Memory Service test."""
    shadow_system_port = allocate_port(8100)
    primary_system_port = allocate_port(8101)
    frontend_port = allocate_port(8200)
    shadow_sglang_port = allocate_port(30000)
    primary_sglang_port = allocate_port(30001)
    shadow_bootstrap_port = allocate_port(8998)
    primary_bootstrap_port = allocate_port(8999)

    ports = {
        "shadow_system_port": shadow_system_port,
        "primary_system_port": primary_system_port,
        "frontend_port": frontend_port,
        "shadow_sglang_port": shadow_sglang_port,
        "primary_sglang_port": primary_sglang_port,
        "shadow_bootstrap_port": shadow_bootstrap_port,
        "primary_bootstrap_port": primary_bootstrap_port,
    }

    yield ports

    deallocate_ports(
        [
            shadow_system_port,
            primary_system_port,
            frontend_port,
            shadow_sglang_port,
            primary_sglang_port,
            shadow_bootstrap_port,
            primary_bootstrap_port,
        ]
    )


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.timeout(600)
@pytest.mark.skipif(not HAS_SGLANG, reason="SGLang not installed")
def test_gpu_memory_service_shadow_engine_failover(
    request,
    runtime_services,
    gpu_memory_service_ports,
    predownload_models,
):
    """
    End-to-end test for GPU Memory Service shadow engine failover (SGLang).

    This test validates the full shadow engine architecture:
    1. GPU Memory Service persists model weights across engine restarts
    2. Shadow engines can be put to sleep and woken up with VA-stable weights
    3. Primary engine failure can be recovered by waking shadow engine
    """
    ports = gpu_memory_service_ports
    tp = GPU_MEMORY_SERVICE_TP
    frontend_port = ports["frontend_port"]

    socket_path_template = "/tmp/gpu_memory_service_{device}.sock"

    # Step 1: Start GPU Memory Service for each device
    logger.info(f"Step 1: Starting GPU Memory Service for {tp} device(s)")
    gpu_mem_services = []

    for device in range(tp):
        socket_path = socket_path_template.format(device=device)
        gpu_mem_service = GPUMemoryServiceProcess(
            request,
            device=device,
            socket_path=socket_path,
        )
        gpu_mem_service.__enter__()
        gpu_mem_services.append(gpu_mem_service)
        logger.info(f"GPU Memory Service for device {device} started")

    try:
        # Step 2: Start frontend
        logger.info("Step 2: Starting frontend")
        with DynamoFrontendProcess(request, frontend_port=frontend_port):
            logger.info(f"Frontend started on port {frontend_port}")

            # Step 3: Start shadow engine
            logger.info("Step 3: Starting shadow engine (SGLang)")
            shadow_engine = create_engine_process(
                request=request,
                engine_id="shadow_engine",
                socket_path_template=socket_path_template,
                system_port=ports["shadow_system_port"],
                ports=ports,
                is_primary=False,
                tp=tp,
            )
            with shadow_engine:
                logger.info(
                    f"Shadow engine started on port {ports['shadow_system_port']}"
                )

                time.sleep(7)

                # Test that shadow engine works before sleeping
                logger.info("Testing shadow engine before sleep...")
                result = send_completion_request(frontend_port)
                assert result["choices"], "Shadow engine should respond before sleep"

                memory_before_shadow_sleep, _, _ = get_gpu_memory_usage(device=0)
                logger.info(
                    f"GPU memory before shadow sleep: {bytes_to_mb(memory_before_shadow_sleep):.1f} MB"
                )

                # Put shadow engine to sleep
                logger.info("Putting shadow engine to sleep")
                sleep_result = shadow_engine.sleep()
                assert (
                    sleep_result.get("status") == "ok"
                ), f"Sleep failed: {sleep_result}"
                logger.info("Shadow engine is now sleeping")

                time.sleep(2)

                memory_after_shadow_sleep, _, _ = get_gpu_memory_usage(device=0)
                logger.info(
                    f"GPU memory after shadow sleep: {bytes_to_mb(memory_after_shadow_sleep):.1f} MB"
                )

                memory_freed_by_shadow_sleep = (
                    memory_before_shadow_sleep - memory_after_shadow_sleep
                )
                logger.info(
                    f"Memory freed by shadow sleep: {bytes_to_mb(memory_freed_by_shadow_sleep):.1f} MB"
                )
                assert memory_after_shadow_sleep < memory_before_shadow_sleep, (
                    f"Shadow sleep should reduce GPU memory! "
                    f"Before: {bytes_to_mb(memory_before_shadow_sleep):.1f} MB, "
                    f"After: {bytes_to_mb(memory_after_shadow_sleep):.1f} MB"
                )

                # Step 4: Start primary engine
                logger.info("Step 4: Starting primary engine (SGLang)")
                primary_engine = create_engine_process(
                    request=request,
                    engine_id="primary_engine",
                    socket_path_template=socket_path_template,
                    system_port=ports["primary_system_port"],
                    ports=ports,
                    is_primary=True,
                    tp=tp,
                )
                with primary_engine:
                    logger.info(
                        f"Primary engine started on port {ports['primary_system_port']}"
                    )

                    time.sleep(7)

                    # Step 5: Run inference with primary engine
                    logger.info("Step 5: Running inference with primary engine")
                    result = send_completion_request(frontend_port)
                    assert result["choices"], "Primary engine should respond"
                    logger.info("Primary engine inference successful")

                    # Step 6: Kill primary engine
                    logger.info("Step 6: Killing primary engine to simulate failure")
                    primary_pid = primary_engine.get_pid()
                    logger.info(f"Terminating primary engine (PID: {primary_pid})")

                logger.info("Primary engine terminated")

                time.sleep(7)

                memory_before_shadow_wake, _, _ = get_gpu_memory_usage(device=0)
                logger.info(
                    f"GPU memory before shadow wake: {bytes_to_mb(memory_before_shadow_wake):.1f} MB"
                )

                # Step 7: Wake up shadow engine
                logger.info("Step 7: Waking up shadow engine")
                wake_result = shadow_engine.wake()
                assert wake_result.get("status") == "ok", f"Wake failed: {wake_result}"
                logger.info("Shadow engine woke up successfully")

                time.sleep(7)

                memory_after_shadow_wake, _, _ = get_gpu_memory_usage(device=0)
                logger.info(
                    f"GPU memory after shadow wake: {bytes_to_mb(memory_after_shadow_wake):.1f} MB"
                )

                memory_restored_by_wake = (
                    memory_after_shadow_wake - memory_before_shadow_wake
                )
                logger.info(
                    f"Memory restored by shadow wake: {bytes_to_mb(memory_restored_by_wake):.1f} MB"
                )

                # Step 8: Verify shadow engine can serve inference
                logger.info(
                    "Step 8: Verifying shadow engine can serve inference after failover"
                )
                result = send_completion_request(frontend_port)
                assert result["choices"], "Shadow engine should respond after wake"
                logger.info("Shadow engine inference successful after failover!")

                for i in range(3):
                    result = send_completion_request(
                        frontend_port,
                        prompt=f"Count to {i + 3}:",
                        max_tokens=20,
                    )
                    assert result["choices"], f"Request {i + 1} failed"
                logger.info("All verification requests successful")

                # Summary
                logger.info("=" * 60)
                logger.info("GPU MEMORY ACCOUNTING SUMMARY (Shadow Engine Failover - SGLang):")
                logger.info(
                    f"  Before shadow sleep:  {bytes_to_mb(memory_before_shadow_sleep):.1f} MB"
                )
                logger.info(
                    f"  After shadow sleep:   {bytes_to_mb(memory_after_shadow_sleep):.1f} MB"
                )
                logger.info(
                    f"  Before shadow wake:   {bytes_to_mb(memory_before_shadow_wake):.1f} MB"
                )
                logger.info(
                    f"  After shadow wake:    {bytes_to_mb(memory_after_shadow_wake):.1f} MB"
                )
                logger.info(
                    f"  Memory freed by sleep: {bytes_to_mb(memory_freed_by_shadow_sleep):.1f} MB"
                )
                logger.info(
                    f"  Memory restored by wake: {bytes_to_mb(memory_restored_by_wake):.1f} MB"
                )
                logger.info("=" * 60)

    finally:
        for gpu_mem_service in reversed(gpu_mem_services):
            try:
                gpu_mem_service.__exit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error stopping GPU Memory Service: {e}")

        for device in range(tp):
            socket_path = socket_path_template.format(device=device)
            try:
                if os.path.exists(socket_path):
                    os.unlink(socket_path)
            except Exception as e:
                logger.warning(f"Error removing socket file {socket_path}: {e}")


@pytest.mark.timeout(300)
@pytest.mark.gpu_1
@pytest.mark.skipif(not HAS_SGLANG, reason="SGLang not installed")
def test_gpu_memory_service_basic_sleep_wake(
    request,
    runtime_services,
    predownload_models,
):
    """
    Basic test for GPU Memory Service sleep/wake functionality (SGLang).

    This is a simpler test that validates:
    1. Engine can be started with GPU Memory Service
    2. Engine can be put to sleep
    3. Engine can be woken up
    4. Engine works after wake
    """
    system_port = allocate_port(8100)
    frontend_port = allocate_port(8200)
    sglang_port = allocate_port(30000)
    bootstrap_port = allocate_port(8998)

    socket_path_template = "/tmp/gpu_memory_service_{device}.sock"
    socket_path = socket_path_template.format(device=0)

    try:
        logger.info("Starting GPU Memory Service")
        with GPUMemoryServiceProcess(
            request,
            device=0,
            socket_path=socket_path,
        ):
            logger.info("GPU Memory Service started")

            logger.info("Starting frontend")
            with DynamoFrontendProcess(request, frontend_port=frontend_port):
                logger.info(f"Frontend started on port {frontend_port}")

                ports = {
                    "shadow_system_port": system_port,
                    "primary_system_port": system_port,
                    "shadow_sglang_port": sglang_port,
                    "shadow_bootstrap_port": bootstrap_port,
                    "primary_sglang_port": sglang_port,
                    "primary_bootstrap_port": bootstrap_port,
                }

                logger.info("Starting SGLang engine with GPU Memory Service")
                engine = create_engine_process(
                    request=request,
                    engine_id="test_engine",
                    socket_path_template=socket_path_template,
                    system_port=system_port,
                    ports=ports,
                    is_primary=False,
                    tp=1,
                )
                with engine:
                    logger.info("Engine started")

                    time.sleep(7)

                    # Test 1: Initial inference
                    logger.info("Test 1: Initial inference")
                    result = send_completion_request(
                        frontend_port, prompt="Hello, I am"
                    )
                    assert result["choices"], "Initial inference failed"

                    memory_before_sleep, _, _ = get_gpu_memory_usage(device=0)
                    logger.info(
                        f"GPU memory before sleep: {bytes_to_mb(memory_before_sleep):.1f} MB "
                        f"({bytes_to_gb(memory_before_sleep):.2f} GB)"
                    )

                    # Test 2: Sleep
                    logger.info("Test 2: Sleep engine")
                    sleep_result = engine.sleep()
                    assert (
                        sleep_result.get("status") == "ok"
                    ), f"Sleep failed: {sleep_result}"
                    time.sleep(2)

                    memory_after_sleep, _, _ = get_gpu_memory_usage(device=0)
                    logger.info(
                        f"GPU memory after sleep: {bytes_to_mb(memory_after_sleep):.1f} MB "
                        f"({bytes_to_gb(memory_after_sleep):.2f} GB)"
                    )

                    memory_freed = memory_before_sleep - memory_after_sleep
                    logger.info(
                        f"Memory freed by sleep: {bytes_to_mb(memory_freed):.1f} MB "
                        f"({bytes_to_gb(memory_freed):.2f} GB)"
                    )
                    assert memory_after_sleep < memory_before_sleep, (
                        f"Sleep should reduce GPU memory usage! "
                        f"Before: {bytes_to_mb(memory_before_sleep):.1f} MB, "
                        f"After: {bytes_to_mb(memory_after_sleep):.1f} MB"
                    )

                    # Test 3: Wake
                    logger.info("Test 3: Wake engine")
                    wake_result = engine.wake()
                    assert (
                        wake_result.get("status") == "ok"
                    ), f"Wake failed: {wake_result}"
                    time.sleep(2)

                    memory_after_wake, _, _ = get_gpu_memory_usage(device=0)
                    logger.info(
                        f"GPU memory after wake: {bytes_to_mb(memory_after_wake):.1f} MB "
                        f"({bytes_to_gb(memory_after_wake):.2f} GB)"
                    )

                    logger.info(
                        f"Memory restored by wake: {bytes_to_mb(memory_after_wake - memory_after_sleep):.1f} MB"
                    )

                    # Test 4: Inference after wake
                    logger.info("Test 4: Inference after wake")
                    result = send_completion_request(
                        frontend_port, prompt="Goodbye, I will"
                    )
                    assert result["choices"], "Inference after wake failed"

                    # Summary
                    logger.info("=" * 60)
                    logger.info("GPU MEMORY ACCOUNTING SUMMARY (SGLang):")
                    logger.info(
                        f"  Before sleep: {bytes_to_mb(memory_before_sleep):.1f} MB"
                    )
                    logger.info(
                        f"  After sleep:  {bytes_to_mb(memory_after_sleep):.1f} MB"
                    )
                    logger.info(
                        f"  After wake:   {bytes_to_mb(memory_after_wake):.1f} MB"
                    )
                    logger.info(
                        f"  Memory freed by sleep: {bytes_to_mb(memory_freed):.1f} MB"
                    )
                    logger.info("=" * 60)

                    logger.info("All basic sleep/wake tests passed (SGLang)!")

    finally:
        deallocate_port(system_port)
        deallocate_port(frontend_port)
        deallocate_port(sglang_port)
        deallocate_port(bootstrap_port)
        if os.path.exists(socket_path):
            try:
                os.unlink(socket_path)
            except Exception:
                pass
