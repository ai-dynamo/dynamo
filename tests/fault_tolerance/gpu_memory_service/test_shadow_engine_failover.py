# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
GPU Memory Service Shadow Engine Failover Tests.

This test validates the GPU Memory Service shadow engine architecture for fault tolerance:
1. Start GPU Memory Service servers for each GPU device
2. Start a shadow engine and put it to sleep
3. Start a primary engine and run inference
4. Kill the primary engine (simulating failure)
5. Start a new shadow engine
6. Wake the original shadow engine and verify it can handle inference

Supports both vLLM and SGLang backends through parametrization.

Based on:
- components/src/dynamo/vllm/gpu_memory_service_adapters/TESTING.md
- components/src/dynamo/sglang/gpu_memory_service_adapters/TESTING.md

Test Execution Notes:
- Requires 2+ GPUs (for TP=2 model configurations)
- Uses Qwen/Qwen3-0.6B by default for faster testing (can use Qwen/Qwen3-14B for full validation)
- GPU Memory Service enables VA-stable sleep/wake for weights
- Tests are automatically skipped if the backend is not installed
"""

import importlib.util
import logging
import os
import shutil
import time
from abc import ABC, abstractmethod
from enum import Enum
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


HAS_VLLM = _check_backend_available("vllm")
HAS_SGLANG = _check_backend_available("sglang")


class Backend(Enum):
    """Supported inference backends."""

    VLLM = "vllm"
    SGLANG = "sglang"


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
        """Put the engine to sleep via HTTP API.

        Args:
            timeout: Request timeout in seconds

        Returns:
            Response JSON from the sleep endpoint
        """
        pass

    @abstractmethod
    def wake(self, *, timeout: int = 30) -> dict:
        """Wake the engine from sleep via HTTP API.

        Args:
            timeout: Request timeout in seconds

        Returns:
            Response JSON from the wake endpoint
        """
        pass


class VLLMWithGPUMemoryServiceProcess(EngineWithGPUMemoryServiceProcess):
    """Process manager for vLLM engine with GPU Memory Service integration.

    This starts a vLLM engine configured to use GPU Memory Service for weight loading,
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
        nixl_port: int = 5600,
        kv_event_port: int = 20080,
        timeout: int = 300,
        gms_load_mode: str = "write",
    ):
        self.nixl_port = nixl_port
        self.kv_event_port = kv_event_port

        # Socket path template uses {device} placeholder that gets substituted per GPU
        # Example: /tmp/gpu_memory_service_{device}.sock -> /tmp/gpu_memory_service_0.sock for device 0
        socket_path_config = socket_path_template

        # Build model_loader_extra_config with socket path and load mode
        extra_config = {
            "gpu_memory_service_socket_path": socket_path_config,
            "gpu_memory_service_load_mode": gms_load_mode,
        }
        import json

        extra_config_str = json.dumps(extra_config)

        command = [
            "python3",
            "-m",
            "dynamo.vllm",
            "--model",
            model,
            "-tp",
            str(tp),
            "--load-format",
            "gpu_memory_service",
            "--enable-sleep-mode",
            "--gpu-memory-utilization",
            "0.9",  # Test multiple engine sleep/wake
            "--model-loader-extra-config",
            extra_config_str,
        ]

        env = os.environ.copy()
        env["DYN_LOG"] = "debug"
        env["DYN_SYSTEM_PORT"] = str(system_port)
        env["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(nixl_port)
        env["DYN_VLLM_KV_EVENT_PORT"] = str(kv_event_port)

        super().__init__(
            request=request,
            engine_id=engine_id,
            system_port=system_port,
            timeout=timeout,
            command=command,
            env=env,
        )

    def sleep(self, *, level: int = 1, timeout: int = 30) -> dict:
        """Put the engine to sleep via HTTP API.

        Args:
            level: Sleep level (1=weights only, 2=weights+buffers, 3=everything)
            timeout: Request timeout in seconds

        Returns:
            Response JSON from the sleep endpoint
        """
        url = f"http://localhost:{self.system_port}/engine/sleep"
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json={"level": level},
            timeout=timeout,
        )
        response.raise_for_status()
        result = response.json()
        logger.info(f"{self.engine_id} sleep response: {result}")
        return result

    def wake(self, *, timeout: int = 30) -> dict:
        """Wake the engine from sleep via HTTP API.

        Args:
            timeout: Request timeout in seconds

        Returns:
            Response JSON from the wake endpoint
        """
        url = f"http://localhost:{self.system_port}/engine/wake"
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json={},
            timeout=timeout,
        )
        response.raise_for_status()
        result = response.json()
        logger.info(f"{self.engine_id} wake response: {result}")
        return result


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
        gms_load_mode: str = "write",
    ):
        self.sglang_port = sglang_port
        self.disaggregation_bootstrap_port = disaggregation_bootstrap_port

        # Socket path template uses {device} placeholder that gets substituted per GPU
        socket_path_config = socket_path_template

        # Build model_loader_extra_config with socket path
        # Note: For SGLang, load mode is set via GPU_MEMORY_SERVICE_LOAD_MODE env var
        import json

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
        # SGLang uses env var for GPU Memory Service load mode
        env["GPU_MEMORY_SERVICE_LOAD_MODE"] = gms_load_mode

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
    """Send a completion request to the frontend.

    Args:
        frontend_port: Port of the frontend server
        prompt: The prompt to send
        max_tokens: Maximum tokens to generate
        timeout: Request timeout in seconds

    Returns:
        Response JSON from the completions endpoint
    """
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
    backend: Backend,
    request,
    engine_id: str,
    socket_path_template: str,
    system_port: int,
    ports: dict,
    is_primary: bool,
    tp: int = GPU_MEMORY_SERVICE_TP,
    gms_load_mode: Optional[str] = None,
) -> EngineWithGPUMemoryServiceProcess:
    """Factory function to create the appropriate engine process for a backend.

    Args:
        backend: The backend to use (vLLM or SGLang)
        request: pytest request fixture
        engine_id: Identifier for the engine
        socket_path_template: Template for GPU Memory Service socket paths
        system_port: Port for the system health endpoint
        ports: Dictionary of allocated ports
        is_primary: True if this is the primary engine (affects port selection)
        tp: Tensor parallelism degree
        gms_load_mode: GPU Memory Service load mode ("write" or "read").
                      If None, uses "write" for first engine (shadow), "read" for subsequent (primary).

    Returns:
        An engine process instance
    """
    # Determine load mode: first engine (shadow, is_primary=False) uses write,
    # subsequent engines (primary, is_primary=True) use read to import existing weights
    if gms_load_mode is None:
        gms_load_mode = "read" if is_primary else "write"

    if backend == Backend.VLLM:
        kwargs = {
            "request": request,
            "engine_id": engine_id,
            "socket_path_template": socket_path_template,
            "system_port": system_port,
            "tp": tp,
            "gms_load_mode": gms_load_mode,
        }
        if is_primary:
            kwargs["nixl_port"] = ports["primary_nixl_port"]
            kwargs["kv_event_port"] = ports["primary_kv_event_port"]
        return VLLMWithGPUMemoryServiceProcess(**kwargs)
    elif backend == Backend.SGLANG:
        kwargs = {
            "request": request,
            "engine_id": engine_id,
            "socket_path_template": socket_path_template,
            "system_port": system_port,
            "tp": tp,
            "gms_load_mode": gms_load_mode,
        }
        if is_primary:
            kwargs["sglang_port"] = ports["primary_sglang_port"]
            kwargs["disaggregation_bootstrap_port"] = ports["primary_bootstrap_port"]
        else:
            kwargs["sglang_port"] = ports["shadow_sglang_port"]
            kwargs["disaggregation_bootstrap_port"] = ports["shadow_bootstrap_port"]
        return SGLangWithGPUMemoryServiceProcess(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def gpu_memory_service_ports(request):
    """Allocate ports for GPU Memory Service test."""
    # Allocate ports: system ports for workers, one frontend port
    shadow_system_port = allocate_port(8100)
    primary_system_port = allocate_port(8101)

    # Single frontend port (frontend discovers workers via runtime)
    frontend_port = allocate_port(8200)

    # vLLM-specific ports: NIXL and KV event ports for primary (shadow uses defaults)
    primary_nixl_port = allocate_port(5601)
    primary_kv_event_port = allocate_port(20081)

    # SGLang-specific ports: internal server port and disaggregation bootstrap
    shadow_sglang_port = allocate_port(30000)
    primary_sglang_port = allocate_port(30001)
    shadow_bootstrap_port = allocate_port(8998)
    primary_bootstrap_port = allocate_port(8999)

    ports = {
        "shadow_system_port": shadow_system_port,
        "primary_system_port": primary_system_port,
        "frontend_port": frontend_port,
        # vLLM-specific
        "primary_nixl_port": primary_nixl_port,
        "primary_kv_event_port": primary_kv_event_port,
        # SGLang-specific
        "shadow_sglang_port": shadow_sglang_port,
        "primary_sglang_port": primary_sglang_port,
        "shadow_bootstrap_port": shadow_bootstrap_port,
        "primary_bootstrap_port": primary_bootstrap_port,
    }

    yield ports

    # Cleanup
    deallocate_ports(
        [
            shadow_system_port,
            primary_system_port,
            frontend_port,
            primary_nixl_port,
            primary_kv_event_port,
            shadow_sglang_port,
            primary_sglang_port,
            shadow_bootstrap_port,
            primary_bootstrap_port,
        ]
    )


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.timeout(600)  # 10 minutes for full test
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(
            Backend.VLLM,
            id="vllm",
            marks=[
                pytest.mark.vllm,
                pytest.mark.skipif(not HAS_VLLM, reason="vLLM not installed"),
            ],
        ),
        pytest.param(
            Backend.SGLANG,
            id="sglang",
            marks=[
                pytest.mark.sglang,
                pytest.mark.skipif(not HAS_SGLANG, reason="SGLang not installed"),
            ],
        ),
    ],
)
def test_gpu_memory_service_shadow_engine_failover(
    request,
    runtime_services,
    gpu_memory_service_ports,
    predownload_models,
    backend: Backend,
):
    """
    End-to-end test for GPU Memory Service shadow engine failover.

    This test validates the full shadow engine architecture:
    1. GPU Memory Service persists model weights across engine restarts
    2. Shadow engines can be put to sleep and woken up with VA-stable weights
    3. Primary engine failure can be recovered by waking shadow engine

    Test Steps (from TESTING.md):
    1. Start GPU Memory Service for each device
    2. Start frontend (serves /v1/completions, discovers workers via runtime)
    3. Start shadow engine and put it to sleep
    4. Start primary engine
    5. Run inference with primary engine
    6. Kill primary engine (simulate failure)
    7. Wake up shadow engine
    8. Verify shadow engine can serve inference

    Note: This is a simplified version that uses TP=1 for single GPU testing.
    For full TP=2 testing, set GPU_MEMORY_SERVICE_TP=2 environment variable.

    Args:
        backend: The inference backend to test (vLLM or SGLang)
    """
    ports = gpu_memory_service_ports
    tp = GPU_MEMORY_SERVICE_TP
    frontend_port = ports["frontend_port"]

    # Socket path template - {device} gets substituted per GPU
    # Use the default socket path to match README.md and GPU Memory Service defaults
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
        # Step 2: Start frontend (serves OpenAI-compatible API)
        logger.info("Step 2: Starting frontend")
        with DynamoFrontendProcess(request, frontend_port=frontend_port):
            logger.info(f"Frontend started on port {frontend_port}")

            # Step 3: Start shadow engine
            logger.info(f"Step 3: Starting shadow engine ({backend.value})")
            shadow_engine = create_engine_process(
                backend=backend,
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

                # Wait for frontend to discover the shadow engine
                time.sleep(7)

                # Test that shadow engine works before sleeping
                logger.info("Testing shadow engine before sleep...")
                result = send_completion_request(frontend_port)
                assert result["choices"], "Shadow engine should respond before sleep"

                # Measure memory before shadow engine sleeps
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

                # Give time for sleep to complete
                time.sleep(2)

                # Measure memory after shadow engine sleeps
                memory_after_shadow_sleep, _, _ = get_gpu_memory_usage(device=0)
                logger.info(
                    f"GPU memory after shadow sleep: {bytes_to_mb(memory_after_shadow_sleep):.1f} MB"
                )

                # CRITICAL ASSERTION: Shadow sleep should reduce GPU memory
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

                # Step 4: Start primary engine (with different ports to avoid conflicts)
                logger.info(f"Step 4: Starting primary engine ({backend.value})")
                primary_engine = create_engine_process(
                    backend=backend,
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

                    # Wait for frontend to discover the primary engine
                    time.sleep(7)

                    # Step 5: Run inference with primary engine
                    logger.info("Step 5: Running inference with primary engine")
                    result = send_completion_request(frontend_port)
                    assert result["choices"], "Primary engine should respond"
                    logger.info("Primary engine inference successful")

                    # Step 6: Kill primary engine (simulate failure)
                    logger.info("Step 6: Killing primary engine to simulate failure")
                    primary_pid = primary_engine.get_pid()
                    logger.info(f"Terminating primary engine (PID: {primary_pid})")

                # Primary engine context exited (process terminated)
                logger.info("Primary engine terminated")

                # Give time for cleanup and frontend to notice
                time.sleep(7)

                # Measure memory before shadow wake (after primary is dead)
                memory_before_shadow_wake, _, _ = get_gpu_memory_usage(device=0)
                logger.info(
                    f"GPU memory before shadow wake: {bytes_to_mb(memory_before_shadow_wake):.1f} MB"
                )

                # Step 7: Wake up shadow engine
                logger.info("Step 7: Waking up shadow engine")
                wake_result = shadow_engine.wake()
                assert wake_result.get("status") == "ok", f"Wake failed: {wake_result}"
                logger.info("Shadow engine woke up successfully")

                # Give time for wake to complete and frontend to route to shadow
                time.sleep(7)

                # Measure memory after shadow wake
                memory_after_shadow_wake, _, _ = get_gpu_memory_usage(device=0)
                logger.info(
                    f"GPU memory after shadow wake: {bytes_to_mb(memory_after_shadow_wake):.1f} MB"
                )

                # Memory should increase after wake (weights restored)
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

                # Additional verification: multiple requests
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
                logger.info(
                    f"GPU MEMORY ACCOUNTING SUMMARY (Shadow Engine Failover - {backend.value}):"
                )
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
        # Cleanup: Stop all GPU Memory Service instances
        for gpu_mem_service in reversed(gpu_mem_services):
            try:
                gpu_mem_service.__exit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error stopping GPU Memory Service: {e}")

        # Cleanup socket files
        for device in range(tp):
            socket_path = socket_path_template.format(device=device)
            try:
                if os.path.exists(socket_path):
                    os.unlink(socket_path)
            except Exception as e:
                logger.warning(f"Error removing socket file {socket_path}: {e}")


@pytest.mark.timeout(300)  # 5 minutes
@pytest.mark.gpu_1  # Can run with single GPU
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(
            Backend.VLLM,
            id="vllm",
            marks=[
                pytest.mark.vllm,
                pytest.mark.skipif(not HAS_VLLM, reason="vLLM not installed"),
            ],
        ),
        pytest.param(
            Backend.SGLANG,
            id="sglang",
            marks=[
                pytest.mark.sglang,
                pytest.mark.skipif(not HAS_SGLANG, reason="SGLang not installed"),
            ],
        ),
    ],
)
def test_gpu_memory_service_basic_sleep_wake(
    request,
    runtime_services,
    predownload_models,
    backend: Backend,
):
    """
    Basic test for GPU Memory Service sleep/wake functionality.

    This is a simpler test that validates:
    1. Engine can be started with GPU Memory Service
    2. Engine can be put to sleep
    3. Engine can be woken up
    4. Engine works after wake

    Uses single GPU (TP=1) for broader compatibility.

    Args:
        backend: The inference backend to test (vLLM or SGLang)
    """
    # Allocate ports
    system_port = allocate_port(8100)
    frontend_port = allocate_port(8200)

    # Backend-specific ports
    extra_ports = []
    if backend == Backend.SGLANG:
        sglang_port = allocate_port(30000)
        bootstrap_port = allocate_port(8998)
        extra_ports = [sglang_port, bootstrap_port]

    # Socket path template - for single GPU just use device 0
    socket_path_template = "/tmp/gpu_memory_service_{device}.sock"
    socket_path = socket_path_template.format(device=0)

    try:
        # Start GPU Memory Service
        logger.info("Starting GPU Memory Service")
        with GPUMemoryServiceProcess(
            request,
            device=0,
            socket_path=socket_path,
        ):
            logger.info("GPU Memory Service started")

            # Start frontend (serves OpenAI-compatible API)
            logger.info("Starting frontend")
            with DynamoFrontendProcess(request, frontend_port=frontend_port):
                logger.info(f"Frontend started on port {frontend_port}")

                # Create ports dict for the factory function
                ports = {
                    "shadow_system_port": system_port,
                    "primary_system_port": system_port,
                    "primary_nixl_port": 5600,
                    "primary_kv_event_port": 20080,
                    "shadow_sglang_port": extra_ports[0] if extra_ports else 30000,
                    "shadow_bootstrap_port": extra_ports[1]
                    if len(extra_ports) > 1
                    else 8998,
                    "primary_sglang_port": extra_ports[0] if extra_ports else 30000,
                    "primary_bootstrap_port": extra_ports[1]
                    if len(extra_ports) > 1
                    else 8998,
                }

                # Start engine with GPU Memory Service
                logger.info(f"Starting {backend.value} engine with GPU Memory Service")
                engine = create_engine_process(
                    backend=backend,
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

                    # Wait for frontend to discover the engine
                    time.sleep(7)

                    # Test 1: Initial inference
                    logger.info("Test 1: Initial inference")
                    result = send_completion_request(
                        frontend_port, prompt="Hello, I am"
                    )
                    assert result["choices"], "Initial inference failed"

                    # Measure memory before sleep
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
                    time.sleep(2)  # Give time for memory to be released

                    # Measure memory after sleep
                    memory_after_sleep, _, _ = get_gpu_memory_usage(device=0)
                    logger.info(
                        f"GPU memory after sleep: {bytes_to_mb(memory_after_sleep):.1f} MB "
                        f"({bytes_to_gb(memory_after_sleep):.2f} GB)"
                    )

                    # CRITICAL ASSERTION: Sleep should reduce GPU memory usage
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

                    # Measure memory after wake
                    memory_after_wake, _, _ = get_gpu_memory_usage(device=0)
                    logger.info(
                        f"GPU memory after wake: {bytes_to_mb(memory_after_wake):.1f} MB "
                        f"({bytes_to_gb(memory_after_wake):.2f} GB)"
                    )

                    # Memory after wake should be similar to before sleep
                    # (weights are restored)
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
                    logger.info(f"GPU MEMORY ACCOUNTING SUMMARY ({backend.value}):")
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

                    logger.info(f"All basic sleep/wake tests passed ({backend.value})!")

    finally:
        deallocate_port(system_port)
        deallocate_port(frontend_port)
        for port in extra_ports:
            deallocate_port(port)
        if os.path.exists(socket_path):
            try:
                os.unlink(socket_path)
            except Exception:
                pass
