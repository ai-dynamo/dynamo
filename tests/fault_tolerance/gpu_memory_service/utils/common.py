# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Shared utilities for GPU Memory Service tests.

This module provides process managers and helper functions that are
backend-agnostic and can be used by vLLM, SGLang, or other backends.
"""

import logging
import os
import shutil
import signal
import time
from typing import Callable

import pynvml
import requests
from gpu_memory_service.common.utils import get_socket_path

from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME
from tests.utils.managed_process import DynamoFrontendProcess, ManagedProcess

logger = logging.getLogger(__name__)


def get_gpu_memory_used(device: int = 0) -> int:
    """Get GPU memory usage in bytes for the specified device."""
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        return pynvml.nvmlDeviceGetMemoryInfo(handle).used
    finally:
        pynvml.nvmlShutdown()


def kill_force(
    process: ManagedProcess,
    timeout_s: float = 30.0,
    poll_interval_s: float = 0.5,
) -> None:
    """SIGKILL a process group and wait for GPU memory reclamation.

    Snapshots GPU memory before the kill, sends SIGKILL to the entire
    process group, reaps the zombie, then polls pynvml until the CUDA
    driver finishes asynchronous cleanup (memory drops below the
    pre-kill snapshot).
    """
    mem_before = get_gpu_memory_used()

    pid = process.get_pid()
    if pid is None:
        logger.warning("kill_force: no PID available")
        return

    try:
        pgid = os.getpgid(pid)
        logger.info(f"kill_force: sending SIGKILL to process group {pgid} (pid={pid})")
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        logger.warning(f"kill_force: process {pid} already dead")
        return

    # Reap the process to avoid zombies
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        pass

    # Wait for CUDA driver to asynchronously reclaim GPU memory
    start = time.time()
    mem_after = mem_before
    while time.time() - start < timeout_s:
        mem_after = get_gpu_memory_used()
        if mem_after < mem_before:
            break
        time.sleep(poll_interval_s)

    freed_mb = (mem_before - mem_after) / (1 << 20)
    logger.info(
        f"kill_force: before={mem_before / (1 << 30):.2f} GiB, "
        f"after={mem_after / (1 << 30):.2f} GiB, freed={freed_mb:.0f} MB"
    )


def send_completion(
    port: int, prompt: str = "Hello", max_retries: int = 3, retry_delay: float = 1.0
) -> dict:
    """Send a completion request to the frontend.

    Includes retry logic to handle transient failures from stale routing
    (e.g., after failover when etcd still has dead instance entries).
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            r = requests.post(
                f"http://localhost:{port}/v1/completions",
                json={
                    "model": FAULT_TOLERANCE_MODEL_NAME,
                    "prompt": prompt,
                    "max_tokens": 20,
                },
                timeout=120,
            )
            r.raise_for_status()
            result = r.json()
            assert result.get("choices"), "No choices in response"
            if attempt > 0:
                logger.info(f"send_completion succeeded after {attempt + 1} attempts")
            return result
        except (requests.exceptions.RequestException, AssertionError) as e:
            last_error = e
            if attempt < max_retries - 1:
                logger.debug(
                    f"send_completion attempt {attempt + 1}/{max_retries} failed: {e}"
                )
                time.sleep(retry_delay)
    raise last_error  # type: ignore


class GMSServerProcess(ManagedProcess):
    """Manages GMS server lifecycle for tests."""

    def __init__(self, request, device: int):
        self.device = device
        self.socket_path = get_socket_path(device)

        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        log_dir = f"{request.node.name}_gms_{device}"
        shutil.rmtree(log_dir, ignore_errors=True)

        super().__init__(
            command=["python3", "-m", "gpu_memory_service", "--device", str(device)],
            env={**os.environ, "DYN_LOG": "debug"},
            timeout=60,
            display_output=True,
            terminate_all_matching_process_names=False,
            log_dir=log_dir,
            health_check_funcs=[self._socket_ready],
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            return super().__exit__(exc_type, exc_val, exc_tb)
        finally:
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)

    def _socket_ready(self, timeout: float = 30) -> bool:
        start = time.time()
        while time.time() - start < timeout:
            if os.path.exists(self.socket_path):
                return True
            time.sleep(0.1)
        return False


def run_shadow_failover_test(
    request,
    ports: dict,
    make_shadow: Callable[[], ManagedProcess],
    make_primary: Callable[[], ManagedProcess],
) -> None:
    """Shared shadow-engine failover flow for both vLLM and SGLang.

    1. Start shadow -> verify inference
    2. Sleep shadow -> log memory freed
    3. Start primary -> verify inference
    4. kill -9 primary -> wait for GPU memory reclamation
    5. Wake shadow -> verify inference x 3
    """
    frontend_port = ports["frontend"]

    with GMSServerProcess(request, device=0):
        with DynamoFrontendProcess(request, frontend_port=frontend_port):
            with make_shadow() as shadow:
                # Shadow inference
                result = send_completion(frontend_port)
                assert result["choices"], "Shadow inference failed"
                logger.info(f"Shadow inference OK: {result}")

                # Sleep shadow
                mem_before = get_gpu_memory_used()
                assert shadow.sleep()["status"] == "ok"
                mem_after = get_gpu_memory_used()
                logger.info(
                    f"Shadow sleep: {mem_before / (1 << 30):.2f} -> "
                    f"{mem_after / (1 << 30):.2f} GiB "
                    f"(freed {(mem_before - mem_after) / (1 << 20):.0f} MB)"
                )

                # Primary: start, verify, kill -9
                with make_primary() as primary:
                    result = send_completion(frontend_port, "Primary test")
                    assert result["choices"], "Primary inference failed"
                    logger.info(f"Primary inference OK: {result}")
                    kill_force(primary)

                # Wake shadow, verify 3x
                assert shadow.wake()["status"] == "ok"
                for i in range(3):
                    result = send_completion(frontend_port, f"Verify {i}")
                    assert result["choices"], f"Verification {i} failed"
                logger.info("All verification passed")
