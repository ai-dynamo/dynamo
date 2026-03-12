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
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from typing import Callable

import pynvml
import requests
from gpu_memory_service.common.utils import get_socket_path

from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME
from tests.utils.managed_process import DynamoFrontendProcess, ManagedProcess

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[4]
DYNAMO_BIN = REPO_ROOT / "dynamo" / "bin"
MIN_EXPECTED_MEMORY_RETURN_FRACTION = 0.6


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
) -> None:
    """SIGKILL a process group and reap it."""
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

    logger.info("kill_force: reaped process group for pid=%d", pid)


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


def assert_log_contains_in_order(log_text: str, parts: list[str]) -> None:
    offset = 0
    for part in parts:
        index = log_text.find(part, offset)
        assert index >= 0, f"Missing log line in order: {part}\n\n{log_text}"
        offset = index + len(part)


class GMSServerProcess(ManagedProcess):
    """Manages GMS server lifecycle for tests."""

    def __init__(self, request, device: int, scope: str = "weights"):
        self.device = device
        self.scope = scope
        self.socket_path = get_socket_path(device, scope)

        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        log_dir = f"{request.node.name}_gms_{scope}_{device}"
        shutil.rmtree(log_dir, ignore_errors=True)

        super().__init__(
            command=[
                "python",
                "-m",
                "gpu_memory_service",
                "--device",
                str(device),
                "--scope",
                scope,
            ],
            env={
                **os.environ,
                "PATH": f"{DYNAMO_BIN}:{os.environ.get('PATH', '')}",
                "DYN_LOG": "debug",
            },
            timeout=60,
            display_output=True,
            terminate_all_matching_process_names=False,
            log_dir=log_dir,
            display_name=f"gms_{scope}",
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
    4. Start shadow wake while primary is still alive and verify it blocks
    5. kill -9 primary
    6. Shadow wake completes in the next KV epoch, then verify inference x 3
    """
    frontend_port = ports["frontend"]

    with ExitStack() as stack:
        weights_gms = stack.enter_context(
            GMSServerProcess(request, device=0, scope="weights")
        )
        kv_cache_gms = stack.enter_context(
            GMSServerProcess(request, device=0, scope="kv_cache")
        )
        stack.enter_context(
            DynamoFrontendProcess(
                request,
                frontend_port=frontend_port,
                display_name="frontend",
            )
        )
        with make_shadow() as shadow:
            result = send_completion(frontend_port)
            assert result["choices"], "Shadow inference failed"
            logger.info(f"Shadow inference OK: {result}")

            shadow_memory_before_sleep = get_gpu_memory_used()
            assert shadow.sleep()["status"] == "ok"
            shadow_memory_after_sleep = get_gpu_memory_used()
            shadow_released_bytes = (
                shadow_memory_before_sleep - shadow_memory_after_sleep
            )
            logger.info(
                f"Shadow sleep: {shadow_memory_before_sleep / (1 << 30):.2f} -> "
                f"{shadow_memory_after_sleep / (1 << 30):.2f} GiB "
                f"(freed {shadow_released_bytes / (1 << 20):.0f} MB)"
            )
            assert shadow_memory_after_sleep < shadow_memory_before_sleep
            assert shadow_released_bytes > 0

            weights_log = weights_gms.read_logs()
            kv_cache_log = kv_cache_gms.read_logs()
            assert_log_contains_in_order(
                weights_log,
                [
                    "RW connected; opened active epoch 1",
                    "Committed epoch 1",
                ],
            )
            assert "RW connected; opened active epoch 2" not in weights_log
            assert_log_contains_in_order(
                kv_cache_log,
                [
                    "RW connected; opened active epoch 1",
                    "RW aborted; clearing active epoch 1",
                    "allocations from epoch 1",
                ],
            )

            with make_primary() as primary:
                result = send_completion(frontend_port, "Primary test")
                assert result["choices"], "Primary inference failed"
                logger.info(f"Primary inference OK: {result}")

                primary_memory_in_use = get_gpu_memory_used()
                logger.info(
                    "Primary active memory: %.2f GiB",
                    primary_memory_in_use / (1 << 30),
                )
                assert primary_memory_in_use > shadow_memory_after_sleep
                assert (
                    primary_memory_in_use - shadow_memory_after_sleep
                ) >= shadow_released_bytes * MIN_EXPECTED_MEMORY_RETURN_FRACTION

                assert_log_contains_in_order(
                    kv_cache_gms.read_logs(),
                    [
                        "RW connected; opened active epoch 1",
                        "RW aborted; clearing active epoch 1",
                        "allocations from epoch 1",
                        "RW connected; opened active epoch 2",
                    ],
                )

                with ThreadPoolExecutor(max_workers=1) as executor:
                    wake_future = executor.submit(shadow.wake, 180)
                    blocked_started_at = time.monotonic()
                    while time.monotonic() - blocked_started_at < 10:
                        if wake_future.done():
                            break
                        time.sleep(0.2)
                    assert not wake_future.done(), (
                        "Shadow wake completed before the primary died; "
                        "KV cache RW handoff did not block as expected"
                    )
                    assert (
                        "RW connected; opened active epoch 3"
                        not in kv_cache_gms.read_logs()
                    )

                    primary_memory_before_kill = get_gpu_memory_used()
                    kill_force(primary)
                    primary_memory_after_kill = get_gpu_memory_used()
                    logger.info(
                        "Primary kill snapshot: %.2f -> %.2f GiB",
                        primary_memory_before_kill / (1 << 30),
                        primary_memory_after_kill / (1 << 30),
                    )

                    log_wait_started_at = time.monotonic()
                    while time.monotonic() - log_wait_started_at < 30:
                        if (
                            "RW connected; opened active epoch 3"
                            in kv_cache_gms.read_logs()
                        ):
                            break
                        time.sleep(0.2)
                    assert (
                        "RW connected; opened active epoch 3"
                        in kv_cache_gms.read_logs()
                    )

                    wake_result = wake_future.result(timeout=180)

            assert wake_result["status"] == "ok"
            shadow_memory_after_wake = get_gpu_memory_used()
            shadow_reacquired_bytes = (
                shadow_memory_after_wake - shadow_memory_after_sleep
            )
            logger.info(
                "Shadow wake memory: %.2f GiB (reacquired %.0f MB)",
                shadow_memory_after_wake / (1 << 30),
                shadow_reacquired_bytes / (1 << 20),
            )
            assert shadow_memory_after_wake > shadow_memory_after_sleep
            assert (
                shadow_reacquired_bytes
            ) >= shadow_released_bytes * MIN_EXPECTED_MEMORY_RETURN_FRACTION

            assert_log_contains_in_order(
                kv_cache_gms.read_logs(),
                [
                    "RW connected; opened active epoch 1",
                    "RW aborted; clearing active epoch 1",
                    "allocations from epoch 1",
                    "RW connected; opened active epoch 2",
                    "RW aborted; clearing active epoch 2",
                    "allocations from epoch 2",
                    "RW connected; opened active epoch 3",
                ],
            )
            assert "RW connected; opened active epoch 2" not in weights_gms.read_logs()

            for i in range(3):
                result = send_completion(frontend_port, f"Verify {i}")
                assert result["choices"], f"Verification {i} failed"
            logger.info("All verification passed")
