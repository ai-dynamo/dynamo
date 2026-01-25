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
import time

import pynvml
import requests
from gpu_memory_service.common.utils import get_socket_path

from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME
from tests.utils.managed_process import ManagedProcess

logger = logging.getLogger(__name__)


def bytes_to_mb(b: int) -> float:
    """Convert bytes to megabytes."""
    return b / (1024 * 1024)


def get_gpu_memory_used(device: int = 0) -> int:
    """Get GPU memory usage in bytes for the specified device."""
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        return pynvml.nvmlDeviceGetMemoryInfo(handle).used
    finally:
        pynvml.nvmlShutdown()


def send_completion(port: int, prompt: str = "Hello") -> dict:
    """Send a completion request to the frontend."""
    r = requests.post(
        f"http://localhost:{port}/v1/completions",
        json={"model": FAULT_TOLERANCE_MODEL_NAME, "prompt": prompt, "max_tokens": 20},
        timeout=120,
    )
    r.raise_for_status()
    result = r.json()
    assert result.get("choices"), "No choices in response"
    return result


class GMSServerProcess(ManagedProcess):
    """GPU Memory Service server process."""

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
            terminate_existing=False,
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
