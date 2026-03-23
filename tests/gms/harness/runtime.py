# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pynvml
import requests

from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[3]
DYNAMO_BIN = REPO_ROOT / "dynamo" / "bin"
MIN_EXPECTED_MEMORY_RETURN_FRACTION = 0.6


def _default_nvml_device() -> int:
    """Return the NVML physical device index for CUDA device 0.

    With CUDA_VISIBLE_DEVICES set (e.g. "1,2"), CUDA device 0 maps to the
    first entry, which is NVML physical device 1.  NVML always uses physical
    indices, so we parse the first entry from CUDA_VISIBLE_DEVICES to get the
    correct NVML index.
    """
    cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_vis:
        first = cuda_vis.split(",")[0].strip()
        try:
            return int(first)
        except ValueError:
            pass
    return 0


_GMS_NVML_DEVICE: int = _default_nvml_device()


def get_gpu_memory_used(device: int = _GMS_NVML_DEVICE) -> int:
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        return pynvml.nvmlDeviceGetMemoryInfo(handle).used
    finally:
        pynvml.nvmlShutdown()


def send_completion(
    port: int,
    prompt: str = "Hello",
    *,
    model: str = FAULT_TOLERANCE_MODEL_NAME,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> dict:
    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"http://localhost:{port}/v1/completions",
                json={
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": 20,
                },
                timeout=120,
            )
            response.raise_for_status()
            result = response.json()
            assert result.get("choices"), "No choices in response"
            if attempt > 0:
                logger.info("send_completion succeeded after %d attempts", attempt + 1)
            return result
        except (requests.exceptions.RequestException, AssertionError) as exc:
            last_error = exc
            if attempt < max_retries - 1:
                logger.debug(
                    "send_completion attempt %d/%d failed: %s",
                    attempt + 1,
                    max_retries,
                    exc,
                )
                time.sleep(retry_delay)
    raise last_error  # type: ignore[misc]


def wait_for_memory_drop(
    baseline_bytes: int,
    *,
    timeout_s: float = 30.0,
    poll_interval_s: float = 0.5,
    device: int = _GMS_NVML_DEVICE,
) -> int:
    """Poll until GPU memory drops below *baseline_bytes*, then return current usage.

    Returns the last observed usage (which may still be >= baseline if timeout fired).
    """
    deadline = time.monotonic() + timeout_s
    current = get_gpu_memory_used(device)
    while time.monotonic() < deadline:
        if current < baseline_bytes:
            return current
        time.sleep(poll_interval_s)
        current = get_gpu_memory_used(device)
    return current
