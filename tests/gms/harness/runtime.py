# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import time
from pathlib import Path

import requests

from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[3]
DYNAMO_BIN = REPO_ROOT / "dynamo" / "bin"
MIN_EXPECTED_MEMORY_RETURN_FRACTION = 0.6


def get_gpu_memory_used(device: int = 0) -> int:
    import pynvml

    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        return pynvml.nvmlDeviceGetMemoryInfo(handle).used
    finally:
        pynvml.nvmlShutdown()


def send_completion(
    port: int,
    prompt: str = "Hello",
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> dict:
    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"http://localhost:{port}/v1/completions",
                json={
                    "model": FAULT_TOLERANCE_MODEL_NAME,
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
