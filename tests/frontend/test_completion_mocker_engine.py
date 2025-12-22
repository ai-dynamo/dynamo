# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Parallelization: Hermetic test (xdist-safe via dynamic ports).
# Tested on: Linux (Ubuntu 24.04 container), Intel(R) Core(TM) i9-14900K, 32 vCPU.
# post_merge wall time:
# - Serialized: 97.29s.
# - Parallel (-n auto): 30.29s (67.00s saved, 3.21x).
# GPU Requirement: gpu_0 (CPU-only, mocker does not use GPU)

from __future__ import annotations

import logging
import time
from typing import Any, Dict

import pytest
import requests

from tests.utils.constants import QWEN

logger = logging.getLogger(__name__)

TEST_MODEL = QWEN

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu_0,  # Mocker is CPU-only (no GPU required)
    pytest.mark.post_merge,
    pytest.mark.parallel,
    pytest.mark.model(TEST_MODEL),
]


def _send_completion_request(
    payload: Dict[str, Any],
    frontend_port: int,
    timeout: int = 180,
) -> requests.Response:
    """Send a text completion request"""

    headers = {"Content-Type": "application/json"}
    print(f"Sending request: {time.time()}")

    response = requests.post(
        f"http://localhost:{frontend_port}/v1/completions",
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    return response


def test_completion_string_prompt(start_services_with_mocker) -> None:
    frontend_port = start_services_with_mocker
    payload: Dict[str, Any] = {
        "model": TEST_MODEL,
        "prompt": "Tell me about Mars",
        "max_tokens": 2000,
    }

    response = _send_completion_request(payload, frontend_port)

    assert response.status_code == 200, (
        f"Completion request failed with status "
        f"{response.status_code}: {response.text}"
    )


def test_completion_empty_array_prompt(start_services_with_mocker) -> None:
    frontend_port = start_services_with_mocker
    payload: Dict[str, Any] = {
        "model": TEST_MODEL,
        "prompt": [],
        "max_tokens": 2000,
    }

    response = _send_completion_request(payload, frontend_port)

    assert response.status_code == 400, (
        f"Completion request should failed with status 400 but got"
        f"{response.status_code}: {response.text}"
    )


def test_completion_single_element_array_prompt(start_services_with_mocker) -> None:
    frontend_port = start_services_with_mocker
    payload: Dict[str, Any] = {
        "model": TEST_MODEL,
        "prompt": ["Tell me about Mars"],
        "max_tokens": 2000,
    }

    response = _send_completion_request(payload, frontend_port)

    assert response.status_code == 200, (
        f"Completion request failed with status "
        f"{response.status_code}: {response.text}"
    )


def test_completion_multi_element_array_prompt(start_services_with_mocker) -> None:
    frontend_port = start_services_with_mocker
    payload: Dict[str, Any] = {
        "model": TEST_MODEL,
        "prompt": [
            "Tell me about Mars",
            "Tell me about Ceres",
            "Tell me about Jupiter",
        ],
        "max_tokens": 300,
    }

    response = _send_completion_request(payload, frontend_port)
    response_data = response.json()

    assert response.status_code == 200, (
        f"Completion request failed with status "
        f"{response.status_code}: {response.text}"
    )

    expected_choices = len(payload.get("prompt"))  # type: ignore
    choices = len(response_data.get("choices", []))

    assert (
        expected_choices == choices
    ), f"Expected {expected_choices} choices, got {choices}"
