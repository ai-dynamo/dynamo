# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import time

import pytest

from tests.fault_tolerance.test_request_cancellation import (
    read_log_content,
    send_request_and_cancel,
    strip_ansi_codes,
)
from tests.serve.test_trtllm import TRTLLMConfig
from tests.utils.engine_process import EngineProcess

logger = logging.getLogger(__name__)


def verify_request_cancelled_trtllm(
    trtllm_process: EngineProcess,
    trtllm_log_offset: int = 0,
) -> int:
    """Simplified verification for TRTLLM cancellation messages

    Returns:
        int: new_trtllm_log_length
    """

    # Check TRTLLM log for cancellation pattern
    trtllm_log_content = read_log_content(trtllm_process.log_path)
    new_trtllm_content = trtllm_log_content[trtllm_log_offset:]

    # Look for TRTLLM-specific cancellation pattern
    has_trtllm_cancellation = False
    cancellation_pattern = "Aborted Request ID: "

    for line in new_trtllm_content.split("\n"):
        clean_line = strip_ansi_codes(line).strip()
        if cancellation_pattern in clean_line:
            has_trtllm_cancellation = True
            logger.info(f"Found TRTLLM cancellation message: {clean_line}")
            break

    if not has_trtllm_cancellation:
        pytest.fail(f"Could not find '{cancellation_pattern}' in TRTLLM backend log")

    return len(trtllm_log_content)


@pytest.mark.e2e
@pytest.mark.gpu_1
@pytest.mark.trtllm_marker
@pytest.mark.slow
def test_request_cancellation_trtllm_aggregated(request, runtime_services):
    """
    End-to-end test for request cancellation functionality with TRTLLM aggregated backend.

    This test verifies that when a request is cancelled by the client,
    the system properly handles the cancellation and cleans up resources
    on the TRTLLM backend side. Tests three scenarios:
    1. Completion request
    2. Chat completion request (non-streaming)
    3. Chat completion request (streaming)
    """

    trtllm_dir = os.environ.get("TRTLLM_DIR", "/workspace/components/backends/trtllm")

    # Configuration for TRTLLM aggregated cancellation test
    trtllm_config = TRTLLMConfig(
        name="aggregated_cancellation_test",
        directory=trtllm_dir,
        marks=[],
        request_payloads=[],
        model="Qwen/Qwen3-0.6B",
        script_name="agg.sh",
    )

    # Start TRTLLM aggregated backend
    logger.info("Starting TRTLLM aggregated backend...")
    extra_env = {
        "MODEL_PATH": trtllm_config.model,
        "SERVED_MODEL_NAME": trtllm_config.model,
        "DYN_LOG": "debug",
    }
    with EngineProcess.from_config(
        trtllm_config, request, extra_env=extra_env
    ) as trtllm_process:
        logger.info("TRTLLM aggregated backend started successfully")

        # Test request cancellation scenarios
        test_scenarios = [
            ("completion", "Completion request cancellation"),
            ("chat_completion", "Chat completion request cancellation"),
            (
                "chat_completion_stream",
                "Chat completion streaming request cancellation",
            ),
        ]

        trtllm_log_offset = 0

        for i, (request_type, description) in enumerate(test_scenarios, 1):
            logger.info(f"Testing {description.lower()} with TRTLLM...")
            send_request_and_cancel(request_type, timeout=1, model="Qwen/Qwen3-0.6B")

            logger.info("Checking for cancellation messages in TRTLLM logs...")
            time.sleep(1)  # Give more time for TRTLLM logs to be written
            trtllm_log_offset = verify_request_cancelled_trtllm(
                trtllm_process,
                trtllm_log_offset=trtllm_log_offset,
            )

            logger.info(f"{description} with TRTLLM detected successfully")

        logger.info("All TRTLLM request cancellation tests completed successfully")


@pytest.mark.e2e
@pytest.mark.gpu_2
@pytest.mark.trtllm_marker
@pytest.mark.slow
def test_request_cancellation_trtllm_disaggregated_decode_first(
    request, runtime_services
):
    """
    End-to-end test for request cancellation functionality with TRTLLM disaggregated backend
    using decode_first strategy.

    This test verifies that when a request is cancelled by the client,
    the system properly handles the cancellation and cleans up resources
    on the TRTLLM disaggregated backend side. Tests three scenarios:
    1. Completion request
    2. Chat completion request (non-streaming)
    3. Chat completion request (streaming)
    """

    trtllm_dir = os.environ.get("TRTLLM_DIR", "/workspace/components/backends/trtllm")

    # Configuration for TRTLLM disaggregated cancellation test with decode_first strategy
    trtllm_config = TRTLLMConfig(
        name="disaggregated_decode_first_cancellation_test",
        directory=trtllm_dir,
        marks=[],
        request_payloads=[],
        model="Qwen/Qwen3-0.6B",
        script_name="disagg.sh",
    )

    # Start TRTLLM disaggregated backend with decode_first strategy
    logger.info("Starting TRTLLM disaggregated backend with decode_first strategy...")
    extra_env = {
        "MODEL_PATH": trtllm_config.model,
        "SERVED_MODEL_NAME": trtllm_config.model,
        "DYN_LOG": "debug",
        "DISAGGREGATION_STRATEGY": "decode_first",
    }
    with EngineProcess.from_config(
        trtllm_config, request, extra_env=extra_env
    ) as trtllm_process:
        logger.info("TRTLLM disaggregated backend (decode_first) started successfully")

        # Test request cancellation scenarios
        test_scenarios = [
            ("completion", "Completion request cancellation"),
            ("chat_completion", "Chat completion request cancellation"),
            (
                "chat_completion_stream",
                "Chat completion streaming request cancellation",
            ),
        ]

        trtllm_log_offset = 0

        for i, (request_type, description) in enumerate(test_scenarios, 1):
            logger.info(
                f"Testing {description.lower()} with TRTLLM disaggregated (decode_first)..."
            )
            send_request_and_cancel(request_type, timeout=1, model="Qwen/Qwen3-0.6B")

            logger.info("Checking for cancellation messages in TRTLLM logs...")
            time.sleep(1)  # Give more time for TRTLLM logs to be written
            trtllm_log_offset = verify_request_cancelled_trtllm(
                trtllm_process,
                trtllm_log_offset=trtllm_log_offset,
            )

            logger.info(
                f"{description} with TRTLLM disaggregated (decode_first) detected successfully"
            )

        logger.info(
            "All TRTLLM disaggregated (decode_first) request cancellation tests completed successfully"
        )
