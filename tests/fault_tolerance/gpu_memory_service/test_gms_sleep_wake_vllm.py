# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
GPU Memory Service Basic Sleep/Wake Test for vLLM.

Tests the basic sleep/wake cycle of a single vLLM engine using the GPU Memory
Service for VA-stable weight offloading.
"""

import logging

import pytest

from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME
from tests.utils.managed_process import DynamoFrontendProcess

from .utils.common import GMSServerProcess, get_gpu_memory_used, send_completion
from .utils.vllm import VLLMWithGMSProcess

logger = logging.getLogger(__name__)


@pytest.mark.vllm
@pytest.mark.e2e
@pytest.mark.gpu_1
@pytest.mark.fault_tolerance
@pytest.mark.nightly
@pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME)
@pytest.mark.timeout(300)
def test_gms_basic_sleep_wake(request, runtime_services, gms_ports, predownload_models):
    """Test basic sleep/wake with GPU Memory Service.

    1. Start GMS server and vLLM engine with GMS integration
    2. Run initial inference to verify engine works
    3. Put engine to sleep and verify GPU memory is freed
    4. Wake engine and verify inference still works
    """
    ports = gms_ports

    with GMSServerProcess(request, device=0):
        with DynamoFrontendProcess(request, frontend_port=ports["frontend"]):
            with VLLMWithGMSProcess(
                request,
                "engine",
                ports["shadow_system"],
                ports["shadow_kv_event"],
                ports["shadow_nixl"],
                ports["frontend"],
            ) as engine:
                # Initial inference
                result = send_completion(ports["frontend"])
                logger.info(f"Initial inference result: {result}")
                assert result["choices"]

                mem_before = get_gpu_memory_used()
                logger.info(f"Memory before sleep: {mem_before / (1 << 20):.0f} MB")

                # Sleep
                sleep_result = engine.sleep()
                assert sleep_result["status"] == "ok"

                mem_after_sleep = get_gpu_memory_used()
                logger.info(f"Memory after sleep: {mem_after_sleep / (1 << 20):.0f} MB")
                assert mem_after_sleep < mem_before, "Sleep should reduce memory"

                # Wake
                wake_result = engine.wake()
                assert wake_result["status"] == "ok"

                # Inference after wake
                result = send_completion(ports["frontend"], "Goodbye")
                logger.info(f"Post-wake inference result: {result}")
                assert result["choices"]

                logger.info(
                    f"Memory freed: {(mem_before - mem_after_sleep) / (1 << 20):.0f} MB"
                )
