# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
GPU Memory Service Basic Sleep/Wake Test for SGLang.

Tests the basic sleep/wake cycle of a single SGLang engine using the GPU Memory
Service for VA-stable weight offloading.
"""

import logging
from contextlib import ExitStack

import pytest

from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME
from tests.utils.managed_process import DynamoFrontendProcess

from .utils.common import (
    MIN_EXPECTED_MEMORY_RETURN_FRACTION,
    GMSServerProcess,
    assert_log_contains_in_order,
    get_gpu_memory_used,
    send_completion,
)
from .utils.sglang import SGLangWithGMSProcess

logger = logging.getLogger(__name__)


@pytest.mark.sglang
@pytest.mark.e2e
@pytest.mark.gpu_1
@pytest.mark.fault_tolerance
@pytest.mark.nightly
@pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME)
@pytest.mark.timeout(300)
def test_gms_basic_sleep_wake(request, runtime_services, gms_ports, predownload_models):
    """Test basic sleep/wake with GPU Memory Service.

    1. Start GMS server and SGLang engine with GMS integration
    2. Run initial inference to verify engine works
    3. Put engine to sleep and verify GPU memory is freed
    4. Wake engine and verify inference still works
    """
    ports = gms_ports

    with ExitStack() as stack:
        weights_gms = stack.enter_context(
            GMSServerProcess(request, device=0, scope="weights")
        )
        kv_cache_gms = stack.enter_context(
            GMSServerProcess(request, device=0, scope="kv_cache")
        )
        stack.enter_context(
            DynamoFrontendProcess(request, frontend_port=ports["frontend"])
        )
        with SGLangWithGMSProcess(
            request,
            "engine",
            ports["shadow_system"],
            ports["shadow_sglang"],
            ports["frontend"],
        ) as engine:
            result = send_completion(ports["frontend"])
            logger.info(f"Initial inference result: {result}")
            assert result["choices"]

            mem_before = get_gpu_memory_used()
            logger.info(f"Memory before sleep: {mem_before / (1 << 20):.0f} MB")

            sleep_result = engine.sleep()
            assert sleep_result["status"] == "ok"

            mem_after_sleep = get_gpu_memory_used()
            released_bytes = mem_before - mem_after_sleep
            logger.info(f"Memory after sleep: {mem_after_sleep / (1 << 20):.0f} MB")
            assert mem_after_sleep < mem_before, "Sleep should reduce memory"
            assert released_bytes > 0

            assert_log_contains_in_order(
                weights_gms.read_logs(),
                [
                    "RW connected; opened active epoch 1",
                    "Committed epoch 1",
                ],
            )
            assert_log_contains_in_order(
                kv_cache_gms.read_logs(),
                [
                    "RW connected; opened active epoch 1",
                    "RW aborted; clearing active epoch 1",
                    "allocations from epoch 1",
                ],
            )

            wake_result = engine.wake()
            assert wake_result["status"] == "ok"

            mem_after_wake = get_gpu_memory_used()
            reacquired_bytes = mem_after_wake - mem_after_sleep
            logger.info(f"Memory after wake: {mem_after_wake / (1 << 20):.0f} MB")
            assert mem_after_wake > mem_after_sleep, "Wake should reacquire memory"
            assert (
                reacquired_bytes
            ) >= released_bytes * MIN_EXPECTED_MEMORY_RETURN_FRACTION

            assert "RW connected; opened active epoch 2" not in weights_gms.read_logs()
            assert_log_contains_in_order(
                kv_cache_gms.read_logs(),
                [
                    "RW connected; opened active epoch 1",
                    "RW aborted; clearing active epoch 1",
                    "allocations from epoch 1",
                    "RW connected; opened active epoch 2",
                ],
            )

            result = send_completion(ports["frontend"], "Goodbye")
            logger.info(f"Post-wake inference result: {result}")
            assert result["choices"]

            logger.info(
                f"Memory freed: {(mem_before - mem_after_sleep) / (1 << 20):.0f} MB"
            )
