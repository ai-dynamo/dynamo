# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
GPU Memory Service Shadow Engine Failover Test for SGLang.

Tests the shadow engine failover scenario where a sleeping shadow engine can
wake up and take over when the primary engine fails.
"""

import logging

import pytest

from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME
from tests.utils.managed_process import DynamoFrontendProcess

from .utils.common import GMSServerProcess, get_gpu_memory_used, send_completion
from .utils.sglang import SGLangWithGMSProcess

logger = logging.getLogger(__name__)


@pytest.mark.sglang
@pytest.mark.e2e
@pytest.mark.gpu_1
@pytest.mark.fault_tolerance
@pytest.mark.nightly
@pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME)
@pytest.mark.timeout(600)
def test_gms_shadow_engine_failover(
    request, runtime_services, gms_ports, predownload_models
):
    """Test shadow engine failover with GPU Memory Service.

    1. Start shadow engine and put it to sleep
    2. Start primary engine and serve inference
    3. Kill primary engine
    4. Wake shadow engine and verify it handles inference
    """
    ports = gms_ports

    with GMSServerProcess(request, device=0):
        with DynamoFrontendProcess(request, frontend_port=ports["frontend"]):
            # Start shadow engine
            with SGLangWithGMSProcess(
                request,
                "shadow",
                ports["shadow_system"],
                ports["shadow_sglang"],
                ports["frontend"],
            ) as shadow:
                # Verify shadow works
                result = send_completion(ports["frontend"])
                logger.info(f"Shadow inference result: {result}")
                assert result["choices"]
                logger.info("Shadow inference OK")

                # Sleep shadow (release memory occupation)
                mem_before = get_gpu_memory_used()
                sleep_result = shadow.sleep()
                assert sleep_result["status"] == "ok"

                mem_after_sleep = get_gpu_memory_used()
                logger.info(
                    f"Shadow sleep freed {(mem_before - mem_after_sleep) / (1 << 20):.0f} MB"
                )
                assert mem_after_sleep < mem_before

                # Start primary engine
                with SGLangWithGMSProcess(
                    request,
                    "primary",
                    ports["primary_system"],
                    ports["primary_sglang"],
                    ports["frontend"],
                ):
                    result = send_completion(ports["frontend"], "Primary test")
                    logger.info(f"Primary inference result: {result}")
                    assert result["choices"]
                    logger.info("Primary inference OK")

                # Primary is dead (exited context manager)

                # Wake shadow (resume memory occupation)
                wake_result = shadow.wake()
                assert wake_result["status"] == "ok"

                # Verify shadow handles failover
                result = send_completion(ports["frontend"], "After failover")
                logger.info(f"Failover inference result: {result}")
                assert result["choices"]
                logger.info("Shadow handles failover OK")

                for i in range(3):
                    result = send_completion(ports["frontend"], f"Verify {i}")
                    logger.info(f"Verification {i} result: {result}")
                    assert result["choices"]
                logger.info("All verification passed")
