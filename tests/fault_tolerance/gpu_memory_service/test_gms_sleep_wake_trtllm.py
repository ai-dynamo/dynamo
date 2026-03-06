# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU Memory Service basic sleep/wake tests for TensorRT-LLM."""

import logging

import pytest

from tests.utils.managed_process import DynamoFrontendProcess

from .utils.common import GMSServerProcess, get_gpu_memory_used, send_completion
from .utils.trtllm import (
    TRTLLM_GMS_MODEL_NAME,
    TRTLLM_GMS_READ_ONLY_CONFIG,
    TRTLLMWithGMSProcess,
)

logger = logging.getLogger(__name__)


@pytest.mark.trtllm
@pytest.mark.e2e
@pytest.mark.gpu_1
@pytest.mark.fault_tolerance
@pytest.mark.nightly
@pytest.mark.model(TRTLLM_GMS_MODEL_NAME)
@pytest.mark.timeout(300)
def test_gms_basic_sleep_wake(request, runtime_services, gms_ports, predownload_models):
    """Validate TRT-LLM sleep/wake with GMS-backed weights."""
    ports = gms_ports

    with GMSServerProcess(request, device=0):
        with DynamoFrontendProcess(request, frontend_port=ports["frontend"]):
            with TRTLLMWithGMSProcess(
                request,
                "engine",
                ports["shadow_system"],
                ports["frontend"],
            ) as engine:
                result = send_completion(ports["frontend"], model=TRTLLM_GMS_MODEL_NAME)
                logger.info("Initial inference result: %s", result)
                assert result["choices"]

                mem_before = get_gpu_memory_used()
                logger.info("Memory before sleep: %.0f MB", mem_before / (1 << 20))

                sleep_result = engine.sleep()
                assert sleep_result["status"] == "ok"

                mem_after_sleep = get_gpu_memory_used()
                logger.info("Memory after sleep: %.0f MB", mem_after_sleep / (1 << 20))
                if "kv_cache" in sleep_result.get("skipped_tags", []):
                    logger.info(
                        "Skipping strict memory reduction assertion: "
                        "kv_cache sleep is unsupported in this TRT-LLM runtime."
                    )
                else:
                    assert mem_after_sleep < mem_before, "Sleep should reduce memory"

                wake_result = engine.wake()
                assert wake_result["status"] == "ok"

                result = send_completion(
                    ports["frontend"], "Goodbye", model=TRTLLM_GMS_MODEL_NAME
                )
                logger.info("Post-wake inference result: %s", result)
                assert result["choices"]

                logger.info(
                    "Memory freed: %.0f MB", (mem_before - mem_after_sleep) / (1 << 20)
                )


@pytest.mark.trtllm
@pytest.mark.e2e
@pytest.mark.gpu_1
@pytest.mark.fault_tolerance
@pytest.mark.nightly
@pytest.mark.model(TRTLLM_GMS_MODEL_NAME)
@pytest.mark.timeout(420)
def test_gms_read_only_import_mode(
    request, runtime_services, gms_ports, predownload_models
):
    """Validate TRT-LLM imports committed weights in gms_read_only mode."""
    ports = gms_ports
    system_port = ports["shadow_system"]

    with GMSServerProcess(request, device=0):
        with DynamoFrontendProcess(request, frontend_port=ports["frontend"]):
            with TRTLLMWithGMSProcess(
                request,
                "writer",
                system_port,
                ports["frontend"],
            ):
                result = send_completion(ports["frontend"], model=TRTLLM_GMS_MODEL_NAME)
                logger.info("Writer inference result: %s", result)
                assert result["choices"]

            with TRTLLMWithGMSProcess(
                request,
                "reader_ro",
                system_port,
                ports["frontend"],
                model_loader_extra_config=TRTLLM_GMS_READ_ONLY_CONFIG,
            ):
                result = send_completion(
                    ports["frontend"],
                    "Read-only import validation",
                    model=TRTLLM_GMS_MODEL_NAME,
                )
                logger.info("Read-only inference result: %s", result)
                assert result["choices"]
