#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Determinism test for language model API using pytest.

This test suite checks if the model produces deterministic outputs
when given the same inputs with fixed seed and temperature=0.

The test uses comprehensive server warmup (sending all test prompts
before validation) to avoid server initialization effects that could
impact determinism measurements.

This is a TensorRTLLM only test.
"""

import logging
import os
import shutil

import pytest
import requests

from tests.utils.engine_process import FRONTEND_PORT
from tests.utils.managed_process import DynamoFrontendProcess, ManagedProcess
from tests.utils.payloads import check_models_api

from .common import check_module_available

logger = logging.getLogger(__name__)


HAS_TRTLLM = check_module_available("tensorrt_llm")

# Just need a model to show the config works rather than any stress of the system.
MODEL_PATH = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
SERVED_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

PROMPT = "In the heart of Eldoria, an ancient land of boundless magic and mysterious creatures, lies the long-forgotten city of Aeloria. Once a beacon of knowledge and power, Aeloria was buried beneath the shifting sands of time, lost to the world for centuries. You are an intrepid explorer, known for your unparalleled curiosity and courage, who has stumbled upon an ancient map hinting at ests that Aeloria holds a secret so profound that it has the potential to reshape the very fabric of reality. Your journey will take you through treacherous deserts, enchanted forests, and across perilous mountain ranges. Your Task: Character Background: Develop a detailed background for your character. Describe their motivations for seeking out Aeloria, their skills and weaknesses, and any personal connections to the ancient city or its legends. Are they driven by a quest for knowledge, a search for lost familt clue is hidden."


class DynamoWorkerProcess(ManagedProcess):
    """Process manager for Dynamo worker with TRTLLM backend"""

    def __init__(self, request, worker_id: str, engine_config: str):
        self.worker_id = worker_id

        command = [
            "python3",
            "-m",
            "dynamo.trtllm",
            "--model",
            MODEL_PATH,
            "--served-model-name",
            SERVED_MODEL_NAME,
            "--extra-engine-args",
            engine_config,
        ]

        # Set debug logging environment
        env = os.environ.copy()
        env["DYN_LOG"] = "debug"
        env["DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS"] = '["generate"]'
        env["DYN_SYSTEM_PORT"] = "9345"
        env["DYN_KVBM_CPU_CACHE_GB"] = "20"
        env["DYN_KVBM_DISK_CACHE_GB"] = "60"
        env["DYN_KVBM_LEADER_WORKER_INIT_TIMEOUT_SECS"] = "1200"

        # TODO: Have the managed process take a command name explicitly to distinguish
        #       between processes started with the same command.
        log_dir = f"{request.node.name}_{worker_id}"

        # Clean up any existing log directory from previous runs
        try:
            shutil.rmtree(log_dir)
            logger.info(f"Cleaned up existing log directory: {log_dir}")
        except FileNotFoundError:
            # Directory doesn't exist, which is fine
            pass

        super().__init__(
            command=command,
            env=env,
            health_check_urls=[
                (f"http://localhost:{FRONTEND_PORT}/v1/models", check_models_api),
                ("http://localhost:9345/health", self.is_ready),
            ],
            timeout=300,
            display_output=True,
            terminate_all_matching_process_names=False,
            log_dir=log_dir,
        )

    def get_pid(self) -> int | None:
        """Get the PID of the worker process"""
        return self.proc.pid if hasattr(self, "proc") and self.proc else None

    def is_ready(self, response) -> bool:
        """Check the health of the worker process"""
        try:
            data = response.json()
            if data.get("status") == "ready":
                logger.info(
                    f"{self.__class__.__name__} {{ name: {self.worker_id} }} status is ready"
                )
                return True
            logger.warning(
                f"{self.__class__.__name__} {{ name: {self.worker_id} }} status is not ready: {data.get('status')}"
            )
        except ValueError:
            logger.warning(
                f"{self.__class__.__name__} {{ name: {self.worker_id} }} health response is not valid JSON"
            )
        return False


def send_completion_request(
    prompt: str, max_tokens: int, timeout: int = 120
) -> requests.Response:
    """Send a completion request to the frontend"""
    payload = {
        "model": SERVED_MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "max_tokens": max_tokens,
    }

    headers = {"Content-Type": "application/json"}

    logger.info(
        f"Sending completion request with prompt: '{prompt[:50]}...' and max_tokens: {max_tokens}"
    )

    try:
        response = requests.post(
            "http://localhost:8000/v1/completions",
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        return response
    except requests.exceptions.Timeout:
        logger.error(f"Request timed out after {timeout} seconds")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed with error: {e}")
        raise


# Test markers to align with repository conventions
# Todo: enable the rest when kvbm is built in the ci


@pytest.mark.kvbm
@pytest.mark.trtllm
@pytest.mark.e2e
@pytest.mark.nightly
@pytest.mark.slow
@pytest.mark.gpu_1
@pytest.mark.skipif(not HAS_TRTLLM, reason="requires tensorrt_llm")
def test_kvbm_without_cuda_graph_enabled(request, runtime_services):
    """
    End-to-end test for TRTLLM worker with cuda_graph_config not defined and
    KVBM enabled.

    This test verifies a TRTLLM worker is able to serve requests when
    cuda graphs are not enabled in pytorch. KVBM should be able to offload
    blocks regardless.
    """

    logger.info("Starting frontend...")
    with DynamoFrontendProcess(request):
        logger.info("Frontend started.")

        engine_config_without_cuda_graph_and_kvbm = (
            "tests/kvbm_integration/engine_config_without_cuda_graph_and_kvbm.yaml"
        )
        logger.info("Starting worker...")
        with DynamoWorkerProcess(
            request, "decode", engine_config_without_cuda_graph_and_kvbm
        ) as worker:
            logger.info(f"Worker PID: {worker.get_pid()}")

            response = send_completion_request(PROMPT, 100, timeout=10)
            assert (
                response.ok
            ), f"Expected successful status, got {response.status_code}"
            logger.info(f"Completion request succeeded: {response.status_code}")


@pytest.mark.kvbm
@pytest.mark.trtllm
@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.nightly
@pytest.mark.gpu_1
@pytest.mark.skipif(not HAS_TRTLLM, reason="requires tensorrt_llm")
def test_kvbm_with_cuda_graph_enabled(request, runtime_services):
    """
    End-to-end test for TRTLLM worker with cuda_graph_config defined and
    KVBM enabled.

    This test verifies a TRTLLM worker is able to serve requests when
    cuda graphs are enabled in pytorch. KVBM should be able to offload
    blocks regardless.
    """

    logger.info("Starting frontend...")
    with DynamoFrontendProcess(request):
        logger.info("Frontend started.")

        engine_config_with_cuda_graph_and_kvbm = (
            "tests/kvbm_integration/engine_config_with_cuda_graph_and_kvbm.yaml"
        )
        logger.info("Starting worker...")
        with DynamoWorkerProcess(
            request, "decode", engine_config_with_cuda_graph_and_kvbm
        ) as worker:
            logger.info(f"Worker PID: {worker.get_pid()}")

            response = send_completion_request(PROMPT, 100, timeout=10)
            assert (
                response.ok
            ), f"Expected successful status, got {response.status_code}"
            logger.info(f"Completion request succeeded: {response.status_code}")
