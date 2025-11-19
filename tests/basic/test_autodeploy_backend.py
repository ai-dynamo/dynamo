# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration test for the autodeploy backend in TRTLLM."""

import logging
import os
import pathlib
import shutil

import pytest
import requests

from tests.utils.engine_process import FRONTEND_PORT
from tests.utils.managed_process import DynamoFrontendProcess, ManagedProcess
from tests.utils.payloads import check_models_api

logger = logging.getLogger(__name__)

# Just need a model to show the config works rather than any stress of the system.
MODEL_PATH = "Qwen/Qwen3-0.6B"
SERVED_MODEL_NAME = MODEL_PATH

PROMPT = "Takes skill to be real"


# TODO: turn into a fixture that _many_ tests can benefit from.
class DynamoWorkerProcess(ManagedProcess):
    """Process manager for Dynamo worker with TRTLLM backend."""

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
            timeout=360,
            display_output=True,
            terminate_existing=False,
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

    def __enter__(self):
        """Start the process and perform warmup request to trigger compilation.

        Without a build cache, the autodeploy LLM engine will have to run some compilation before
        being able to actually execute requests. We add a warmup stage here so that we can have
        tighter timeouts on the requests sent during the actual tests.
        """
        result = super().__enter__()

        logger.info(
            f"Sending warmup request to {self.worker_id} to trigger compilation..."
        )
        try:
            warmup_response = send_completion_request(
                prompt=PROMPT,
                max_tokens=1,
                timeout=300,
            )
            if warmup_response.ok:
                logger.info(
                    f"Warmup request completed successfully for {self.worker_id}"
                )
            else:
                raise RuntimeError(
                    f"Warmup request returned status {warmup_response.status_code} for {self.worker_id}"
                )
        except Exception as e:
            logger.error(f"Warmup request failed for {self.worker_id}: {e}")
            raise

        return result


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


@pytest.mark.trtllm_marker
@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.gpu_1
def test_smoke(request, runtime_services):
    """End-to-end test for TRTLLM worker with autodeploy backend in its most basic form."""

    logger.info("Starting frontend...")
    with DynamoFrontendProcess(request):
        logger.info("Frontend started.")

        engine_config_path = str(
            pathlib.Path(__file__).parent / "autodeploy_engine_config.yaml"
        )
        logger.info("Starting worker...")
        with DynamoWorkerProcess(request, "decode", engine_config_path) as worker:
            logger.info(f"Worker PID: {worker.get_pid()}")

            response = send_completion_request(
                prompt=PROMPT, max_tokens=100, timeout=20
            )
            assert (
                response.ok
            ), f"Expected successful status, got {response.status_code}"
            logger.info(f"Completion request succeeded: {response.status_code}")
