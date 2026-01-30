# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM-specific utilities for GPU Memory Service tests."""

import logging
import os
import shutil

import requests

from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME
from tests.utils.managed_process import ManagedProcess
from tests.utils.payloads import check_health_generate, check_models_api

logger = logging.getLogger(__name__)


class VLLMWithGMSProcess(ManagedProcess):
    """vLLM engine with GPU Memory Service integration."""

    def __init__(
        self,
        request,
        engine_id: str,
        system_port: int,
        kv_event_port: int,
        nixl_port: int,
        frontend_port: int,
    ):
        self.engine_id = engine_id
        self.system_port = system_port

        log_dir = f"{request.node.name}_{engine_id}"
        shutil.rmtree(log_dir, ignore_errors=True)

        super().__init__(
            command=[
                "python3",
                "-m",
                "dynamo.vllm",
                "--model",
                FAULT_TOLERANCE_MODEL_NAME,
                "--load-format",
                "gms",
                "--enable-sleep-mode",
                "--gpu-memory-utilization",
                "0.8",
            ],
            env={
                **os.environ,
                "DYN_LOG": "debug",
                "DYN_SYSTEM_PORT": str(system_port),
                "DYN_VLLM_KV_EVENT_PORT": str(kv_event_port),
                "VLLM_NIXL_SIDE_CHANNEL_PORT": str(nixl_port),
            },
            health_check_urls=[
                (f"http://localhost:{system_port}/health", self._is_ready),
                (f"http://localhost:{frontend_port}/v1/models", check_models_api),
                (f"http://localhost:{frontend_port}/health", check_health_generate),
            ],
            timeout=300,
            display_output=True,
            terminate_existing=False,
            stragglers=[],
            log_dir=log_dir,
        )

    def _is_ready(self, response) -> bool:
        try:
            return response.json().get("status") == "ready"
        except ValueError:
            return False

    def sleep(self) -> dict:
        """Put the engine to sleep, offloading weights from GPU memory."""
        r = requests.post(
            f"http://localhost:{self.system_port}/engine/sleep",
            json={"level": 1},
            timeout=30,
        )
        r.raise_for_status()
        logger.info(f"{self.engine_id} sleep: {r.json()}")
        return r.json()

    def wake(self) -> dict:
        """Wake the engine, reloading weights to GPU memory."""
        r = requests.post(
            f"http://localhost:{self.system_port}/engine/wake_up", json={}, timeout=30
        )
        r.raise_for_status()
        logger.info(f"{self.engine_id} wake: {r.json()}")
        return r.json()
