# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang-specific utilities for GPU Memory Service tests."""

import logging
import os
import shutil

import requests

from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME
from tests.utils.managed_process import ManagedProcess
from tests.utils.payloads import check_health_generate, check_models_api

logger = logging.getLogger(__name__)


class SGLangWithGMSProcess(ManagedProcess):
    """SGLang engine with GPU Memory Service integration."""

    def __init__(
        self,
        request,
        engine_id: str,
        system_port: int,
        sglang_port: int,
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
                "dynamo.sglang",
                "--model-path",
                FAULT_TOLERANCE_MODEL_NAME,
                "--load-format",
                "gms",
                "--enable-memory-saver",
                "--mem-fraction-static",
                "0.8",
                "--port",
                str(sglang_port),
            ],
            env={
                **os.environ,
                "DYN_LOG": "debug",
                "DYN_SYSTEM_PORT": str(system_port),
            },
            health_check_urls=[
                (f"http://localhost:{system_port}/health", self._is_ready),
                (f"http://localhost:{frontend_port}/v1/models", check_models_api),
                (f"http://localhost:{frontend_port}/health", check_health_generate),
            ],
            timeout=300,
            display_output=True,
            terminate_all_matching_process_names=False,
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
            f"http://localhost:{self.system_port}/engine/release_memory_occupation",
            json={},
            timeout=30,
        )
        r.raise_for_status()
        logger.info(f"{self.engine_id} release_memory_occupation: {r.json()}")
        return r.json()

    def wake(self) -> dict:
        """Wake the engine, reloading weights to GPU memory."""
        r = requests.post(
            f"http://localhost:{self.system_port}/engine/resume_memory_occupation",
            json={},
            timeout=30,
        )
        r.raise_for_status()
        logger.info(f"{self.engine_id} resume_memory_occupation: {r.json()}")
        return r.json()
