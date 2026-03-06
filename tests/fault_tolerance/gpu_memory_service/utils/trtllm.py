# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT-LLM-specific utilities for GPU Memory Service tests."""

import logging
import os
import shutil

import requests

from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME
from tests.utils.managed_process import ManagedProcess
from tests.utils.payloads import check_health_generate, check_models_api

logger = logging.getLogger(__name__)

TRTLLM_GMS_MODEL_NAME = os.environ.get(
    "TRTLLM_GMS_MODEL_NAME", FAULT_TOLERANCE_MODEL_NAME
)
TRTLLM_GMS_READ_ONLY_CONFIG = '{"gms_read_only": true}'
TRTLLM_GMS_FREE_GPU_MEMORY_FRACTION = os.environ.get(
    "TRTLLM_GMS_FREE_GPU_MEMORY_FRACTION", "0.9"
)
TRTLLM_GMS_MAX_SEQ_LEN = os.environ.get("TRTLLM_GMS_MAX_SEQ_LEN", "256")
TRTLLM_GMS_MAX_NUM_TOKENS = os.environ.get("TRTLLM_GMS_MAX_NUM_TOKENS", "256")
TRTLLM_GMS_OVERRIDE_ENGINE_ARGS = os.environ.get(
    "TRTLLM_GMS_OVERRIDE_ENGINE_ARGS",
    '{"kv_cache_config":{"max_tokens":4096}}',
)


def _build_trtllm_env(system_port: int) -> dict[str, str]:
    env = {**os.environ}
    env["DYN_LOG"] = "debug"
    env["DYN_SYSTEM_PORT"] = str(system_port)
    if "CUDA_VISIBLE_DEVICES" not in env:
        env["CUDA_VISIBLE_DEVICES"] = "0"
    env["TLLM_WORKER_USE_SINGLE_PROCESS"] = "1"
    env["MPI4PY_MPIABI"] = "openmpi"
    env["OMPI_MCA_coll_ucc_enable"] = "0"

    venv_path = env.get("VIRTUAL_ENV")
    if venv_path:
        venv_lib = os.path.join(venv_path, "lib")
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            f"{venv_lib}:{existing}" if existing else venv_lib
        )

    return env


class TRTLLMWithGMSProcess(ManagedProcess):
    """TensorRT-LLM engine with GPU Memory Service integration."""

    def __init__(
        self,
        request,
        engine_id: str,
        system_port: int,
        frontend_port: int,
        model_loader_extra_config: str | None = None,
    ):
        self.engine_id = engine_id
        self.system_port = system_port

        log_dir = f"{request.node.name}_{engine_id}"
        shutil.rmtree(log_dir, ignore_errors=True)
        command = [
            "python3",
            "-m",
            "dynamo.trtllm",
            "--model",
            TRTLLM_GMS_MODEL_NAME,
            "--gpus-per-node",
            "1",
            "--load-format",
            "gms",
            "--enable-sleep",
            "--free-gpu-memory-fraction",
            TRTLLM_GMS_FREE_GPU_MEMORY_FRACTION,
            "--max-seq-len",
            TRTLLM_GMS_MAX_SEQ_LEN,
            "--max-num-tokens",
            TRTLLM_GMS_MAX_NUM_TOKENS,
            "--override-engine-args",
            TRTLLM_GMS_OVERRIDE_ENGINE_ARGS,
        ]
        if model_loader_extra_config is not None:
            command.extend(
                [
                    "--model-loader-extra-config",
                    model_loader_extra_config,
                ]
            )

        super().__init__(
            command=command,
            env=_build_trtllm_env(system_port),
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
        """Put the engine to sleep, offloading GPU memory."""
        response = requests.post(
            f"http://localhost:{self.system_port}/engine/release_memory_occupation",
            json={},
            timeout=30,
        )
        response.raise_for_status()
        logger.info("%s release_memory_occupation: %s", self.engine_id, response.json())
        return response.json()

    def wake(self) -> dict:
        """Wake the engine, restoring GPU memory mappings."""
        response = requests.post(
            f"http://localhost:{self.system_port}/engine/resume_memory_occupation",
            json={},
            timeout=30,
        )
        response.raise_for_status()
        logger.info("%s resume_memory_occupation: %s", self.engine_id, response.json())
        return response.json()
