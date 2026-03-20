# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT-LLM harness for GPU Memory Service integration tests."""

import logging
import os
import shutil

import requests

from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME
from tests.utils.managed_process import ManagedProcess
from tests.utils.payloads import check_health_generate, check_models_api

from .runtime import DYNAMO_BIN

logger = logging.getLogger(__name__)

# Override via environment variables for CI or custom setups.
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


def _build_env(system_port: int) -> dict[str, str]:
    env = {**os.environ}
    env["DYN_LOG"] = "debug"
    env["DYN_SYSTEM_PORT"] = str(system_port)
    env["PATH"] = f"{DYNAMO_BIN}:{env.get('PATH', '')}"
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    # Required for single-process TRT-LLM workers
    env["TLLM_WORKER_USE_SINGLE_PROCESS"] = "1"
    env["MPI4PY_MPIABI"] = "openmpi"
    env["OMPI_MCA_coll_ucc_enable"] = "0"
    # Ensure the venv libs are on LD_LIBRARY_PATH so TRT-LLM can find them
    venv = env.get("VIRTUAL_ENV")
    if venv:
        venv_lib = os.path.join(venv, "lib")
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{venv_lib}:{existing}" if existing else venv_lib
    env.pop("HF_HUB_OFFLINE", None)
    return env


class TRTLLMWithGMSProcess(ManagedProcess):
    """TensorRT-LLM engine with GMS weights + sleep/wake enabled."""

    def __init__(
        self,
        request,
        engine_id: str,
        system_port: int,
        frontend_port: int,
        *,
        model_loader_extra_config: str | None = None,
    ):
        self.engine_id = engine_id
        self.system_port = system_port

        log_dir = f"{request.node.name}_{engine_id}"
        shutil.rmtree(log_dir, ignore_errors=True)

        command = [
            "python",
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
            command.extend(["--model-loader-extra-config", model_loader_extra_config])

        super().__init__(
            command=command,
            env=_build_env(system_port),
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
            display_name=engine_id,
        )

    def _is_ready(self, response) -> bool:
        try:
            return response.json().get("status") == "ready"
        except ValueError:
            return False

    def sleep(self) -> dict:
        """Call /engine/release_memory_occupation to free GPU memory."""
        r = requests.post(
            f"http://localhost:{self.system_port}/engine/release_memory_occupation",
            json={},
            timeout=30,
        )
        r.raise_for_status()
        result = r.json()
        logger.info("%s sleep: %s", self.engine_id, result)
        return result

    def wake(self, timeout: int = 180) -> dict:
        """Call /engine/resume_memory_occupation to restore GPU memory."""
        r = requests.post(
            f"http://localhost:{self.system_port}/engine/resume_memory_occupation",
            json={},
            timeout=timeout,
        )
        r.raise_for_status()
        result = r.json()
        logger.info("%s wake: %s", self.engine_id, result)
        return result
