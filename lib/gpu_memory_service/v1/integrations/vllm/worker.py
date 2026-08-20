# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ownership-based GMS V1 worker for vLLM's normal model loader.

Dynamo selects this worker when ``DYN_GMS_USE_V1=true``.
"""

from __future__ import annotations

from contextlib import AbstractContextManager

from gpu_memory_service.v1.integrations.vllm.backend import BACKEND_NAME
from vllm.v1.worker.gpu_worker import Worker


class GMSV1Worker(Worker):
    """Route vLLM allocator scopes to the selected GMS V1 backend."""

    def init_device(self) -> None:
        model_config = self.vllm_config.model_config
        if not model_config.enable_sleep_mode:
            raise RuntimeError("GMS V1 requires vLLM sleep mode")
        model_config.sleep_mode_backend = BACKEND_NAME

        super().init_device()
        self._get_sleep_mode_backend()

    def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager[None]:
        backend = self._get_sleep_mode_backend()
        if tag == "weights":
            return backend.capture_weights(self.model_runner.get_model)
        if tag == "kv_cache":
            return backend.capture_kv_cache()
        return super()._maybe_get_memory_pool_context(tag)
