# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ownership-based GMS V1 worker for vLLM's normal model loader.

Select explicitly with::

    --worker-cls gpu_memory_service.v1.integrations.vllm.worker.GMSV1Worker
"""

from __future__ import annotations

from contextlib import AbstractContextManager

from vllm.v1.worker.gpu_worker import Worker

from .backend import BACKEND_NAME


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
        if tag == "weights":
            backend = self._get_sleep_mode_backend()
            return backend.capture_weights(self.model_runner.get_model)
        return super()._maybe_get_memory_pool_context(tag)
