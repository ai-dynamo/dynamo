# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ownership-based GMS V1 worker for vLLM's normal model loader.

Select explicitly with::

    --worker-cls gpu_memory_service.v1.integrations.vllm.worker.GMSV1Worker
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from time import monotonic

from gpu_memory_service.v1.integrations.vllm import checkpoint_timing
from vllm.logger import init_logger
from gpu_memory_service.v1.integrations.vllm.backend import BACKEND_NAME
from vllm.v1.worker.gpu_worker import Worker

# Installed at import so every worker process is wrapped before vLLM dispatches
# the first collective_rpc. Inert unless DYN_GMS_CHECKPOINT_TIMING is set.
checkpoint_timing.install()

# See checkpoint_timing.py: this must be a "vllm.*" logger or nothing is emitted.
logger = init_logger("vllm.gpu_memory_service.v1.timing")


class GMSV1Worker(Worker):
    """Route vLLM allocator scopes to the selected GMS V1 backend."""

    def checkpoint_restore(self) -> None:
        """Time the worker-side restore, which is otherwise silent.

        GMS logs its own wake, so the wake half of promotion is visible; this
        half is not, and under load it is the larger of the two.
        """
        t0 = monotonic()
        try:
            super().checkpoint_restore()
        finally:
            logger.info(
                "[ckpt-timing] worker.checkpoint_restore rank=%d elapsed=%.3fs",
                self.rank if hasattr(self, "rank") else -1,
                monotonic() - t0,
            )

    def checkpoint_prepare(self) -> None:
        """Counterpart to the above; never measured, and it runs at capture."""
        t0 = monotonic()
        try:
            super().checkpoint_prepare()
        finally:
            logger.info(
                "[ckpt-timing] worker.checkpoint_prepare rank=%d elapsed=%.3fs",
                self.rank if hasattr(self, "rank") else -1,
                monotonic() - t0,
            )

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
