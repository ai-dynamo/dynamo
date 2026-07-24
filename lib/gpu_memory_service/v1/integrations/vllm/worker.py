# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ownership-based GMS V1 worker for vLLM's normal model loader.

Select explicitly with::

    --worker-cls gpu_memory_service.v1.integrations.vllm.worker.GMSV1Worker
"""

from __future__ import annotations

import logging
from contextlib import AbstractContextManager, nullcontext

from gpu_memory_service.common.utils import get_socket_path
from gpu_memory_service.common.vmm import get_vmm
from vllm.v1.worker.gpu_worker import Worker
from vllm.v1.worker.workspace import current_workspace_manager

from ...client.memory_manager import SnapshotMemoryManager
from ...client.rpc import AllocationClient
from ...client.torch import SnapshotTorchPool
from .backend import BACKEND_NAME
from .patches import install_vllm_integration
from .runtime import VllmSnapshotRuntime, install_runtime

logger = logging.getLogger(__name__)


class GMSV1Worker(Worker):
    """Use GMS V1 only for vLLM's normal BaseModelLoader execution."""

    def init_device(self) -> None:
        model_config = self.vllm_config.model_config
        if not model_config.enable_sleep_mode:
            raise RuntimeError("GMS V1 requires vLLM sleep mode")
        model_config.sleep_mode_backend = BACKEND_NAME

        super().init_device()

        device = self.device.index
        if device is None:
            raise RuntimeError("GMS V1 requires an indexed CUDA device")
        client = AllocationClient(get_socket_path(device, "snapshot-v1"))
        try:
            manager = SnapshotMemoryManager(client, get_vmm(), device)
            pool = SnapshotTorchPool(manager)
            install_vllm_integration(current_workspace_manager(), pool)
            runtime = install_runtime(manager, pool)
        except BaseException:
            client.close()
            raise
        self._gms_v1_runtime: VllmSnapshotRuntime = runtime

    def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager[None]:
        return nullcontext()

    def sleep(self, level: int = 1) -> None:
        if level != 1:
            raise ValueError("GMS V1 supports only whole-engine level 1 suspend")
        try:
            super().sleep(level)
        except Exception as cause:
            logger.exception("GMS V1 suspend failed; terminating the worker process")
            raise SystemExit(1) from cause

    def wake_up(self, tags: list[str] | None = None) -> None:
        if tags is not None:
            raise ValueError("GMS V1 does not support partial-tag resume")
        try:
            super().wake_up(tags)
        except Exception as cause:
            logger.exception("GMS V1 resume failed; terminating the worker process")
            raise SystemExit(1) from cause
