# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Confine GMS V1 to vLLM's normal model loader."""

from __future__ import annotations

from types import MethodType
from typing import Any

from ...client.torch import SnapshotTorchPool


def install_vllm_integration(
    workspace_manager: Any,
    pool: SnapshotTorchPool,
) -> None:
    """Wrap the current BaseModelLoader and native workspace growth paths."""
    from vllm.model_executor.model_loader.base_loader import BaseModelLoader
    from vllm.v1.worker.workspace import dbo_current_ubatch_id

    original_load_model = BaseModelLoader.load_model

    def load_model(loader: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            with pool.model_load_pool():
                model = original_load_model(loader, *args, **kwargs)
        except Exception as cause:
            pool.abort_model_load(cause)
        pool.finalize_model_load(model)
        return model

    BaseModelLoader.load_model = load_model

    original_workspace_growth = workspace_manager._ensure_workspace_size

    def ensure_workspace_size(self: Any, required_bytes: int) -> Any:
        current = self._current_workspaces[dbo_current_ubatch_id()]
        if self._workspace_size_bytes(current) >= required_bytes:
            return original_workspace_growth(required_bytes)
        with pool.native_workspace_pool():
            return original_workspace_growth(required_bytes)

    workspace_manager._ensure_workspace_size = MethodType(
        ensure_workspace_size, workspace_manager
    )
