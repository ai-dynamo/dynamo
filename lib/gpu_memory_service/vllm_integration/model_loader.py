# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM model loader for GPU Memory Service integration.

Provides a model loader that loads weights via GMS for cross-process sharing.
The loader uses RW_OR_RO mode: first process loads from disk (RW), subsequent
processes import from GMS metadata (RO).
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from gpu_memory_service import get_or_create_gms_client_memory_manager
from gpu_memory_service.client.torch.module import (
    materialize_module_from_gms,
    register_module_tensors,
)
from gpu_memory_service.common.types import GrantedLockType, RequestedLockType
from gpu_memory_service.common.utils import get_socket_path

if TYPE_CHECKING:
    from gpu_memory_service.client.memory_manager import GMSClientMemoryManager

logger = logging.getLogger(__name__)

# Track imported weights for memory accounting
_last_imported_weights_bytes: int = 0


def get_imported_weights_bytes() -> int:
    """Return bytes of weights imported in the last load_model call."""
    return _last_imported_weights_bytes


def register_gms_loader(load_format: str = "gms") -> None:
    """Register the GMS model loader with vLLM's loader registry."""
    from vllm.model_executor.model_loader import register_model_loader
    from vllm.model_executor.model_loader.base_loader import BaseModelLoader
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    @register_model_loader(load_format)
    class GMSModelLoader(BaseModelLoader):
        """vLLM model loader that loads weights via GPU Memory Service."""

        def __init__(self, load_config):
            super().__init__(load_config)
            self.default_loader = DefaultModelLoader(
                replace(load_config, load_format="auto")
            )

        def download_model(self, model_config) -> None:
            self.default_loader.download_model(model_config)

        def load_weights(self, model: torch.nn.Module, model_config) -> None:
            self.default_loader.load_weights(model, model_config)

        def load_model(self, vllm_config, model_config) -> torch.nn.Module:
            device = torch.cuda.current_device()
            gms_client, pool = get_or_create_gms_client_memory_manager(
                get_socket_path(device),
                device,
                mode=RequestedLockType.RW_OR_RO,
                tag="weights",
            )

            if gms_client.mode == GrantedLockType.RO:
                return _load_read_mode(gms_client, vllm_config, model_config, device)
            else:
                return _load_write_mode(
                    gms_client,
                    pool,
                    vllm_config,
                    model_config,
                    self.default_loader,
                    torch.device("cuda", device),
                )


# =============================================================================
# Helper functions
# =============================================================================


def _load_read_mode(
    gms_client: "GMSClientMemoryManager",
    vllm_config,
    model_config,
    device_index: int,
) -> torch.nn.Module:
    """Load model by importing weights from GMS (RO mode)."""
    global _last_imported_weights_bytes

    try:
        model = _create_meta_model(vllm_config, model_config)
        materialize_module_from_gms(gms_client, model, device_index=device_index)

        _last_imported_weights_bytes = gms_client.total_bytes
        logger.info(
            "[GMS] Read mode: imported %.2f GiB",
            _last_imported_weights_bytes / (1 << 30),
        )
        return model.eval()
    except Exception:
        gms_client.close()
        raise


def _load_write_mode(
    gms_client: "GMSClientMemoryManager",
    pool,
    vllm_config,
    model_config,
    default_loader,
    target_device: torch.device,
) -> torch.nn.Module:
    """Load model from disk and publish weights to GMS (RW mode).

    Initializes model using GMS memory pool, loads weights from disk,
    registers tensors with GMS, and commits for cross-process sharing.
    """
    global _last_imported_weights_bytes

    from torch.cuda.memory import use_mem_pool
    from vllm.model_executor.model_loader.utils import (
        initialize_model,
        process_weights_after_loading,
    )
    from vllm.utils.torch_utils import set_default_torch_dtype

    gms_client.clear_all()

    # Allocate model tensors using GMS memory pool
    with set_default_torch_dtype(model_config.dtype):
        with use_mem_pool(pool, device=target_device):
            with target_device:
                model = initialize_model(
                    vllm_config=vllm_config, model_config=model_config
                )

            default_loader.load_weights(model, model_config)
            process_weights_after_loading(model, model_config, target_device)
            torch.cuda.empty_cache()

    # Update GMS metadata store with model tensors
    register_module_tensors(gms_client, model)
    _last_imported_weights_bytes = gms_client.total_bytes

    # Ensure all writes to GPU memory are finished before we unmap
    torch.cuda.synchronize()

    if not gms_client.commit():
        raise RuntimeError("Allocation Server commit failed")

    gms_client.switch_to_read()

    logger.info(
        "[GMS] Write mode: published %.2f GiB (%d mappings)",
        _last_imported_weights_bytes / (1 << 30),
        len(gms_client._mappings),
    )
    return model.eval()


def _create_meta_model(vllm_config, model_config) -> torch.nn.Module:
    """Create model on meta device for RO mode materialization."""
    from vllm.model_executor.model_loader.utils import (
        initialize_model,
        process_weights_after_loading,
    )
    from vllm.utils.torch_utils import set_default_torch_dtype

    meta_device = torch.device("meta")

    # Enable meta tensor workaround for torch.nonzero() etc.
    try:
        import torch.fx.experimental._config as fx_config

        fx_config.meta_nonzero_assume_all_nonzero = True
    except (ImportError, AttributeError):
        pass

    with set_default_torch_dtype(model_config.dtype):
        with meta_device:
            model = initialize_model(vllm_config=vllm_config, model_config=model_config)

    try:
        process_weights_after_loading(model, model_config, meta_device)
    except Exception as e:
        logger.debug("[GMS] Post-processing on meta tensors: %s", e)

    return model
