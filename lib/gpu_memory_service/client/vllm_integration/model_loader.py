# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM model loader for GPU Memory Service integration.

This module registers a vLLM `load_format` that loads model weights into
GPU Memory Service allocations (RW session), publishes via Commit(), and then
holds an RO lock for inference lifetime.

The model loader uses RW_OR_RO mode to connect to the GPU Memory Service:
- First process to connect gets RW lock and loads weights from disk
- Subsequent processes get RO lock and import weights from metadata store
This enables weight sharing across processes without explicit configuration.

Configuration via model_loader_extra_config:
- gpu_memory_service_socket_path: Unix socket path for the Allocation Server (per GPU).
  You may include `{device}` which will be formatted with the GPU device index.
  Default: /tmp/gpu_memory_service_{device}.sock

IMPORTANT: Sleep/Wake Memory Behavior
-------------------------------------
When using GPU Memory Service with vLLM's sleep/wake functionality, the sleep/wake does NOT
actually free GPU memory. The physical memory for model weights remains allocated
by the Allocation Server. This is by design for weight sharing:

- The Allocation Server owns the physical memory for weights
- On sleep, the client unmaps its local VA mappings but the server keeps the memory
- On wake, the client remaps the same weights without reloading from disk

This enables fast context switching between inference instances. If you need to
actually free GPU memory during sleep, use native vLLM sleep/wake (without GPU Memory Service).
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Optional

import torch

from gpu_memory_service import get_or_create_gms_client_memory_manager
from gpu_memory_service.client.vllm_integration.config import (
    get_local_rank,
    resolve_socket_path,
    strip_gms_extra_config,
)
from gpu_memory_service.common.types import GrantedLockType, RequestedLockType

logger = logging.getLogger(__name__)


# Module-level storage for imported weights bytes (for memory accounting)
_gpu_memory_service_imported_weights_bytes: int = 0


def get_imported_weights_bytes() -> int:
    """Return last recorded weights bytes for vLLM memory accounting."""
    global _gpu_memory_service_imported_weights_bytes
    return _gpu_memory_service_imported_weights_bytes


def _create_meta_model(vllm_config, model_config) -> torch.nn.Module:
    """Create model structure on meta device with post-processed parameters.

    Used for RO mode where weights are materialized from GMS metadata.
    This creates the model skeleton without loading weights from disk,
    then runs quantization post-processing to ensure the parameter structure
    matches what was registered during write mode.
    """
    from vllm.model_executor.model_loader.utils import (
        initialize_model,
        process_weights_after_loading,
    )
    from vllm.utils.torch_utils import set_default_torch_dtype

    meta_device = torch.device("meta")

    # Enable meta tensor workaround for operations like torch.nonzero()
    try:
        import torch.fx.experimental._config as fx_config
        fx_config.meta_nonzero_assume_all_nonzero = True
    except (ImportError, AttributeError):
        pass

    # Create model on meta device - no GPU memory allocated
    with set_default_torch_dtype(model_config.dtype):
        with meta_device:
            model = initialize_model(vllm_config=vllm_config, model_config=model_config)

    # Run quantization post-processing to get the FINAL parameter structure
    try:
        process_weights_after_loading(model, model_config, meta_device)
    except Exception as e:
        logger.debug("[GMS] Post-processing on meta tensors: %s", e)

    return model


def _load_model_read_mode(
    gms_client_memory_manager,
    vllm_config,
    model_config,
    device_index: int,
) -> torch.nn.Module:
    """Load model in read mode (import weights from GMS metadata).

    Args:
        gms_client_memory_manager: GMS client with RO lock
        vllm_config: vLLM configuration
        model_config: Model configuration
        device_index: CUDA device index

    Returns:
        Loaded model in eval mode
    """
    global _gpu_memory_service_imported_weights_bytes

    from gpu_memory_service.client.torch.module import materialize_module_from_gms
    from gpu_memory_service.client.vllm_integration.patches import patch_sleep_wake

    try:
        model = _create_meta_model(vllm_config, model_config)

        materialize_module_from_gms(
            gms_client_memory_manager,
            model,
            device_index=device_index,
        )

        imported_bytes = gms_client_memory_manager.total_bytes
        _gpu_memory_service_imported_weights_bytes = imported_bytes

        # Apply sleep/wake patches
        patch_sleep_wake()

        logger.info(
            "[GMS] Read mode: imported %.2f GiB from GPU memory service",
            imported_bytes / (1 << 30),
        )

        return model.eval()
    except Exception:
        gms_client_memory_manager.close()
        raise


def _load_model_write_mode(
    gms_client_memory_manager,
    pool,
    vllm_config,
    model_config,
    load_config,
    target_device: torch.device,
) -> torch.nn.Module:
    """Load model in write mode (load from disk, publish to GMS).

    Args:
        gms_client_memory_manager: GMS client with RW lock
        pool: Memory pool for allocations
        vllm_config: vLLM configuration
        model_config: Model configuration
        load_config: Load configuration
        target_device: Target CUDA device

    Returns:
        Loaded model in eval mode
    """
    global _gpu_memory_service_imported_weights_bytes

    from torch.cuda.memory import use_mem_pool

    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
    from vllm.model_executor.model_loader.utils import (
        initialize_model,
        process_weights_after_loading,
    )
    from vllm.utils.torch_utils import set_default_torch_dtype

    from gpu_memory_service.client.torch.module import register_module_tensors
    from gpu_memory_service.client.vllm_integration.patches import patch_sleep_wake

    # Start fresh (weights model load is authoritative)
    gms_client_memory_manager.clear_all()

    # Prepare disk loader config
    disk_load_config = strip_gms_extra_config(load_config)
    disk_load_config = replace(disk_load_config, load_format="auto")

    # Load model with allocations routed to GMS pool
    with set_default_torch_dtype(model_config.dtype):
        with use_mem_pool(pool, device=target_device):
            with target_device:
                model = initialize_model(vllm_config=vllm_config, model_config=model_config)

            DefaultModelLoader(disk_load_config).load_weights(model, model_config)
            process_weights_after_loading(model, model_config, target_device)

            # Release cached blocks while still in use_mem_pool context
            torch.cuda.empty_cache()

    # Register tensors and commit
    register_module_tensors(gms_client_memory_manager, model)
    total_bytes = gms_client_memory_manager.total_bytes
    _gpu_memory_service_imported_weights_bytes = total_bytes

    torch.cuda.synchronize()

    ok = gms_client_memory_manager.commit()
    if not ok:
        raise RuntimeError("Allocation Server commit failed")

    gms_client_memory_manager.switch_to_read()

    # Apply sleep/wake patches
    patch_sleep_wake()

    logger.info(
        "[GMS] Write mode: published %.2f GiB, switched to read mode (%d mappings)",
        total_bytes / (1 << 30),
        len(gms_client_memory_manager._mappings),
    )

    return model.eval()


def register_gpu_memory_service_loader(load_format: str = "gpu_memory_service") -> None:
    """Register vLLM loader that allocates via GPU Memory Service."""
    try:
        from vllm.model_executor.model_loader import register_model_loader
        from vllm.model_executor.model_loader.base_loader import BaseModelLoader
        from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
    except ImportError as e:
        raise RuntimeError(f"vLLM not installed or incompatible: {e}")

    @register_model_loader(load_format)
    class GPUServiceModelLoader(BaseModelLoader):
        """vLLM model loader that publishes weights to GPU Memory Service."""

        _imported_weights_bytes: int = 0

        def __init__(self, load_config):
            super().__init__(load_config)
            self._socket_path: Optional[str] = None

        def download_model(self, model_config) -> None:
            disk_load_config = strip_gms_extra_config(self.load_config)
            disk_load_config = replace(disk_load_config, load_format="auto")
            DefaultModelLoader(disk_load_config).download_model(model_config)

        def load_weights(self, model: torch.nn.Module, model_config) -> None:
            disk_load_config = strip_gms_extra_config(self.load_config)
            disk_load_config = replace(disk_load_config, load_format="auto")
            DefaultModelLoader(disk_load_config).load_weights(model, model_config)

        def load_model(self, vllm_config, model_config) -> torch.nn.Module:
            logger.info("[GMS] load_model() called")

            device_config = vllm_config.device_config
            load_config = vllm_config.load_config

            # Determine target device FIRST (before resolving socket path)
            load_device = (
                device_config.device
                if load_config.device is None
                else load_config.device
            )
            target_device = torch.device(load_device)

            if target_device.type == "cuda" and target_device.index is None:
                device_index = get_local_rank()
                target_device = torch.device("cuda", device_index)
            else:
                device_index = target_device.index if target_device.index is not None else 0

            # Resolve socket path using the determined device_index
            socket_path = resolve_socket_path(load_config, device_index=device_index)
            self._socket_path = socket_path
            logger.debug("[GMS] Socket path: %s (device_index=%d)", socket_path, device_index)

            # Connect to GMS with RW_OR_RO mode
            logger.debug(
                "[GMS] Connecting (socket=%s, device=%d, mode=RW_OR_RO)",
                socket_path, device_index
            )

            gms_client_memory_manager, pool = get_or_create_gms_client_memory_manager(
                socket_path, device_index, mode=RequestedLockType.RW_OR_RO, tag="weights"
            )

            granted_mode = gms_client_memory_manager.mode
            logger.info("[GMS] Connection established, mode=%s", granted_mode)

            # Dispatch to appropriate loading method
            if granted_mode == GrantedLockType.RO:
                return _load_model_read_mode(
                    gms_client_memory_manager,
                    vllm_config,
                    model_config,
                    device_index,
                )
            else:
                return _load_model_write_mode(
                    gms_client_memory_manager,
                    pool,
                    vllm_config,
                    model_config,
                    load_config,
                    target_device,
                )
