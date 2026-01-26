# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang model loader for GPU Memory Service integration.

This module provides a custom SGLang `load_format` class that integrates with
the GPUMemoryServiceMemorySaverImpl for weight sharing.

The model loader uses "auto" mode to connect to the GPU Memory Service:
- First process to connect gets RW lock and loads weights from disk
- Subsequent processes get RO lock and import weights from metadata

This enables weight sharing across processes without explicit configuration.

Flow:
1. torch_memory_saver patch creates GPUMemoryServiceMemorySaverImpl (owns allocator with auto mode)
2. SGLang's ModelRunner calls region("weights") which sets up use_mem_pool() in WRITE mode
3. GPUServiceModelLoader.load_model() delegates to DefaultModelLoader (allocations routed via mempool)
4. After loading, GPUServiceModelLoader calls impl.finalize_write_mode() to commit and switch to read

For import-only mode (READ - granted when another process already committed weights):
- GPUServiceModelLoader creates a meta model and materializes from metadata
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import replace
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

# Apply patches at module import time - this ensures patches are active
# in SGLang's scheduler subprocess BEFORE torch_memory_saver is initialized
def _apply_gms_patches():
    """Apply GPU Memory Service patches at module import time."""
    if os.environ.get("GPU_MEMORY_SERVICE_SGLANG_AUTO_REGISTER") == "1":
        from gpu_memory_service.sglang_integration.patches import (
            patch_empty_cache,
            patch_torch_memory_saver,
        )
        patch_empty_cache()
        patch_torch_memory_saver()
        logger.info("[GMS] Applied patches at model_loader module import")

_apply_gms_patches()


def _get_local_rank() -> int:
    """Get the local rank (GPU device index) for the current worker."""
    try:
        if torch.cuda.is_initialized():
            current_device = torch.cuda.current_device()
            if current_device != 0 or os.environ.get("LOCAL_RANK", "0") == "0":
                return current_device
    except Exception:
        pass

    try:
        import torch.distributed as dist

        if dist.is_initialized():
            return int(dist.get_rank() % torch.cuda.device_count())
    except Exception:
        pass

    return int(os.environ.get("LOCAL_RANK", "0"))


def _parse_extra_config(load_config: Any = None) -> dict:
    """Parse model_loader_extra_config from load_config."""
    if load_config is None:
        return {}
    raw_config = getattr(load_config, "model_loader_extra_config", None)
    if not raw_config:
        return {}
    if isinstance(raw_config, str):
        return json.loads(raw_config)
    return raw_config


# Keys that GPU Memory Service adds to model_loader_extra_config
GPU_MEMORY_SERVICE_EXTRA_CONFIG_KEYS = {
    "gpu_memory_service_socket_path",
}


def _strip_gms_extra_config(load_config: Any) -> Any:
    """Return load_config with GMS keys removed from model_loader_extra_config."""
    extra = _parse_extra_config(load_config)
    filtered = {k: v for k, v in extra.items() if k not in GPU_MEMORY_SERVICE_EXTRA_CONFIG_KEYS}

    if hasattr(load_config, "model_loader_extra_config"):
        # Always use JSON string - SGLang's DefaultModelLoader expects a dict-like config
        return replace(load_config, model_loader_extra_config=json.dumps(filtered))
    return load_config


class GPUServiceModelLoader:
    """SGLang model loader that loads/imports weights via GPU Memory Service.

    This class implements SGLang's model loader interface and routes to either:
    - WRITE mode: Load from disk using DefaultModelLoader, allocations via GMS mempool
    - READ mode: Create meta model and materialize from GMS metadata
    """

    def __init__(self, load_config):
        """Initialize with load_config from SGLang."""
        self.load_config = load_config
        self.extra_config = _parse_extra_config(load_config)
        self._default_loader = None

    def _get_default_loader(self):
        """Lazily create the default loader."""
        if self._default_loader is None:
            from sglang.srt.model_loader.loader import DefaultModelLoader

            stripped_config = _strip_gms_extra_config(self.load_config)
            stripped_config = replace(stripped_config, load_format="auto")
            self._default_loader = DefaultModelLoader(stripped_config)
        return self._default_loader

    def load_model(
        self,
        *,
        model_config,
        device_config,
    ) -> torch.nn.Module:
        """Load or import model weights.

        Checks the GMS impl mode:
        - WRITE mode: Delegate to DefaultModelLoader, then finalize
        - READ mode: Create meta model and materialize from metadata
        """
        from gpu_memory_service.sglang_integration.torch_memory_saver_impl import (
            get_gpu_memory_service_impl,
        )

        impl = get_gpu_memory_service_impl()
        if impl is None:
            raise RuntimeError(
                "GPU Memory Service impl not initialized. "
                "Ensure torch_memory_saver_patch was imported before model loading."
            )

        mode = impl.get_mode()
        logger.info("[GMS Loader] Loading model in %s mode", mode.upper())

        if mode == "read":
            return self._load_import_only(model_config, device_config, impl)
        else:
            return self._load_write_mode(model_config, device_config, impl)

    def _load_write_mode(self, model_config, device_config, impl) -> torch.nn.Module:
        """Load model from disk and register with GMS (WRITE mode)."""
        default_loader = self._get_default_loader()

        # DefaultModelLoader.load_model() handles the weights region context
        model = default_loader.load_model(
            model_config=model_config,
            device_config=device_config,
        )

        # Finalize: register tensors, commit, switch to read mode
        impl.finalize_write_mode(model)

        return model

    def _load_import_only(self, model_config, device_config, impl) -> torch.nn.Module:
        """Import model weights from GMS metadata (READ mode)."""
        from gpu_memory_service.client.torch.module import materialize_module_from_gms

        allocator = impl.get_allocator()
        if allocator is None:
            raise RuntimeError("GMS allocator is None in READ mode")

        device_index = _get_local_rank()

        # Create meta model (no GPU memory)
        model = self._create_meta_model(model_config, device_config)

        # Materialize weights from GMS
        materialize_module_from_gms(allocator, model, device_index=device_index)

        # Track imported bytes
        impl.set_imported_weights_bytes(allocator.total_bytes)

        logger.info(
            "[GMS Loader] READ mode: imported %.2f GiB from metadata",
            allocator.total_bytes / (1 << 30),
        )

        return model.eval()

    def _create_meta_model(self, model_config, device_config) -> torch.nn.Module:
        """Create model on meta device for import-only mode."""
        from sglang.srt.model_loader import get_model

        # Enable meta tensor workaround for torch.nonzero() etc.
        try:
            import torch.fx.experimental._config as fx_config

            fx_config.meta_nonzero_assume_all_nonzero = True
        except (ImportError, AttributeError):
            pass

        original_device = torch.cuda.current_device()
        meta_device = torch.device("meta")

        with meta_device:
            model = get_model(
                model_config=model_config,
                load_config=replace(self.load_config, load_format="dummy"),
                device_config=device_config,
            )

        # Restore device
        torch.cuda.set_device(original_device)

        # Try post-processing on meta tensors (may fail, that's OK)
        try:
            from sglang.srt.model_loader.utils import process_model_weights_after_loading

            process_model_weights_after_loading(model, model_config)
        except Exception as e:
            logger.debug("[GMS Loader] Post-processing on meta tensors: %s", e)

        return model


def patch_model_runner_for_gpu_memory_service() -> None:
    """Patch SGLang's ModelRunner to fix memory accounting with pre-loaded weights.

    When weights are pre-loaded via GMS (import-only mode), SGLang's min_per_gpu_memory
    captured before loading is lower than device total. This causes under-reservation
    of overhead memory in KV cache calculation.

    This patch ensures ModelRunner.init_memory_pool() always uses device total memory
    for overhead calculation.
    """
    try:
        from sglang.srt.model_executor.model_runner import ModelRunner
    except ImportError:
        logger.warning("[GMS Patch] Could not import ModelRunner, skipping patch")
        return

    if hasattr(ModelRunner, "_gms_patched"):
        return

    original_init_memory_pool = ModelRunner.init_memory_pool

    def patched_init_memory_pool(self, *args, **kwargs):
        """Patched init_memory_pool that uses device total for overhead calculation."""
        # Override min_per_gpu_memory to device total if weights were pre-loaded
        from gpu_memory_service.sglang_integration.torch_memory_saver_impl import (
            get_gpu_memory_service_impl,
        )

        impl = get_gpu_memory_service_impl()
        if impl is not None and impl.get_imported_weights_bytes() > 0:
            total_memory = torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).total_memory
            if hasattr(self, "min_per_gpu_memory"):
                old_value = self.min_per_gpu_memory
                self.min_per_gpu_memory = total_memory
                logger.info(
                    "[GMS Patch] Adjusted min_per_gpu_memory: %.2f GiB -> %.2f GiB",
                    old_value / (1 << 30),
                    total_memory / (1 << 30),
                )

        return original_init_memory_pool(self, *args, **kwargs)

    ModelRunner.init_memory_pool = patched_init_memory_pool
    ModelRunner._gms_patched = True
    logger.info("[GMS Patch] Patched ModelRunner.init_memory_pool")
