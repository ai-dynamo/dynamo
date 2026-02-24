# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang model loader for GPU Memory Service integration.

Provides a model loader that loads weights via GMS for cross-process sharing.
The loader uses RW_OR_RO mode: first process loads from disk (RW), subsequent
processes import from GMS metadata (RO).

Usage:
    Set --load-format gms when launching SGLang.
"""

from __future__ import annotations

import logging
from dataclasses import replace

import torch
from gpu_memory_service.integrations.common import patch_empty_cache
from gpu_memory_service.integrations.common.utils import setup_meta_tensor_workaround
from gpu_memory_service.integrations.sglang.patches import (
    patch_model_runner,
    patch_torch_memory_saver,
)

logger = logging.getLogger(__name__)

# Apply patches at module import time.
# This module is only imported when load_format="gms" is used.
patch_empty_cache()
patch_torch_memory_saver()
patch_model_runner()
logger.info("[GMS] Applied patches")


class GMSModelLoader:
    """SGLang model loader that loads/imports weights via GPU Memory Service."""

    def __init__(self, load_config):
        self.load_config = load_config
        self._default_loader = None

    def _get_default_loader(self):
        if self._default_loader is None:
            from sglang.srt.model_loader.loader import DefaultModelLoader

            config = replace(self.load_config, load_format="auto")
            self._default_loader = DefaultModelLoader(config)
        return self._default_loader

    def load_model(
        self,
        *,
        model_config,
        device_config,
    ) -> torch.nn.Module:
        """Load or import model weights."""
        from gpu_memory_service.integrations.sglang.memory_saver import (
            get_gms_memory_saver_impl,
        )

        impl = get_gms_memory_saver_impl()
        if impl is None:
            raise RuntimeError(
                "GMS impl not initialized. "
                "Ensure torch_memory_saver patch was applied before model loading."
            )

        mode = impl.get_mode()
        logger.info("[GMS] Loading model in %s mode", mode.upper())

        if mode == "read":
            return self._load_import_only(model_config, device_config, impl)
        else:
            return self._load_write_mode(model_config, device_config, impl)

    def _load_write_mode(self, model_config, device_config, impl) -> torch.nn.Module:
        """Load model from disk and register with GMS (WRITE mode)."""
        default_loader = self._get_default_loader()

        model = default_loader.load_model(
            model_config=model_config,
            device_config=device_config,
        )

        impl.finalize_write_mode(model)
        return model

    def _load_import_only(self, model_config, device_config, impl) -> torch.nn.Module:
        """Import model weights from GMS metadata (READ mode)."""
        from gpu_memory_service.client.torch.module import materialize_module_from_gms

        allocator = impl.get_allocator()
        if allocator is None:
            raise RuntimeError("GMS allocator is None in READ mode")

        device_index = torch.cuda.current_device()
        model = self._create_meta_model(model_config, device_config)

        materialize_module_from_gms(allocator, model, device_index=device_index)
        impl.set_imported_weights_bytes(allocator.total_bytes)

        logger.info(
            "[GMS] READ mode: imported %.2f GiB from metadata",
            allocator.total_bytes / (1 << 30),
        )
        return model.eval()

    def _create_meta_model(self, model_config, device_config) -> torch.nn.Module:
        """Create model on meta device for import-only mode."""
        from sglang.srt.model_loader import get_model

        setup_meta_tensor_workaround()

        original_device = torch.cuda.current_device()
        meta_device = torch.device("meta")

        with meta_device:
            model = get_model(
                model_config=model_config,
                load_config=replace(self.load_config, load_format="dummy"),
                device_config=device_config,
            )

        torch.cuda.set_device(original_device)

        try:
            from sglang.srt.model_loader.utils import (
                process_model_weights_after_loading,
            )

            process_model_weights_after_loading(model, model_config)
        except Exception as e:
            logger.debug("[GMS] Post-processing on meta tensors: %s", e)

        return model
