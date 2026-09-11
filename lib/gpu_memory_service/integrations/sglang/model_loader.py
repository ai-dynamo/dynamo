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

import torch
from gpu_memory_service.client.torch.module import materialize_module_from_gms
from gpu_memory_service.common.locks import GrantedLockType
from gpu_memory_service.integrations.common import patch_empty_cache
from gpu_memory_service.integrations.common.utils import (
    setup_meta_tensor_workaround,
    strip_gms_model_loader_config,
)
from gpu_memory_service.integrations.sglang import (
    install_gms_unified_cache,
    install_kv_leases,
    install_vmm_ipc_kv,
)
from gpu_memory_service.integrations.sglang.memory_saver import (
    get_gms_memory_saver_impl,
)
from gpu_memory_service.integrations.sglang.patches import (
    patch_failover_extend_warmup_for_gms,
    patch_model_runner,
    patch_serving_collective_timeout_for_gms,
    patch_shared_kv_pool_geometry,
    patch_static_state_for_gms,
    patch_torch_memory_saver,
)

logger = logging.getLogger(__name__)

# Apply patches at module import time.
# This module is only imported when load_format="gms" is used.
# Because SGLang scheduler processes use multiprocessing spawn, these patches
# must run inside the child process.  The import chain that triggers this is:
#   child unpickles server_args.load_format -> imports GMSModelLoader -> here.
patch_empty_cache()
patch_torch_memory_saver()
patch_model_runner()
patch_shared_kv_pool_geometry()
patch_failover_extend_warmup_for_gms()
patch_static_state_for_gms()
patch_serving_collective_timeout_for_gms()
install_vmm_ipc_kv.install()
install_kv_leases.install()
install_gms_unified_cache.install()
logger.info("[GMS] Applied patches")


class GMSModelLoader:
    """SGLang model loader that loads/imports weights via GPU Memory Service."""

    def __init__(self, load_config):
        self.load_config = load_config
        self._default_loader = None
        self.preloaded_weights_bytes = 0

    def _get_default_loader(self):
        if self._default_loader is None:
            from sglang.srt.model_loader.loader import DefaultModelLoader

            config = strip_gms_model_loader_config(
                self.load_config,
                load_format="auto",
            )
            self._default_loader = DefaultModelLoader(config)
        return self._default_loader

    def load_model(
        self,
        *,
        model_config,
        device_config,
    ) -> torch.nn.Module:
        """Load or import model weights."""
        impl = get_gms_memory_saver_impl()
        if impl is None:
            raise RuntimeError(
                "GMS impl not initialized. "
                "Ensure torch_memory_saver patch was applied before model loading."
            )

        mode = impl.allocators["weights"].granted_lock_type
        logger.info("[GMS] Loading model in %s mode", mode.name)

        if mode == GrantedLockType.RO:
            return self._load_import_only(model_config, device_config, impl)
        return self._load_write_mode(model_config, device_config, impl)

    def _load_write_mode(self, model_config, device_config, impl) -> torch.nn.Module:
        """Load model from disk and register with GMS (WRITE mode)."""
        default_loader = self._get_default_loader()

        model = default_loader.load_model(
            model_config=model_config,
            device_config=device_config,
        )

        # ModelRunner owns the outer torch_memory_saver weights region. A
        # finalization here would reconnect the GMS allocator read-only while
        # that region is still active, routing private buffer clones through a
        # read-only GMS pool and surfacing as a false CUDA OOM. The ModelRunner
        # patch publishes immediately after the region exits.
        impl.finalize_write_mode(model)
        return model

    def _load_import_only(self, model_config, device_config, impl) -> torch.nn.Module:
        """Import model weights from GMS metadata (READ mode)."""
        allocator = impl.allocators["weights"]

        device_index = torch.cuda.current_device()
        model = self._create_meta_model(model_config, device_config)

        materialize_module_from_gms(allocator, model, device_index=device_index)
        self._restore_import_runtime_state(model)
        impl.imported_weights_bytes = allocator.total_bytes
        impl.preloaded_weights_bytes = allocator.total_bytes
        self.preloaded_weights_bytes = allocator.total_bytes

        logger.info(
            "[GMS] READ mode: imported %.2f GiB from metadata",
            allocator.total_bytes / (1 << 30),
        )
        return model.eval()

    @staticmethod
    def _restore_import_runtime_state(model: torch.nn.Module) -> None:
        """Recreate small backend state that is not part of shared weights."""
        restored = 0
        for module in model.modules():
            quant_method = getattr(module, "quant_method", None)
            prepare = getattr(
                quant_method,
                "_prepare_flashinfer_trtllm_activation_params",
                None,
            )
            if prepare is None or hasattr(module, "_flashinfer_trtllm_gemm1_alpha"):
                continue
            prepare(module)
            restored += 1
        if restored:
            logger.info(
                "[GMS] Restored FlashInfer TRT-LLM runtime state for %d modules",
                restored,
            )

    @staticmethod
    def _validate_import_model(model: torch.nn.Module) -> None:
        """Reject model-specific mutations that cannot be replayed safely."""
        if callable(getattr(model, "post_load_weights", None)):
            raise RuntimeError(
                "GMS SGLang read-only weight import does not yet support models "
                "with custom post_load_weights processing. The writer publishes "
                "the final tensor graph, so replaying that hook after import can "
                "mutate or requantize shared weights; use a supported model until "
                "an explicit runtime-state restoration adapter is available."
            )

    def _create_meta_model(self, model_config, device_config) -> torch.nn.Module:
        """Create model on meta device for import-only mode."""
        from sglang.srt.model_loader.loader import (
            _get_quantization_config,
            _initialize_model,
        )
        from sglang.srt.model_loader.utils import set_default_torch_dtype

        setup_meta_tensor_workaround()

        meta_device = torch.device("meta")
        load_config = strip_gms_model_loader_config(
            self.load_config,
            load_format="auto",
        )
        quant_config = _get_quantization_config(model_config, load_config)

        # DummyModelLoader opens its own CUDA device context and initializes
        # random weights. Build only the module tree on meta; GMS then binds the
        # final parameters and tensor attributes published by the writer.
        with set_default_torch_dtype(model_config.dtype):
            with meta_device:
                model = _initialize_model(
                    model_config,
                    load_config,
                    quant_config,
                )
        self._validate_import_model(model)

        if not str(getattr(device_config, "device", "")).startswith("cuda"):
            raise RuntimeError(
                "GMS SGLang weight import currently requires a CUDA device"
            )

        return model
