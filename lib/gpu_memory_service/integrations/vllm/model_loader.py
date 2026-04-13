# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM model loader for GPU Memory Service integration.

Provides a model loader that loads weights via GMS for cross-process sharing.
The loader uses RW_OR_RO mode: first process loads from disk (RW), subsequent
processes import from GMS metadata (RO).
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import torch
from gpu_memory_service.client.torch.allocator import (
    get_or_create_gms_client_memory_manager,
    gms_use_mem_pool,
)
from gpu_memory_service.client.torch.module import materialize_module_from_gms
from gpu_memory_service.common.locks import GrantedLockType
from gpu_memory_service.common.utils import get_socket_path
from gpu_memory_service.integrations.common.utils import (
    finalize_gms_write,
    get_gms_lock_mode,
    setup_meta_tensor_workaround,
    strip_gms_model_loader_config,
)

if TYPE_CHECKING:
    from gpu_memory_service.client.memory_manager import GMSClientMemoryManager

logger = logging.getLogger(__name__)

# Track imported weights for memory accounting
_last_imported_weights_bytes: int = 0


def get_imported_weights_bytes() -> int:
    """Return bytes of weights imported in the last load_model call."""
    return _last_imported_weights_bytes


# =============================================================================
# MX (ModelExpress) Integration — Optional P2P weight transfer
#
# Write mode: delegates to LoadStrategyChain which handles weight loading
#   (RDMA P2P -> ModelStreamer -> GDS -> disk), post-processing, NIXL
#   registration, and metadata publishing.
# Read mode: uses register_tensors + publish_metadata directly to make
#   GMS-imported tensors available as a P2P source.
# =============================================================================

_mx_ctx = None  # type: LoadContext | None


def get_mx_ctx(
    vllm_config=None,
    model_config=None,
    device_id: int | None = None,
):
    """Get or create the process-global MX LoadContext singleton.

    With no arguments, returns the existing instance (or None).
    When all arguments are provided, creates the singleton on first call.
    Checks MX_ENABLED env var, modelexpress installation, and NIXL
    availability.
    """
    global _mx_ctx
    if _mx_ctx is not None:
        return _mx_ctx

    if vllm_config is None or model_config is None or device_id is None:
        return None

    if os.environ.get("MX_ENABLED", "0") != "1":
        return None

    try:
        from modelexpress.client import MxClient
        from modelexpress.load_strategy import LoadContext
        from modelexpress.metadata import build_source_identity
        from modelexpress.nixl_transfer import is_nixl_available
    except ImportError:
        logger.warning(
            "[GMS-MX] MX_ENABLED=1 but modelexpress is not installed. "
            "Skipping MX integration."
        )
        return None

    if not is_nixl_available():
        logger.warning(
            "[GMS-MX] MX_ENABLED=1 but NIXL is not available. "
            "Skipping MX integration."
        )
        return None

    import uuid

    try:
        import torch.distributed as dist

        global_rank = dist.get_rank() if dist.is_initialized() else device_id
    except Exception:
        global_rank = device_id

    _mx_ctx = LoadContext(
        vllm_config=vllm_config,
        model_config=model_config,
        load_config=vllm_config.load_config,
        target_device=torch.device("cuda", device_id),
        global_rank=global_rank,
        device_id=device_id,
        identity=build_source_identity(vllm_config, model_config),
        mx_client=MxClient(),
        worker_id=uuid.uuid4().hex[:8],
    )
    logger.info(
        "[GMS-MX] Created MX context (rank=%d, device=%d)",
        global_rank,
        device_id,
    )
    return _mx_ctx


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
            # Strip GMS-specific keys before creating the fallback loader,
            # otherwise DefaultModelLoader rejects unknown extra config.
            self.default_loader = DefaultModelLoader(
                strip_gms_model_loader_config(
                    load_config,
                    load_format="auto",
                )
            )

        def download_model(self, model_config) -> None:
            self.default_loader.download_model(model_config)

        def load_weights(self, model: torch.nn.Module, model_config) -> None:
            self.default_loader.load_weights(model, model_config)

        def load_model(self, vllm_config, model_config, prefix="") -> torch.nn.Module:
            device = torch.cuda.current_device()
            extra = getattr(self.load_config, "model_loader_extra_config", {}) or {}
            mode = get_gms_lock_mode(extra)
            gms_client = get_or_create_gms_client_memory_manager(
                get_socket_path(device, "weights"),
                device,
                mode=mode,
                tag="weights",
            )

            if gms_client.granted_lock_type == GrantedLockType.RO:
                return _load_read_mode(gms_client, vllm_config, model_config, device)
            else:
                return _load_write_mode(
                    gms_client,
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
    """Load model by importing weights from GMS (RO mode).

    When MX is active, registers materialized tensors with NIXL so this
    node is discoverable as a P2P source (e.g. for shadow engine failover).
    """
    global _last_imported_weights_bytes

    try:
        model = _create_meta_model(vllm_config, model_config)
        materialize_module_from_gms(gms_client, model, device_index=device_index)

        # MX: register materialized tensors (available for P2P transfer)
        mx_ctx = get_mx_ctx(vllm_config, model_config, device_index)
        if mx_ctx is not None:
            from modelexpress.load_strategy import publish_metadata, register_tensors

            try:
                register_tensors(model, mx_ctx)
                publish_metadata(mx_ctx)
            except Exception as e:
                logger.warning("[GMS-MX] Registration failed: %s", e)

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
    vllm_config,
    model_config,
    default_loader,
    target_device: torch.device,
) -> torch.nn.Module:
    """Load model from disk and publish weights to GMS (RW mode).

    Initializes model using GMS memory pool, loads weights from disk,
    registers tensors with GMS, and commits for cross-process sharing.

    When MX is active, uses LoadStrategyChain for automatic weight source
    detection (RDMA P2P -> ModelStreamer -> GDS -> disk) with fallback.
    The chain also handles NIXL registration and metadata publishing.
    """
    global _last_imported_weights_bytes

    from vllm.model_executor.model_loader.utils import (
        initialize_model,
        process_weights_after_loading,
    )
    from vllm.utils.torch_utils import set_default_torch_dtype

    device_id = target_device.index
    mx_ctx = get_mx_ctx(vllm_config, model_config, device_id)

    # Allocate model tensors using GMS memory pool
    with set_default_torch_dtype(model_config.dtype):
        with gms_use_mem_pool("weights", target_device):
            with target_device:
                model = initialize_model(
                    vllm_config=vllm_config, model_config=model_config
                )

            if mx_ctx is not None:
                # Full MX strategy chain: RDMA -> ModelStreamer -> GDS -> Default
                # Chain handles weight loading, post-processing, NIXL registration,
                # and metadata publishing.
                from modelexpress.load_strategy import LoadStrategyChain

                LoadStrategyChain.run(model, mx_ctx)
            else:
                # No MX: load from disk (existing behavior)
                default_loader.load_weights(model, model_config)
                process_weights_after_loading(model, model_config, target_device)

            torch.cuda.empty_cache()

    _last_imported_weights_bytes = finalize_gms_write(gms_client, model)

    logger.info(
        "[GMS] Write mode: published %.2f GiB",
        _last_imported_weights_bytes / (1 << 30),
    )
    return model.eval()


def _create_meta_model(vllm_config, model_config) -> torch.nn.Module:
    """Create model on meta device for RO mode materialization."""
    from vllm.model_executor.model_loader.utils import (
        initialize_model,
        process_weights_after_loading,
    )
    from vllm.utils.torch_utils import set_default_torch_dtype

    setup_meta_tensor_workaround()
    meta_device = torch.device("meta")

    with set_default_torch_dtype(model_config.dtype):
        with meta_device:
            model = initialize_model(vllm_config=vllm_config, model_config=model_config)

    try:
        process_weights_after_loading(model, model_config, meta_device)
    except Exception as e:
        logger.debug("[GMS] Post-processing on meta tensors: %s", e)

    return model
