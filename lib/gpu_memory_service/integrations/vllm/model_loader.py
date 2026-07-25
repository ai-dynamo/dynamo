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
from gpu_memory_service.client.torch.module import (
    materialize_module_from_gms,
    rebind_nonparameter_tensors,
)
from gpu_memory_service.common.locks import GrantedLockType
from gpu_memory_service.common.utils import get_socket_path
from gpu_memory_service.integrations.common.utils import (
    GMSCommittedMemoryStats,
    get_gms_lock_mode,
    prepare_gms_write,
    publish_gms_write,
    setup_meta_tensor_workaround,
    strip_gms_model_loader_config,
)

if os.environ.get("MX_ENABLED", "0") == "1":
    try:
        from modelexpress.engines.vllm.adapter import build_vllm_load_context
        from modelexpress.load_strategy import (
            LoadStrategyChain,
            publish_metadata,
            register_tensors,
        )
    except ImportError as e:
        raise ImportError(
            "MX_ENABLED=1 but modelexpress is not installed. "
            "Install with: pip install modelexpress"
        ) from e

if TYPE_CHECKING:
    from gpu_memory_service.client.memory_manager import GMSClientMemoryManager

logger = logging.getLogger(__name__)

# Track imported weights plus the vLLM-local model_memory_usage adjustment.
_last_imported_weights_bytes: int = 0
_last_model_memory_usage_offset_bytes: int = 0

# First writer's GMS client awaiting publication after vLLM memory profiling.
# The retained tensors keep rebound-away GMS pool allocations alive until
# commit; see rebind_nonparameter_tensors.
_pending_gms_client: "GMSClientMemoryManager | None" = None
_pending_retained_gms_tensors: list[torch.Tensor] = []


def get_imported_weights_bytes() -> int:
    """Return bytes of weights imported in the last load_model call."""
    return _last_imported_weights_bytes


def get_model_memory_usage_offset_bytes() -> int:
    """Return the offset to add to imported bytes for vLLM model_memory_usage."""
    return _last_model_memory_usage_offset_bytes


def has_pending_gms_write() -> bool:
    """Return whether this process still owns an unpublished GMS write."""
    return _pending_gms_client is not None


def _store_pending_gms_write(
    gms_client: "GMSClientMemoryManager",
    stats: GMSCommittedMemoryStats,
    rebound_bytes: int,
    retained_gms_tensors: list[torch.Tensor],
) -> None:
    global _last_imported_weights_bytes, _last_model_memory_usage_offset_bytes
    global _pending_gms_client, _pending_retained_gms_tensors

    if _pending_gms_client is not None:
        raise RuntimeError("A GMS write is already awaiting publication")
    _pending_gms_client = gms_client
    _pending_retained_gms_tensors = retained_gms_tensors
    _last_imported_weights_bytes = stats.committed_bytes
    _last_model_memory_usage_offset_bytes = stats.pruned_bytes + rebound_bytes


def _take_pending_gms_write() -> "GMSClientMemoryManager | None":
    global _pending_gms_client, _pending_retained_gms_tensors

    gms_client = _pending_gms_client
    _pending_gms_client = None
    _pending_retained_gms_tensors = []
    return gms_client


def publish_pending_gms_write() -> bool:
    """Publish and clear the pending vLLM first-writer state, if any.

    On publication failure the writer is released best-effort and the
    original error propagates; the engine cannot serve without published
    weights, and process teardown lets GMS clear the aborted layout.
    """
    gms_client = _take_pending_gms_write()
    if gms_client is None:
        return False

    try:
        publish_gms_write(gms_client)
    except BaseException:
        try:
            gms_client.close(best_effort=True)
        except BaseException:
            logger.exception("[GMS] Failed to release a failed pending write")
        raise

    logger.info(
        "[GMS] Published %.2f GiB after vLLM memory profiling and switched "
        "to read mode",
        _last_imported_weights_bytes / (1 << 30),
    )
    return True


def abort_pending_gms_write() -> bool:
    """Abort and clear the pending vLLM first-writer state, if any.

    Releases the RPC lease best-effort without CUDA cleanup: an abort may
    run with CUDA in an error state where a normal close synchronizes and
    calls os._exit.
    """
    gms_client = _take_pending_gms_write()
    if gms_client is None:
        return False
    gms_client.close(best_effort=True)
    return True


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


def get_mx_load_context(
    vllm_config=None,
    model_config=None,
):
    """Get or create the process-global MX LoadContext singleton.

    With no arguments, returns the existing instance (or None).
    When both arguments are provided, creates the singleton on first call.
    Checks MX_ENABLED env var, modelexpress installation, and NIXL
    availability.
    """
    global _mx_ctx
    if _mx_ctx is not None:
        return _mx_ctx

    if vllm_config is None or model_config is None:
        return None

    if os.environ.get("MX_ENABLED", "0") != "1":
        return None

    _mx_ctx = build_vllm_load_context(vllm_config, model_config)
    logger.info(
        "[GMS-MX] Created MX context (rank=%d, device=%d)",
        _mx_ctx.global_rank,
        _mx_ctx.device_id,
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


# NVFP4 MoE backends whose step-5 (fused_experts.process_weights_after_loading)
# is a pure in-place activation-scale fuse that the RW writer captures in the
# committed weights, so the RO kernel rebuild (which skips step-5) is provably
# correct: the experts __init__ recomputes any derived scales from the already
# fused, committed quant config. FLASHINFER_TRTLLM is validated end-to-end;
# FLASHINFER_CUTLASS is audited by inspection. Any other backend may do work
# (workspace alloc, non-idempotent transforms, EPLB param registration) the RO
# path does not reproduce -> refuse rather than silently serve wrong weights.
# See failover-kimi/EPOCH-0251-PORT.md (R1). Override at your own risk with
# DYN_GMS_ALLOW_UNVALIDATED_MOE_BACKEND=1.
_RO_VALIDATED_NVFP4_BACKENDS = frozenset(
    {"FLASHINFER_TRTLLM", "FLASHINFER_CUTLASS"}
)


def _rebuild_ro_moe_kernels(model: torch.nn.Module) -> int:
    """Rebuild modular-MoE kernel objects on the RO (import) path.

    The RW writer commits weights *after* the quant method's
    ``process_weights_after_loading`` has run — i.e. the committed tensors are
    already in final kernel format. The RO path builds the model on meta and
    materializes those committed tensors, but the meta-side
    ``process_weights_after_loading`` aborts on the first MoE layer (meta
    tensors are unsupported), so every FusedMoE quant method is left with
    ``moe_kernel is None`` and the MoE forward asserts.

    Re-running the full ``process_weights_after_loading`` on RO would
    double-transform the already-final weights. Instead we rebuild *only* the
    kernel object (``get_fused_moe_quant_config`` + ``make_nvfp4_moe_kernel``)
    from the materialized weights, and deliberately skip the weight
    (re-)transform and the ``fused_experts.process_weights_after_loading``
    repack the writer already committed. Analog of the 0.23.0 RO MoE-kernel
    rebuild, adapted to 0.25.1's modular MoE.
    """
    try:
        from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
            make_nvfp4_moe_kernel,
        )
    except Exception as e:
        logger.debug("[GMS] No NVFP4 modular-MoE kernel builder available: %s", e)
        return 0

    built = 0
    for layer in model.modules():
        qm = getattr(layer, "quant_method", None)
        if qm is None or getattr(qm, "moe_kernel", "missing") is not None:
            # Not a quant method, or kernel already built.
            continue
        # NVFP4 modular FusedMoE method exposes these; skip everything else.
        if not (
            hasattr(qm, "nvfp4_backend")
            and hasattr(qm, "experts_cls")
            and hasattr(qm, "get_fused_moe_quant_config")
        ):
            continue
        backend_name = getattr(qm.nvfp4_backend, "name", str(qm.nvfp4_backend))
        if (
            backend_name not in _RO_VALIDATED_NVFP4_BACKENDS
            and os.environ.get("DYN_GMS_ALLOW_UNVALIDATED_MOE_BACKEND") != "1"
        ):
            raise RuntimeError(
                f"[GMS] RO MoE-kernel rebuild is not validated for NVFP4 backend "
                f"{backend_name!r} (validated: {sorted(_RO_VALIDATED_NVFP4_BACKENDS)}). "
                f"The RO path skips fused_experts.process_weights_after_loading, which "
                f"is only proven safe when step-5 is an in-place scale fuse captured in "
                f"the commit. Set DYN_GMS_ALLOW_UNVALIDATED_MOE_BACKEND=1 to override at "
                f"your own risk (may serve numerically wrong weights). See "
                f"failover-kimi/EPOCH-0251-PORT.md (R1)."
            )
        if built == 0:
            logger.info(
                "[GMS] Read mode: rebuilding MoE kernels for NVFP4 backend %s",
                backend_name,
            )
        try:
            qm.moe_quant_config = qm.get_fused_moe_quant_config(layer)
            qm.moe_kernel = make_nvfp4_moe_kernel(
                moe_quant_config=qm.moe_quant_config,
                moe_config=qm.moe,
                experts_cls=qm.experts_cls,
                backend=qm.nvfp4_backend,
                routing_tables=layer._expert_routing_tables(),
                layer=layer,
            )
            built += 1
        except Exception:
            logger.exception(
                "[GMS] RO moe_kernel rebuild failed for %s",
                layer.__class__.__name__,
            )
    if built:
        logger.info("[GMS] Read mode: rebuilt %d MoE kernel(s)", built)
    return built


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
    global _last_imported_weights_bytes, _last_model_memory_usage_offset_bytes

    try:
        model = _create_meta_model(vllm_config, model_config)
        materialize_module_from_gms(gms_client, model, device_index=device_index)

        # Modular-MoE kernels are not committed (they are runtime objects, not
        # weights); rebuild them from the materialized weights. See
        # _rebuild_ro_moe_kernels for why we skip the weight re-transform.
        _rebuild_ro_moe_kernels(model)

        # MX: register materialized tensors (available for P2P transfer)
        mx_ctx = get_mx_load_context(vllm_config, model_config)
        if mx_ctx is not None:
            register_tensors(model, mx_ctx)
            publish_metadata(mx_ctx)

        _last_imported_weights_bytes = gms_client.total_bytes
        _last_model_memory_usage_offset_bytes = 0
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
    """Load model from disk and prepare weights for GMS publication (RW mode).

    Initializes model using GMS memory pool, loads weights from disk,
    registers tensors with GMS, and prepares a write that is published only
    after vLLM memory profiling (see GMSWorker.determine_available_memory).
    Deferring the commit keeps waiting RO consumers (snapshot saver, peer
    engines) off the device while vLLM profiles memory.

    When MX is active, uses LoadStrategyChain for automatic weight source
    detection (RDMA P2P -> ModelStreamer -> GDS -> disk) with fallback.
    The chain also handles NIXL registration and metadata publishing.
    """
    if _pending_gms_client is not None:
        raise RuntimeError("A GMS write is already awaiting publication")

    from vllm.model_executor.model_loader.utils import (
        initialize_model,
        process_weights_after_loading,
    )
    from vllm.utils.torch_utils import set_default_torch_dtype

    mx_ctx = get_mx_load_context(vllm_config, model_config)

    # Allocate model tensors using GMS memory pool
    with set_default_torch_dtype(model_config.dtype):
        with gms_use_mem_pool("weights", target_device):
            with target_device:
                model = initialize_model(
                    vllm_config=vllm_config, model_config=model_config
                )

            if mx_ctx is not None:
                # Full MX load strategy chain: RDMA -> ModelStreamer -> GDS -> Default
                LoadStrategyChain.run(model, mx_ctx)
            else:
                default_loader.load_weights(model, model_config)
                process_weights_after_loading(model, model_config, target_device)

            torch.cuda.empty_cache()

    stats = prepare_gms_write(gms_client, model)
    # The private clones must exist before vLLM profiles memory so the
    # profiled peak covers them. The retained GMS originals stay alive until
    # commit, for readers to materialize from.
    retained_gms_tensors: list[torch.Tensor] = []
    rebound_bytes = rebind_nonparameter_tensors(
        gms_client, model, retain_gms_tensors=retained_gms_tensors
    )
    _store_pending_gms_write(gms_client, stats, rebound_bytes, retained_gms_tensors)

    logger.info(
        "[GMS] Write mode: prepared %.2f GiB for publication after profiling "
        "(vLLM memory offset %.2f GiB)",
        _last_imported_weights_bytes / (1 << 30),
        _last_model_memory_usage_offset_bytes / (1 << 30),
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
