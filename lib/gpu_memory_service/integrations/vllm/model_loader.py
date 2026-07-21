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
    get_gms_client_memory_manager,
    get_or_create_gms_client_memory_manager,
    get_or_create_scratch_manager,
    gms_use_mem_pool,
)
from gpu_memory_service.client.torch.module import (
    _iter_module_tensors,
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

# Allocator tag for the transient single-block scratch pool that backs RO model
# construction (see _create_device_model). Torn down after materialize.
_CONSTRUCT_TAG = "ro_construct"

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
        # Construct the RO skeleton on a REAL device (over aliased scratch)
        # rather than on `meta`. Building on a real device sidesteps the
        # meta-safety hazards the old meta path guarded against (.item()/
        # .to(device)/data-dependent ops during __init__, fake-op signatures,
        # post-load hooks that touch the device) — the source of most per-model
        # RO patches. Construction allocates over a single-block scratch pool so
        # the garbage skeleton costs one aliased physical block, not a
        # weights-sized copy (see _create_device_model).
        model = _create_device_model(vllm_config, model_config, device_index)
        # materialize runs OUTSIDE the construct scratch pool: parameters are
        # repointed to committed GMS views and buffers/attrs are cloned into
        # ordinary (non-scratch) device memory, so nothing the model retains
        # references construction scratch afterward.
        materialize_module_from_gms(gms_client, model, device_index=device_index)
        # Hard-assert every construction-scratch tensor was replaced, then free
        # the scratch pool. Any tensor still on scratch would dangle once the
        # pool is torn down, so this fails loudly rather than serving garbage.
        _finalize_construction_scratch(model)

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


def _create_device_model(
    vllm_config, model_config, device_index: int
) -> torch.nn.Module:
    """Create the RO model on a real CUDA device over aliased scratch backing.

    Mirrors KV shadow-mode construction: parameters (and the derived tensors
    ``process_weights_after_loading`` builds) are allocated through a
    single-block scratch pool, so the whole garbage skeleton is backed by ONE
    aliased physical block (~scratch_size) instead of a weights-sized copy —
    which is what makes this fit the failover colocation budget.
    ``materialize_module_from_gms`` then repoints each parameter to its committed
    GMS view; the scratch pool is torn down afterward (see
    ``_finalize_construction_scratch``).

    Building on a real device (instead of ``meta``) lets model ``__init__`` and
    ``process_weights_after_loading`` run on real tensors, so the meta-device
    hazards the old meta path guarded against (``.item()``/``.to(device)``/data
    dependent ops, custom-op fake signatures, post-load hooks that initialize
    device scratch) simply do not arise.
    """
    from vllm.model_executor.model_loader.utils import (
        initialize_model,
        process_weights_after_loading,
    )
    from vllm.utils.torch_utils import set_default_torch_dtype

    device = torch.device("cuda", device_index)
    # Client-local scratch only (no GMS server session): one shared physical
    # block aliased across every construction allocation.
    get_or_create_scratch_manager(
        get_socket_path(device_index, _CONSTRUCT_TAG),
        device_index,
        tag=_CONSTRUCT_TAG,
        single_block=True,
    )
    with set_default_torch_dtype(model_config.dtype):
        with gms_use_mem_pool(_CONSTRUCT_TAG, device):
            with device:
                model = initialize_model(
                    vllm_config=vllm_config, model_config=model_config
                )
            # Runs on real tensors, so post-load hooks that build derived tensors
            # (MLA projections, MoE kernels, ...) execute normally. Values are
            # garbage and are overwritten by materialize; only the resulting
            # structure matters, and the scratch-coverage check backstops any
            # tensor materialize does not replace. A garbage-value-sensitive hook
            # is unexpected but must not abort bring-up.
            try:
                process_weights_after_loading(model, model_config, device)
            except Exception as e:
                logger.warning(
                    "[GMS] Read mode: process_weights_after_loading raised during "
                    "device construction (continuing; materialize fills committed "
                    "tensors, scratch-coverage check guards the rest): %s",
                    e,
                )

    return model


def _finalize_construction_scratch(model: torch.nn.Module) -> None:
    """Assert construction scratch was fully replaced, then free the pool.

    Unlike KV scratch (reallocated in place at the same VA on wake), construction
    scratch is torn down after materialize. A parameter or buffer still backed by
    it would dangle into freed memory once the pool is released, so any leftover
    is a hard error — an uncommitted weight, or a derived tensor materialize never
    filled — rather than a silently-served garbage/use-after-free model. On
    success the shared scratch block and all its VA reservations are released and
    the tag is evicted.
    """
    manager = get_gms_client_memory_manager(_CONSTRUCT_TAG)
    if manager is None:
        return

    ranges = [
        (base_va, scratch.va_reserved_size)
        for base_va, scratch in manager.scratch_mappings.items()
    ]

    def _in_scratch(t: torch.Tensor) -> bool:
        ptr = int(t.data_ptr())
        return any(va <= ptr < va + size for va, size in ranges)

    try:
        # Enumerate exactly the tensor set materialize operates on (parameters,
        # buffers, and bare tensor attributes) so a scratch-backed tensor_attr
        # materialize never replaced is caught too, not just params/buffers.
        dangling = [
            name
            for name, tensor, _ in _iter_module_tensors(model)
            if tensor.numel() > 0 and _in_scratch(tensor)
        ]
        if dangling:
            raise RuntimeError(
                f"[GMS] Read mode: {len(dangling)} tensor(s) still backed by "
                "construction scratch after materialize (uncommitted weight or "
                f"unfilled derived tensor): {dangling[:10]}"
            )
    finally:
        # Free the shared scratch block + VA reservations and evict the tag,
        # whether or not the assertion above fired.
        manager.close()
