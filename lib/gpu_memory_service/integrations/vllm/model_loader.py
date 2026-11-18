# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM model loader for GPU Memory Service integration.

Provides a model loader that loads weights via GMS for cross-process sharing.
The loader uses RW_OR_RO mode: first process loads from disk (RW), subsequent
processes import from GMS metadata (RO).
"""

from __future__ import annotations

import gc
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

import torch
from gpu_memory_service.client.torch.allocator import (
    get_or_create_gms_client_memory_manager,
    gms_use_mem_pool,
    raise_if_gms_allocator_failed,
    release_gms_mem_pool,
)
from gpu_memory_service.client.torch.module import (
    _iter_module_tensors,
    materialize_module_from_gms,
    register_module_tensors,
)
from gpu_memory_service.common.locks import GrantedLockType
from gpu_memory_service.common.utils import get_socket_path
from gpu_memory_service.common.vmm import get_vmm_device_type
from gpu_memory_service.integrations.common.utils import (
    GMSCommittedMemoryStats,
    get_gms_lock_mode,
    publish_gms_write,
    setup_meta_tensor_workaround,
    strip_gms_model_loader_config,
    torch_device,
)
from gpu_memory_service.integrations.vllm.load_diff import (
    descriptor as load_diff_descriptor,
)
from gpu_memory_service.integrations.vllm.load_diff import writer as load_diff_writer
from gpu_memory_service.integrations.vllm.load_diff.state import (
    snapshot as load_diff_snapshot,
)
from gpu_memory_service.v1.client.parameter_storage import (
    copy_non_parameter_tensors_to_default_allocator,
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
# Storage owners preserve the immutable initial values used by reader metadata.
# They must outlive publication independently of the mutable TensorImpls.
_pending_gms_client: "GMSClientMemoryManager | None" = None
_pending_retained_gms_tensors: list[torch.UntypedStorage] = []
_published_gms_tensor_owners: list[torch.UntypedStorage] = []


@dataclass(frozen=True)
class _StorageSpan:
    """GMS allocation bounds in the shape the shared copy-out helper reads."""

    base: int
    aligned_size: int


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
    retained_gms_tensors: list[torch.UntypedStorage],
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
    retained = list(_pending_retained_gms_tensors)
    gms_client = _take_pending_gms_write()
    if gms_client is None:
        return False

    try:
        # Profiling can release live pool storage after preparation's check.
        raise_if_gms_allocator_failed(gms_client)
        publish_gms_write(gms_client)
        _published_gms_tensor_owners.extend(retained)
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
            device = torch_device().current_device()
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
                    torch.device(get_vmm_device_type().value, device),
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

    With ``gms_reconstruction_mode=load_diff`` the model's Python-side state
    is rebuilt from the writer's load diff instead of by post-processing a
    meta model; see ``load_diff.reader``.

    When MX is active, registers materialized tensors with NIXL so this
    node is discoverable as a P2P source (e.g. for shadow engine failover).
    """
    global _last_imported_weights_bytes, _last_model_memory_usage_offset_bytes

    try:
        if load_diff_writer.selected(vllm_config):
            # Imports vLLM, which this module must not need at import time.
            from gpu_memory_service.integrations.vllm.load_diff.reader import restore

            model = restore(gms_client, vllm_config, model_config, device_index)
        else:
            model = _create_meta_model(vllm_config, model_config)
            materialize_module_from_gms(gms_client, model, device_index=device_index)

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
    try:
        return _prepare_write(
            gms_client, vllm_config, model_config, default_loader, target_device
        )
    except BaseException:
        # Release the RW lease of a write that never became pending, without
        # CUDA cleanup (as abort_pending_gms_write); a pending write belongs
        # to publish/abort.
        if _pending_gms_client is None:
            try:
                gms_client.close(best_effort=True)
            except BaseException:
                logger.exception("[GMS] Failed to release a failed write")
        raise


def _prepare_write(
    gms_client: "GMSClientMemoryManager",
    vllm_config,
    model_config,
    default_loader,
    target_device: torch.device,
) -> torch.nn.Module:
    if _pending_gms_client is not None:
        raise RuntimeError("A GMS write is already awaiting publication")

    from vllm.model_executor.model_loader.utils import (
        initialize_model,
        process_weights_after_loading,
    )
    from vllm.utils.torch_utils import set_default_torch_dtype

    mx_ctx = get_mx_load_context(vllm_config, model_config)
    publish_load_diff = load_diff_writer.selected(vllm_config)
    # Readers check identity before they construct; construction can change
    # the HF config.
    load_diff_identity = (
        load_diff_descriptor.identity(vllm_config, model_config)
        if publish_load_diff
        else None
    )

    # Allocate model tensors using GMS memory pool
    with set_default_torch_dtype(model_config.dtype):
        with gms_use_mem_pool("weights", target_device):
            with target_device:
                model = initialize_model(
                    vllm_config=vllm_config, model_config=model_config
                )
            construct = load_diff_snapshot(model) if publish_load_diff else None

            if mx_ctx is not None:
                # Full MX load strategy chain: RDMA -> ModelStreamer -> GDS -> Default
                LoadStrategyChain.run(model, mx_ctx)
            else:
                default_loader.load_weights(model, model_config)
                process_weights_after_loading(model, model_config, target_device)

            torch_device().empty_cache()

    referenced = register_module_tensors(gms_client, model)
    if publish_load_diff:
        load_diff_writer.publish(gms_client, model, construct, load_diff_identity)
    stats, rebound_bytes, retained_gms_tensors = _copy_out_and_release_pool(
        gms_client, model
    )
    _raise_if_unreferenced_allocations(gms_client, referenced)
    _store_pending_gms_write(gms_client, stats, rebound_bytes, retained_gms_tensors)

    logger.info(
        "[GMS] Write mode: prepared %.2f GiB for publication after profiling "
        "(vLLM memory offset %.2f GiB)",
        _last_imported_weights_bytes / (1 << 30),
        _last_model_memory_usage_offset_bytes / (1 << 30),
    )
    return model.eval()


def _registered_non_parameters(
    gms_client: "GMSClientMemoryManager", model: torch.nn.Module
) -> tuple[list[torch.UntypedStorage], set[int]]:
    """GMS storages behind registered non-Parameter metadata, and tensor ids.

    The metadata references the initial values, so one owner per storage must
    outlive the rebinding of the live tensors, including after commit.
    Returning also drops tensors that properties created during the walk,
    before the heap scan could mistake them for live ones.
    """
    retained: list[torch.UntypedStorage] = []
    seen: set[int] = set()
    in_tree: set[int] = set()
    for _, tensor, kind in _iter_module_tensors(model):
        if kind == "parameter":
            continue
        in_tree.add(int(tensor._cdata))
        storage = tensor.untyped_storage()
        if int(storage._cdata) in seen:
            continue
        if not any(
            va <= tensor.data_ptr() < va + m.aligned_size
            for va, m in gms_client.mappings.items()
        ):
            continue
        seen.add(int(storage._cdata))
        retained.append(storage)
    return retained, in_tree


def _held_non_parameters(model: torch.nn.Module) -> Iterator[torch.Tensor]:
    """Non-Parameter tensors that modules hold directly."""
    for module in model.modules():
        for value in (*module._buffers.values(), *vars(module).values()):
            for item in value if isinstance(value, (list, tuple)) else (value,):
                if torch.is_tensor(item) and not isinstance(item, torch.nn.Parameter):
                    yield item


def _copy_out_and_release_pool(
    gms_client: "GMSClientMemoryManager", model: torch.nn.Module
) -> tuple[GMSCommittedMemoryStats, int, list[torch.UntypedStorage]]:
    """Move live non-Parameters to private memory, then drop the weight pool.

    Runs after registration. Returns the committed/pruned stats, the bytes
    copied out, and the GMS storages that back registered non-Parameter
    metadata (they must outlive publication).
    """
    before_bytes = gms_client.total_bytes
    before_count = len(gms_client.mappings)
    retained_gms_tensors, in_tree = _registered_non_parameters(gms_client, model)
    torch_device().synchronize(gms_client.device)
    # One full collection serves both steps: the heap scan must not find dead
    # cyclic tensors, and their blocks must be free when the pool is dropped.
    gc.collect()
    copied: list[tuple[int, tuple[torch.Tensor, ...]]] = []
    _, rebound_bytes = copy_non_parameter_tensors_to_default_allocator(
        (model,),
        tuple(
            _StorageSpan(base=m.va, aligned_size=m.aligned_size)
            for m in gms_client.mappings.values()
        ),
        collect_garbage=False,
        copied=copied,
        # The heap scan cannot see tensors frozen by gc.freeze(). Frozen
        # tensors held outside the module tree stay in GMS memory.
        known_tensors=_held_non_parameters(model),
    )
    torch_device().synchronize(gms_client.device)
    # Dropping the pool after the copy makes Torch release the blocks the copy
    # freed together with the ones that were never used, so every allocation
    # left holds live storage. empty_cache() cannot do this: GMS workers patch
    # it to do nothing while mappings are live. An allocation that stayed
    # committed with no tensor referencing it would fail a reader's remap.
    release_gms_mem_pool(gms_client, collect_garbage=False)
    stats = GMSCommittedMemoryStats(
        committed_bytes=gms_client.total_bytes,
        pruned_bytes=before_bytes - gms_client.total_bytes,
        pruned_count=before_count - len(gms_client.mappings),
    )
    _log_copy_out(copied, in_tree, stats)
    return stats, rebound_bytes, retained_gms_tensors


def _raise_if_unreferenced_allocations(
    gms_client: "GMSClientMemoryManager", referenced: set[str]
) -> None:
    """Fail the write if an allocation is left that no registered tensor is in.

    A reader maps only the allocations its tensor metadata references, and
    its remap fails if the committed layout has more. Storage that stays in
    GMS unregistered keeps its allocation: a Parameter held outside the
    module tree (only non-Parameters are copied out), or a tensor the
    copy-out could not find.
    """
    unreferenced = [
        mapping
        for mapping in gms_client.mappings.values()
        if mapping.allocation_id not in referenced
    ]
    if unreferenced:
        size = sum(mapping.aligned_size for mapping in unreferenced)
        raise RuntimeError(
            f"[GMS] Unregistered storage keeps {len(unreferenced)} allocation(s) "
            f"({size / (1 << 20):.2f} MiB) alive, for example a Parameter held "
            "outside the module tree. No tensor metadata references them, so a "
            "reader could not remap this layout; it is not published"
        )


def _log_copy_out(
    copied: list[tuple[int, tuple[torch.Tensor, ...]]],
    in_tree: set[int],
    stats: GMSCommittedMemoryStats,
) -> None:
    """Log what the writer copied out of GMS, split by module-tree membership.

    In-tree tensors are the module tree's buffers and tensor attributes;
    out-of-tree tensors are held elsewhere (kernel workspaces, caches). A copy
    that contains any in-tree tensor counts as in-tree bytes.
    """
    counts = [0, 0]
    copied_bytes = [0, 0]
    for nbytes, tensors in copied:
        tree = [int(tensor._cdata) in in_tree for tensor in tensors]
        counts[0] += sum(tree)
        counts[1] += len(tree) - sum(tree)
        copied_bytes[0 if any(tree) else 1] += nbytes
    mib = 1 << 20
    logger.info(
        "[GMS] Write mode: copied %d non-Parameter tensors (%.2f MiB) out of "
        "GMS: in-tree %d (%.2f MiB), out-of-tree %d (%.2f MiB); pruned %d "
        "unused allocations (%.2f GiB)",
        sum(counts),
        sum(copied_bytes) / mib,
        counts[0],
        copied_bytes[0] / mib,
        counts[1],
        copied_bytes[1] / mib,
        stats.pruned_count,
        stats.pruned_bytes / (1 << 30),
    )


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
