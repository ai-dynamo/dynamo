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
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from gpu_memory_service.client.torch.allocator import (
    get_or_create_gms_client_memory_manager,
    gms_use_mem_pool,
)
from gpu_memory_service.client.torch.module import (
    _swap_tensor_contents,
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

_MLA_DERIVED_TENSOR_ATTRS = (
    "W_UK_T",
    "W_UV",
    "W_K",
    "W_K_scale",
    "W_V",
    "W_V_scale",
)


@dataclass(frozen=True)
class _MLANormalizationStats:
    private_bytes: int
    source_allocation_ids: frozenset[str]


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
    memory_usage_offset_bytes: int,
    retained_gms_tensors: list[torch.Tensor],
) -> None:
    global _last_imported_weights_bytes, _last_model_memory_usage_offset_bytes
    global _pending_gms_client, _pending_retained_gms_tensors

    if _pending_gms_client is not None:
        raise RuntimeError("A GMS write is already awaiting publication")
    _pending_gms_client = gms_client
    _pending_retained_gms_tensors = retained_gms_tensors
    _last_imported_weights_bytes = stats.committed_bytes
    _last_model_memory_usage_offset_bytes = memory_usage_offset_bytes


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
        model = _create_meta_model(vllm_config, model_config)
        mla_modules, cleared_attrs = _clear_mla_derived_tensors(model)
        materialize_module_from_gms(gms_client, model, device_index=device_index)
        _process_mla_weights_after_gms_materialization(
            mla_modules,
            model_config,
            torch.device("cuda", device_index),
        )
        mla_private_bytes = _mla_private_storage_bytes(gms_client, mla_modules)
        if mla_modules:
            logger.info(
                "[GMS] Read mode: rebuilt %d MLA modules after clearing %d "
                "derived tensor attrs",
                len(mla_modules),
                cleared_attrs,
            )

        # MX: register materialized tensors (available for P2P transfer)
        mx_ctx = get_mx_load_context(vllm_config, model_config)
        if mx_ctx is not None:
            register_tensors(model, mx_ctx)
            publish_metadata(mx_ctx)

        _last_imported_weights_bytes = gms_client.total_bytes
        _last_model_memory_usage_offset_bytes = mla_private_bytes
        logger.info(
            "[GMS] Read mode: imported %.2f GiB with %.2f MiB of private "
            "MLA tensors",
            _last_imported_weights_bytes / (1 << 30),
            mla_private_bytes / (1 << 20),
        )
        return model.eval()
    except Exception:
        gms_client.close()
        raise


def _is_mla_post_load_module(module: torch.nn.Module) -> bool:
    """Identify vLLM MLA modules without depending on their import location."""
    return (
        hasattr(module, "kv_b_proj")
        and hasattr(module, "kv_lora_rank")
        and hasattr(module, "num_heads")
        and callable(getattr(module, "process_weights_after_loading", None))
    )


def _mla_modules(
    model: torch.nn.Module,
) -> list[tuple[str, torch.nn.Module]]:
    """Capture every strict MLA module in model traversal order."""
    return [
        (module_name, module)
        for module_name, module in model.named_modules()
        if _is_mla_post_load_module(module)
    ]


def _mla_derived_tensors(
    mla_modules: list[tuple[str, torch.nn.Module]],
) -> list[tuple[str, torch.Tensor]]:
    """Return unique known derived tensors without traversing helper objects."""
    seen: set[int] = set()
    tensors = []
    for module_name, module in mla_modules:
        for attr_name in _MLA_DERIVED_TENSOR_ATTRS:
            tensor = module.__dict__.get(attr_name)
            if not torch.is_tensor(tensor) or id(tensor) in seen:
                continue
            seen.add(id(tensor))
            qualified_name = f"{module_name}.{attr_name}" if module_name else attr_name
            tensors.append((qualified_name, tensor))
    return tensors


def _storage_key(tensor: torch.Tensor) -> tuple[torch.device, int, int]:
    storage = tensor.untyped_storage()
    return storage.device, int(storage.data_ptr()), storage.nbytes()


def _mla_private_storage_bytes(
    gms_client: "GMSClientMemoryManager",
    mla_modules: list[tuple[str, torch.nn.Module]],
) -> int:
    """Count unique live MLA storage not backed by parameters or GMS."""
    parameter_storage = {
        _storage_key(parameter)
        for _name, module in mla_modules
        for parameter in module.parameters()
    }
    private_storage: dict[tuple[torch.device, int, int], int] = {}
    for _name, tensor in _mla_derived_tensors(mla_modules):
        storage = tensor.untyped_storage()
        if storage.nbytes() == 0:
            continue
        storage_key = _storage_key(tensor)
        if storage_key in parameter_storage or _is_gms_mapped_tensor(
            gms_client, tensor
        ):
            continue
        private_storage[storage_key] = storage.nbytes()
    return sum(private_storage.values())


def _is_gms_mapped_tensor(
    gms_client: "GMSClientMemoryManager",
    tensor: torch.Tensor,
) -> bool:
    ptr = int(tensor.data_ptr())
    return any(
        va <= ptr < va + mapping.aligned_size
        for va, mapping in gms_client.mappings.items()
    )


def _normalize_mla_derived_tensors(
    gms_client: "GMSClientMemoryManager",
    model: torch.nn.Module,
) -> _MLANormalizationStats:
    """Move writer MLA post-load views to ordinary identity-preserving clones.

    ``torch.preserve_format`` retains strides for dense tensors. PyTorch
    normalizes non-dense split views because they cannot preserve their layout
    in an independent allocation.
    """
    prepared: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    mla_modules = _mla_modules(model)
    source_allocation_ids = set()
    for qualified_name, tensor in _mla_derived_tensors(mla_modules):
        for va, mapping in gms_client.mappings.items():
            if va <= int(tensor.data_ptr()) < va + mapping.aligned_size:
                source_allocation_ids.add(mapping.allocation_id)
                break
        private = tensor.detach().clone(memory_format=torch.preserve_format)
        prepared.append((qualified_name, tensor, private))

    completed: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    try:
        for name, tensor, private in prepared:
            _swap_tensor_contents(tensor, private, name=name)
            completed.append((name, tensor, private))

        for name, tensor, _private in completed:
            if tensor._base is not None or _is_gms_mapped_tensor(gms_client, tensor):
                raise RuntimeError(
                    f"MLA derived tensor {name!r} did not move to private storage"
                )
    except BaseException as swap_error:
        rollback_error: BaseException | None = None
        for name, tensor, private in reversed(completed):
            try:
                _swap_tensor_contents(tensor, private, name=name)
            except BaseException as exc:
                rollback_error = rollback_error or exc
        if rollback_error is not None:
            raise RuntimeError(
                "GMS MLA tensor normalization failed and rollback was incomplete"
            ) from rollback_error
        raise swap_error

    # The replacement objects now own the displaced GMS views. Do not retain
    # them: temporary dequantization allocations should be pruned, while any
    # parameter-backed allocation remains retained through parameter metadata.
    prepared.clear()
    completed.clear()
    return _MLANormalizationStats(
        private_bytes=_mla_private_storage_bytes(gms_client, mla_modules),
        source_allocation_ids=frozenset(source_allocation_ids),
    )


def _clear_mla_derived_tensors(
    model: torch.nn.Module,
) -> tuple[list[tuple[str, torch.nn.Module]], int]:
    """Release known meta MLA views and their direct exact-object aliases."""
    modules = _mla_modules(model)
    derived_tensors = {
        id(tensor): tensor
        for _module_name, module in modules
        for attr_name in _MLA_DERIVED_TENSOR_ATTRS
        if torch.is_tensor(tensor := module.__dict__.get(attr_name))
    }
    aliases = [
        (module, attr_name)
        for module in model.modules()
        for attr_name, value in module.__dict__.items()
        if torch.is_tensor(value) and derived_tensors.get(id(value)) is value
    ]
    known_attrs = [
        (module, attr_name)
        for _module_name, module in modules
        for attr_name in _MLA_DERIVED_TENSOR_ATTRS
        if attr_name in module.__dict__
    ]
    cleared_attrs = set()
    for module, attr_name in known_attrs + aliases:
        key = id(module), attr_name
        if key in cleared_attrs:
            continue
        del module.__dict__[attr_name]
        cleared_attrs.add(key)
    return modules, len(cleared_attrs)


def _pruned_mla_source_bytes(
    allocation_sizes_before_prune: dict[str, int],
    retained_allocation_ids: set[str],
    source_allocation_ids: frozenset[str],
) -> int:
    """Count MLA source allocations removed by writer normalization."""
    return sum(
        size
        for allocation_id, size in allocation_sizes_before_prune.items()
        if (
            allocation_id in source_allocation_ids
            and allocation_id not in retained_allocation_ids
        )
    )


def _process_mla_weights_after_gms_materialization(
    mla_modules: list[tuple[str, torch.nn.Module]],
    model_config,
    target_device: torch.device,
) -> None:
    """Rebuild captured MLA state after registered weights materialize."""
    from vllm.utils.torch_utils import set_default_torch_dtype

    with set_default_torch_dtype(model_config.dtype):
        with target_device:
            for module_name, module in mla_modules:
                try:
                    module.process_weights_after_loading(model_config.dtype)
                except Exception as exc:
                    name = module_name or "<root>"
                    raise RuntimeError(
                        f"Failed to rebuild MLA post-load tensors for {name!r}"
                    ) from exc


def _load_write_mode(
    gms_client: "GMSClientMemoryManager",
    vllm_config,
    model_config,
    default_loader,
    target_device: torch.device,
) -> torch.nn.Module:
    """Load a writer model, releasing its lease on every failed preparation."""
    try:
        return _load_write_mode_impl(
            gms_client,
            vllm_config,
            model_config,
            default_loader,
            target_device,
        )
    except BaseException:
        try:
            if _pending_gms_client is gms_client:
                abort_pending_gms_write()
            else:
                gms_client.close(best_effort=True)
        except BaseException:
            logger.exception("[GMS] Failed to release failed write-mode load")
        raise


def _load_write_mode_impl(
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

    mla_stats = _normalize_mla_derived_tensors(gms_client, model)
    allocation_sizes_before_prune = {
        mapping.allocation_id: mapping.aligned_size
        for mapping in gms_client.mappings.values()
    }
    stats = prepare_gms_write(gms_client, model)
    pruned_mla_source_bytes = _pruned_mla_source_bytes(
        allocation_sizes_before_prune,
        {mapping.allocation_id for mapping in gms_client.mappings.values()},
        mla_stats.source_allocation_ids,
    )
    # The private clones must exist before vLLM profiles memory so the
    # profiled peak covers them. The retained GMS originals stay alive until
    # commit, for readers to materialize from.
    retained_gms_tensors: list[torch.Tensor] = []
    rebound_bytes = rebind_nonparameter_tensors(
        gms_client, model, retain_gms_tensors=retained_gms_tensors
    )
    private_bytes = mla_stats.private_bytes + rebound_bytes
    # imported + offset represents vLLM's pre-prune allocation accounting,
    # replacing any pruned MLA source allocation with its live private cache:
    # offset = pruned - pruned_mla_source + private.
    memory_usage_offset_bytes = (
        stats.pruned_bytes - pruned_mla_source_bytes + private_bytes
    )
    _store_pending_gms_write(
        gms_client,
        stats,
        memory_usage_offset_bytes,
        retained_gms_tensors,
    )

    logger.info(
        "[GMS] Write mode: prepared %.2f GiB for publication after profiling "
        "(vLLM memory offset %.2f GiB; private MLA tensors %.2f MiB)",
        _last_imported_weights_bytes / (1 << 30),
        _last_model_memory_usage_offset_bytes / (1 << 30),
        mla_stats.private_bytes / (1 << 20),
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
