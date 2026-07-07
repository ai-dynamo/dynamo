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
from typing import TYPE_CHECKING, Callable, TypeVar

import torch
from gpu_memory_service.client.torch.allocator import (
    get_gms_client_memory_manager,
    get_or_create_gms_client_memory_manager,
    gms_use_mem_pool,
)
from gpu_memory_service.client.torch.module import materialize_module_from_gms
from gpu_memory_service.common.locks import GrantedLockType
from gpu_memory_service.common.utils import get_socket_path
from gpu_memory_service.integrations.common.utils import (
    PreparedGMSWrite,
    get_gms_lock_mode,
    prepare_gms_write,
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
_pending_gms_write: PreparedGMSWrite | None = None
_T = TypeVar("_T")


def get_imported_weights_bytes() -> int:
    """Return bytes of weights imported in the last load_model call."""
    return _last_imported_weights_bytes


def get_model_memory_usage_offset_bytes() -> int:
    """Return the offset to add to imported bytes for vLLM model_memory_usage."""
    return _last_model_memory_usage_offset_bytes


def has_pending_gms_write() -> bool:
    """Return whether this process still owns an unpublished GMS write."""
    return _pending_gms_write is not None


def _store_pending_gms_write(prepared: PreparedGMSWrite) -> None:
    global _last_imported_weights_bytes, _last_model_memory_usage_offset_bytes
    global _pending_gms_write

    if _pending_gms_write is not None:
        raise RuntimeError("A GMS write is already awaiting publication")
    _pending_gms_write = prepared
    _last_imported_weights_bytes = prepared.stats.committed_bytes
    _last_model_memory_usage_offset_bytes = (
        prepared.stats.pruned_bytes + prepared.rebound_bytes
    )


def publish_pending_gms_write() -> bool:
    """Publish and clear the pending vLLM first-writer state, if any."""
    global _pending_gms_write

    prepared = _pending_gms_write
    if prepared is None:
        return False
    _pending_gms_write = None

    try:
        prepared.publish()
    except BaseException:
        try:
            prepared.abort()
        except BaseException:
            logger.exception("[GMS] Failed to close a failed pending write")
        raise

    logger.info(
        "[GMS] Published %.2f GiB after vLLM memory profiling and switched "
        "to read mode",
        prepared.stats.committed_bytes / (1 << 30),
    )
    return True


def abort_pending_gms_write() -> bool:
    """Abort and clear the pending vLLM first-writer state, if any."""
    global _pending_gms_write

    prepared = _pending_gms_write
    if prepared is None:
        return False
    _pending_gms_write = None
    prepared.abort()
    return True


def _abort_pending_gms_write_without_masking(context: str) -> None:
    try:
        if abort_pending_gms_write():
            return
    except BaseException:
        logger.exception("[GMS] Failed to release pending write %s", context)
        # Ownership was already detached from the pending slot. Do not also
        # close through the registry and risk cleaning up the same client twice.
        return

    gms_client = get_gms_client_memory_manager("weights")
    if gms_client is None or gms_client.granted_lock_type != GrantedLockType.RW:
        return
    try:
        gms_client.close(best_effort=True)
    except BaseException:
        logger.exception("[GMS] Failed to release original write %s", context)


def _all_ranks_succeeded(local_success: bool) -> bool:
    """Reduce phase status over vLLM's participating world group."""
    distributed = torch.distributed
    if not distributed.is_available() or not distributed.is_initialized():
        return local_success

    world_size = distributed.get_world_size()
    if world_size == 1:
        return local_success

    try:
        from vllm.distributed.parallel_state import get_world_group

        world_group = get_world_group()
    except (AssertionError, ImportError) as exc:
        raise RuntimeError(
            "Cannot coordinate GMS phase: vLLM world group is not initialized"
        ) from exc

    status = torch.tensor(int(local_success), dtype=torch.int32, device="cpu")
    distributed.all_reduce(
        status,
        op=distributed.ReduceOp.MIN,
        group=world_group.cpu_group,
    )
    return bool(status.item())


def _run_coordinated_gms_phase(
    phase: Callable[[], _T],
    *,
    phase_name: str,
    coordinate: Callable[[bool], bool],
) -> _T:
    """Run a fallible phase, exchange status, then clean up and raise."""
    result: _T
    local_error: BaseException | None = None
    try:
        result = phase()
    except BaseException as exc:
        # Every rank must reach the same collective even after a local error.
        local_error = exc

    try:
        all_success = coordinate(local_error is None)
    except BaseException:
        _abort_pending_gms_write_without_masking(
            f"after {phase_name} coordination failure"
        )
        if local_error is not None:
            raise local_error.with_traceback(local_error.__traceback__)
        raise

    if local_error is not None:
        _abort_pending_gms_write_without_masking(f"after {phase_name} failure")
        raise local_error.with_traceback(local_error.__traceback__)

    if not all_success:
        _abort_pending_gms_write_without_masking(f"after peer {phase_name} failure")
        raise RuntimeError(f"GMS {phase_name} failed on a peer rank")

    return result


def run_gms_load_phase(
    load: Callable[[], _T],
    *,
    coordinate: Callable[[bool], bool] = _all_ranks_succeeded,
) -> _T:
    """Coordinate complete vLLM model loading before any rank returns."""
    return _run_coordinated_gms_phase(
        load,
        phase_name="model loading",
        coordinate=coordinate,
    )


def profile_before_gms_write_publication(
    profile: Callable[[], _T],
    *,
    coordinate: Callable[[bool], bool] = _all_ranks_succeeded,
) -> _T:
    """Profile locally, coordinate success across ranks, then publish."""
    result = _run_coordinated_gms_phase(
        profile,
        phase_name="profiling",
        coordinate=coordinate,
    )
    publish_pending_gms_write()
    return result


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

    Initializes model using GMS memory pool, loads weights from disk, then
    prepares a write for publication after vLLM memory profiling.

    When MX is active, uses LoadStrategyChain for automatic weight source
    detection (RDMA P2P -> ModelStreamer -> GDS -> disk) with fallback.
    The chain also handles NIXL registration and metadata publishing.
    """
    if _pending_gms_write is not None:
        raise RuntimeError("A GMS write is already awaiting publication")

    prepared: PreparedGMSWrite | None = None
    try:
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

        prepared = prepare_gms_write(gms_client, model)
        # These private clones must exist before vLLM profiles memory or
        # captures graphs. Their registered GMS copies remain for readers.
        prepared.rebind_nonparameter_tensors()
        model.eval()
        _store_pending_gms_write(prepared)

        logger.info(
            "[GMS] Write mode: prepared %.2f GiB for publication after profiling "
            "(vLLM memory offset %.2f GiB)",
            _last_imported_weights_bytes / (1 << 30),
            _last_model_memory_usage_offset_bytes / (1 << 30),
        )
        return model
    except BaseException:
        if prepared is not None and _pending_gms_write is prepared:
            _abort_pending_gms_write_without_masking("during model loading")
        else:
            try:
                gms_client.close(best_effort=True)
            except BaseException:
                logger.exception("[GMS] Failed to close write-mode model loader")
        raise


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
