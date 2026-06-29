# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM model loader for GPU Memory Service integration.

Provides a model loader that loads weights via GMS for cross-process sharing.
The loader uses RW_OR_RO mode: first process loads from disk (RW), subsequent
processes import from GMS metadata (RO).
"""

from __future__ import annotations

import inspect
import logging
import os
from typing import TYPE_CHECKING

import torch
from gpu_memory_service.client.torch.allocator import (
    get_gms_client_memory_manager,
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


def patch_vllm_worker_memory_accounting() -> None:
    """Avoid double-counting RW GMS weights in vLLM KV memory profiling.

    Spawned vLLM worker processes always import the GMS model loader for
    ``--load-format gms``, but they may not instantiate ``GMSWorker`` in every
    executor path. Patch the base GPU worker here so the accounting fix follows
    the load-format path that is guaranteed to be present in child workers.
    """
    try:
        from vllm.v1.worker.gpu_worker import Worker
    except Exception:
        logger.debug(
            "[GMS] vLLM GPU Worker unavailable for memory patch", exc_info=True
        )
        return

    original = Worker.determine_available_memory
    if getattr(original, "_gms_weight_accounting_patched", False):
        return

    def patched_determine_available_memory(self, *args, **kwargs):
        manager = get_gms_client_memory_manager("weights")
        model_runner = getattr(self, "model_runner", None)
        if (
            manager is None
            or manager.granted_lock_type != GrantedLockType.RW
            or model_runner is None
            or not hasattr(model_runner, "model_memory_usage")
        ):
            return original(self, *args, **kwargs)

        old_usage = int(getattr(model_runner, "model_memory_usage") or 0)
        if old_usage <= 0:
            return original(self, *args, **kwargs)

        logger.info(
            "[GMS] Suppressing %.2f GiB RW GMS weight bytes from vLLM "
            "weights_memory during KV profiling; cudaMemGetInfo already "
            "accounts for them as non-torch memory",
            old_usage / (1 << 30),
        )
        model_runner.model_memory_usage = 0
        try:
            return original(self, *args, **kwargs)
        finally:
            model_runner.model_memory_usage = old_usage

    patched_determine_available_memory._gms_weight_accounting_patched = True
    patched_determine_available_memory._gms_original = original
    Worker.determine_available_memory = patched_determine_available_memory
    logger.info("[GMS] Patched vLLM Worker.determine_available_memory")


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
    patch_vllm_worker_memory_accounting()
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
            # Spawned vLLM workers can import the loader while gpu_worker is still
            # in a circular import. Retry here, after worker initialization and
            # before KV memory profiling.
            patch_vllm_worker_memory_accounting()
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
        target_device = torch.device("cuda", device_index)

        logger.info("[GMS] Read mode: creating meta model")
        model = _create_meta_model(vllm_config, model_config)

        logger.info("[GMS] Read mode: materializing tensors")
        materialize_module_from_gms(gms_client, model, device_index=device_index)

        _refresh_fused_moe_router_tensors_after_gms_materialization(model)
        _process_fused_moe_kernels_after_gms_materialization(
            model,
            model_config,
            target_device,
        )
        _process_mla_weights_after_gms_materialization(
            model,
            model_config,
            target_device,
        )

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
        logger.exception("[GMS] Read mode failed while importing weights")
        gms_client.close(best_effort=True)
        raise


def _is_mla_post_load_module(module: torch.nn.Module) -> bool:
    return (
        hasattr(module, "kv_b_proj")
        and hasattr(module, "kv_lora_rank")
        and hasattr(module, "num_heads")
        and callable(getattr(module, "process_weights_after_loading", None))
    )


def _call_with_supported_kwargs(factory, **kwargs):
    signature = inspect.signature(factory)
    supported_kwargs = {
        name: value for name, value in kwargs.items() if name in signature.parameters
    }
    return factory(**supported_kwargs)


def _is_meta_tensor(value) -> bool:
    return isinstance(value, torch.Tensor) and value.is_meta


def _refresh_fused_moe_router_tensors_after_gms_materialization(
    model: torch.nn.Module,
) -> None:
    """Refresh vLLM FusedMoE tensor references captured from a meta model.

    Some MoE constructors keep non-owning references to gate tensors in the
    FusedMoE layer and router. GMS RO mode materializes the owning gate
    parameters/buffers after construction, so these cached references must be
    rebound before vLLM's profile run executes routing kernels.
    """

    refreshed: list[str] = []
    stale_meta: list[str] = []

    for name, module in model.named_modules():
        gate = getattr(module, "gate", None)
        experts = getattr(module, "experts", None)
        if gate is None or experts is None:
            continue

        e_score_correction_bias = getattr(gate, "e_score_correction_bias", None)
        hash_indices_table = getattr(gate, "tid2eid", None)
        router = getattr(experts, "router", None)

        changed = False
        if hasattr(experts, "e_score_correction_bias"):
            if _is_meta_tensor(getattr(experts, "e_score_correction_bias", None)):
                stale_meta.append(f"{name}.experts.e_score_correction_bias")
            experts.e_score_correction_bias = e_score_correction_bias
            changed = True
        if hasattr(experts, "hash_indices_table"):
            if _is_meta_tensor(getattr(experts, "hash_indices_table", None)):
                stale_meta.append(f"{name}.experts.hash_indices_table")
            experts.hash_indices_table = hash_indices_table
            changed = True
        if router is not None and hasattr(router, "e_score_correction_bias"):
            if _is_meta_tensor(getattr(router, "e_score_correction_bias", None)):
                stale_meta.append(f"{name}.experts.router.e_score_correction_bias")
            router.e_score_correction_bias = e_score_correction_bias
            changed = True
        if router is not None and hasattr(router, "_hash_indices_table"):
            if _is_meta_tensor(getattr(router, "_hash_indices_table", None)):
                stale_meta.append(f"{name}.experts.router._hash_indices_table")
            router._hash_indices_table = hash_indices_table
            changed = True

        runner = getattr(experts, "runner", None)
        if runner is not None and getattr(runner, "router", None) is not router:
            runner.router = router
            changed = True

        if changed:
            refreshed.append(name)

    if refreshed:
        logger.info(
            "[GMS] Read mode: refreshed %d FusedMoE router tensor references: %s",
            len(refreshed),
            refreshed[:8],
        )
    if stale_meta:
        logger.info(
            "[GMS] Read mode: replaced stale meta FusedMoE router tensors: %s",
            stale_meta[:16],
        )


def _make_fused_moe_kernel(module: torch.nn.Module, quant_method) -> bool:
    experts_cls = getattr(quant_method, "experts_cls", None)
    if experts_cls is None:
        return False

    quant_config = quant_method.get_fused_moe_quant_config(module)
    if quant_config is None:
        return False
    quant_method.moe_quant_config = quant_config

    routing_tables = None
    maybe_routing_tables = getattr(module, "_maybe_init_expert_routing_tables", None)
    if callable(maybe_routing_tables):
        routing_tables = maybe_routing_tables()
    shared_experts = getattr(module, "shared_experts", None)

    if hasattr(quant_method, "fp8_backend"):
        from vllm.model_executor.layers.fused_moe.oracle.fp8 import make_fp8_moe_kernel

        quant_method.moe_kernel = _call_with_supported_kwargs(
            make_fp8_moe_kernel,
            moe_quant_config=quant_config,
            moe_config=module.moe_config,
            fp8_backend=quant_method.fp8_backend,
            experts_cls=experts_cls,
            routing_tables=routing_tables,
            shared_experts=shared_experts,
            layer=module,
        )
        return True

    if hasattr(quant_method, "mxfp4_backend"):
        from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
            make_mxfp4_moe_kernel,
        )

        quant_method.moe_kernel = _call_with_supported_kwargs(
            make_mxfp4_moe_kernel,
            moe_quant_config=quant_config,
            moe_config=module.moe_config,
            mxfp4_backend=quant_method.mxfp4_backend,
            experts_cls=experts_cls,
            routing_tables=routing_tables,
            shared_experts=shared_experts,
            layer=module,
        )
        return True

    if hasattr(quant_method, "nvfp4_backend"):
        from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
            make_nvfp4_moe_kernel,
        )

        quant_method.moe_kernel = _call_with_supported_kwargs(
            make_nvfp4_moe_kernel,
            moe_quant_config=quant_config,
            moe_config=module.moe_config,
            experts_cls=experts_cls,
            routing_tables=routing_tables,
            shared_experts=shared_experts,
            layer=module,
        )
        return True

    if hasattr(quant_method, "unquantized_backend"):
        from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
            make_unquantized_moe_kernel,
        )

        quant_method.moe_kernel = _call_with_supported_kwargs(
            make_unquantized_moe_kernel,
            quant_config=quant_config,
            moe_config=module.moe_config,
            backend=quant_method.unquantized_backend,
            experts_cls=experts_cls,
            routing_tables=routing_tables,
            shared_experts=shared_experts,
            layer=module,
        )
        return True

    return False


def _process_fused_moe_kernels_after_gms_materialization(
    model: torch.nn.Module,
    model_config,
    target_device: torch.device,
) -> None:
    """Rebuild vLLM MoE runtime kernels around imported GMS weights."""
    from vllm.utils.torch_utils import set_default_torch_dtype

    rebuilt: list[str] = []
    with set_default_torch_dtype(model_config.dtype):
        with target_device:
            for name, module in model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is None:
                    continue
                if getattr(quant_method, "moe_kernel", None) is not None:
                    continue
                if not callable(
                    getattr(quant_method, "get_fused_moe_quant_config", None)
                ):
                    continue
                if not hasattr(module, "moe_config"):
                    continue
                if _make_fused_moe_kernel(module, quant_method):
                    rebuilt.append(name)

    if rebuilt:
        logger.info(
            "[GMS] Read mode: rebuilt %d FusedMoE kernels: %s",
            len(rebuilt),
            rebuilt[:8],
        )


def _process_mla_weights_after_gms_materialization(
    model: torch.nn.Module,
    model_config,
    target_device: torch.device,
) -> None:
    """Rebuild derived MLA projection tensors skipped from GMS metadata."""
    from vllm.utils.torch_utils import set_default_torch_dtype

    processed: list[str] = []
    with set_default_torch_dtype(model_config.dtype):
        with target_device:
            for name, module in model.named_modules():
                if not _is_mla_post_load_module(module):
                    continue
                module.process_weights_after_loading(model_config.dtype)
                processed.append(name)

    if processed:
        logger.info(
            "[GMS] Read mode: rebuilt %d MLA post-load modules: %s",
            len(processed),
            processed[:8],
        )


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
    from vllm.model_executor.model_loader.utils import initialize_model
    from vllm.utils.torch_utils import set_default_torch_dtype

    setup_meta_tensor_workaround()
    meta_device = torch.device("meta")

    with set_default_torch_dtype(model_config.dtype):
        with meta_device:
            model = initialize_model(vllm_config=vllm_config, model_config=model_config)

    # Do not run vLLM post-load hooks on the RO meta model. Some DSV4
    # quantization and attention hooks initialize CUDA runtime state even when
    # tensors are still meta. GMS imports the writer's final parameter metadata
    # first, then rebuilds supported derived runtime state on the target CUDA
    # device in _load_read_mode().

    return model
