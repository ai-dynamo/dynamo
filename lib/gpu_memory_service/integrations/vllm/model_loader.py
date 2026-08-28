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
    get_or_create_gms_client_memory_manager,
    gms_use_mem_pool,
)
from gpu_memory_service.client.torch.module import (
    _resolve_module_attr,
    copy_unmapped_tensors_into_gms,
    repoint_non_module_tensor_caches,
    materialize_module_from_gms,
    rebind_nonparameter_tensors,
)
from gpu_memory_service.common.locks import GrantedLockType
from gpu_memory_service.common.utils import get_socket_path, is_truthy_env
from gpu_memory_service.integrations.common.utils import (
    GMSCommittedMemoryStats,
    get_gms_lock_mode,
    prepare_gms_write,
    publish_gms_write,
    setup_meta_tensor_workaround,
    strip_gms_model_loader_config,
)
from gpu_memory_service.integrations.vllm.upstream_workarounds import (
    vllm_meta_init_workarounds,
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
        torch.cuda.synchronize()
        logger.info("[GMS] Read mode: cuda sync ok after materialize")

        # Rebuild vLLM runtime helpers that the RO meta constructor skipped.
        # These are best-effort: vLLM internals move, and a missing helper
        # should not take down an otherwise materialized shadow.
        try:
            _process_fused_moe_kernels_after_gms_materialization(
                model, model_config, target_device
            )
            torch.cuda.synchronize()
            logger.info("[GMS] Read mode: cuda sync ok after FusedMoE rebuild")
        except Exception:
            logger.exception(
                "[GMS] Read mode: FusedMoE kernel rebuild failed; continuing"
            )
        # Re-deriving MLA W_UK_T/W_UV on the reader means running vLLM's
        # NVFP4 dequant, which for this quant method is a full GEMM through
        # `quant_method.apply(layer, eye)`. That kernel illegal-addresses
        # against GMS VMM mappings and takes engine startup down with it, so
        # it stays opt-in until the writer publishes the derived tensors.
        if is_truthy_env("DYN_GMS_RO_MLA_POSTLOAD"):
            try:
                _process_mla_weights_after_gms_materialization(
                    model, model_config, target_device
                )
                torch.cuda.synchronize()
                logger.info("[GMS] Read mode: cuda sync ok after MLA post-load")
            except Exception:
                logger.exception(
                    "[GMS] Read mode: MLA post-load failed; continuing with "
                    "the published tensors (expect degraded output)"
                )
        else:
            logger.warning(
                "[GMS] Read mode: skipping MLA process_weights_after_loading "
                "(GMS VMM IMA in NVFP4 dequant GEMM); set "
                "DYN_GMS_RO_MLA_POSTLOAD=1 to attempt it"
            )
        try:
            _rebuild_fp8_linear_kernels_after_gms_materialization(
                model, target_device
            )
            torch.cuda.synchronize()
            logger.info("[GMS] Read mode: cuda sync ok after fp8_linear rebuild")
        except Exception:
            logger.exception(
                "[GMS] Read mode: fp8_linear kernel rebuild failed; continuing"
            )
        # Not best-effort: the Triton `dsv4_topk` router loads
        # `correction_bias` by pointer and illegal-addresses against a VMM
        # mapping, so a failure here means the first request takes the
        # engine down. Fail the load instead.
        _clone_triton_incompatible_params_off_gms(model)
        torch.cuda.synchronize()
        logger.info("[GMS] Read mode: cuda sync ok after triton-param clone")

        # Last, after every pass that rebinds module tensors by name. The
        # FusedMoE rebuild and the Triton clone both replace tensors --
        # `expert_map` among them -- which re-breaks the aliases plain
        # holder objects keep. Repairing any earlier would be undone here.
        repoint_non_module_tensor_caches(model)

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
            # process_weights_after_loading allocates MLA q/k/v/prob scales
            # outside the GMS pool. Copy them in before register/publish so
            # RO import does not zero-fill 312 leftover Parameters.
            copy_unmapped_tensors_into_gms(gms_client, model)

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


def _is_mla_post_load_module(module: torch.nn.Module) -> bool:
    return (
        hasattr(module, "kv_b_proj")
        and hasattr(module, "kv_lora_rank")
        and hasattr(module, "num_heads")
        and callable(getattr(module, "process_weights_after_loading", None))
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

    kernel_makers = (
        (
            "fp8_backend",
            "vllm.model_executor.layers.fused_moe.oracle.fp8",
            "make_fp8_moe_kernel",
        ),
        (
            "mxfp4_backend",
            "vllm.model_executor.layers.fused_moe.oracle.mxfp4",
            "make_mxfp4_moe_kernel",
        ),
        (
            "nvfp4_backend",
            "vllm.model_executor.layers.fused_moe.oracle.nvfp4",
            "make_nvfp4_moe_kernel",
        ),
        (
            "unquantized_backend",
            "vllm.model_executor.layers.fused_moe.oracle.unquantized",
            "make_unquantized_moe_kernel",
        ),
    )
    for attr, module_name, fn_name in kernel_makers:
        if not hasattr(quant_method, attr):
            continue
        try:
            maker = getattr(__import__(module_name, fromlist=[fn_name]), fn_name)
        except (ImportError, AttributeError):
            continue
        kwargs = {
            "moe_quant_config": quant_config,
            "quant_config": quant_config,
            "moe_config": module.moe_config,
            attr: getattr(quant_method, attr),
            "backend": getattr(quant_method, attr),
            "experts_cls": experts_cls,
            "routing_tables": routing_tables,
            "shared_experts": shared_experts,
        }
        accepted = set(inspect.signature(maker).parameters)
        try:
            quant_method.moe_kernel = maker(
                **{k: v for k, v in kwargs.items() if k in accepted}
            )
            return True
        except TypeError:
            continue
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


def _clone_module_tensors_off_gms(module: torch.nn.Module | None) -> None:
    """Move ``module``'s CUDA parameters/buffers off GMS VMM in place.

    Must not run under ``gms_use_mem_pool`` or the clones stay on VMM.
    """
    if module is None:
        return
    for name, param in list(module.named_parameters(recurse=True)):
        if param is None or param.is_meta or not param.is_cuda:
            continue
        mod, attr = _resolve_module_attr(module, name)
        mod._parameters[attr] = torch.nn.Parameter(
            param.detach().contiguous().clone(),
            requires_grad=param.requires_grad,
        )
    for name, buf in list(module.named_buffers(recurse=True)):
        if buf is None or buf.is_meta or not buf.is_cuda:
            continue
        mod, attr = _resolve_module_attr(module, name)
        mod._buffers[attr] = buf.detach().contiguous().clone()


def _process_mla_weights_after_gms_materialization(
    model: torch.nn.Module,
    model_config,
    target_device: torch.device,
) -> None:
    """Re-derive MLA projections (``W_UK_T``/``W_UV``) on the RO path.

    vLLM computes these by dequantizing ``kv_b_proj``; they exist only after
    ``process_weights_after_loading`` runs, so a reader that skips it serves
    whatever the writer happened to publish.

    Two GMS-specific hazards are handled here:

    * The dequant kernels address ``kv_b_proj`` by pointer, and CUDA VMM
      mappings illegal-address on B200 (same failure mode as Triton
      ``dsv4_topk``; see ``_clone_triton_incompatible_params_off_gms``).
      Its tensors are cloned into ordinary CUDA memory first.
    * ``materialize_module_from_gms`` zero-fills the leftover ``q_scale`` /
      ``k_scale`` / ``v_scale`` / ``prob_scale`` staging Parameters. Zero is
      neither the ">0 loaded" nor the "<0 absent" sentinel, so the KV-cache
      quant method asserts. Deleting them reproduces the reference path,
      which for a checkpoint carrying no attention scales resolves to 1.0.
    """
    from vllm.utils.torch_utils import set_default_torch_dtype

    processed: list[str] = []
    with set_default_torch_dtype(model_config.dtype):
        with target_device:
            for name, module in model.named_modules():
                if not _is_mla_post_load_module(module):
                    continue
                _clone_module_tensors_off_gms(getattr(module, "kv_b_proj", None))
                for leaf in ("q_scale", "k_scale", "v_scale", "prob_scale"):
                    if isinstance(
                        getattr(module, leaf, None), torch.nn.Parameter
                    ):
                        delattr(module, leaf)
                module.process_weights_after_loading(model_config.dtype)
                processed.append(name)

    if processed:
        logger.info(
            "[GMS] Read mode: rebuilt %d MLA post-load modules: %s",
            len(processed),
            processed[:8],
        )


def _rebuild_fp8_linear_kernels_after_gms_materialization(
    model: torch.nn.Module,
    target_device: torch.device,
) -> None:
    """Re-select FP8 GEMM kernels on CUDA after RO GMS materialize.

    Meta-device construction can pick a non-DeepGEMM kernel (no compute
    capability). Packed GMS weights then hit the wrong apply path and IMA.
    Skip re-packing: GMS already holds the writer's processed tensors.
    """
    from vllm.model_executor.kernels.linear import init_fp8_linear_kernel

    rebuilt: list[str] = []
    with target_device:
        for name, module in model.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is None or not hasattr(quant_method, "fp8_linear"):
                continue
            weight = getattr(module, "weight", None)
            if not torch.is_tensor(weight) or weight.is_meta:
                continue
            shape = tuple(weight.shape)
            if len(shape) > 2:
                # is_bmm packed layout (g, r, d) -> DeepGEMM config wants 2D
                shape = (int(weight.numel() // shape[-1]), int(shape[-1]))
            keys = (
                "activation_quant_key",
                "weight_quant_key",
                "input_dtype",
                "out_dtype",
            )
            if any(not hasattr(quant_method, k) for k in keys):
                continue
            try:
                kernel = init_fp8_linear_kernel(
                    activation_quant_key=quant_method.activation_quant_key,
                    weight_quant_key=quant_method.weight_quant_key,
                    input_dtype=quant_method.input_dtype,
                    out_dtype=quant_method.out_dtype,
                    weight_shape=shape,
                    module_name=type(quant_method).__name__,
                )
            except Exception:
                logger.debug(
                    "[GMS] fp8_linear rebuild skipped for %s", name, exc_info=True
                )
                continue
            quant_method.fp8_linear = kernel
            rebuilt.append(f"{name}:{type(kernel).__name__}")
    if rebuilt:
        logger.info(
            "[GMS] Read mode: rebuilt %d fp8_linear kernels: %s",
            len(rebuilt),
            rebuilt[:8],
        )


def _clone_triton_incompatible_params_off_gms(model: torch.nn.Module) -> None:
    """Copy tiny router/bias params off GMS VMM into ordinary CUDA memory.

    Triton ``dsv4_topk`` loads ``correction_bias`` by pointer. CUDA VMM
    mappings are not Triton-loadable and illegal-address on B200.
    """
    cloned: list[str] = []
    # vLLM aliases some of these across modules -- `e_score_correction_bias`
    # is one tensor referenced by both `mlp.gate` and
    # `mlp.experts.routed_experts`. Cloning each name separately would hand
    # the router and the experts different copies of the routing bias, so
    # keep one clone per source storage and reuse it.
    by_storage: dict[tuple[int, int], torch.Tensor] = {}

    def _off_gms(tensor: torch.Tensor) -> torch.Tensor:
        # Must not run under gms_use_mem_pool or the clone stays on VMM.
        key = (tensor.untyped_storage().data_ptr(), tensor.storage_offset())
        clone = by_storage.get(key)
        if clone is None:
            clone = tensor.detach().contiguous().clone()
            by_storage[key] = clone
        return clone

    def _want(name: str, tensor: torch.Tensor) -> bool:
        if tensor is None or not torch.is_tensor(tensor):
            return False
        if tensor.is_meta or not tensor.is_cuda:
            return False
        key = name.rsplit(".", 1)[-1].lower()
        if "bias" not in key and "expert_map" not in key and "hash_indices" not in key:
            return False
        return tensor.numel() * tensor.element_size() <= 16 * (1 << 20)

    for name, param in list(model.named_parameters()):
        if not _want(name, param):
            continue
        try:
            mod, attr = _resolve_module_attr(model, name)
        except Exception:
            continue
        if hasattr(mod, "_parameters") and attr in mod._parameters:
            mod._parameters[attr] = torch.nn.Parameter(
                _off_gms(param),
                requires_grad=param.requires_grad,
            )
            cloned.append(name)
    for name, buf in list(model.named_buffers()):
        if not _want(name, buf):
            continue
        try:
            mod, attr = _resolve_module_attr(model, name)
        except Exception:
            continue
        if hasattr(mod, "_buffers") and attr in mod._buffers:
            mod._buffers[attr] = _off_gms(buf)
            cloned.append(name)
    # FusedTopkBiasRouter stores e_score_correction_bias as a plain tensor
    # alias of Parameter.data. Replacing _parameters does not update it.
    for mod_name, module in model.named_modules():
        for attr in ("e_score_correction_bias", "_hash_indices_table"):
            if not hasattr(module, attr):
                continue
            t = getattr(module, attr)
            if not torch.is_tensor(t) or t.is_meta or not t.is_cuda:
                continue
            off = _off_gms(t)
            if hasattr(module, "_parameters") and attr in module._parameters:
                module._parameters[attr] = torch.nn.Parameter(
                    off, requires_grad=False
                )
            elif hasattr(module, "_buffers") and attr in module._buffers:
                module._buffers[attr] = off
            else:
                setattr(module, attr, off)
            cloned.append(f"{mod_name}.{attr}" if mod_name else attr)
    if cloned:
        logger.info(
            "[GMS] Read mode: cloned %d Triton-incompatible params off GMS: %s",
            len(cloned),
            cloned[:8],
        )


def _create_meta_model(vllm_config, model_config) -> torch.nn.Module:
    """Create model on meta device for RO mode materialization.

    Constructor-time vLLM bugs that fire on meta tensors (DeepSeek V4 RoPE
    device, FusedMoE expert-map logging, Module.to(cuda) of meta params) are
    monkey-patched for the duration of this call. Full vLLM post-load is
    skipped here: some quantization/attention hooks allocate CUDA scratch
    even when tensors are meta. GMS imports the writer's final parameters
    and rebuilds supported MLA/MoE helpers afterward.
    """
    from vllm.model_executor.model_loader.utils import initialize_model
    from vllm.utils.torch_utils import set_default_torch_dtype

    setup_meta_tensor_workaround()
    meta_device = torch.device("meta")

    with vllm_meta_init_workarounds():
        with set_default_torch_dtype(model_config.dtype):
            with meta_device:
                model = initialize_model(
                    vllm_config=vllm_config, model_config=model_config
                )

    return model
