# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM model loader for GPU Memory Service integration.

Provides a model loader that loads weights via GMS for cross-process sharing.
The loader uses RW_OR_RO mode: first process loads from disk (RW), subsequent
processes import from GMS metadata (RO).
"""

from __future__ import annotations

import copy
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
    materialize_module_from_gms,
    rebind_nonparameter_tensors,
)
from gpu_memory_service.common.locks import GrantedLockType
from gpu_memory_service.common.utils import get_socket_path
from gpu_memory_service.common.vmm import get_vmm_device_type
from gpu_memory_service.integrations.common.utils import (
    GMSCommittedMemoryStats,
    get_gms_lock_mode,
    prepare_gms_write,
    publish_gms_write,
    setup_meta_tensor_workaround,
    strip_gms_model_loader_config,
    torch_device,
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


_KV_SCALE_PLACEHOLDERS = (
    "k_scale",
    "v_scale",
    "q_scale",
    "prob_scale",
    "k_zero_point",
    "v_zero_point",
    "q_zero_point",
)
_DEFERRED_QUANT_MODULES_ATTR = "_gms_deferred_quant_modules"
_DEFERRED_HOOK_MODULES_ATTR = "_gms_deferred_hook_modules"
_META_TENSOR_ERRORS = (NotImplementedError, RuntimeError)


def _pop_attr_list(model: torch.nn.Module, attr: str) -> list[torch.nn.Module]:
    return model.__dict__.pop(attr, [])


def _import_is_deferred_attention_layer():
    """Resolve ``is_deferred_attention_layer`` across vLLM 0.33 and 0.34+."""
    # Deferred import: gpu_memory_service can be imported without vLLM installed.
    try:
        from vllm.model_executor.layers.attention import (
            is_deferred_attention_layer,
        )

        return is_deferred_attention_layer
    except ImportError:
        from vllm.model_executor.model_loader.utils import (
            is_deferred_attention_layer,
        )

        return is_deferred_attention_layer


def _iter_children_first(
    root: torch.nn.Module, quant_base: type
) -> list[torch.nn.Module]:
    """Traverse modules child-first, stopping before children of quantized layers."""
    seen: set[torch.nn.Module] = set()
    ordered: list[torch.nn.Module] = []

    def _visit(node: torch.nn.Module) -> None:
        quantized = isinstance(getattr(node, "quant_method", None), quant_base)
        for child in () if quantized else node.children():
            if child not in seen:
                seen.add(child)
                _visit(child)
        ordered.append(node)

    _visit(root)
    return ordered


def _call_post_load_hook(hook, dtype, *, is_attn: bool) -> bool:
    """Invoke a module post-load hook matching 0-arg (0.34+) or 1-arg (0.33) signatures."""
    try:
        sig = inspect.signature(hook)
    except (TypeError, ValueError):
        return False

    if is_attn:
        for args in ((dtype,), ()):
            try:
                sig.bind(*args)
            except TypeError:
                continue
            hook(*args)
            return True
        return False

    try:
        sig.bind()
    except TypeError:
        return False
    hook()
    return True


def _run_deferred_module_hooks(
    model: torch.nn.Module,
    model_config,
    quant_base: type,
    *,
    fail_on_error: bool = False,
    target_modules: set[int] | None = None,
) -> list[torch.nn.Module]:
    """Invoke module-level post-load hooks child-first, isolating or raising errors."""
    is_deferred_attention = _import_is_deferred_attention_layer()
    dtype = getattr(model_config, "dtype", None)
    failed_modules: list[torch.nn.Module] = []

    for module in _iter_children_first(model, quant_base):
        if target_modules is not None and id(module) not in target_modules:
            continue
        hook = getattr(module, "process_weights_after_loading", None)
        if not callable(hook):
            continue
        is_attn = is_deferred_attention(module)
        if not is_attn and module is not model:
            try:
                inspect.signature(hook).bind()
            except (TypeError, ValueError):
                continue
        try:
            _call_post_load_hook(hook, dtype, is_attn=is_attn)
        except _META_TENSOR_ERRORS as e:
            if fail_on_error:
                raise
            failed_modules.append(module)
            logger.debug(
                "[GMS] Deferred post-processing hook for %s (%s): %s",
                type(module).__name__,
                "attention" if is_attn else "module",
                e,
            )
        except Exception:
            logger.exception(
                "[GMS] Unexpected error in post-processing hook for %s",
                type(module).__name__,
            )
            raise

    return failed_modules


def _finalize_kv_cache_layer(
    layer: torch.nn.Module,
    quant_method,
    base_kv_cls: type,
) -> None:
    """Rebuild host-side FP8 KV-cache scale state from materialized GMS tensors."""
    for placeholder_name in _KV_SCALE_PLACEHOLDERS:
        for registry in (layer._parameters, layer._buffers):
            tensor = registry.get(placeholder_name)
            if tensor is not None and tensor.is_meta:
                del registry[placeholder_name]

    for scale_name in ("_k_scale", "_v_scale", "_q_scale", "_prob_scale"):
        scale = getattr(layer, scale_name, None)
        if torch.is_tensor(scale):
            if scale.is_meta:
                raise RuntimeError(
                    f"[GMS] Read-only attach left KV-cache scale {scale_name!r} "
                    "on meta device after materialization"
                )
            if scale.numel() != 1:
                raise RuntimeError(
                    "[GMS] Read-only attach cannot rebuild per-head KV-cache "
                    "scales: vLLM's pre-processed weights path supports only "
                    "per-tensor scales"
                )

    # Call BaseKVCacheMethod explicitly: CompressedTensorsKVCacheMethod
    # overrides process_weights_after_loading without checking
    # is_weights_pre_processed().
    base_kv_cls.process_weights_after_loading(quant_method, layer)

    prob_scale = getattr(layer, "_prob_scale", None)
    if torch.is_tensor(prob_scale) and prob_scale.numel() == 1:
        layer._prob_scale_float = prob_scale.item()


def _finalize_post_materialized_modules(
    model: torch.nn.Module,
    deferred_quant_modules: list[torch.nn.Module],
    deferred_hook_modules: list[torch.nn.Module],
    model_config,
) -> None:
    """Finalize KV scales, pre-processed quant modules, and deferred hooks after attach."""
    deferred_quant_ids = {id(m) for m in deferred_quant_modules}
    deferred_hook_ids = {id(m) for m in deferred_hook_modules}
    has_quant_methods = bool(deferred_quant_ids) or any(
        getattr(mod, "quant_method", None) is not None for mod in model.modules()
    )
    if not has_quant_methods and not deferred_hook_ids:
        return

    # Deferred imports: gpu_memory_service can be imported without vLLM installed.
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizeMethodBase,
    )
    from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
    from vllm.model_executor.utils import weights_already_processed

    rebuilt_kv = 0
    rebuilt_quant = 0
    kv_modules: list[tuple[torch.nn.Module, object]] = []
    quant_modules: list[tuple[torch.nn.Module, object]] = []

    for _mod_name, mod in model.named_modules():
        qm = getattr(mod, "quant_method", None)
        if qm is None:
            continue
        if isinstance(qm, BaseKVCacheMethod):
            kv_modules.append((mod, qm))
        elif getattr(qm, "supports_pre_processed_weights", False) and (
            id(mod) in deferred_quant_ids
            or any(
                hasattr(qm, attr) and getattr(qm, attr) is None
                for attr in ("moe_kernel", "kernel")
            )
        ):
            quant_modules.append((mod, qm))

    with weights_already_processed():
        # Finalize KV-cache scales before any deferred hooks read host float copies.
        for mod, qm in kv_modules:
            _finalize_kv_cache_layer(mod, qm, BaseKVCacheMethod)
            rebuilt_kv += 1

        for mod, qm in quant_modules:
            qm.process_weights_after_loading(mod)
            if hasattr(mod, "update_param_tp_status"):
                mod.update_param_tp_status()
            rebuilt_quant += 1

    # Replay only hooks that previously failed on meta tensors; re-running
    # already-succeeded attention hooks would overwrite read-only GMS weights.
    if deferred_hook_ids:
        with weights_already_processed():
            _run_deferred_module_hooks(
                model,
                model_config,
                QuantizeMethodBase,
                fail_on_error=True,
                target_modules=deferred_hook_ids,
            )

    if rebuilt_quant:
        logger.info(
            "[GMS] Finalized %d pre-processed quantized module(s) after "
            "GMS materialization",
            rebuilt_quant,
        )
    if rebuilt_kv:
        logger.info(
            "[GMS] Finalized %d KV-cache scale module(s) after GMS materialization",
            rebuilt_kv,
        )


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
        deferred_quant_modules = _pop_attr_list(model, _DEFERRED_QUANT_MODULES_ATTR)
        deferred_hook_modules = _pop_attr_list(model, _DEFERRED_HOOK_MODULES_ATTR)
        materialize_module_from_gms(gms_client, model, device_index=device_index)
        _finalize_post_materialized_modules(
            model, deferred_quant_modules, deferred_hook_modules, model_config
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

            torch_device().empty_cache()

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
    # Deferred imports: gpu_memory_service can be imported without vLLM installed.
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizeMethodBase,
    )
    from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
    from vllm.model_executor.model_loader import utils as vllm_loader_utils
    from vllm.utils.torch_utils import set_default_torch_dtype

    setup_meta_tensor_workaround()
    meta_device = torch.device("meta")

    with set_default_torch_dtype(model_config.dtype):
        with meta_device:
            model = vllm_loader_utils.initialize_model(
                vllm_config=vllm_config, model_config=model_config
            )

    # The writer already decided whether lm_head ties to embed_tokens; comparing
    # meta tensors inside maybe_retie_word_embeddings would raise and abort the
    # entire post-processing pass.
    meta_config = copy.copy(model_config)
    if hasattr(meta_config, "word_embeddings_untied_by_checkpoint"):
        meta_config.word_embeddings_untied_by_checkpoint = False
    maybe_retie = getattr(vllm_loader_utils, "maybe_retie_word_embeddings", None)
    if callable(maybe_retie):
        try:
            maybe_retie(model, meta_config)
        except _META_TENSOR_ERRORS as e:
            logger.debug("[GMS] Meta word-embedding retie skipped: %s", e)
        except Exception:
            logger.exception("[GMS] Unexpected error in meta word-embedding retie")
            raise

    deferred_quant_modules: list[torch.nn.Module] = []
    skipped_non_preprocessed = 0
    for mod_name, module in model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if not isinstance(quant_method, QuantizeMethodBase):
            continue
        # KV-cache quant methods read scale values (unavailable on meta) or
        # consume/overwrite placeholders; finalize them after GMS materialization.
        if isinstance(quant_method, BaseKVCacheMethod):
            continue
        try:
            quant_method.process_weights_after_loading(module)
            if hasattr(module, "update_param_tp_status"):
                module.update_param_tp_status()
        except _META_TENSOR_ERRORS as e:
            if getattr(quant_method, "supports_pre_processed_weights", False):
                deferred_quant_modules.append(module)
                logger.debug(
                    "[GMS] Deferred pre-processed quant module %s from meta pass: %s",
                    mod_name,
                    e,
                )
            else:
                skipped_non_preprocessed += 1
                if skipped_non_preprocessed <= 5:
                    logger.debug(
                        "[GMS] Meta quant post-processing skipped for %s: %s",
                        mod_name,
                        e,
                    )
        except Exception:
            logger.exception(
                "[GMS] Unexpected error in meta quant post-processing for %s",
                mod_name,
            )
            raise

    if skipped_non_preprocessed:
        logger.info(
            "[GMS] Meta pass skipped %d quantized module(s) needing concrete tensors",
            skipped_non_preprocessed,
        )

    deferred_hook_modules = _run_deferred_module_hooks(
        model, meta_config, QuantizeMethodBase, fail_on_error=False
    )

    if deferred_quant_modules:
        setattr(model, _DEFERRED_QUANT_MODULES_ATTR, deferred_quant_modules)
    if deferred_hook_modules:
        setattr(model, _DEFERRED_HOOK_MODULES_ATTR, deferred_hook_modules)

    return model
