# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM monkey-patches applied at GMSWorker import.

Patches:
  - MemorySnapshot.measure: adds GMS-committed bytes to free_memory in RO mode.
  - FusedMoE routing-buffer initialization: creates deterministic expert maps
    on CPU during RO/meta model construction.
  - request_memory: bypasses the free>=requested check during deferred-KV init.
  - NixlConnector.register_kv_caches: defers registration during the scratch
    phase and stashes the dict for replay at wake.

The torch.cuda.empty_cache patch lives in integrations/common/patches.py.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from gpu_memory_service.client.torch.allocator import get_gms_client_memory_manager
from gpu_memory_service.common.locks import GrantedLockType
from gpu_memory_service.common.utils import is_scratch_kv_enabled

logger = logging.getLogger(__name__)

_memory_snapshot_patched = False
_request_memory_patched = False
_register_kv_caches_patched = False
_moe_wna16_fake_patched = False
_kv_cache_alloc_patched = False
_kv_cache_alloc_lock = threading.Lock()
_fused_moe_patch_lock = threading.Lock()
_fused_moe_patch_depth = 0
_fused_moe_layer_module: Any | None = None
_fused_moe_original_determine_expert_map: Any | None = None
_fused_moe_determine_expert_map_wrapper: Any | None = None
_module_to_patch_lock = threading.Lock()
_module_to_patch_depth = 0
_module_to_module: Any | None = None
_module_to_original: Any | None = None
_module_to_wrapper: Any | None = None


# =============================================================================
# Core GMS patch (always applied)
# =============================================================================


def patch_memory_snapshot() -> None:
    """Add committed GMS bytes to MemorySnapshot.free_memory"""
    global _memory_snapshot_patched

    if _memory_snapshot_patched:
        return

    try:
        from vllm.utils.mem_utils import MemorySnapshot
    except ImportError:
        logger.debug("[GMS Patch] MemorySnapshot not available")
        return

    original_measure = MemorySnapshot.measure

    def patched_measure(self):
        original_measure(self)

        manager = get_gms_client_memory_manager("weights")
        assert manager is not None, "GMS client is not initialized"

        if manager.granted_lock_type == GrantedLockType.RO:
            allocations = manager.list_handles()
            committed_bytes = sum(alloc.aligned_size for alloc in allocations)
        else:
            # NOTE: by design, we want to assume we have the whole GPU when writing
            # weights for the first time, so we don't make an adjustment.
            committed_bytes = 0
            logger.info("[GMS] RW mode - skipping committed memory adjustment")

        original_free = self.free_memory
        self.free_memory += committed_bytes

        if committed_bytes > 0:
            logger.info(
                "[GMS Patch] Adjusted free_memory: %.2f GiB + %.2f GiB = %.2f GiB",
                original_free / (1 << 30),
                committed_bytes / (1 << 30),
                self.free_memory / (1 << 30),
            )

    MemorySnapshot.measure = patched_measure
    _memory_snapshot_patched = True
    logger.info("[GMS Patch] Patched MemorySnapshot.measure")


def patch_moe_wna16_marlin_gemm_fake_impl() -> None:
    """Fix vLLM 0.21.0's stale fake signature for WNA16 Marlin MoE."""
    global _moe_wna16_fake_patched

    if _moe_wna16_fake_patched:
        return

    try:
        import torch
        from torch.library import register_fake

        import vllm._custom_ops  # noqa: F401
    except ImportError:
        logger.debug("[GMS Patch] vLLM custom ops not available")
        return

    if not hasattr(torch.ops, "_moe_C") or not hasattr(
        torch.ops._moe_C,
        "moe_wna16_marlin_gemm",
    ):
        logger.debug("[GMS Patch] moe_wna16_marlin_gemm op not available")
        return

    def fake_impl(
        input: torch.Tensor,
        output: torch.Tensor | None,
        b_qweight: torch.Tensor,
        b_bias: torch.Tensor | None,
        b_scales: torch.Tensor,
        a_scales: torch.Tensor | None,
        global_scale: torch.Tensor | None,
        b_qzeros: torch.Tensor | None,
        g_idx: torch.Tensor | None,
        perm: torch.Tensor | None,
        workspace: torch.Tensor,
        sorted_token_ids: torch.Tensor,
        expert_ids: torch.Tensor,
        num_tokens_past_padded: torch.Tensor,
        topk_weights: torch.Tensor,
        moe_block_size: int,
        top_k: int,
        mul_topk_weights: bool,
        b_q_type: Any,
        size_m: int,
        size_n: int,
        size_k: int,
        is_k_full: bool,
        use_atomic_add: bool,
        use_fp32_reduce: bool,
        is_zp_float: bool,
        thread_k: int = -1,
        thread_n: int = -1,
        blocks_per_sm: int = -1,
    ) -> torch.Tensor:
        del (
            b_qweight,
            b_bias,
            b_scales,
            a_scales,
            global_scale,
            b_qzeros,
            g_idx,
            perm,
            workspace,
            sorted_token_ids,
            expert_ids,
            num_tokens_past_padded,
            topk_weights,
            moe_block_size,
            mul_topk_weights,
            b_q_type,
            size_k,
            is_k_full,
            use_atomic_add,
            use_fp32_reduce,
            is_zp_float,
            thread_k,
            thread_n,
            blocks_per_sm,
        )
        if output is not None:
            return output
        return torch.empty(
            (size_m * top_k, size_n), dtype=input.dtype, device=input.device
        )

    @register_fake("_moe_C::moe_wna16_marlin_gemm", allow_override=True)
    def moe_wna16_marlin_gemm_fake(
        input: torch.Tensor,
        output: torch.Tensor | None,
        b_qweight: torch.Tensor,
        b_bias: torch.Tensor | None,
        b_scales: torch.Tensor,
        a_scales: torch.Tensor | None,
        global_scale: torch.Tensor | None,
        b_qzeros: torch.Tensor | None,
        g_idx: torch.Tensor | None,
        perm: torch.Tensor | None,
        workspace: torch.Tensor,
        sorted_token_ids: torch.Tensor,
        expert_ids: torch.Tensor,
        num_tokens_past_padded: torch.Tensor,
        topk_weights: torch.Tensor,
        moe_block_size: int,
        top_k: int,
        mul_topk_weights: bool,
        b_q_type: Any,
        size_m: int,
        size_n: int,
        size_k: int,
        is_k_full: bool,
        use_atomic_add: bool,
        use_fp32_reduce: bool,
        is_zp_float: bool,
        thread_k: int = -1,
        thread_n: int = -1,
        blocks_per_sm: int = -1,
    ) -> torch.Tensor:
        return fake_impl(
            input,
            output,
            b_qweight,
            b_bias,
            b_scales,
            a_scales,
            global_scale,
            b_qzeros,
            g_idx,
            perm,
            workspace,
            sorted_token_ids,
            expert_ids,
            num_tokens_past_padded,
            topk_weights,
            moe_block_size,
            top_k,
            mul_topk_weights,
            b_q_type,
            size_m,
            size_n,
            size_k,
            is_k_full,
            use_atomic_add,
            use_fp32_reduce,
            is_zp_float,
            thread_k,
            thread_n,
            blocks_per_sm,
        )

    try:
        from torch._library.fake_impl import Kernel
        from torch._library.simple_registry import singleton

        entry = singleton.find("_moe_C::moe_wna16_marlin_gemm")
        if len(entry.fake_impl.kernels) > 1:
            entry.fake_impl.kernels[:] = [
                Kernel(fake_impl, "[GMS] moe_wna16_marlin_gemm fake override")
            ]
    except Exception:
        logger.debug(
            "[GMS Patch] Could not prune stale WNA16 fake impls", exc_info=True
        )

    _moe_wna16_fake_patched = True
    logger.info("[GMS Patch] Patched moe_wna16_marlin_gemm fake impl")


# =============================================================================
# Read-only/meta model construction helpers
# =============================================================================


@contextmanager
def fused_moe_cpu_routing_buffers_during_meta_init() -> Iterator[None]:
    """Keep vLLM FusedMoE expert maps off meta tensors during GMS RO load.

    vLLM builds ``_expert_map`` on the default device and immediately reads it
    with ``Tensor.item()``. During GMS read-only loading the default device is
    ``meta``, so construct this small deterministic buffer on CPU until GMS
    materializes the saved CUDA buffers.

    TODO: replace with an upstream vLLM meta-safe initialization hook.
    """
    try:
        import torch
        import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer
    except ImportError:
        logger.debug("[GMS Patch] vLLM FusedMoE layer not available")
        yield
        return

    global _fused_moe_determine_expert_map_wrapper
    global _fused_moe_layer_module
    global _fused_moe_original_determine_expert_map
    global _fused_moe_patch_depth

    with _fused_moe_patch_lock:
        if _fused_moe_patch_depth == 0:

            def determine_expert_map_on_cpu(*args, **kwargs):
                assert _fused_moe_original_determine_expert_map is not None
                with torch.device("cpu"):
                    return _fused_moe_original_determine_expert_map(*args, **kwargs)

            _fused_moe_layer_module = fused_moe_layer
            _fused_moe_original_determine_expert_map = (
                fused_moe_layer.determine_expert_map
            )
            _fused_moe_determine_expert_map_wrapper = determine_expert_map_on_cpu
            fused_moe_layer.determine_expert_map = determine_expert_map_on_cpu

        _fused_moe_patch_depth += 1

    try:
        yield
    finally:
        with _fused_moe_patch_lock:
            _fused_moe_patch_depth -= 1
            if _fused_moe_patch_depth == 0:
                if (
                    _fused_moe_layer_module is not None
                    and _fused_moe_layer_module.determine_expert_map
                    is _fused_moe_determine_expert_map_wrapper
                ):
                    _fused_moe_layer_module.determine_expert_map = (
                        _fused_moe_original_determine_expert_map
                    )
                _fused_moe_layer_module = None
                _fused_moe_original_determine_expert_map = None
                _fused_moe_determine_expert_map_wrapper = None


@contextmanager
def meta_safe_module_to_during_meta_init() -> Iterator[None]:
    """Keep meta tensors on meta when constructors call ``Module.to(cuda)``."""
    try:
        import torch
    except ImportError:
        yield
        return

    def has_meta_tensors(module):
        return any(
            getattr(tensor, "is_meta", False)
            for tensor in module.parameters(recurse=True)
        ) or any(
            getattr(tensor, "is_meta", False) for tensor in module.buffers(recurse=True)
        )

    global _module_to_module
    global _module_to_original
    global _module_to_patch_depth
    global _module_to_wrapper

    with _module_to_patch_lock:
        if _module_to_patch_depth == 0:
            _module_to_module = torch.nn.Module
            _module_to_original = torch.nn.Module.to

            def meta_safe_to(self, *args, **kwargs):
                assert _module_to_original is not None
                try:
                    device, dtype, non_blocking, memory_format = (
                        torch._C._nn._parse_to(*args, **kwargs)
                    )
                except Exception:
                    return _module_to_original(self, *args, **kwargs)

                if (
                    device is None
                    or device.type == "meta"
                    or not has_meta_tensors(self)
                ):
                    return _module_to_original(self, *args, **kwargs)

                def convert(tensor):
                    target_device = torch.device("meta") if tensor.is_meta else device
                    target_dtype = (
                        dtype
                        if dtype is not None
                        and (tensor.is_floating_point() or tensor.is_complex())
                        else None
                    )
                    if memory_format is not None and tensor.dim() in (4, 5):
                        return tensor.to(
                            target_device,
                            target_dtype,
                            non_blocking,
                            memory_format=memory_format,
                        )
                    return tensor.to(target_device, target_dtype, non_blocking)

                return self._apply(convert)

            _module_to_wrapper = meta_safe_to
            torch.nn.Module.to = meta_safe_to

        _module_to_patch_depth += 1

    try:
        yield
    finally:
        with _module_to_patch_lock:
            _module_to_patch_depth -= 1
            if _module_to_patch_depth == 0:
                if (
                    _module_to_module is not None
                    and _module_to_module.to is _module_to_wrapper
                ):
                    _module_to_module.to = _module_to_original
                _module_to_module = None
                _module_to_original = None
                _module_to_wrapper = None


# =============================================================================
# Shadow mode patches
# =============================================================================


def patch_request_memory() -> None:
    """Bypass free >= requested check (shadow shares GPU with active engine)."""
    global _request_memory_patched

    if _request_memory_patched:
        return

    try:
        from vllm.v1.worker import utils as worker_utils
    except ImportError:
        logger.debug("[GMS Patch] vllm.v1.worker.utils not available")
        return

    def patched_request_memory(init_snapshot, cache_config):
        requested_memory = int(
            init_snapshot.total_memory * cache_config.gpu_memory_utilization
        )
        logger.info(
            "[GMS Patch] Shadow mode: bypassing memory check "
            "(requested=%.2f GiB, free=%.2f GiB)",
            requested_memory / (1 << 30),
            init_snapshot.free_memory / (1 << 30),
        )
        return requested_memory

    worker_utils.request_memory = patched_request_memory
    _request_memory_patched = True
    logger.info("[GMS Patch] Patched request_memory for shadow mode")


def patch_register_kv_caches() -> None:
    """Defer NixlConnector.register_kv_caches while KV backing is scratch-aliased.

    Registering NIXL MRs over scratch would pin a soon-stale page into the NIC;
    sleep tears down scratch and wake remaps real backing at the same VAs.
    Stash the dict during the scratch phase and let GMSWorker.wake_up replay
    it after remap.
    """
    global _register_kv_caches_patched

    if _register_kv_caches_patched:
        return

    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import (
            NixlConnector,
        )
    except ImportError:
        logger.debug("[GMS Patch] NixlConnector not available")
        return

    original_register = NixlConnector.register_kv_caches

    def patched_register_kv_caches(self, kv_caches):
        from gpu_memory_service.client.torch.allocator import (
            get_gms_client_memory_manager,
            is_scratch,
        )

        # Fail closed on lookup errors: falling through to original_register
        # would pin an MR onto a scratch page that sleep is about to free,
        # exactly the bug this patch exists to prevent.
        try:
            kv_mgr = get_gms_client_memory_manager("kv_cache")
            has_deferred = kv_mgr is not None and is_scratch(kv_mgr)
        except (LookupError, AttributeError, RuntimeError) as exc:
            logger.warning(
                "[GMS Patch] Cannot determine deferred-KV state — "
                "raising to avoid pinning a stale scratch MR: %s",
                exc,
                exc_info=True,
            )
            raise

        if has_deferred:
            self._scratch_kv_pending = kv_caches
            logger.info(
                "[GMS Patch] Deferring NIXL KV cache registration "
                "(stashed %d layers for wake replay)",
                len(kv_caches),
            )
            return
        return original_register(self, kv_caches)

    NixlConnector.register_kv_caches = patched_register_kv_caches
    _register_kv_caches_patched = True
    logger.info("[GMS Patch] Patched NixlConnector.register_kv_caches")


def patch_kv_cache_allocation_for_scratch() -> None:
    """Route only raw KV tensors through scratch and avoid full-KV writes.

    GMS scratch-KV reserves the final virtual address range but only backs a
    small prefix until wake-up. vLLM's raw KV allocation uses ``torch.zeros``,
    which touches the full future range. Reuse the upstream allocation logic,
    but make those raw scratch allocations ``torch.empty`` and keep transient
    metadata/cudagraph buffers on PyTorch's normal allocator.
    """
    global _kv_cache_alloc_patched

    if _kv_cache_alloc_patched:
        return

    try:
        import torch

        from vllm.v1.worker import gpu_model_runner
        from vllm.v1.worker import kv_connector_model_runner_mixin
    except ImportError:
        logger.debug("[GMS Patch] vLLM KV allocation helpers not available")
        return

    original_allocate = gpu_model_runner.GPUModelRunner._allocate_kv_cache_tensors
    mixin_cls = kv_connector_model_runner_mixin.KVConnectorModelRunnerMixin
    original_uniform_allocate = mixin_cls.allocate_uniform_kv_caches

    def allocate_kv_tensor(torch_mod, original_zeros, args, kwargs):
        from gpu_memory_service.client.torch.allocator import (
            get_gms_client_memory_manager,
            gms_use_mem_pool,
            is_scratch,
        )

        kv_mgr = get_gms_client_memory_manager("kv_cache")
        if kv_mgr is None:
            return original_zeros(*args, **kwargs)

        device = kwargs.get("device")
        if device is None or "out" in kwargs:
            return original_zeros(*args, **kwargs)

        # Validate the registered tag before routing through the shared
        # pluggable allocator. If lookup fails, fail closed instead of silently
        # allocating KV tensors outside GMS.
        use_empty = is_scratch(kv_mgr)
        mempool_device = (
            torch_mod.device("cuda", device)
            if isinstance(device, int)
            else torch_mod.device(device)
        )
        with gms_use_mem_pool("kv_cache", device=mempool_device):
            if use_empty:
                return torch_mod.empty(*args, **kwargs)
            return original_zeros(*args, **kwargs)

    @contextmanager
    def scratch_aware_zeros(func):
        func_globals = func.__globals__
        original_torch = func_globals.get("torch", torch)

        class TorchProxy:
            def __getattr__(self, name):
                return getattr(original_torch, name)

            def zeros(self, *args, **kwargs):
                return allocate_kv_tensor(
                    original_torch,
                    original_torch.zeros,
                    args,
                    kwargs,
                )

        with _kv_cache_alloc_lock:
            previous_torch = func_globals.get("torch")
            func_globals["torch"] = TorchProxy()
            try:
                yield
            finally:
                if previous_torch is None:
                    func_globals.pop("torch", None)
                else:
                    func_globals["torch"] = previous_torch

    def patched_allocate_kv_cache_tensors(self, kv_cache_config):
        with scratch_aware_zeros(original_allocate):
            return original_allocate(self, kv_cache_config)

    def patched_allocate_uniform_kv_caches(*args, **kwargs):
        with scratch_aware_zeros(original_uniform_allocate):
            return original_uniform_allocate(*args, **kwargs)

    gpu_model_runner.GPUModelRunner._allocate_kv_cache_tensors = (
        patched_allocate_kv_cache_tensors
    )
    mixin_cls.allocate_uniform_kv_caches = staticmethod(
        patched_allocate_uniform_kv_caches,
    )
    _kv_cache_alloc_patched = True
    logger.info(
        "[GMS Patch] Patched vLLM KV cache allocation to avoid full scratch zero-fill"
    )


# =============================================================================
# Patch application helper
# =============================================================================


def apply_scratch_kv_patches() -> None:
    """Apply scratch-KV monkey-patches. No-ops when scratch KV is disabled."""
    if not is_scratch_kv_enabled():
        return

    patch_request_memory()
    patch_register_kv_caches()
    patch_kv_cache_allocation_for_scratch()
    logger.info("[GMS Patch] applied")
