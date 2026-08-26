# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Temporary vLLM compatibility workarounds used by the GMS integration.

These shims cover vLLM bugs that are upstreamable and should not be confused
with GMS memory-management logic.  Keep their scope narrow and remove each one
as the corresponding vLLM fix is available in the runtime image.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

_moe_wna16_fake_patched = False
_fused_moe_patch_lock = threading.Lock()
_fused_moe_patch_depth = 0
_fused_moe_layer_module: Any | None = None
_fused_moe_original_determine_expert_map: Any | None = None
_fused_moe_determine_expert_map_wrapper: Any | None = None
_deepseek_rope_patch_lock = threading.Lock()
_deepseek_rope_compute_lock = threading.RLock()
_deepseek_rope_patch_depth = 0
_deepseek_rope_original_methods: dict[Any, Any] | None = None
_deepseek_rope_wrappers: dict[Any, Any] | None = None
_module_to_patch_lock = threading.Lock()
_module_to_patch_depth = 0
_module_to_module: Any | None = None
_module_to_original: Any | None = None
_module_to_wrapper: Any | None = None


def patch_moe_wna16_marlin_gemm_fake_impl() -> None:
    """Fix vLLM's stale fake signature for WNA16 Marlin MoE."""
    global _moe_wna16_fake_patched

    if _moe_wna16_fake_patched:
        return

    try:
        import torch
        import vllm._custom_ops  # noqa: F401
        from torch.library import register_fake
    except ImportError:
        logger.debug("[GMS vLLM Workaround] vLLM custom ops not available")
        return

    if not hasattr(torch.ops, "_moe_C") or not hasattr(
        torch.ops._moe_C,
        "moe_wna16_marlin_gemm",
    ):
        logger.debug("[GMS vLLM Workaround] moe_wna16_marlin_gemm op not available")
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
            "[GMS vLLM Workaround] Could not prune stale WNA16 fake impls",
            exc_info=True,
        )

    _moe_wna16_fake_patched = True
    logger.info("[GMS vLLM Workaround] Patched moe_wna16_marlin_gemm fake impl")


@contextmanager
def fused_moe_cpu_routing_buffers_during_meta_init() -> Iterator[None]:
    """Keep vLLM FusedMoE expert maps off meta during GMS RO construction."""
    try:
        import torch
        import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer
    except ImportError:
        logger.debug("[GMS vLLM Workaround] vLLM FusedMoE layer not available")
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


class _CpuPlatformProxy:
    device_type = "cpu"

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


@contextmanager
def deepseek_rope_cpu_cache_during_meta_init() -> Iterator[None]:
    """Build DeepSeek RoPE constructor caches on CPU during GMS RO load."""
    try:
        import torch
        import vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope as deepseek_rope
    except ImportError:
        logger.debug("[GMS vLLM Workaround] vLLM DeepSeek RoPE layer not available")
        yield
        return

    classes = [
        cls
        for cls in (
            getattr(deepseek_rope, "DeepseekScalingRotaryEmbedding", None),
            getattr(deepseek_rope, "DeepseekV4ScalingRotaryEmbedding", None),
        )
        if cls is not None
    ]
    # Preserve order while avoiding duplicate class objects.
    classes = list(dict.fromkeys(classes))
    if not classes:
        yield
        return

    global _deepseek_rope_original_methods
    global _deepseek_rope_patch_depth
    global _deepseek_rope_wrappers

    with _deepseek_rope_patch_lock:
        if _deepseek_rope_patch_depth == 0:
            _deepseek_rope_original_methods = {}
            _deepseek_rope_wrappers = {}

            def make_wrapper(original):
                def compute_cache_on_cpu(self, *args, **kwargs):
                    with _deepseek_rope_compute_lock:
                        original_platform = getattr(
                            deepseek_rope,
                            "current_platform",
                            None,
                        )
                        if original_platform is not None:
                            deepseek_rope.current_platform = _CpuPlatformProxy(
                                original_platform,
                            )
                        try:
                            with torch.device("cpu"):
                                return original(self, *args, **kwargs)
                        finally:
                            if original_platform is not None:
                                deepseek_rope.current_platform = original_platform

                return compute_cache_on_cpu

            for cls in classes:
                original = cls._compute_cos_sin_cache
                wrapper = make_wrapper(original)
                _deepseek_rope_original_methods[cls] = original
                _deepseek_rope_wrappers[cls] = wrapper
                cls._compute_cos_sin_cache = wrapper

        _deepseek_rope_patch_depth += 1

    try:
        yield
    finally:
        with _deepseek_rope_patch_lock:
            _deepseek_rope_patch_depth -= 1
            if _deepseek_rope_patch_depth == 0:
                assert _deepseek_rope_original_methods is not None
                assert _deepseek_rope_wrappers is not None
                for cls, original in _deepseek_rope_original_methods.items():
                    if cls._compute_cos_sin_cache is _deepseek_rope_wrappers[cls]:
                        cls._compute_cos_sin_cache = original
                _deepseek_rope_original_methods = None
                _deepseek_rope_wrappers = None


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
                    device, dtype, non_blocking, memory_format = torch._C._nn._parse_to(
                        *args, **kwargs
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


@contextmanager
def vllm_meta_init_workarounds() -> Iterator[None]:
    """Apply temporary vLLM meta-init workarounds in one narrow scope."""
    with (
        fused_moe_cpu_routing_buffers_during_meta_init(),
        deepseek_rope_cpu_cache_during_meta_init(),
        meta_safe_module_to_during_meta_init(),
    ):
        yield
