# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Temporary Dynamo compatibility patches for vLLM snapshot restore."""

from __future__ import annotations

import functools
import logging
from collections.abc import Iterator
from typing import Any

logger = logging.getLogger(__name__)

_PATCH_SENTINEL = "_dynamo_snapshot_nested_kv_patch"
_GPU_MODEL_RUNNER_MODULES = (
    "vllm.v1.worker.gpu_model_runner",
    "vllm.v1.worker.gpu.model_runner",
)


def patch_vllm_quantized_kv_cache_wake_up() -> bool:
    """Patch vLLM V1 quantized KV-cache wake-up for nested containers.

    Returns:
        True if a vLLM ``GPUModelRunner.init_fp8_kv_scales`` method was found
        and is patched (including already-patched methods), otherwise False.
    """
    patched_any = False
    for module_name in _GPU_MODEL_RUNNER_MODULES:
        try:
            module = __import__(module_name, fromlist=["GPUModelRunner"])
        except ImportError:
            continue

        gpu_model_runner = getattr(module, "GPUModelRunner", None)
        if gpu_model_runner is None:
            continue

        method = getattr(gpu_model_runner, "init_fp8_kv_scales", None)
        if method is None:
            continue

        if getattr(method, _PATCH_SENTINEL, False):
            patched_any = True
            continue

        # TODO: Temporary compatibility for hybrid/Mamba KV caches, which can
        # be nested during quantized KV-cache wake-up after snapshot restore.
        # Remove once an upstream vLLM PR lands.
        _patched_init_fp8_kv_scales = _wrap_init_fp8_kv_scales(method)
        setattr(_patched_init_fp8_kv_scales, _PATCH_SENTINEL, True)
        gpu_model_runner.init_fp8_kv_scales = _patched_init_fp8_kv_scales
        logger.info(
            "Patched %s.GPUModelRunner.init_fp8_kv_scales for nested "
            "quantized KV-cache wake-up during snapshot restore.",
            module_name,
        )
        patched_any = True

    if not patched_any:
        logger.debug(
            "No vLLM GPUModelRunner.init_fp8_kv_scales method found for "
            "nested quantized KV-cache wake-up patch."
        )
    return patched_any


def _wrap_init_fp8_kv_scales(method: Any) -> Any:
    @functools.wraps(method)
    def _patched_init_fp8_kv_scales(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not _is_quantized_kv_cache_dtype(
            getattr(getattr(self, "cache_config", None), "cache_dtype", None)
        ):
            return method(self, *args, **kwargs)

        if not hasattr(self, "kv_caches"):
            return method(self, *args, **kwargs)

        original_kv_caches = self.kv_caches
        flattened_kv_caches = list(_iter_kv_cache_tensors(original_kv_caches))
        try:
            self.kv_caches = flattened_kv_caches
            return method(self, *args, **kwargs)
        finally:
            self.kv_caches = original_kv_caches

    return _patched_init_fp8_kv_scales


def _is_quantized_kv_cache_dtype(cache_dtype: Any) -> bool:
    if not isinstance(cache_dtype, str):
        return False

    try:
        from vllm.utils.torch_utils import is_quantized_kv_cache
    except ImportError:
        return _is_quantized_kv_cache_dtype_fallback(cache_dtype)

    return is_quantized_kv_cache(cache_dtype)


def _is_quantized_kv_cache_dtype_fallback(cache_dtype: str) -> bool:
    return (
        cache_dtype.startswith("fp8")
        or cache_dtype.endswith("per_token_head")
        or cache_dtype == "nvfp4"
    )


def _iter_kv_cache_tensors(kv_cache: Any) -> Iterator[Any]:
    if kv_cache is None:
        return

    import torch

    if isinstance(kv_cache, torch.Tensor):
        yield kv_cache
        return

    if isinstance(kv_cache, (list, tuple)):
        for cache_entry in kv_cache:
            yield from _iter_kv_cache_tensors(cache_entry)
        return

    raise TypeError(
        "Unsupported KV cache entry type during quantized KV-cache wake-up "
        "reset: "
        f"{type(kv_cache).__name__}"
    )
