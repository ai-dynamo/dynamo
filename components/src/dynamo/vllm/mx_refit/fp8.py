# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP8 helpers for Dynamo vLLM ModelExpress refit.

This module owns the small set of vLLM monkey patches needed when Dynamo owns
the vLLM engine process and external trainers refit FP8 weights through
ModelExpress. It intentionally does not import NeMo-RL.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from unittest.mock import patch

import torch
from vllm.model_executor.layers.linear import LinearBase
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.engine.utils import CoreEngineProcManager

try:
    import ray
except ModuleNotFoundError:
    ray = None


FP8_BLOCK_QUANT_KWARGS = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "weight_block_size": [128, 128],
}


@dataclass(frozen=True)
class FP8Config:
    """Runtime FP8 refit settings shared with patched vLLM workers."""

    use_weight_pow2_scale: bool = False
    model_parallel_size: int | None = None
    kv_cache_dtype: str = "auto"
    use_fp8_weights: bool = True


@dataclass
class FP8State:
    """Mutable process-local cache used by FP8 weight refit."""

    seen_params: set[str] = field(default_factory=set)
    fp8_param_names: set[str] = field(default_factory=set)
    vllm_patches: list[patch] = field(default_factory=list)


global_fp8_config: FP8Config | None = None
fp8_state = FP8State()
fp8_patches_applied = False

_ORIGINAL_RUN_ENGINE_CORE = EngineCoreProc.run_engine_core
_ORIGINAL_CORE_MANAGER_INIT = CoreEngineProcManager.__init__


def _core_manager_init_with_fp8(*args, **kwargs):
    kwargs["vllm_config"].dynamo_mx_fp8_cfg = global_fp8_config
    return _ORIGINAL_CORE_MANAGER_INIT(*args, **kwargs)


def _run_engine_core_with_fp8(*args, **kwargs):
    fp8_config = kwargs["vllm_config"].dynamo_mx_fp8_cfg
    del kwargs["vllm_config"].dynamo_mx_fp8_cfg
    monkey_patch_vllm_ray_executor(fp8_config)
    return _ORIGINAL_RUN_ENGINE_CORE(*args, **kwargs)


def _copy_scalar(dest: torch.Tensor, value: float | torch.Tensor) -> None:
    source = torch.as_tensor(value, device=dest.device, dtype=dest.dtype)
    dest.copy_(source.reshape(dest.shape))


def ensure_fp8_config(
    *,
    model_parallel_size: int,
    kv_cache_dtype: str,
    use_fp8_weights: bool = True,
) -> FP8Config:
    """Initialize the process-local FP8 config if startup did not already do so."""
    global global_fp8_config
    if global_fp8_config is None:
        global_fp8_config = FP8Config(
            model_parallel_size=model_parallel_size,
            kv_cache_dtype=kv_cache_dtype,
            use_fp8_weights=use_fp8_weights,
        )
    return global_fp8_config


def monkey_patch_vllm_ray_executor(fp8_config: FP8Config) -> None:
    """Apply FP8 patches in local or Ray worker processes before model init."""
    if fp8_config.model_parallel_size and fp8_config.model_parallel_size > 1 and ray:
        from vllm.v1.executor.ray_distributed_executor import RayDistributedExecutor

        original_collective_rpc = RayDistributedExecutor.collective_rpc

        def patched_collective_rpc(self, *args, **kwargs):
            global fp8_patches_applied
            if not fp8_patches_applied:
                futures = [
                    worker.execute_method.remote(apply_fp8_patches, fp8_config)
                    for worker in self.workers
                ]
                [ray.get(future) for future in futures]
                fp8_patches_applied = True
            return original_collective_rpc(self, *args, **kwargs)

        RayDistributedExecutor.collective_rpc = patched_collective_rpc
        return

    apply_fp8_patches(None, fp8_config)


def apply_fp8_patches(_worker, fp8_config: FP8Config) -> None:
    """Patch vLLM FP8 post-load hooks so parameters remain refittable."""
    global fp8_patches_applied, global_fp8_config
    if fp8_patches_applied:
        return

    global_fp8_config = fp8_config

    if global_fp8_config.use_fp8_weights:
        patch_paths = (
            (
                "vllm.model_executor.layers.quantization.fp8."
                "Fp8LinearMethod.process_weights_after_loading",
                process_weights_after_loading,
            ),
            (
                "vllm.model_executor.layers.quantization.kv_cache."
                "BaseKVCacheMethod.process_weights_after_loading",
                process_weights_after_loading_kv,
            ),
        )
        for path, replacement in patch_paths:
            patcher = patch(path, replacement)
            fp8_state.vllm_patches.append(patcher)

    for patcher in fp8_state.vllm_patches:
        patcher.start()

    fp8_patches_applied = True


def init_fp8(
    vllm_cfg: dict[str, object],
    model_name: str,
    model_parallel_size: int,
) -> dict[str, object]:
    """Initialize Dynamo's FP8 refit patches before vLLM engine construction."""
    del model_name

    precision = str(vllm_cfg.get("precision", ""))
    use_fp8_weights = precision == "fp8"
    kv_cache_dtype = str(vllm_cfg.get("kv_cache_dtype", "auto"))
    if kv_cache_dtype not in {"auto", "fp8", "fp8_e4m3"}:
        raise ValueError(
            "kv_cache_dtype must be one of ['auto', 'fp8', 'fp8_e4m3'], "
            f"but got {kv_cache_dtype}"
        )
    if kv_cache_dtype.startswith("fp8") and not use_fp8_weights:
        raise ValueError(
            f"kv_cache_dtype='{kv_cache_dtype}' requires quantization='fp8'"
        )

    ensure_fp8_config(
        model_parallel_size=model_parallel_size,
        kv_cache_dtype=kv_cache_dtype,
        use_fp8_weights=use_fp8_weights,
    )

    if bool(vllm_cfg.get("use_deep_gemm", False)):
        os.environ["VLLM_USE_DEEP_GEMM"] = "1"
        os.environ["VLLM_USE_DEEP_GEMM_E8M0"] = "0"

    if bool(vllm_cfg.get("async_engine", True)):
        EngineCoreProc.run_engine_core = _run_engine_core_with_fp8
        CoreEngineProcManager.__init__ = _core_manager_init_with_fp8
    else:
        monkey_patch_vllm_ray_executor(global_fp8_config)

    return {
        "quantization": "fp8",
        "kv_cache_dtype": kv_cache_dtype,
        "hf_overrides": {"quantization_config": dict(FP8_BLOCK_QUANT_KWARGS)},
    }


def _get_module_from_param_name(model: torch.nn.Module, name: str):
    path_parts = name.split(".")
    module_path = path_parts[:-1]
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    reverse_mapping = {
        original_name: fused_name
        for fused_name, original_names in packed_modules_mapping.items()
        for original_name in original_names
    }
    if module_path and module_path[-1] in reverse_mapping:
        module_path[-1] = reverse_mapping[module_path[-1]]

    current_module = model
    try:
        for part in module_path:
            if isinstance(current_module, torch.nn.ModuleList):
                current_module = current_module[int(part)]
            else:
                current_module = getattr(current_module, part)
    except (AttributeError, IndexError, ValueError):
        return current_module
    return current_module


def _is_fp8_weight(name: str, model: torch.nn.Module) -> bool:
    if name not in fp8_state.seen_params:
        fp8_state.seen_params.add(name)
        if name.endswith("weight"):
            module = _get_module_from_param_name(model, name)
            if isinstance(module, LinearBase) and (
                module.weight.dtype == torch.float8_e4m3fn
            ):
                fp8_state.fp8_param_names.add(name)
    return name in fp8_state.fp8_param_names


def load_weights(
    weights: list[tuple[str, torch.Tensor]],
    model_runner,
) -> None:
    """Quantize BF16/FP32 refit tensors into vLLM's FP8 checkpoint layout."""
    quantized_weights = []
    model = model_runner.model

    for name, tensor in weights:
        if not _is_fp8_weight(name, model):
            quantized_weights.append((name, tensor))
            continue
        weight_fp8, scale = cast_tensor_to_fp8_blockwise(
            tensor.to(torch.float),
            weight_block_size=FP8_BLOCK_QUANT_KWARGS["weight_block_size"],
        )
        scale = torch.squeeze(scale, dim=-1)
        quantized_weights.append((name, weight_fp8))
        quantized_weights.append((f"{name}_scale_inv", scale))

    model.load_weights(quantized_weights)


def cast_tensor_to_fp8_blockwise(
    data_hp: torch.Tensor,
    *,
    weight_block_size: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cast a 2D tensor to blockwise FP8 plus inverse scale tensor."""
    if len(data_hp.shape) != 2:
        raise AssertionError("Only 2D input tensor is supported")

    block_size_1 = weight_block_size[1]
    block_size_0 = weight_block_size[0]
    shape_before_padding = data_hp.shape
    if data_hp.shape[1] % block_size_1 != 0 or data_hp.shape[0] % block_size_0 != 0:
        pad_1 = (
            0
            if data_hp.shape[1] % block_size_1 == 0
            else block_size_1 - data_hp.shape[1] % block_size_1
        )
        pad_0 = (
            0
            if data_hp.shape[0] % block_size_0 == 0
            else block_size_0 - data_hp.shape[0] % block_size_0
        )
        data_hp = torch.nn.functional.pad(
            data_hp,
            (0, pad_1, 0, pad_0),
            mode="constant",
            value=data_hp[-1, -1],
        )

    max_dtype = torch.finfo(torch.float8_e4m3fn).max
    original_shape = data_hp.shape
    blk_m = data_hp.shape[0] // block_size_0
    blk_n = data_hp.shape[1] // block_size_1
    data_hp = data_hp.reshape(blk_m, block_size_0, blk_n, block_size_1)
    data_hp = data_hp.permute(0, 2, 1, 3)
    data_hp = data_hp.to(torch.float32).contiguous().flatten(start_dim=2)

    max_abs = torch.amax(torch.abs(data_hp), dim=-1, keepdim=True)
    scale_fp = max_dtype / max_abs
    scale_fp = torch.where(max_abs == 0, 1.0, scale_fp)
    scale_fp = torch.where(max_abs == torch.inf, 1.0, scale_fp)
    descale_fp = torch.reciprocal(scale_fp)

    data_lp = torch.clamp(data_hp * scale_fp, min=-1 * max_dtype, max=max_dtype)
    fp_data = data_lp.to(torch.float8_e4m3fn)
    fp_data = (
        fp_data.reshape(blk_m, blk_n, block_size_0, block_size_1)
        .permute(0, 2, 1, 3)
        .reshape(original_shape)
    )

    if fp_data.shape != shape_before_padding:
        fp_data = fp_data[: shape_before_padding[0], : shape_before_padding[1]]

    return fp_data, descale_fp


def maybe_post_process_fp8_weight_block(layer: torch.nn.Module) -> None:
    """Run vLLM's optional DeepGEMM FP8 post-processing in-place."""
    if getattr(layer, "weight_block_size", None) is None:
        return

    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        deepgemm_post_process_fp8_weight_block,
    )
    from vllm.utils.deep_gemm import (
        is_deep_gemm_e8m0_used,
        should_use_deepgemm_for_fp8_linear,
    )

    should_use_deepgemm = should_use_deepgemm_for_fp8_linear(
        layer.orig_dtype,
        layer.weight,
    )
    if not should_use_deepgemm:
        return

    dg_weight, dg_weight_scale = deepgemm_post_process_fp8_weight_block(
        wq=layer.weight.data,
        ws=layer.weight_scale.data,
        quant_block_shape=tuple(layer.weight_block_size),
        use_e8m0=is_deep_gemm_e8m0_used(),
    )
    layer.weight.data.copy_(dg_weight)
    layer.weight_scale.data.copy_(dg_weight_scale)


def process_weights_after_loading(self, layer) -> None:
    """Patch vLLM FP8 linear post-load to preserve refittable Parameters."""
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        process_fp8_weight_block_strategy,
    )

    weight_scale = layer.weight_scale_inv
    weight, weight_scale = process_fp8_weight_block_strategy(layer.weight, weight_scale)
    layer.weight.data = weight.data
    if hasattr(layer, "weight_scale"):
        layer.weight_scale.copy_(weight_scale.data)
    else:
        layer.weight_scale = torch.nn.Parameter(weight_scale.data, requires_grad=False)
        layer.update_param_tp_status()

    maybe_post_process_fp8_weight_block(layer)
    if not hasattr(layer, "input_scale"):
        layer.input_scale = None


def process_weights_after_loading_kv(self, layer) -> None:
    """Patch vLLM FP8 KV-cache post-load to keep Q/K/V scale Parameters."""
    from vllm.platforms import current_platform

    if layer.kv_cache_dtype != "auto" and not layer.calculate_kv_scales:
        if layer.k_scale > 0.0 and layer.v_scale > 0.0:
            k_scale = layer.k_scale.to("cpu").tolist()
            v_scale = layer.v_scale.to("cpu").tolist()
            if current_platform.is_fp8_fnuz():
                k_scale *= 2
                v_scale *= 2
        elif layer.k_scale < 0.0 and layer.v_scale < 0.0:
            k_scale = 1.0
            v_scale = 1.0
        else:
            scale_to_duplicate = max(layer.k_scale, layer.v_scale)
            k_scale = scale_to_duplicate.to("cpu").tolist()
            v_scale = scale_to_duplicate.to("cpu").tolist()
            if current_platform.is_fp8_fnuz():
                k_scale *= 2
                v_scale *= 2

        if layer.q_scale < 0.0:
            _copy_scalar(layer._q_scale, k_scale)
            layer._q_scale_float = float(k_scale)

        _copy_scalar(layer._k_scale, k_scale)
        _copy_scalar(layer._v_scale, v_scale)
        layer._k_scale_float = float(k_scale)
        layer._v_scale_float = float(v_scale)

    if layer.q_scale > 0.0:
        q_scale = layer.q_scale
        if current_platform.is_fp8_fnuz():
            q_scale *= 2
        layer.calculate_kv_scales = False
    else:
        q_scale = 1.0

    if layer.prob_scale > 0.0:
        prob_scale = layer.prob_scale
        if current_platform.is_fp8_fnuz():
            prob_scale *= 2
    else:
        prob_scale = 1.0

    if isinstance(q_scale, torch.Tensor):
        q_scale_float = float(q_scale.item())
    else:
        q_scale_float = float(q_scale)
    _copy_scalar(layer._q_scale, q_scale)
    layer._q_scale_float = q_scale_float
    _copy_scalar(layer._prob_scale, prob_scale)
