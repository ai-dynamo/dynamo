# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo vLLM MX refit extension with Megatron publisher support.

The DTensor path receives HF-named tensors from NeMo-RL publishers. The
Megatron path consumes Modelexpress Megatron metadata and translates
Megatron-native trainer shards into HF-named tensors before calling vLLM's
loader.
"""

from __future__ import annotations

import gc
import logging
import os
import socket
import traceback
import types
from dataclasses import dataclass, field
from typing import Any

import torch

logger = logging.getLogger(__name__)
_VLLM_WEIGHT_LOADER_V2_PATCHED = False


@dataclass
class MxConfig:
    """Wire-compatible subset of ``nemo_rl.distributed.mx_helpers.MxConfig``."""

    enabled: bool = True
    mx_server_url: str = "modelexpress-server:8001"
    timeout_seconds: float = 300.0
    same_rank_only: bool = True
    tree_scale_out: bool = True
    moe_expert_filter: bool = True
    register_self_buffers: list[str] = field(default_factory=list)
    nic_pin: str = "auto"
    retain_latest_k: int = 1

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "MxConfig":
        if not d:
            return cls()
        return cls(
            enabled=bool(d.get("enabled", True)),
            mx_server_url=str(d.get("mx_server_url", "modelexpress-server:8001")),
            timeout_seconds=float(d.get("timeout_seconds", 300.0)),
            same_rank_only=bool(d.get("same_rank_only", True)),
            tree_scale_out=bool(d.get("tree_scale_out", True)),
            moe_expert_filter=bool(d.get("moe_expert_filter", True)),
            register_self_buffers=list(d.get("register_self_buffers", []) or []),
            nic_pin=str(d.get("nic_pin", "auto")),
            retain_latest_k=int(d.get("retain_latest_k", 1)),
        )


def _pin_local_nic(*, device_id: int, mode: str = "auto") -> None:
    if mode == "off":
        return
    try:
        from modelexpress.ucx_utils import apply_nic_pin_for_device

        if mode == "auto":
            apply_nic_pin_for_device(device_id=device_id)
            logger.info("[mx] pinned NIC for device %d (auto)", device_id)
        else:
            os.environ["UCX_NET_DEVICES"] = mode
            os.environ["MX_RDMA_NIC_PIN"] = "off"
            logger.info("[mx] pinned NIC explicitly: %s", mode)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[mx] NIC pin failed (mode=%s): %s", mode, exc)


def _device_index(device: Any) -> int:
    index = getattr(device, "index", None)
    if index is not None:
        return int(index)
    if isinstance(device, int):
        return device
    return int(torch.cuda.current_device())


def _model_name(worker: Any) -> str:
    vllm_config = getattr(worker.model_runner, "vllm_config", None)
    model_config = getattr(vllm_config, "model_config", None)
    return str(getattr(model_config, "model", "unknown"))


def _parallel_config(worker: Any) -> Any:
    if hasattr(worker, "parallel_config"):
        return worker.parallel_config
    vllm_config = getattr(worker.model_runner, "vllm_config", None)
    return getattr(vllm_config, "parallel_config", None)


def _target_tp(worker: Any) -> tuple[int, int]:
    parallel_config = _parallel_config(worker)
    tp_size = int(getattr(parallel_config, "tensor_parallel_size", 1) or 1)
    if tp_size <= 1 and torch.distributed.is_initialized():
        tp_size = int(torch.distributed.get_world_size())
    if torch.distributed.is_initialized():
        # Dynamo vLLM workers are single-DP in this smoke. For TP>1, global
        # rank modulo TP gives the worker's TP rank.
        tp_rank = int(torch.distributed.get_rank() % tp_size)
    else:
        tp_rank = 0
    return tp_size, tp_rank


def _param_for_loaded_weight(
    name: str,
    params: dict[str, torch.Tensor],
) -> torch.Tensor | None:
    candidates = [name]
    if name.startswith("backbone."):
        candidates.append(f"model.{name[len('backbone.') :]}")
    for candidate in candidates:
        param = params.get(candidate)
        if param is not None:
            return param

    # vLLM maps separate q/k/v checkpoint names onto qkv_proj internally.
    # The shape adapter below still needs to inspect the fused parameter.
    for candidate in candidates:
        for shard_name in ("q_proj", "k_proj", "v_proj"):
            if shard_name not in candidate:
                continue
            mapped_name = candidate.replace(shard_name, "qkv_proj")
            param = params.get(mapped_name)
            if param is not None:
                return param
    return None


def _maybe_copy_tp_local_weight(
    *,
    name: str,
    weight: torch.Tensor,
    params: dict[str, torch.Tensor],
) -> bool:
    """Copy exact TP-local linear shards directly.

    The Megatron matched-TP path receives local shards. vLLM's standard
    loaders usually expect checkpoint-global tensors and slice again, which
    is wrong for exact-shaped row-parallel and fused local layouts such as
    Nemotron-H Mamba ``conv1d``/``in_proj``.
    """
    param = _param_for_loaded_weight(name, params)
    if param is None or tuple(param.shape) != tuple(weight.shape):
        return False
    if ".experts." in name:
        return False

    is_linear_shard = (
        getattr(param, "input_dim", None) is not None
        or getattr(param, "output_dim", None) is not None
    )
    if not is_linear_shard:
        return False

    with torch.no_grad():
        param.copy_(weight, non_blocking=True)
    return True


def _maybe_expand_tp_local_weight(
    *,
    name: str,
    weight: torch.Tensor,
    params: dict[str, torch.Tensor],
    tp_size: int,
    tp_rank: int,
) -> torch.Tensor:
    """Wrap a local TP shard in a checkpoint-global tensor for vLLM loaders."""
    if tp_size <= 1 or weight.ndim == 0:
        return weight

    param = _param_for_loaded_weight(name, params)
    if param is None:
        return weight

    is_sharded_weight = bool(getattr(param, "is_sharded_weight", False))
    use_bitsandbytes_4bit = bool(getattr(param, "use_bitsandbytes_4bit", False))
    if is_sharded_weight or use_bitsandbytes_4bit:
        return weight

    dim = getattr(param, "output_dim", None)
    if dim is None:
        dim = getattr(param, "input_dim", None)
    if dim is None:
        return weight
    dim = int(dim)
    if dim < 0:
        dim += weight.ndim
    if dim < 0 or dim >= weight.ndim:
        return weight

    local_extent = int(weight.shape[dim])
    if dim < param.ndim and local_extent > int(param.shape[dim]):
        return weight

    expanded_shape = list(weight.shape)
    expanded_shape[dim] = local_extent * tp_size
    expanded = torch.empty(
        expanded_shape,
        dtype=weight.dtype,
        device=weight.device,
    )
    expanded.narrow(dim, tp_rank * local_extent, local_extent).copy_(
        weight,
        non_blocking=True,
    )
    return expanded


def _is_1d_and_scalar(
    param: torch.Tensor,
    loaded_weight: torch.Tensor,
) -> bool:
    return (
        param.data.ndim == 1
        and param.data.numel() == 1
        and loaded_weight.ndim == 0
        and loaded_weight.numel() == 1
    )


def _copy_loaded_weight(
    param: torch.Tensor,
    loaded_weight: torch.Tensor,
) -> None:
    if loaded_weight.ndim == 0 and loaded_weight.numel() == 1:
        loaded_weight = loaded_weight.reshape(1)
    assert param.data.shape == loaded_weight.shape or _is_1d_and_scalar(
        param,
        loaded_weight,
    ), (
        f"Attempted to load weight {tuple(loaded_weight.shape)} into "
        f"parameter {tuple(param.data.shape)}"
    )
    param.data.copy_(loaded_weight)


def _fp8_qkv_scale_target_name(name: str) -> str | None:
    """Return the live vLLM Qwen attention scale parameter for a scale tensor."""
    replacements = (
        (".self_attn.attn.attn.q_scale", ".self_attn.attn.q_scale"),
        (".self_attn.attn.q_scale", ".self_attn.attn.q_scale"),
        (".self_attn.q_scale", ".self_attn.attn.q_scale"),
        (".self_attn.attn.attn.k_scale", ".self_attn.attn.k_scale"),
        (".self_attn.attn.k_scale", ".self_attn.attn.k_scale"),
        (".self_attn.k_scale", ".self_attn.attn.k_scale"),
        (".self_attn.attn.attn.v_scale", ".self_attn.attn.v_scale"),
        (".self_attn.attn.v_scale", ".self_attn.attn.v_scale"),
        (".self_attn.v_scale", ".self_attn.attn.v_scale"),
    )
    for old, new in replacements:
        if name.endswith(old):
            return f"{name[: -len(old)]}{new}"
    return None


def _copy_fp8_scale_tensor(
    target: torch.Tensor,
    source: torch.Tensor,
) -> None:
    if source.numel() != 1:
        raise ValueError(
            f"Expected singleton FP8 scale tensor, got shape {tuple(source.shape)}"
        )
    source = source.detach().to(device=target.device, dtype=target.dtype)
    source = source.reshape(target.shape)
    with torch.no_grad():
        target.copy_(source, non_blocking=True)


def _ensure_fp8_prob_scale_parameter(
    *,
    module: torch.nn.Module,
    module_name: str,
    device: torch.device,
    params: dict[str, torch.Tensor],
) -> None:
    existing = getattr(module, "prob_scale", None)
    if isinstance(existing, torch.Tensor):
        params[f"{module_name}.prob_scale"] = existing
        return

    parameter = torch.nn.Parameter(
        torch.full((), -1.0, device=device, dtype=torch.float32),
        requires_grad=False,
    )
    module.register_parameter("prob_scale", parameter)
    params[f"{module_name}.prob_scale"] = parameter


def _ensure_fp8_scale_parameter(
    *,
    model: torch.nn.Module,
    target_name: str,
    source: torch.Tensor,
    params: dict[str, torch.Tensor],
) -> torch.Tensor | None:
    module_name, _, attr_name = target_name.rpartition(".")
    module = dict(model.named_modules()).get(module_name)

    target = params.get(target_name)
    if target is not None:
        if module is not None and attr_name in {"q_scale", "k_scale", "v_scale"}:
            _ensure_fp8_prob_scale_parameter(
                module=module,
                module_name=module_name,
                device=target.device,
                params=params,
            )
        return target

    if module is None or attr_name not in {"q_scale", "k_scale", "v_scale"}:
        return None

    existing = getattr(module, attr_name, None)
    if isinstance(existing, torch.Tensor):
        params[target_name] = existing
        _ensure_fp8_prob_scale_parameter(
            module=module,
            module_name=module_name,
            device=existing.device,
            params=params,
        )
        return existing

    parameter = torch.nn.Parameter(
        torch.full((), -1.0, device=source.device, dtype=torch.float32),
        requires_grad=False,
    )
    module.register_parameter(attr_name, parameter)
    params[target_name] = parameter
    _ensure_fp8_prob_scale_parameter(
        module=module,
        module_name=module_name,
        device=parameter.device,
        params=params,
    )
    return parameter


def _load_fp8_qkv_scale_weights(
    weights: list[tuple[str, torch.Tensor]],
    model: torch.nn.Module,
) -> list[tuple[str, torch.Tensor]]:
    """Load FP8 Q/K/V scale tensors directly into vLLM attention modules."""
    params = dict(model.named_parameters(remove_duplicate=False))
    remaining = []
    loaded_count = 0
    unresolved: list[tuple[str, str]] = []
    for name, tensor in weights:
        target_name = _fp8_qkv_scale_target_name(name)
        if target_name is None:
            remaining.append((name, tensor))
            continue

        target = _ensure_fp8_scale_parameter(
            model=model,
            target_name=target_name,
            source=tensor,
            params=params,
        )
        if target is None:
            unresolved.append((name, target_name))
            remaining.append((name, tensor))
        else:
            _copy_fp8_scale_tensor(target, tensor)
            loaded_count += 1

    if loaded_count:
        logger.info("[mx] directly loaded %d FP8 Q/K/V scale tensors", loaded_count)
    if unresolved:
        sample = ", ".join(f"{name}->{target}" for name, target in unresolved[:3])
        logger.warning(
            "[mx] could not directly load %d FP8 Q/K/V scale tensors; "
            "leaving them for the vLLM loader (sample: %s)",
            len(unresolved),
            sample,
        )
    return remaining


def _tp_rank(param: torch.Tensor) -> int:
    return int(getattr(param, "tp_rank", 0) or 0)


def _block_scale_extent(
    *,
    param_extent: int,
    loaded_extent: int,
    shard_size: int,
    shard_offset: int,
    tp_size: int,
) -> tuple[int, int]:
    if shard_size <= 0:
        return shard_size, shard_offset
    if param_extent == shard_size and loaded_extent >= shard_offset + shard_size:
        return shard_size, shard_offset

    expected_global_blocks = param_extent * max(tp_size, 1)
    if loaded_extent == expected_global_blocks:
        return param_extent, shard_offset // shard_size * param_extent
    if loaded_extent <= expected_global_blocks and shard_size > param_extent:
        blocks = max(1, shard_size // max(param_extent, 1))
        return max(1, (shard_size + blocks - 1) // blocks), shard_offset // blocks
    return shard_size, shard_offset


def _load_column_parallel_weight(
    param: torch.Tensor,
    loaded_weight: torch.Tensor,
) -> None:
    if tuple(param.data.shape) == tuple(loaded_weight.shape):
        _copy_loaded_weight(param, loaded_weight)
        return

    output_dim = getattr(param, "output_dim", None)
    if output_dim is None:
        _copy_loaded_weight(param, loaded_weight)
        return

    output_dim = int(output_dim)
    shard_size = int(param.data.shape[output_dim])
    start_idx = _tp_rank(param) * shard_size
    loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
    _copy_loaded_weight(param, loaded_weight)


def _load_row_parallel_weight(
    param: torch.Tensor,
    loaded_weight: torch.Tensor,
) -> None:
    if tuple(param.data.shape) == tuple(loaded_weight.shape):
        _copy_loaded_weight(param, loaded_weight)
        return

    input_dim = getattr(param, "input_dim", None)
    if input_dim is None:
        _copy_loaded_weight(param, loaded_weight)
        return

    input_dim = int(input_dim)
    shard_size = int(param.data.shape[input_dim])
    start_idx = _tp_rank(param) * shard_size
    loaded_weight = loaded_weight.narrow(input_dim, start_idx, shard_size)
    _copy_loaded_weight(param, loaded_weight)


def _load_merged_column_weight(
    param: torch.Tensor,
    loaded_weight: torch.Tensor,
    **kwargs: Any,
) -> None:
    output_dim = getattr(param, "output_dim", None)
    if output_dim is None:
        _copy_loaded_weight(param, loaded_weight)
        return

    output_dim = int(output_dim)
    shard_offset = int(kwargs["shard_offset"])
    shard_size = int(kwargs["shard_size"])
    param_extent = int(param.data.shape[output_dim])
    loaded_extent = int(loaded_weight.shape[output_dim])
    if loaded_extent <= param_extent:
        local_offset = 0 if shard_offset == 0 else param_extent - loaded_extent
        param_data = param.data.narrow(output_dim, local_offset, loaded_extent)
        assert param_data.shape == loaded_weight.shape, (
            f"Attempted to load local merged shard {tuple(loaded_weight.shape)} "
            f"into parameter shard {tuple(param_data.shape)}"
        )
        param_data.copy_(loaded_weight)
        return

    shard_size, shard_offset = _block_scale_extent(
        param_extent=param_extent,
        loaded_extent=loaded_extent,
        shard_size=shard_size,
        shard_offset=shard_offset,
        tp_size=int(getattr(param, "tp_size", 1) or 1),
    )

    param_data = param.data.narrow(output_dim, shard_offset, shard_size)
    loaded_weight = loaded_weight.narrow(
        output_dim,
        _tp_rank(param) * shard_size,
        shard_size,
    )
    assert param_data.shape == loaded_weight.shape, (
        f"Attempted to load merged shard {tuple(loaded_weight.shape)} into "
        f"parameter shard {tuple(param_data.shape)}"
    )
    param_data.copy_(loaded_weight)


def _shard_id_as_int(shard_id: str | int) -> int:
    if isinstance(shard_id, int):
        return shard_id
    qkv_idxs = {"q": 0, "k": 1, "v": 2}
    return qkv_idxs[shard_id]


def _load_qkv_weight(
    param: torch.Tensor,
    loaded_weight: torch.Tensor,
    **kwargs: Any,
) -> None:
    output_dim = getattr(param, "output_dim", None)
    if output_dim is None:
        _copy_loaded_weight(param, loaded_weight)
        return

    output_dim = int(output_dim)
    shard_offset = int(kwargs.get("shard_offset", 0))
    shard_size = int(kwargs.get("shard_size", param.data.shape[output_dim]))
    shard_id = kwargs.get("shard_id", "q")
    num_heads = int(kwargs.get("num_heads", 1) or 1)
    param_extent = int(param.data.shape[output_dim])
    loaded_extent = int(loaded_weight.shape[output_dim])
    if isinstance(shard_id, str) and loaded_extent <= param_extent:
        if shard_id == "q":
            local_offset = 0
        elif shard_id == "k":
            local_offset = param_extent - 2 * loaded_extent
        else:
            local_offset = param_extent - loaded_extent
        param_data = param.data.narrow(output_dim, local_offset, loaded_extent)
        assert param_data.shape == loaded_weight.shape, (
            f"Attempted to load local qkv shard {tuple(loaded_weight.shape)} into "
            f"parameter shard {tuple(param_data.shape)}"
        )
        param_data.copy_(loaded_weight)
        return

    shard_size, shard_offset = _block_scale_extent(
        param_extent=param_extent,
        loaded_extent=loaded_extent,
        shard_size=shard_size,
        shard_offset=shard_offset,
        tp_size=int(getattr(param, "tp_size", 1) or 1),
    )

    param_data = param.data.narrow(output_dim, shard_offset, shard_size)
    shard_id_int = _tp_rank(param) if shard_id == "q" else _tp_rank(param) // num_heads
    loaded_weight = loaded_weight.narrow(
        output_dim,
        shard_id_int * shard_size,
        shard_size,
    )
    assert param_data.shape == loaded_weight.shape, (
        f"Attempted to load qkv shard {tuple(loaded_weight.shape)} into "
        f"parameter shard {tuple(param_data.shape)}"
    )
    param_data.copy_(loaded_weight)


def _infer_parallel_dim(
    *,
    param: torch.Tensor,
    loaded_weight: torch.Tensor,
    tp_size: int,
    prefer_last: bool,
) -> int | None:
    if param.data.ndim != loaded_weight.ndim:
        return None

    dims = range(param.data.ndim - 1, -1, -1) if prefer_last else range(param.data.ndim)
    fallback: int | None = None
    for dim in dims:
        if any(
            param.data.shape[other_dim] != loaded_weight.shape[other_dim]
            for other_dim in range(param.data.ndim)
            if other_dim != dim
        ):
            continue
        param_extent = int(param.data.shape[dim])
        loaded_extent = int(loaded_weight.shape[dim])
        if loaded_extent == param_extent * max(tp_size, 1):
            return dim
        if loaded_extent >= param_extent:
            fallback = dim
        elif fallback is None:
            fallback = dim
    return fallback


def _prepare_plain_vllm_parameter_loader(
    *,
    param: torch.Tensor,
    loaded_weight: torch.Tensor,
    tp_size: int,
    tp_rank: int,
    parallel_attr: str,
) -> bool:
    if hasattr(param, "load_row_parallel_weight") and hasattr(
        param,
        "load_column_parallel_weight",
    ):
        return False

    if not hasattr(param, "tp_size"):
        param.tp_size = tp_size
    if not hasattr(param, "tp_rank"):
        param.tp_rank = tp_rank

    if getattr(param, parallel_attr, None) is None:
        dim = _infer_parallel_dim(
            param=param,
            loaded_weight=loaded_weight,
            tp_size=tp_size,
            prefer_last=parallel_attr == "input_dim",
        )
        if dim is not None:
            setattr(param, parallel_attr, dim)

    param.load_column_parallel_weight = types.MethodType(
        _load_column_parallel_weight,
        param,
    )
    param.load_row_parallel_weight = types.MethodType(
        _load_row_parallel_weight,
        param,
    )
    param.load_merged_column_weight = types.MethodType(
        _load_merged_column_weight,
        param,
    )
    param.load_qkv_weight = types.MethodType(_load_qkv_weight, param)
    return True


def _patch_vllm_weight_loader_v2_fallbacks() -> bool:
    global _VLLM_WEIGHT_LOADER_V2_PATCHED
    if _VLLM_WEIGHT_LOADER_V2_PATCHED:
        return False

    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear,
        MergedColumnParallelLinear,
        QKVParallelLinear,
        RowParallelLinear,
    )

    def _patch_simple(cls: type[Any], *, parallel_attr: str) -> None:
        original = cls.weight_loader_v2

        def patched_weight_loader_v2(
            self: Any,
            param: torch.Tensor,
            loaded_weight: torch.Tensor,
        ) -> Any:
            _prepare_plain_vllm_parameter_loader(
                param=param,
                loaded_weight=loaded_weight,
                tp_size=int(getattr(self, "tp_size", 1) or 1),
                tp_rank=int(getattr(self, "tp_rank", 0) or 0),
                parallel_attr=parallel_attr,
            )
            return original(self, param, loaded_weight)

        cls.weight_loader_v2 = patched_weight_loader_v2

    def _patch_stacked(cls: type[Any]) -> None:
        original = cls.weight_loader_v2

        def patched_weight_loader_v2(
            self: Any,
            param: torch.Tensor,
            loaded_weight: torch.Tensor,
            loaded_shard_id: Any = None,
        ) -> Any:
            _prepare_plain_vllm_parameter_loader(
                param=param,
                loaded_weight=loaded_weight,
                tp_size=int(getattr(self, "tp_size", 1) or 1),
                tp_rank=int(getattr(self, "tp_rank", 0) or 0),
                parallel_attr="output_dim",
            )
            return original(self, param, loaded_weight, loaded_shard_id)

        cls.weight_loader_v2 = patched_weight_loader_v2

    _patch_simple(ColumnParallelLinear, parallel_attr="output_dim")
    _patch_simple(RowParallelLinear, parallel_attr="input_dim")
    _patch_stacked(MergedColumnParallelLinear)
    _patch_stacked(QKVParallelLinear)
    _VLLM_WEIGHT_LOADER_V2_PATCHED = True
    return True


def _patch_plain_vllm_parameter_loaders(
    *,
    model: torch.nn.Module,
    tp_size: int,
    tp_rank: int,
) -> int:
    patched = 0
    for param in model.parameters():
        if hasattr(param, "load_row_parallel_weight") and hasattr(
            param,
            "load_column_parallel_weight",
        ):
            continue
        if (
            getattr(param, "input_dim", None) is None
            and getattr(param, "output_dim", None) is None
        ):
            continue

        if not hasattr(param, "tp_size"):
            param.tp_size = tp_size
        if not hasattr(param, "tp_rank"):
            param.tp_rank = tp_rank
        param.load_column_parallel_weight = types.MethodType(
            _load_column_parallel_weight,
            param,
        )
        param.load_row_parallel_weight = types.MethodType(
            _load_row_parallel_weight,
            param,
        )
        param.load_merged_column_weight = types.MethodType(
            _load_merged_column_weight,
            param,
        )
        param.load_qkv_weight = types.MethodType(_load_qkv_weight, param)
        patched += 1
    return patched


def _patch_parameter_weight_loader_callables(
    *,
    model: torch.nn.Module,
    tp_size: int,
    tp_rank: int,
) -> int:
    patched = 0
    for param in model.parameters():
        if getattr(param, "_mx_weight_loader_wrapped", False):
            continue
        weight_loader = getattr(param, "weight_loader", None)
        if not callable(weight_loader):
            continue

        layer = getattr(weight_loader, "__self__", None)
        layer_name = layer.__class__.__name__ if layer is not None else ""
        parallel_attr = (
            "input_dim" if layer_name == "RowParallelLinear" else "output_dim"
        )

        def wrapped_weight_loader(
            loader_param: torch.Tensor,
            loaded_weight: torch.Tensor,
            *args: Any,
            _weight_loader: Any = weight_loader,
            _layer: Any = layer,
            _parallel_attr: str = parallel_attr,
            **kwargs: Any,
        ) -> Any:
            _prepare_plain_vllm_parameter_loader(
                param=loader_param,
                loaded_weight=loaded_weight,
                tp_size=int(getattr(_layer, "tp_size", tp_size) or tp_size),
                tp_rank=int(getattr(_layer, "tp_rank", tp_rank) or tp_rank),
                parallel_attr=_parallel_attr,
            )
            return _weight_loader(loader_param, loaded_weight, *args, **kwargs)

        param.weight_loader = wrapped_weight_loader
        param._mx_weight_loader_wrapped = True
        patched += 1
    return patched


def _torch_dtype(dtype_name: str) -> torch.dtype:
    dtype_name = str(dtype_name).removeprefix("torch.")
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "half": torch.float16,
        "float32": torch.float32,
        "float": torch.float32,
    }.get(dtype_name, torch.bfloat16)


class MxRefitWorkerExtension:
    """Methods injected into vLLM's Worker via ``worker_extension_cls``."""

    def prepare_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        self._mx_state_dict_info = state_dict_info
        if not hasattr(self, "state_dict_info"):
            self.state_dict_info = state_dict_info

    def _mx_uses_fp8_quantization(self) -> bool:
        vllm_config = getattr(self.model_runner, "vllm_config", None)
        configs = (
            vllm_config,
            getattr(vllm_config, "model_config", None),
            getattr(self.model_runner, "model_config", None),
        )
        for config in configs:
            if config is None:
                continue
            quant_config = getattr(config, "quant_config", None)
            if (
                quant_config is not None
                and quant_config.__class__.__name__ == "Fp8Config"
            ):
                return True
            quantization = getattr(config, "quantization", None)
            if quantization is not None and str(quantization).lower() == "fp8":
                return True
        return False

    def _mx_get_fp8_module(self) -> Any:
        from dynamo.vllm.mx_refit import fp8 as fp8_module

        if getattr(fp8_module, "global_fp8_config", None) is None:
            tp_size, _ = _target_tp(self)
            cache_config = getattr(
                getattr(self.model_runner, "vllm_config", None),
                "cache_config",
                None,
            )
            kv_cache_dtype = getattr(cache_config, "cache_dtype", "auto")
            fp8_module.ensure_fp8_config(
                model_parallel_size=tp_size,
                kv_cache_dtype=str(kv_cache_dtype),
                use_fp8_weights=True,
            )
            logger.info(
                "[mx] initialized Dynamo FP8 loader config for refit "
                "(tp_size=%d, kv_cache_dtype=%s)",
                tp_size,
                kv_cache_dtype,
            )
        return fp8_module

    def _mx_load_fp8_weights(self, weights: list[tuple[str, torch.Tensor]]) -> None:
        fp8_module = self._mx_get_fp8_module()
        model = self.model_runner.model
        tp_size, tp_rank = _target_tp(self)
        weights = _load_fp8_qkv_scale_weights(weights, model)
        if _patch_vllm_weight_loader_v2_fallbacks():
            logger.info("[mx] patched vLLM weight_loader_v2 fallbacks for FP8 refit")
        wrapped = _patch_parameter_weight_loader_callables(
            model=model,
            tp_size=tp_size,
            tp_rank=tp_rank,
        )
        if wrapped:
            logger.info("[mx] wrapped %d vLLM parameter weight loaders", wrapped)
        patched = _patch_plain_vllm_parameter_loaders(
            model=model,
            tp_size=tp_size,
            tp_rank=tp_rank,
        )
        if patched:
            logger.info(
                "[mx] patched %d plain vLLM parameters with refit loaders "
                "(tp_size=%d, tp_rank=%d)",
                patched,
                tp_size,
                tp_rank,
            )
        old_do_torchao_reload = getattr(model, "_do_torchao_reload", None)
        if old_do_torchao_reload is not None:
            model._do_torchao_reload = False
        try:
            fp8_module.load_weights(weights, self.model_runner)
        finally:
            if old_do_torchao_reload is not None:
                model._do_torchao_reload = old_do_torchao_reload

    def _mx_load_weights(self, weights: list[tuple[str, torch.Tensor]]) -> None:
        tp_size, tp_rank = _target_tp(self)
        uses_fp8_quantization = self._mx_uses_fp8_quantization()
        if tp_size > 1:
            params = dict(self.model_runner.model.named_parameters())
            adapted_weights = []
            for name, weight in weights:
                if not uses_fp8_quantization and _maybe_copy_tp_local_weight(
                    name=name,
                    weight=weight,
                    params=params,
                ):
                    continue
                adapted_weights.append(
                    (
                        name,
                        _maybe_expand_tp_local_weight(
                            name=name,
                            weight=weight,
                            params=params,
                            tp_size=tp_size,
                            tp_rank=tp_rank,
                        ),
                    )
                )
            weights = adapted_weights
        if uses_fp8_quantization:
            logger.info("[mx] loading refit tensors through Dynamo FP8 loader")
            self._mx_load_fp8_weights(weights)
        else:
            self.model_runner.model.load_weights(weights=weights)

    def _mx_maybe_process_fp8_kv_cache(self) -> None:
        cache_config = getattr(self.model_runner.vllm_config, "cache_config", None)
        kv_cache_dtype = getattr(cache_config, "cache_dtype", None)
        if kv_cache_dtype is None or "fp8" not in str(kv_cache_dtype).lower():
            return

        from vllm.model_executor.model_loader.utils import (
            process_weights_after_loading,
        )

        target_device = next(self.model_runner.model.parameters()).device
        process_weights_after_loading(
            self.model_runner.model,
            self.model_runner.model_config,
            target_device,
        )

    def _mx_init_receiver(self, mx_config: MxConfig) -> None:
        if getattr(self, "_mx_receiver", None):
            return

        from modelexpress import MxV2RefitReceiver

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        device_id = _device_index(self.device)
        _pin_local_nic(device_id=device_id, mode=mx_config.nic_pin)
        pod_hostname = socket.gethostname()
        self._mx_receiver = MxV2RefitReceiver(
            agent_name=f"dynamo-vllm-{pod_hostname}-r{rank}",
            device_id=device_id,
            mx_server_url=mx_config.mx_server_url,
            worker_rank=rank,
        )

        publish_tensors = (
            dict(self.model_runner.model.named_parameters())
            if mx_config.tree_scale_out
            else None
        )
        self._mx_receiver.initialize(model_tensors=publish_tensors)
        logger.info(
            "[mx] receiver initialized rank=%d device=%d publish_buffers=%d",
            rank,
            device_id,
            len(publish_tensors or {}),
        )

    def update_weights_via_mx(
        self,
        *,
        version: int,
        mx_config: Any = None,
    ) -> bool:
        try:
            if not isinstance(mx_config, MxConfig):
                mx_config = MxConfig.from_dict(mx_config or {})

            self._mx_init_receiver(mx_config)
            candidates = self._mx_receiver.discover_v2_sources(
                model_name=_model_name(self),
                min_version=int(version),
                same_rank_only=mx_config.same_rank_only,
                include_replicas=mx_config.tree_scale_out,
            )
            if not candidates:
                logger.warning(
                    "[mx] no v2 source available for version>=%d on rank %d",
                    version,
                    self._mx_receiver.worker_rank,
                )
                return False

            if any(c.megatron_meta is not None for c in candidates):
                return self._mx_update_weights_via_mx_megatron(
                    candidates=candidates,
                    version=int(version),
                    mx_config=mx_config,
                )

            return self._mx_update_weights_via_mx_dtensor(
                candidates=candidates,
                version=int(version),
                mx_config=mx_config,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "[mx] update_weights_via_mx failed on rank=%s: %s\n%s",
                getattr(getattr(self, "_mx_receiver", None), "worker_rank", -1),
                exc,
                traceback.format_exc(),
            )
            return False

    def _mx_update_weights_via_mx_dtensor(
        self,
        *,
        candidates: list[Any],
        version: int,
        mx_config: MxConfig,
    ) -> bool:
        chosen = self._mx_receiver.pick_best_source(candidates)
        if chosen is None:
            logger.warning(
                "[mx] no candidate covers required experts on rank %d",
                self._mx_receiver.worker_rank,
            )
            return False

        tensor_shapes: dict[str, tuple[int, ...]] = {}
        registry = getattr(chosen, "registry", None)
        if registry:
            for td in registry.get("tensors", []):
                tensor_shapes[td.name] = tuple(int(s) for s in td.global_shape)

        from modelexpress.nemo_rl_v2 import ROLE_INFERENCE_REPLICA

        if chosen.role == ROLE_INFERENCE_REPLICA and not tensor_shapes:
            tensor_shapes = {
                name: tuple(p.shape)
                for name, p in self.model_runner.model.named_parameters()
            }

        weights = list(
            self._mx_receiver._receiver.receive_weights_scratch(
                chosen.ref,
                timeout_seconds=mx_config.timeout_seconds,
                tensor_shapes=tensor_shapes or None,
            )
        )

        if chosen.role == ROLE_INFERENCE_REPLICA:
            model_params = dict(self.model_runner.model.named_parameters())
            for name, tensor in weights:
                param = model_params.get(name)
                if param is not None:
                    with torch.no_grad():
                        param.copy_(tensor, non_blocking=True)
        else:
            self._mx_load_weights(weights)

        torch.cuda.current_stream().synchronize()
        self._mx_maybe_process_fp8_kv_cache()
        if mx_config.tree_scale_out:
            self._mx_receiver.publish_self_as_source(
                version=int(version),
                model_name=_model_name(self),
            )
        gc.collect()
        torch.cuda.empty_cache()
        return True

    def _build_megatron_context(self, candidates: list[Any]) -> Any:
        from modelexpress.megatron_translator import (
            MegatronReceiverContext,
            ReceiveSpec,
            discover_megatron_context,
        )
        from modelexpress.nemo_rl_v2 import TargetTpLayout

        cfg, name_map = discover_megatron_context(candidates)
        if cfg is None:
            raise RuntimeError(
                "Megatron MX sources did not publish megatron_transformer_config"
            )

        tp_size, tp_rank = _target_tp(self)
        receive_specs: dict[str, ReceiveSpec] = {}
        for cand in candidates:
            if cand.megatron_meta is None or cand.registry is None:
                continue
            for td in cand.registry.get("tensors", []):
                if td.name in receive_specs or not td.megatron_role:
                    continue
                lookup_name = (
                    td.name[len("module.") :]
                    if td.name.startswith("module.")
                    else td.name
                )
                hf_names = name_map.get(lookup_name, name_map.get(td.name, [td.name]))
                receive_specs[td.name] = ReceiveSpec(
                    megatron_name=td.name,
                    hf_names=list(hf_names),
                    role=td.megatron_role,
                    target_shape=tuple(int(s) for s in td.global_shape),
                    target_dtype=td.dtype,
                    shard_axis=int(td.shard_axis),
                    pp_rank=cand.megatron_meta.pp_rank,
                    role_descriptor=dict(td.megatron_extras or {}),
                )

        logger.info(
            "[mx-megatron] built context tp=%d rank=%d tensors=%d",
            tp_size,
            tp_rank,
            len(receive_specs),
        )
        return MegatronReceiverContext(
            target_tp_layout=TargetTpLayout(tp_size=tp_size, tp_rank=tp_rank),
            transformer_config=cfg,
            hf_name_map=name_map,
            receive_specs=receive_specs,
        )

    def _mx_pull_megatron_vocab_buffers(
        self,
        *,
        candidates: list[Any],
        ctx: Any,
        mx_config: MxConfig,
    ) -> None:
        vocab_buffers = getattr(self, "_mx_megatron_vocab_buffers", {})
        if not vocab_buffers:
            return

        megatron_cands = sorted(
            [c for c in candidates if c.megatron_meta is not None],
            key=lambda c: c.megatron_meta.tp_rank,
        )
        for cand in megatron_cands:
            batch = []
            for name, dest in vocab_buffers.items():
                spec = ctx.receive_specs[name]
                axis = int(spec.shard_axis)
                if axis != 0:
                    raise RuntimeError(
                        f"vocab_parallel tensor {name!r} uses unsupported "
                        f"shard_axis={axis}; expected 0"
                    )
                rows = int(spec.target_shape[axis])
                lo = int(cand.megatron_meta.tp_rank) * rows
                view = dest.narrow(axis, lo, rows)
                if not view.is_contiguous():
                    raise RuntimeError(
                        f"vocab_parallel destination for {name!r} is not contiguous"
                    )
                batch.append((name, None, view))
            if batch:
                self._mx_receiver._receiver.pull_to(
                    cand.ref,
                    batch,
                    timeout_seconds=mx_config.timeout_seconds,
                )

    def _mx_update_weights_via_mx_megatron(
        self,
        *,
        candidates: list[Any],
        version: int,
        mx_config: MxConfig,
    ) -> bool:
        from modelexpress.megatron_translator import run_refit_cycle
        from modelexpress.nemo_rl_v2 import ROLE_MEGATRON_VOCAB_PARALLEL

        if not getattr(self, "_mx_megatron_ctx", None):
            self._mx_megatron_ctx = self._build_megatron_context(candidates)

        ctx = self._mx_megatron_ctx
        if not hasattr(self, "_mx_megatron_buffers"):
            buffers: dict[str, torch.Tensor] = {}
            vocab_buffers: dict[str, torch.Tensor] = {}
            source_tp_size = next(
                (
                    c.megatron_meta.tp_size
                    for c in candidates
                    if c.megatron_meta is not None and c.megatron_meta.tp_size > 0
                ),
                ctx.target_tp_layout.tp_size,
            )
            for spec in ctx.receive_specs.values():
                if spec.role.startswith("expert_"):
                    continue
                shape = list(spec.target_shape)
                target = buffers
                if spec.role == ROLE_MEGATRON_VOCAB_PARALLEL:
                    shape[int(spec.shard_axis)] *= int(source_tp_size)
                    target = vocab_buffers
                # The Megatron registry shape reflects the published buffer
                # shape. Vocab tensors are the exception: vLLM's loader wants
                # the full vocab tensor and slices it internally for TP.
                target[spec.megatron_name] = torch.empty(
                    shape,
                    dtype=_torch_dtype(spec.target_dtype),
                    device=self.device,
                )
            all_buffers = dict(buffers)
            all_buffers.update(vocab_buffers)
            if all_buffers:
                self._mx_receiver._receiver._nixl.register_tensors(all_buffers)
            self._mx_megatron_buffers = buffers
            self._mx_megatron_vocab_buffers = vocab_buffers
            logger.info(
                "[mx-megatron] registered %d per-rank buffers and %d full-vocab buffers",
                len(buffers),
                len(vocab_buffers),
            )

        matched = next(
            (
                c
                for c in candidates
                if c.megatron_meta is not None
                and c.megatron_meta.tp_rank == ctx.target_tp_layout.tp_rank
            ),
            None,
        )
        source_tp_size = next(
            (
                c.megatron_meta.tp_size
                for c in candidates
                if c.megatron_meta is not None and c.megatron_meta.tp_size > 0
            ),
            None,
        )
        if matched is None or (
            source_tp_size is not None
            and source_tp_size != ctx.target_tp_layout.tp_size
        ):
            raise RuntimeError(
                "Dynamo Megatron MX refit currently supports matched TP only "
                f"(target_tp={ctx.target_tp_layout.tp_size}, source_tp={source_tp_size})."
            )

        self._mx_receiver._receiver._nixl.rebind_tensors(self._mx_megatron_buffers)
        for _name, _tensor in self._mx_receiver.receive_from(
            matched,
            timeout_seconds=mx_config.timeout_seconds,
        ):
            pass

        self._mx_pull_megatron_vocab_buffers(
            candidates=candidates,
            ctx=ctx,
            mx_config=mx_config,
        )

        def _noop_pull(_src: Any, _dest: torch.Tensor) -> None:
            return

        pre_assembled_buffers = dict(self._mx_megatron_buffers)
        pre_assembled_buffers.update(self._mx_megatron_vocab_buffers)
        weights = list(
            run_refit_cycle(
                self._mx_receiver,
                candidates=candidates,
                context=ctx,
                pull=_noop_pull,
                device=self.device,
                pre_assembled_buffers=pre_assembled_buffers,
            )
        )
        if not weights:
            logger.warning(
                "[mx-megatron] no translated tensors for version %d", version
            )
            return False

        self._mx_load_weights(weights)
        torch.cuda.current_stream().synchronize()
        self._mx_maybe_process_fp8_kv_cache()
        if mx_config.tree_scale_out:
            self._mx_receiver.publish_self_as_source(
                version=int(version),
                model_name=_model_name(self),
            )
        gc.collect()
        torch.cuda.empty_cache()
        return True


__all__ = ["MxConfig", "MxRefitWorkerExtension"]
