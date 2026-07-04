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
import time as _time
from dataclasses import dataclass, field
from typing import Any

import torch

logger = logging.getLogger(__name__)


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


def _process_fp8_kv_cache_modules(
    model: torch.nn.Module,
    target_device: torch.device,
) -> int:
    from dynamo.vllm.mx_refit.fp8 import process_weights_after_loading_kv
    from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
    from vllm.model_executor.model_loader.utils import device_loading_context

    processed = 0
    for _, module in model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if not isinstance(quant_method, BaseKVCacheMethod):
            continue
        with device_loading_context(module, target_device):
            process_weights_after_loading_kv(quant_method, module)
        processed += 1
    return processed


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


def _torch_dtype(dtype_name: str) -> torch.dtype:
    dtype_name = str(dtype_name).removeprefix("torch.")
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "half": torch.float16,
        "float32": torch.float32,
        "float": torch.float32,
        "int64": torch.int64,
        "long": torch.long,
    }.get(dtype_name, torch.bfloat16)


def _split_policy_and_draft_weights(
    weights: list[tuple[str, torch.Tensor]],
) -> tuple[list[tuple[str, torch.Tensor]], list[tuple[str, torch.Tensor]]]:
    policy_weights = []
    draft_weights = []
    for name, tensor in weights:
        if name.startswith("draft."):
            draft_weights.append((name.removeprefix("draft."), tensor))
        else:
            policy_weights.append((name, tensor))
    return policy_weights, draft_weights


def _derive_qwen_llama_hf_names(megatron_name: str, role: str | None) -> list[str]:
    name = megatron_name.removeprefix("module.")

    if name.startswith("decoder.layers."):
        parts = name.split(".")
        if len(parts) < 4:
            return [name]
        layer = parts[2]
        suffix = ".".join(parts[3:])
        layer_prefix = f"model.layers.{layer}"

        if suffix == "self_attention.linear_qkv.weight":
            return [
                f"{layer_prefix}.self_attn.q_proj.weight",
                f"{layer_prefix}.self_attn.k_proj.weight",
                f"{layer_prefix}.self_attn.v_proj.weight",
            ]
        if suffix == "self_attention.linear_proj.weight":
            return [f"{layer_prefix}.self_attn.o_proj.weight"]
        if suffix == "self_attention.q_layernorm.weight":
            return [f"{layer_prefix}.self_attn.q_norm.weight"]
        if suffix == "self_attention.k_layernorm.weight":
            return [f"{layer_prefix}.self_attn.k_norm.weight"]
        if suffix == "mlp.linear_fc1.weight" and role == "gated_mlp_column":
            return [
                f"{layer_prefix}.mlp.gate_proj.weight",
                f"{layer_prefix}.mlp.up_proj.weight",
            ]
        if suffix == "mlp.linear_fc2.weight":
            return [f"{layer_prefix}.mlp.down_proj.weight"]
        if suffix == "input_layernorm.weight":
            return [f"{layer_prefix}.input_layernorm.weight"]
        if suffix == "pre_mlp_layernorm.weight":
            return [f"{layer_prefix}.post_attention_layernorm.weight"]

    if name == "decoder.final_layernorm.weight":
        return ["model.norm.weight"]
    if name == "embedding.word_embeddings.weight":
        return ["model.embed_tokens.weight"]
    if name == "output_layer.weight":
        return ["lm_head.weight"]
    return [name]


def _resolve_hf_names(
    megatron_name: str,
    role: str | None,
    name_map: dict[str, list[str]],
) -> list[str]:
    lookup_name = megatron_name.removeprefix("module.")
    hf_names = name_map.get(lookup_name, name_map.get(megatron_name))
    if hf_names is None:
        return _derive_qwen_llama_hf_names(megatron_name, role)

    hf_names = list(hf_names)
    if hf_names in ([megatron_name], [lookup_name]):
        return _derive_qwen_llama_hf_names(megatron_name, role)
    return [name.removeprefix("module.") for name in hf_names]


def _trim_draft_vocab_padding(
    draft_model: torch.nn.Module,
    draft_weights: list[tuple[str, torch.Tensor]],
) -> list[tuple[str, torch.Tensor]]:
    vocab_sizes = {
        name: int(module.org_vocab_size)
        for name, module in draft_model.named_modules()
        if hasattr(module, "org_vocab_size")
    }
    if not vocab_sizes:
        return draft_weights

    trimmed = []
    for name, tensor in draft_weights:
        for module_name, org_vocab_size in vocab_sizes.items():
            leaf_name = module_name.rsplit(".", 1)[-1]
            if leaf_name in name and tensor.shape[0] > org_vocab_size:
                tensor = tensor[:org_vocab_size]
                break
        trimmed.append((name, tensor))
    return trimmed


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
        weights = _load_fp8_qkv_scale_weights(weights, model)
        old_do_torchao_reload = getattr(model, "_do_torchao_reload", None)
        if old_do_torchao_reload is not None:
            model._do_torchao_reload = False
        try:
            fp8_module.load_weights(weights, self.model_runner)
        finally:
            if old_do_torchao_reload is not None:
                model._do_torchao_reload = old_do_torchao_reload


    # ===== ModelExpress net-new refit features (MDL / EP-filter /
    # Phase-0.5 pinned-CPU staging / MoE swizzle guard / byte-identity
    # verify). Additive + env-gated; default behavior unchanged. =====
    _MX_STACKED_GROUPS = (
        ("qkv_proj", "q_proj", 0),
        ("qkv_proj", "k_proj", 1),
        ("qkv_proj", "v_proj", 2),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    )

    def _mx_build_fused_dest_map(
        self,
        weights: list[tuple[str, torch.Tensor]],
    ) -> None:
        """Precompute, per HF tensor, its destination in the vLLM model.

        MDL (Mapped Direct Load): rather than lean on vLLM's stock
        ``load_weights`` (fragile per-arch traversal; the Qwen3-MoE
        fused-layout bug we hit on 2026-07-03 lives there), we resolve
        each received HF tensor to exactly ONE of three destinations,
        computed ONCE. This is our own eager, destination-mapped
        in-place write — inspired by RDT's "don't re-run the loader
        each refit" goal, but NOT RDT: no lazy tensors, no deferred
        narrow(), no dependency on the RDT API. We eagerly RDMA-pull
        then copy into the params' final slots, because NeMo-RL already
        knows its target layout (declared TargetTpLayout) so the dynamic
        layout discovery RDT's laziness buys is unnecessary here.

          * ``direct``  — ``hf_name`` matches a vLLM param 1:1 by shape.
            Warm cycle does ``param.data.copy_(tensor)``.
          * ``fused``   — ``hf_name`` is a member of a stacked param
            (q/k/v -> qkv_proj, gate/up -> gate_up_proj). Warm cycle
            does ``param.data.narrow(0, offset, size).copy_(tensor)``.
            Offsets are derived from the ACTUAL member tensor shapes
            (not model config), so this is version-robust: whatever
            order/size vLLM allocated, we match it by summing member
            rows in canonical (q,k,v / gate,up) order.
          * ``expert``  — per-expert MoE tensor (gate/up/down_proj)
            resolved via vLLM's ``get_expert_mapping()`` to a slot in
            the stacked ``w13_weight`` / ``w2_weight`` param. Warm cycle
            does ``param.data[expert_id].narrow(axis, offset, size).
            copy_(tensor)``. Requires the standard 3D layout
            (``--moe-backend triton``); the swizzled 4D ``auto`` layout
            routes to fallback (and the swizzle guard errors first).
          * ``fallback`` — anything else, or MoE experts under a
            swizzled backend. Warm cycle routes these through vLLM's
            stock loader. On dense + triton-MoE this set is EMPTY.

        Built after cycle 1's stock load so ``named_parameters`` is
        populated. Idempotent — only rebuilds if not present.
        """
        params = self._mx_param_cache
        direct: dict[str, "torch.Tensor"] = {}
        # fused: hf_name -> (fused_param, axis, offset, size)
        fused: dict[str, tuple] = {}
        fallback_names: set[str] = set()

        # expert dest: hf_name -> (fused_param, expert_id, axis, offset, size)
        # for MoE per-expert tensors written into the stacked w13/w2 params.
        expert: dict[str, tuple] = {}

        # Build the MoE expert-name → (fused_param, expert_id, shard_id)
        # lookup from vLLM's own authoritative mapping table, so we don't
        # hardcode name transforms. Empty on dense models / when the
        # backend is swizzled (see _mx_check_moe_swizzle — that guard
        # fires first). shard_id: w1=gate, w3=up, w2=down.
        expert_lookup: dict[str, tuple] = {}  # weight_suffix -> (param_suffix, expert_id, shard_id)
        try:
            model = self.model_runner.model
            if hasattr(model, "get_expert_mapping"):
                for param_suffix, weight_suffix, expert_id, shard_id in model.get_expert_mapping():
                    expert_lookup[weight_suffix] = (param_suffix, int(expert_id), shard_id)
        except Exception as exc:  # noqa: BLE001
            logger.info("[mx-mdl] no expert mapping (dense or unavailable): %s", exc)

        # First, bucket the stacked-group members by their fused param
        # name so we can accumulate offsets in canonical order.
        # group_key = fused_param_name; value = list of
        # (order, hf_name, member_rows).
        groups: dict[str, list[tuple[int, str, int]]] = {}
        name_to_shape = {n: tuple(t.shape) for n, t in weights}

        for hf_name in name_to_shape:
            param = params.get(hf_name)
            if param is not None and tuple(param.shape) == name_to_shape[hf_name]:
                direct[hf_name] = param
                continue
            # MoE per-expert tensor? Resolve via vLLM's expert mapping.
            if ".experts." in hf_name and expert_lookup:
                dest = self._mx_resolve_expert_dest(
                    hf_name, name_to_shape[hf_name], expert_lookup, params,
                )
                if dest is not None:
                    expert[hf_name] = dest
                    continue
            # Try stacked-group membership.
            matched = False
            for fused_suffix, member_suffix, order in self._MX_STACKED_GROUPS:
                if member_suffix + "." in hf_name or hf_name.endswith(member_suffix + ".weight"):
                    fused_name = hf_name.replace(member_suffix, fused_suffix)
                    fused_param = params.get(fused_name)
                    if fused_param is None:
                        continue
                    member_rows = name_to_shape[hf_name][0]
                    groups.setdefault(fused_name, []).append(
                        (order, hf_name, member_rows)
                    )
                    matched = True
                    break
            if not matched:
                fallback_names.add(hf_name)

        # Resolve offsets within each fused group (canonical order).
        for fused_name, members in groups.items():
            fused_param = params[fused_name]
            members.sort(key=lambda m: m[0])
            offset = 0
            for _order, hf_name, member_rows in members:
                fused[hf_name] = (fused_param, 0, offset, member_rows)
                offset += member_rows
            # Sanity: total should equal the fused param's axis-0 size.
            if offset != int(fused_param.shape[0]):
                logger.warning(
                    "[mx-mdl] fused group %s: member rows sum to %d but "
                    "param axis-0 is %d; routing group to fallback",
                    fused_name, offset, int(fused_param.shape[0]),
                )
                for _o, hf_name, _r in members:
                    fused.pop(hf_name, None)
                    fallback_names.add(hf_name)

        self._mx_mdl_direct = direct
        self._mx_mdl_fused = fused
        self._mx_mdl_expert = expert
        self._mx_mdl_fallback = fallback_names
        logger.info(
            "[mx-mdl] dest map built: %d direct, %d fused-slice, "
            "%d expert-slice, %d fallback",
            len(direct), len(fused), len(expert), len(fallback_names),
        )

    def _mx_resolve_expert_dest(
        self,
        hf_name: str,
        hf_shape: tuple,
        expert_lookup: dict,
        params: dict,
    ) -> tuple | None:
        """Resolve a per-expert HF tensor to its slot in the stacked w13/w2 param.

        Returns ``(fused_param, local_expert_idx, axis, offset, size)``
        for the warm-cycle write ``fused_param.data[local_expert_idx].
        narrow(axis, offset, size).copy_(tensor)``, or ``None`` to route
        to fallback.

        Layout (vLLM standard / --moe-backend triton):
          * w1 (gate): w13_weight[E] rows [0, inter)      → axis 0, offset 0
          * w3 (up):   w13_weight[E] rows [inter, 2*inter) → axis 0, offset inter
          * w2 (down): w2_weight[E] full                  → axis 0, offset 0, full

        EP>1 correctness: ``get_expert_mapping()`` yields a GLOBAL expert
        id, but under expert-parallel the vLLM param only holds this
        rank's LOCAL experts, so the param index must be the local slot.
        We map global→local via the owning FusedMoE module's
        ``_map_global_expert_id_to_local_expert_id`` (mirrors vLLM's own
        weight_loader). At EP=1 that map is identity, so this is a no-op
        for the validated single-rank path. A global id not local to
        this rank maps to -1 → routed to fallback/skipped (the EP filter
        should have pruned it from the pull upstream anyway).

        Guards: only accepts if the resolved fused param exists, is 3D
        (standard stacked, NOT the swizzled 4D layout — that routes to
        fallback and the swizzle guard errors), the local index is in
        range, and the destination slice shape matches the received
        tensor exactly (protects the TP assumption; TP>1 per-expert
        sharding would mismatch and route to fallback).
        """
        # Match the received name against vLLM's weight suffixes.
        for weight_suffix, (param_suffix, expert_id, shard_id) in expert_lookup.items():
            if weight_suffix not in hf_name:
                continue
            fused_name = hf_name.replace(weight_suffix, param_suffix)
            fused_param = params.get(fused_name)
            if fused_param is None or fused_param.ndim != 3:
                return None  # swizzled/absent → fallback

            # global → local expert index (identity at EP=1).
            local_idx = self._mx_map_global_to_local_expert(fused_name, expert_id)
            if local_idx is None or local_idx < 0 or local_idx >= int(fused_param.shape[0]):
                return None  # not local to this rank → fallback/skip

            per_expert = fused_param.shape[1]  # axis-0 size of param.data[E]
            rows = hf_shape[0]
            if shard_id == "w1":       # gate → first half
                axis, offset, size = 0, 0, rows
            elif shard_id == "w3":     # up → second half
                axis, offset, size = 0, rows, rows
            else:                       # w2 (down) → full slot
                axis, offset, size = 0, 0, int(per_expert)
            # Shape sanity: the narrowed slot must equal the received tensor.
            if size != rows and shard_id in ("w1", "w3"):
                return None
            if shard_id == "w2" and int(per_expert) != rows:
                return None
            return (fused_param, local_idx, axis, offset, size)
        return None

    def _mx_map_global_to_local_expert(
        self, fused_param_name: str, global_expert_id: int
    ) -> int | None:
        """Map a global expert id to this rank's local slot index.

        Resolves the owning FusedMoE module from the fused param name
        (e.g. ``model.layers.3.mlp.experts.w13_weight`` -> the
        ``...mlp.experts`` module) and calls its
        ``_map_global_expert_id_to_local_expert_id``. Returns the local
        index, ``-1`` if the expert isn't on this rank, or the global id
        unchanged if the module/method can't be resolved (EP=1 identity
        fallback). Cached per fused-param module to avoid re-walking.
        """
        cache = getattr(self, "_mx_moe_module_cache", None)
        if cache is None:
            cache = {}
            self._mx_moe_module_cache = cache
        module = cache.get(fused_param_name, "MISS")
        if module == "MISS":
            mod_path = fused_param_name.rsplit(".", 1)[0]
            obj = self.model_runner.model
            for part in mod_path.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            module = obj
            cache[fused_param_name] = module
        if module is not None and hasattr(
            module, "_map_global_expert_id_to_local_expert_id"
        ):
            try:
                return int(
                    module._map_global_expert_id_to_local_expert_id(global_expert_id)
                )
            except Exception:  # noqa: BLE001
                return global_expert_id
        # No EP map available (EP=1 / dense-style): identity.
        return global_expert_id

    def _mx_buffer_device(self) -> "torch.device":
        """Phase 0.5: ``MX_MEGATRON_BUFFER_LOC=host`` puts the NIXL-registered
        receive buffers in host (pinned) RAM to free ~model-shard-sized HBM
        (8.77 GB on 4B, ~61 GB on 30B — the difference between fitting and OOM
        on a 190 GB GB200). Default ``device`` keeps them on GPU (unchanged).
        """
        loc = os.environ.get("MX_MEGATRON_BUFFER_LOC", "device").lower()
        if loc == "host":
            return torch.device("cpu")
        if loc not in ("device", "host"):
            logger.warning(
                "[mx] MX_MEGATRON_BUFFER_LOC=%r not recognized; using device", loc
            )
        return self.device

    def _mx_mdl_note_cold(self, weights: list[tuple[str, torch.Tensor]]) -> None:
        """MDL cold cycle: after the stock load, cache params + build the
        destination map. No-op unless ``MX_LOAD_MODE=direct``. Gated to the
        validated regime (TP=1, non-FP8); other regimes stay on the stock path.
        """
        if os.environ.get("MX_LOAD_MODE", "stock").lower() != "direct":
            return
        if getattr(self, "_mx_mdl_direct", None) is not None:
            return  # already built
        tp_size, _ = _target_tp(self)
        if tp_size != 1 or self._mx_uses_fp8_quantization():
            return
        self._mx_check_moe_swizzle()
        self._mx_param_cache = dict(self.model_runner.model.named_parameters())
        self._mx_build_fused_dest_map(weights)

    def _mx_try_mdl_warm(self, weights: list[tuple[str, torch.Tensor]]) -> bool:
        """MDL warm cycle: write each tensor to its precomputed destination with
        zero stock-loader calls for mapped tensors (``param.data.copy_`` for
        1:1, ``param.narrow(...).copy_`` for stacked members, per-expert slot
        for MoE). Returns True if it handled the load. No-op (returns False)
        unless ``MX_LOAD_MODE=direct`` and the dest map has been built.
        """
        if os.environ.get("MX_LOAD_MODE", "stock").lower() != "direct":
            return False
        if getattr(self, "_mx_mdl_direct", None) is None:
            return False
        direct_hits = fused_hits = expert_hits = 0
        fallback: list[tuple[str, torch.Tensor]] = []
        _t0 = _time.perf_counter()
        with torch.no_grad():
            for hf_name, tensor in weights:
                dest = self._mx_mdl_fused.get(hf_name)
                if dest is not None:
                    param, axis, offset, size = dest
                    param.data.narrow(axis, offset, size).copy_(tensor, non_blocking=True)
                    fused_hits += 1
                    continue
                edest = self._mx_mdl_expert.get(hf_name)
                if edest is not None:
                    param, eid, axis, offset, size = edest
                    param.data[eid].narrow(axis, offset, size).copy_(tensor, non_blocking=True)
                    expert_hits += 1
                    continue
                param = self._mx_mdl_direct.get(hf_name)
                if param is not None and tuple(param.shape) == tuple(tensor.shape):
                    param.data.copy_(tensor, non_blocking=True)
                    direct_hits += 1
                    continue
                fallback.append((hf_name, tensor))
        if fallback:
            self.model_runner.model.load_weights(weights=fallback)
        logger.info(
            "[mx-mdl] warm-cycle: %d direct + %d fused-slice + %d expert-slice "
            "in %.3fs, %d fallback via stock",
            direct_hits, fused_hits, expert_hits,
            _time.perf_counter() - _t0, len(fallback),
        )
        return True

    def _mx_check_moe_swizzle(self) -> None:
        """Fail loud if a MoE expert param is in a swizzled >3D layout.

        vLLM's ``--moe-backend auto`` selects a batched/packed backend on
        some GPUs (observed on GB200 for Qwen3-MoE) whose
        ``process_weights_after_loading`` repacks ``w13_weight`` into a
        4D kernel layout ``(num_experts, tile, 2*inter, hidden_tile)``.
        The incremental refit ``load_weights`` path can't write raw
        per-expert HF weights back into that swizzled param — it fails
        deep in ``_load_w13`` with an opaque
        ``shard_dim=0 is not a valid data dimension for a 3D tensor``.

        We detect the swizzle up-front (any ``experts...w13_weight`` /
        ``w2_weight`` param with ``ndim > 3``) and raise a directive to
        launch with ``--moe-backend triton`` (standard un-swizzled
        layout), which is our contained fix until vLLM ships a
        refit-aware reload path. See
        pensieve/RL/NemoRL/JulyAlignment/NemoRL_MegaMX_Design.md §3-§5.

        No-op on dense models and on correctly-configured MoE.
        """
        for name, param in self.model_runner.model.named_parameters():
            if ("experts" in name and name.endswith(("w13_weight", "w2_weight"))):
                if param.ndim > 3:
                    raise RuntimeError(
                        f"[mx-megatron] MoE expert param {name!r} is in a "
                        f"swizzled {param.ndim}D layout {tuple(param.shape)}; "
                        f"vLLM's refit load_weights cannot write raw HF "
                        f"weights into it. Relaunch the vLLM worker with "
                        f"'--moe-backend triton' to keep the standard "
                        f"(num_experts, 2*inter, hidden) layout. See "
                        f"NemoRL_MegaMX_Design.md §5 (contained fix)."
                    )
                # First expert param checked is representative; done.
                return

    def _mx_apply_receiver_ep_filter(
        self,
        receive_specs: dict,
    ) -> None:
        """Rewrite ``role_descriptor['local_expert_ids']`` on expert-role
        specs to reflect THIS receiver's EP layout.

        Wired to vLLM's parallel_config (§4.5 of the MX-RL design doc):
        when ``enable_expert_parallel=True``, only the experts routed to
        this inference rank need to be pulled — the planner's
        ``_plan_per_expert`` filters to that set. When EP is disabled
        (default), every rank owns every expert and this method still
        writes the full set (identity result; no filter applied at
        planner time).

        Uses ``modelexpress.rl_expert_layout.compute_local_expert_ids``
        so the placement math (linear vs round_robin) stays in one
        place and matches what a rank-to-rank publisher would advertise
        under the same layout.
        """
        pc = self.model_runner.vllm_config.parallel_config
        # vLLM's EP is enabled via ``enable_expert_parallel``. When on,
        # the effective EP world size equals the TP*PP*DP group's total
        # devices (via get_ep_group); when off, EP is a no-op mesh of 1.
        ep_enabled = bool(getattr(pc, "enable_expert_parallel", False))
        if ep_enabled:
            try:
                from vllm.distributed import parallel_state as _ps
                _ep = _ps.get_ep_group()
                ep_world_size = int(_ep.world_size)
                ep_rank = int(_ep.rank_in_group)
            except Exception:
                # If EP is enabled in config but group isn't up yet,
                # fall back to no filter.
                ep_world_size, ep_rank = 1, 0
        else:
            ep_world_size, ep_rank = 1, 0

        # num_experts from HF config. Present on all MoE architectures
        # under different names; try the common ones. Skip filter if
        # this isn't an MoE model at all.
        hf_cfg = self.model_runner.model_config.hf_config
        num_experts = (
            getattr(hf_cfg, "num_local_experts", None)
            or getattr(hf_cfg, "num_experts", None)
            or getattr(hf_cfg, "n_routed_experts", None)
        )
        if not num_experts:
            return  # not MoE — nothing to filter

        placement = getattr(pc, "expert_placement_strategy", "linear")
        # Only "linear" and "round_robin" are recognised by
        # compute_local_expert_ids; anything else defaults to linear.
        if placement not in ("linear", "round_robin"):
            placement = "linear"

        from modelexpress.rl_expert_layout import compute_local_expert_ids
        local = compute_local_expert_ids(
            ep_rank=ep_rank,
            ep_world_size=ep_world_size,
            num_experts=int(num_experts),
            placement=placement,
        )
        local_str = ",".join(str(e) for e in local)

        # Walk expert-role specs, overwrite the local_expert_ids hint.
        # Non-expert specs are unchanged. This is a receiver-side
        # override — the trainer's own hint (its EP-owned set) stays in
        # the sidecar but the planner uses what we set here.
        touched = 0
        for spec in receive_specs.values():
            if not spec.role.startswith("expert_"):
                continue
            rd = dict(spec.role_descriptor or {})
            rd["local_expert_ids"] = local_str
            spec.role_descriptor = rd
            touched += 1

        logger.info(
            "[mx-megatron] EP filter: ep_enabled=%s ep_rank=%d ep_size=%d "
            "num_experts=%d placement=%s local=%d experts (%s...) "
            "applied to %d expert-role specs",
            ep_enabled, ep_rank, ep_world_size, num_experts, placement,
            len(local), local_str[:60], touched,
        )

    def _mx_verify_byte_identity(
        self,
        weights: list[tuple[str, torch.Tensor]],
        *,
        gt_path: str,
    ) -> None:
        """Compare received HF tensors bitwise against a Bridge ground truth.

        Loads ``gt_path`` (produced by ``bridge.export_hf_weights`` on
        the trainer) as ``{"hf_weights": {name: tensor}, ...}`` OR a bare
        state-dict, and does a per-tensor ``torch.equal`` for every name
        in ``weights``. Logs a summary line

            [mx-verify] byte-identity: N/M tensors match (X mismatches)

        which is grep-able across cycles + workers. On mismatch, logs
        the first-N offending tensor names + shape/dtype/max-abs-diff.

        Only invoked when the ``MX_VERIFY_BYTE_IDENTITY`` env var is set,
        so there's no overhead in the production path. Runs inside the
        vLLM worker process; ``gt_path`` must be visible via a mounted
        volume (e.g. the shared PVC on ``/mnt/rl-workspace``).
        """
        t0 = _time.perf_counter()
        # Use mmap so tensors are backed by the file on disk rather than
        # copied into anonymous RAM up-front. Critical at MoE scale where
        # the GT file is ~60 GB — copying it all into the pod's 128 GB
        # memory limit on top of the pinned-CPU buffer cache (another
        # ~60 GB) OOM-kills the container. With mmap=True torch keeps
        # only page-cache pressure, not per-tensor RSS.
        try:
            gt_blob = torch.load(
                gt_path, map_location="cpu", weights_only=False, mmap=True,
            )
            _mmap_ok = True
        except (TypeError, RuntimeError) as exc:
            # Older torch or non-mmap-compatible pickle format. Fall
            # back to non-mmap load and warn.
            logger.warning(
                "[mx-verify] mmap=True load failed (%s); falling back "
                "to eager load (may OOM on large GTs)",
                exc,
            )
            gt_blob = torch.load(gt_path, map_location="cpu", weights_only=False)
            _mmap_ok = False
        # Accept both the pensieve wrapper ({"hf_weights": {...}, ...})
        # and a bare state-dict.
        if isinstance(gt_blob, dict) and "hf_weights" in gt_blob:
            gt = gt_blob["hf_weights"]
        else:
            gt = gt_blob
        assert isinstance(gt, dict), (
            f"GT file {gt_path!r} did not resolve to a tensor dict "
            f"(got {type(gt).__name__})"
        )
        logger.info(
            "[mx-verify] loaded GT (mmap=%s, %d tensors) in %.2fs",
            _mmap_ok, len(gt), _time.perf_counter() - t0,
        )

        match = 0
        missing_in_gt = 0
        shape_dtype_mismatch = 0
        value_mismatch = 0
        mismatch_examples: list[str] = []
        # Stream compare: pop each GT tensor as we consume it so the
        # (small) receive tensor plus the (streamed) GT page is the
        # only extra live memory. On mmap-backed tensors ``del`` +
        # ``gt.pop`` release the mmap ref immediately.
        for name, recv in weights:
            gt_t = gt.pop(name, None)
            if gt_t is None:
                missing_in_gt += 1
                if len(mismatch_examples) < 5:
                    mismatch_examples.append(f"missing-in-gt: {name}")
                continue
            recv_cpu = recv.detach().to("cpu") if recv.device.type != "cpu" else recv
            if tuple(recv_cpu.shape) != tuple(gt_t.shape) or recv_cpu.dtype != gt_t.dtype:
                shape_dtype_mismatch += 1
                if len(mismatch_examples) < 5:
                    mismatch_examples.append(
                        f"shape/dtype: {name} recv={tuple(recv_cpu.shape)}/{recv_cpu.dtype} "
                        f"gt={tuple(gt_t.shape)}/{gt_t.dtype}"
                    )
                del gt_t
                continue
            if torch.equal(recv_cpu, gt_t):
                match += 1
            else:
                value_mismatch += 1
                if len(mismatch_examples) < 5:
                    diff = (recv_cpu.to(torch.float32) - gt_t.to(torch.float32)).abs()
                    mismatch_examples.append(
                        f"value: {name} shape={tuple(recv_cpu.shape)} "
                        f"max_abs_diff={diff.max().item():.4e} "
                        f"mean_abs_diff={diff.mean().item():.4e}"
                    )
            del gt_t
        total = len(weights)
        mismatches = missing_in_gt + shape_dtype_mismatch + value_mismatch
        elapsed = _time.perf_counter() - t0
        logger.info(
            "[mx-verify] byte-identity: %d/%d tensors match "
            "(%d mismatches: %d missing-in-gt, %d shape/dtype, %d value) "
            "in %.2fs against gt=%s",
            match, total, mismatches, missing_in_gt, shape_dtype_mismatch,
            value_mismatch, elapsed, gt_path,
        )
        if mismatch_examples:
            for line in mismatch_examples:
                logger.info("[mx-verify]   %s", line)

    def _mx_load_weights(self, weights: list[tuple[str, torch.Tensor]]) -> None:
        weights, draft_weights = _split_policy_and_draft_weights(weights)
        # MDL warm fast-path (opt-in MX_LOAD_MODE=direct; no-op otherwise).
        # Writes policy weights straight to their precomputed slots, bypassing
        # the stock loader. Draft weights still go through the normal path.
        if self._mx_try_mdl_warm(weights):
            self._mx_load_draft_weights(draft_weights)
            return
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
            # MDL cold cycle: build the destination map after the stock load
            # so cycle 2+ can take the warm fast-path. No-op unless opt-in.
            self._mx_mdl_note_cold(weights)
        self._mx_load_draft_weights(draft_weights)

    def _mx_load_draft_weights(
        self,
        draft_weights: list[tuple[str, torch.Tensor]],
    ) -> None:
        if not draft_weights:
            return

        drafter = getattr(self.model_runner, "drafter", None)
        draft_model = getattr(drafter, "model", None) if drafter is not None else None
        if draft_model is None:
            speculator = getattr(self.model_runner, "speculator", None)
            draft_model = (
                getattr(speculator, "model", None) if speculator is not None else None
            )
        if draft_model is None:
            raise RuntimeError("Received EAGLE draft weights, but vLLM has no drafter.")

        draft_model.load_weights(
            weights=_trim_draft_vocab_padding(draft_model, draft_weights),
        )

    def _mx_maybe_process_fp8_kv_cache(self) -> None:
        cache_config = getattr(self.model_runner.vllm_config, "cache_config", None)
        kv_cache_dtype = getattr(cache_config, "cache_dtype", None)
        if kv_cache_dtype is None or "fp8" not in str(kv_cache_dtype).lower():
            return

        target_device = next(self.model_runner.model.parameters()).device
        processed = _process_fp8_kv_cache_modules(
            self.model_runner.model,
            target_device,
        )
        logger.debug("[mx] processed FP8 KV cache scales on %d modules", processed)

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
                receive_specs[td.name] = ReceiveSpec(
                    megatron_name=td.name,
                    hf_names=_resolve_hf_names(
                        td.name,
                        td.megatron_role,
                        name_map,
                    ),
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
        # EP filter (opt-in via mx_config.moe_expert_filter): rewrite each
        # expert-role spec's local_expert_ids so the planner pulls only the
        # experts this receiver routes to. Identity no-op at EP=1.
        if getattr(mx_config, "moe_expert_filter", False):
            self._mx_apply_receiver_ep_filter(ctx.receive_specs)
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
                    device=self._mx_buffer_device(),
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
        # Optional byte-identity verify vs a Bridge ground truth (opt-in via
        # MX_VERIFY_BYTE_IDENTITY=<gt_path>; no overhead when unset).
        _gt_path = os.environ.get("MX_VERIFY_BYTE_IDENTITY")
        if _gt_path:
            self._mx_verify_byte_identity(weights, gt_path=_gt_path)
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
