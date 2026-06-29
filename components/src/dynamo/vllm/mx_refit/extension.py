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

    def _mx_load_weights(self, weights: list[tuple[str, torch.Tensor]]) -> None:
        weights, draft_weights = _split_policy_and_draft_weights(weights)
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

        # Detect matched-TP vs mixed-TP up front. The matched-TP context
        # (built via _build_megatron_context) encodes per-rank target_shape;
        # the mixed-TP path needs a *global* target_shape so the slice
        # planner can compute cross-source ranges, so it builds its own ctx.
        megatron_cands = [c for c in candidates if c.megatron_meta is not None]
        if not megatron_cands:
            return False
        target_tp_size, target_tp_rank = _target_tp(self)
        matched = next(
            (
                c
                for c in megatron_cands
                if c.megatron_meta.tp_rank == target_tp_rank
                and c.megatron_meta.tp_size == target_tp_size
            ),
            None,
        )
        source_tp_size = max(
            (
                c.megatron_meta.tp_size
                for c in megatron_cands
                if c.megatron_meta.tp_size > 0
            ),
            default=0,
        )
        is_matched_tp = matched is not None and source_tp_size == target_tp_size

        if not is_matched_tp:
            return self._mx_update_weights_via_mx_megatron_mixed_tp(
                megatron_cands=megatron_cands,
                source_tp_size=source_tp_size,
                target_tp_size=target_tp_size,
                target_tp_rank=target_tp_rank,
                version=version,
                mx_config=mx_config,
            )

        # ---- Matched-TP path ----
        if not getattr(self, "_mx_megatron_ctx", None):
            self._mx_megatron_ctx = self._build_megatron_context(candidates)

        ctx = self._mx_megatron_ctx
        if not hasattr(self, "_mx_megatron_buffers"):
            buffers: dict[str, torch.Tensor] = {}
            vocab_buffers: dict[str, torch.Tensor] = {}
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

    def _mx_update_weights_via_mx_megatron_mixed_tp(
        self,
        *,
        megatron_cands: list[Any],
        source_tp_size: int,
        target_tp_size: int,
        target_tp_rank: int,
        version: int,
        mx_config: MxConfig,
    ) -> bool:
        """Mixed-TP weight refit: v1 sliced-pull where the dest narrow is
        contiguous, v0 scratch + host-copy fallback otherwise.

        Mirrors NeMo-RL's ``vllm_backend.py::_update_weights_via_mx_megatron``
        mixed-TP branch (ported from ai-dynamo/dynamo#10901). Builds receive
        specs with *global* ``target_shape`` (per-rank × ``source_tp_size`` on
        the role's shard axis, except for replicated tensors) so
        ``MxV2RefitReceiver.pick_megatron_slice_plans`` can compute cross-
        source ranges. Caches per-plan dest buffers via
        ``self._mx_megatron_plan_dests``; NIXL re-registers only when new
        allocations land.
        """
        from modelexpress.megatron_translator import (
            MegatronReceiverContext,
            ReceiveSpec,
            assemble_into_destination,
            discover_megatron_context,
            translate_megatron_to_hf,
        )
        from modelexpress.nemo_rl_v2 import MegatronTensorSpec, TargetTpLayout

        sidecar_cfg, name_map = discover_megatron_context(megatron_cands)
        if sidecar_cfg is None:
            logger.warning(
                "[mx-megatron] mixed-TP: sources advertise Megatron but no "
                "transformer_config sidecar found; aborting refit"
            )
            return False

        layout = TargetTpLayout(tp_size=target_tp_size, tp_rank=target_tp_rank)

        # Per-role shard axes; matches modelexpress nemo_rl_v2 convention.
        SHARD_AXIS_BY_ROLE = {
            "column": 0,
            "qkv_column": 0,
            "gated_mlp_column": 0,
            "vocab_parallel": 0,
            "row": 1,
            "expert_column": 0,
            "expert_row": 0,
            "replicated": 0,
        }
        receive_specs: dict[str, ReceiveSpec] = {}
        for c in megatron_cands:
            tensors = c.registry.get("tensors", []) if c.registry else []
            for td in tensors:
                if not td.megatron_role or td.name in receive_specs:
                    continue
                role = td.megatron_role
                shard_axis = SHARD_AXIS_BY_ROLE.get(role, int(td.shard_axis))
                per_rank_shape = list(td.global_shape)
                global_shape = list(per_rank_shape)
                if role != "replicated":
                    global_shape[shard_axis] = (
                        per_rank_shape[shard_axis] * source_tp_size
                    )
                receive_specs[td.name] = ReceiveSpec(
                    megatron_name=td.name,
                    hf_names=_resolve_hf_names(td.name, role, name_map),
                    role=role,
                    target_shape=tuple(int(s) for s in global_shape),
                    target_dtype=td.dtype or "bfloat16",
                    shard_axis=shard_axis,
                    pp_rank=c.megatron_meta.pp_rank,
                    role_descriptor=dict(td.megatron_extras or {}),
                )
        logger.info(
            "[mx-megatron] mixed-TP: %d ReceiveSpecs built; source_tp=%d target_tp=%d",
            len(receive_specs),
            source_tp_size,
            target_tp_size,
        )

        # Plan the cross-source slice transfer.
        target_specs = {
            m_name: MegatronTensorSpec(
                role=rs.role,
                target_shape=rs.target_shape,
                target_dtype=rs.target_dtype,
                shard_axis=rs.shard_axis,
                pp_rank=rs.pp_rank,
                role_descriptor=dict(rs.role_descriptor or {}),
            )
            for m_name, rs in receive_specs.items()
        }
        plans = self._mx_receiver.pick_megatron_slice_plans(
            megatron_cands,
            target_tp_layout=layout,
            target_tensor_specs=target_specs,
        )

        # Cache plan_dests across refit cycles. Plan shapes are deterministic
        # for a fixed (source_tp, target_tp) layout, so cycle-N's allocations
        # match cycle-1's. v1 sliced-pull writes directly into these dest
        # views; cached buffers stay live across pulls.
        cached_plan_dests: dict[str, torch.Tensor] | None = getattr(
            self, "_mx_megatron_plan_dests", None
        )
        plan_dests: dict[str, torch.Tensor] = cached_plan_dests or {}
        v1_batches: dict[str, list[Any]] = {
            c.ref.mx_source_id: [] for c in megatron_cands
        }
        v0_plans: list[Any] = []
        newly_allocated_this_cycle = 0

        for plan in plans:
            if not plan.sources:
                continue
            rs = receive_specs[plan.tensor_name]
            if plan.assembly == "per_expert":
                v0_plans.append(plan)
                continue
            if plan.tensor_name in plan_dests:
                dest = plan_dests[plan.tensor_name]
            else:
                dest = torch.empty(
                    plan.target_shape,
                    dtype=_torch_dtype(rs.target_dtype),
                    device=self.device,
                )
                plan_dests[plan.tensor_name] = dest
                newly_allocated_this_cycle += 1
            axis = 1 if plan.assembly == "concat_dim1" else 0
            routed_v1 = True
            for src in plan.sources:
                target_lo, target_hi = src.target_local_range
                dest_view = dest.narrow(axis, target_lo, target_hi - target_lo)
                if not dest_view.is_contiguous():
                    routed_v1 = False
                    break
                v1_batches[src.mx_source_id].append(
                    (plan.tensor_name, src.source_subslice, dest_view)
                )
            if not routed_v1:
                # Don't drop cached entries — they may be valid for other
                # plans; just route this plan to v0.
                if cached_plan_dests is None:
                    plan_dests.pop(plan.tensor_name, None)
                for sid in v1_batches:
                    v1_batches[sid] = [
                        r for r in v1_batches[sid] if r[0] != plan.tensor_name
                    ]
                v0_plans.append(plan)

        if newly_allocated_this_cycle > 0 and plan_dests:
            self._mx_receiver._receiver._nixl.register_tensors(plan_dests)
            self._mx_megatron_plan_dests = plan_dests
            logger.info(
                "[mx-megatron] mixed-TP: registered %d plan_dests "
                "(%d newly allocated this cycle)",
                len(plan_dests),
                newly_allocated_this_cycle,
            )
        n_v1_slices = sum(len(b) for b in v1_batches.values())
        logger.info(
            "[mx-megatron] mixed-TP: %d v1 slices across %d sources "
            "(plans cached=%d v0=%d)",
            n_v1_slices,
            sum(1 for b in v1_batches.values() if b),
            len(plan_dests),
            len(v0_plans),
        )

        # v1 sliced-pulls write directly into pre-narrowed dest views.
        for cand in megatron_cands:
            batch = v1_batches[cand.ref.mx_source_id]
            if not batch:
                continue
            self._mx_receiver._receiver.pull_to(
                cand.ref,
                batch,
                timeout_seconds=mx_config.timeout_seconds,
            )

        # v0 fallback: scratch per source, host-copy into target.
        scratch: dict[str, dict[str, torch.Tensor]] = {}
        if v0_plans:
            v0_source_ids: set[str] = set()
            for plan in v0_plans:
                for src in plan.sources:
                    v0_source_ids.add(src.mx_source_id)
            for cand in [c for c in megatron_cands if c.ref.mx_source_id in v0_source_ids]:
                buf_dict: dict[str, torch.Tensor] = {}
                for name, t in self._mx_receiver._receiver.receive_weights_scratch(
                    cand.ref, timeout_seconds=mx_config.timeout_seconds,
                ):
                    buf_dict[name] = t
                scratch[cand.ref.mx_source_id] = buf_dict

        ctx = MegatronReceiverContext(
            target_tp_layout=layout,
            transformer_config=sidecar_cfg,
            hf_name_map=name_map,
            receive_specs=receive_specs,
        )
        weights: list[tuple[str, torch.Tensor]] = []
        for plan in plans:
            if not plan.sources:
                continue
            rs = receive_specs[plan.tensor_name]
            if plan.tensor_name in plan_dests:
                assembled = plan_dests[plan.tensor_name]
            else:
                def _pull_factory(name=plan.tensor_name, assembly=plan.assembly):
                    def _pull(src, dest):
                        full = scratch.get(src.mx_source_id, {}).get(name)
                        if full is None:
                            raise RuntimeError(
                                f"mixed-TP v0: scratch missing {name!r} from "
                                f"source {src.mx_source_id}"
                            )
                        axis = 1 if assembly == "concat_dim1" else 0
                        if src.source_subslice is not None:
                            slo, shi = src.source_subslice
                            slice_src = full.narrow(axis, slo, shi - slo)
                        else:
                            slice_src = full
                        dest.copy_(slice_src, non_blocking=True)

                    return _pull

                assembled = assemble_into_destination(
                    plan, pull=_pull_factory(), device=self.device,
                )
            for hf_name, hf_tensor in translate_megatron_to_hf(
                plan,
                assembled,
                transformer_config=ctx.transformer_config,
                hf_names=list(rs.hf_names),
            ):
                weights.append((hf_name, hf_tensor))

        if not weights:
            logger.warning(
                "[mx-megatron] mixed-TP yielded 0 tensors for version %d",
                version,
            )
            return False
        logger.info(
            "[mx-megatron] mixed-TP yielded %d HF tensors; calling vLLM load_weights",
            len(weights),
        )

        self._mx_load_weights(weights)
        torch.cuda.current_stream().synchronize()
        self._mx_maybe_process_fp8_kv_cache()
        if mx_config.tree_scale_out:
            try:
                self._mx_receiver.publish_self_as_source(
                    version=int(version),
                    model_name=_model_name(self),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[mx-megatron] mixed-TP tree-scale-out republish failed: %s",
                    exc,
                )
        gc.collect()
        torch.cuda.empty_cache()
        return True


__all__ = ["MxConfig", "MxRefitWorkerExtension"]
