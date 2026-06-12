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
        candidates.append(f"model.{name[len('backbone.'):]}")
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

    def _mx_load_weights(self, weights: list[tuple[str, torch.Tensor]]) -> None:
        tp_size, tp_rank = _target_tp(self)
        if tp_size > 1:
            params = dict(self.model_runner.model.named_parameters())
            adapted_weights = []
            for name, weight in weights:
                if _maybe_copy_tp_local_weight(
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
        from modelexpress.nemo_rl_v2 import (
            ROLE_MEGATRON_VOCAB_PARALLEL,
            TargetTpLayout,
        )

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
                    td.name[len("module."):] if td.name.startswith("module.") else td.name
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
            logger.warning("[mx-megatron] no translated tensors for version %d", version)
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
