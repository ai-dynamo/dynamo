# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelExpress v2 refit receiver, as a vLLM v1 ``worker_extension_cls``.

Registered into the vLLM ``Worker`` class via ``parallel_config.worker_extension_cls``
so its methods become callable through ``AsyncLLM.collective_rpc``. Each method
runs **inside the worker process**, where ``self.model_runner.model`` and
``self.device`` are available.

Lifetime model (mirrors ``VllmInternalWorkerExtension``):

  * ``prepare_refit_info(state_dict_info)`` is called once per worker before the
    first refit. Stores per-tensor (shape, dtype) for asserts and FP8 paths.
  * ``update_weights_via_mx(version, mx_config)`` is called every refit cycle.
    Lazy-initializes an :class:`modelexpress.MxV2RefitReceiver` on first call,
    registers ``model.named_parameters()`` as NIXL receive buffers, then for
    every subsequent cycle: discover same-rank source → RDMA receive → call
    ``_load_weights`` → optionally republish as inference_replica for tree
    fan-out.

The Dynamo handler in ``components/src/dynamo/vllm/handlers.py`` invokes this
via ``await self.engine_client.collective_rpc("update_weights_via_mx", kwargs=...)``.
"""

from __future__ import annotations

import gc
import logging
import os
import traceback
from dataclasses import dataclass, field
from typing import Any

import torch

logger = logging.getLogger(__name__)


# =============================================================================
# MxConfig — wire-compatible with nemo_rl.distributed.mx_helpers.MxConfig
# =============================================================================


@dataclass
class MxConfig:
    """Subset of nemo-rl's MxConfig needed on the receiver side.

    The trainer sends this as a dict over the Dynamo Endpoint RPC; we parse it
    with :meth:`from_dict`. Field names and defaults must stay in sync with
    nemo_rl.distributed.mx_helpers.MxConfig.
    """

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


# =============================================================================
# NIC pinning (port of nemo_rl.distributed.mx_helpers.pin_local_nic)
# =============================================================================


def _pin_local_nic(*, device_id: int, mode: str = "auto") -> None:
    """Best-effort NUMA-local NIC pinning before NIXL initializes.

    On multi-NIC RDMA fabrics (e.g. GB200/GCP four-subnet RoCE) each rank's
    NIXL agent must bind to the NIC NUMA-closest to its GPU; cross-NIC writes
    are unrouted. Delegated to modelexpress's helper.
    """
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


# =============================================================================
# Worker extension class — injected into vLLM Worker via worker_extension_cls
# =============================================================================


class MxRefitWorkerExtension:
    """Methods added to vLLM's ``Worker`` class via ``worker_extension_cls``.

    Has no ``__init__``: vLLM merges this class's methods into the existing
    ``Worker`` via ``__bases__``, and any state we need is stashed on ``self``
    lazily inside the methods themselves (``self._mx_receiver``,
    ``self._mx_recv_buffers``, ``self._mx_state_dict_info``). No conflict
    checks fire because all our attribute names use the ``_mx_`` prefix.
    """

    # ------------------------------------------------------------------ #
    # Refit-info preparation
    # ------------------------------------------------------------------ #
    def prepare_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        """Record per-tensor (shape, dtype) info from the trainer.

        Called once by the trainer driver before the first refit cycle.
        Mirrors :py:meth:`VllmInternalWorkerExtension.prepare_refit_info` on
        the NeMo-RL side — assigns an instance attribute used by the FP8 path
        and for assertion checks. No-op for now if the worker class already
        had this attribute (the IPC-ZMQ path also stores it).
        """
        self._mx_state_dict_info = state_dict_info  # noqa: SLF001
        if not hasattr(self, "state_dict_info"):
            self.state_dict_info = state_dict_info

    # ------------------------------------------------------------------ #
    # Weight loading (minimal — adds GPT-OSS / FP8 / draft handling later)
    # ------------------------------------------------------------------ #
    def _mx_load_weights(self, weights: list[tuple[str, torch.Tensor]]) -> None:
        """Push refitted weights into the running vLLM model.

        Initial implementation covers the dense BF16/FP16 path only. GPT-OSS
        transpose, FP8 KV-cache post-processing, and Eagle3 draft-weight
        splitting are TODOs — see the NeMo-RL reference
        (``nemo_rl/models/generation/vllm/vllm_backend.py:_load_weights``)
        for those branches.
        """
        # The model parameters are the same buffers we registered with NIXL,
        # so by the time we get here the bytes are already in place. The call
        # below still runs vLLM's model-specific weight loader so any
        # per-tensor renaming / reshape / quant-state updates happen
        # consistently across paths.
        self.model_runner.model.load_weights(weights=weights)

    def _mx_maybe_process_fp8_kv_cache(self) -> None:
        """If the model uses FP8 KV cache, re-run vLLM's weight-loading hook.

        Static FP8 KV scales are computed in ``process_weights_after_loading``;
        they need to be recomputed after every refit so the scales match the
        new weights. Skipped silently for non-FP8 KV cache configurations.
        """
        use_fp8_kv_cache = False
        if hasattr(self.model_runner.vllm_config, "cache_config"):
            kv_cache_dtype = getattr(
                self.model_runner.vllm_config.cache_config, "cache_dtype", None
            )
            use_fp8_kv_cache = (
                kv_cache_dtype is not None and "fp8" in str(kv_cache_dtype).lower()
            )

        if not use_fp8_kv_cache:
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

    # ------------------------------------------------------------------ #
    # The refit RPC entry point
    # ------------------------------------------------------------------ #
    def update_weights_via_mx(
        self,
        *,
        version: int,
        mx_config: Any = None,
    ) -> bool:
        """Receive weights via NIXL RDMA from the MX server (v2 path).

        Mirrors ``VllmInternalWorkerExtension.update_weights_via_mx`` in
        NeMo-RL. The lazy-init path runs on first call per worker; subsequent
        calls reuse the registered NIXL buffers.

        Args:
            version: monotonically-increasing training-step counter; the
                receiver picks sources whose ``training_step >= version``.
            mx_config: an ``MxConfig`` instance or a dict matching its fields.
                The trainer typically sends a dict over the Dynamo Endpoint
                RPC and we parse it here.

        Returns:
            True on successful refit, False on recoverable failures
            (no source found, source doesn't cover required experts).
            Unrecoverable errors are logged with a traceback and return False
            so the caller can decide whether to retry.
        """
        try:
            # Allow dict input — the handler may pass through unparsed JSON
            if not isinstance(mx_config, MxConfig):
                mx_config = MxConfig.from_dict(mx_config or {})

            # ---- Lazy-init receiver (no pre-registered buffers; scratch path) ----
            # We use ``MxRefitReceiver.receive_weights_scratch`` rather than the
            # pre-registered-buffer path because the trainer publishes HF
            # state_dict names (``q_proj``, ``k_proj``, ``v_proj``) but vLLM's
            # internal params are fused (``qkv_proj``). Registering vLLM's
            # ``named_parameters()`` as receive buffers gives a name mismatch
            # that breaks ``model.load_weights`` (it does the HF→fused merge
            # itself, and if you feed it ``qkv_proj`` it produces ``qkqkv_proj``
            # via its stacked_params_mapping). The scratch path allocates temp
            # CUDA buffers sized to the publisher's tensor list, RDMA-pulls
            # into them, and yields ``(hf_name, tensor)`` pairs that
            # ``load_weights`` consumes correctly. Extra GPU memory cost:
            # ~1× model size briefly per refit, freed at end.
            if not getattr(self, "_mx_receiver", None):
                # Import here so workers that never refit via MX don't pay
                # the modelexpress import cost.
                from modelexpress import MxV2RefitReceiver

                rank = (
                    torch.distributed.get_rank()
                    if torch.distributed.is_initialized()
                    else 0
                )
                _pin_local_nic(
                    device_id=self.device.index, mode=mx_config.nic_pin
                )
                self._mx_receiver = MxV2RefitReceiver(  # noqa: SLF001
                    agent_name=f"dynamo-vllm-r{rank}",
                    device_id=self.device.index,
                    mx_server_url=mx_config.mx_server_url,
                    worker_rank=rank,
                )
                self._mx_receiver.initialize(model_tensors=None)
                logger.info(
                    "[mx] receiver initialized (scratch path): rank=%d device=%d",
                    rank,
                    self.device.index,
                )

            # ---- Discover, pick source, RDMA pull ----
            model_name = getattr(
                self.model_runner.vllm_config.model_config, "model", "unknown"
            )
            candidates = self._mx_receiver.discover_v2_sources(
                model_name=model_name,
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

            # ---- Megatron-MX dispatch ----
            # Sources whose ``megatron_meta`` is populated come from a
            # Megatron-Core publisher. The HF state-dict layout requires
            # role-aware translation (QKV un-interleave, gated-MLP split,
            # per-expert grouped split) rather than the DTensor path's
            # bulk-pull-of-HF-tensors. Route to the Megatron handler.
            if any(getattr(c, "megatron_meta", None) is not None for c in candidates):
                return self._update_weights_via_mx_megatron(
                    candidates=candidates,
                    version=int(version),
                    mx_config=mx_config,
                    model_name=model_name,
                )

            chosen = self._mx_receiver.pick_best_source(candidates)
            if chosen is None:
                logger.warning(
                    "[mx] no candidate covers required experts on rank %d",
                    self._mx_receiver.worker_rank,
                )
                return False
            logger.info(
                "[mx] rank=%d chosen role=%s src_rank=%d version=%s",
                self._mx_receiver.worker_rank,
                chosen.role,
                chosen.worker_rank,
                chosen.ref.training_step,
            )

            # Scratch path: allocate temp buffers, RDMA-pull, collect HF-named
            # tensors. ``receive_weights_scratch`` lives on the inner
            # ``MxRefitReceiver``; ``MxV2RefitReceiver.receive_from`` wraps the
            # non-scratch ``receive_weights`` which assumes name parity.
            #
            # Build a tensor_shapes dict from the v2 candidate's registry so
            # the yielded tensors come back with their original shape (not
            # flat 1D). vLLM's ``load_weights`` calls ``.copy_(t)`` into the
            # model param, so shape must match (or .view() must succeed).
            tensor_shapes: dict[str, tuple[int, ...]] = {}
            registry = getattr(chosen, "registry", None)
            if registry:
                for td in registry.get("tensors", []):
                    tensor_shapes[td.name] = tuple(int(s) for s in td.global_shape)
            weights: list[tuple[str, torch.Tensor]] = list(
                self._mx_receiver._receiver.receive_weights_scratch(
                    chosen.ref,
                    timeout_seconds=mx_config.timeout_seconds,
                    tensor_shapes=tensor_shapes or None,
                )
            )

            # ---- vLLM's load_weights handles HF→fused merge ----
            self._mx_load_weights(weights)
            torch.cuda.current_stream().synchronize()
            self._mx_maybe_process_fp8_kv_cache()

            # ---- Tree fan-out: republish self as inference_replica ----
            if mx_config.tree_scale_out:
                try:
                    self._mx_receiver.publish_self_as_source(
                        version=int(version),
                        model_name=model_name,
                    )
                except Exception as exc:  # noqa: BLE001
                    # Non-fatal: refit succeeded, we just can't serve as a
                    # source for downstream receivers this cycle.
                    logger.warning(
                        "[mx] tree-scale-out republish failed: %s", exc
                    )

            gc.collect()
            torch.cuda.empty_cache()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "[mx] update_weights_via_mx failed on rank=%d: %s\n%s",
                getattr(
                    getattr(self, "_mx_receiver", None), "worker_rank", -1
                ),
                exc,
                traceback.format_exc(),
            )
            return False

    # ------------------------------------------------------------------ #
    # Megatron-MX path (cluster-validated 2026-06-10 on Qwen3-MoE-30B-A3B:
    # 18 867 / 18 867 HF tensors byte-identical against bridge ground truth)
    # ------------------------------------------------------------------ #
    def _update_weights_via_mx_megatron(
        self,
        *,
        candidates: list,
        version: int,
        mx_config: Any,
        model_name: str,
    ) -> bool:
        """Megatron-MX path of :meth:`update_weights_via_mx`.

        Megatron-Core trainers publish per-rank native shards (no allgather)
        — column-parallel rows / row-parallel cols / fused QKV / fused
        gated-MLP / per-expert grouped tensors / etc. The receiver-side
        translator (``modelexpress.megatron_translator``) assembles those
        shards into HF-shaped tensors via vendored Bridge helpers,
        without taking a Bridge dependency in this worker image.

        Two paths depending on the trainer's TP layout:

          * **matched-TP** (source_tp == target_tp): one source per
            tp_rank; bulk receive_from into pre-allocated dest buffers
            registered with NIXL; translator walks the filled buffers
            directly (no host-side assembly).

          * **mixed-TP** (source_tp != target_tp): per-source sliced
            pull. Each plan's per-source contribution lands directly in
            the planner's pre-narrowed dest view via the v1
            ``MxRefitReceiver.pull_to`` primitive (one combined NIXL
            transfer with N descriptor pairs per source). Row-parallel
            (axis-1 narrows, non-contiguous in memory) falls back to
            v0 scratch + host copy.

        Cluster-validated:
          * Qwen3-4B-Thinking-2507 matched-TP: 398 / 398 byte-identical
          * Qwen3-MoE-30B-A3B-Instruct-2507 matched-TP: 18 867 / 18 867
          * synthetic TP=2 → TP=1 mixed-TP target-narrower: 8 / 8
          * synthetic TP=1 → TP=2 mixed-TP target-wider (v1 sliced-pull): 16 / 16
        """
        import time as _time
        from modelexpress.megatron_translator import (
            MegatronReceiverContext, ReceiveSpec,
            assemble_into_destination, discover_megatron_context,
            run_refit_cycle, translate_megatron_to_hf,
        )
        from modelexpress.nemo_rl_v2 import MegatronTensorSpec, TargetTpLayout

        megatron_cands = [c for c in candidates if c.megatron_meta is not None]
        if not megatron_cands:
            return False

        # Sidecar (transformer_config + Megatron→HF name map).
        sidecar_cfg, name_map = discover_megatron_context(megatron_cands)
        if sidecar_cfg is None:
            logger.warning(
                "[mx-megatron] sources advertise Megatron but no "
                "transformer_config sidecar found; aborting refit"
            )
            return False

        # Receiver's target layout: vLLM TP world × rank.
        target_tp = getattr(
            self.model_runner.vllm_config.parallel_config,
            "tensor_parallel_size", 1,
        )
        target_tp_rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_initialized() else 0
        )
        layout = TargetTpLayout(tp_size=target_tp, tp_rank=target_tp_rank)

        # Build ReceiveSpecs from candidate registries (union — replicated
        # tensors may only be published by rank 0).
        SHARD_AXIS_BY_ROLE = {
            "column": 0, "qkv_column": 0, "gated_mlp_column": 0,
            "vocab_parallel": 0, "row": 1,
            "expert_column": 0, "expert_row": 0, "replicated": 0,
        }
        receive_specs: dict[str, ReceiveSpec] = {}
        source_tp_size = max(
            c.megatron_meta.tp_size for c in megatron_cands
            if c.megatron_meta.tp_size > 0
        )
        for c in megatron_cands:
            for td in (c.registry.get("tensors", []) if c.registry else []):
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
                lookup_name = (
                    td.name[len("module."):]
                    if td.name.startswith("module.") else td.name
                )
                hf_names = name_map.get(
                    lookup_name, name_map.get(td.name, [td.name])
                )
                receive_specs[td.name] = ReceiveSpec(
                    megatron_name=td.name,
                    hf_names=list(hf_names),
                    role=role,
                    target_shape=tuple(int(s) for s in global_shape),
                    target_dtype=td.dtype or "bfloat16",
                    shard_axis=shard_axis,
                    pp_rank=c.megatron_meta.pp_rank,
                    role_descriptor=dict(td.megatron_extras or {}),
                )

        logger.info(
            "[mx-megatron] %d ReceiveSpecs built; source_tp=%d target_tp=%d",
            len(receive_specs), source_tp_size, target_tp,
        )

        # Matched-TP fast path: bulk receive_from into pre-allocated buffers.
        matched = next(
            (c for c in megatron_cands
             if c.megatron_meta.tp_rank == layout.tp_rank
             and c.megatron_meta.tp_size == layout.tp_size),
            None,
        )
        is_matched_tp = matched is not None

        device = self.device
        dt_map = {
            "bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32,
        }

        weights: list[tuple[str, torch.Tensor]] = []

        if is_matched_tp:
            # Pre-allocate + NIXL-register the per-rank buffers ONCE per
            # worker lifetime, not once per refit cycle. The receive_specs
            # are deterministic given the source's TP layout, so cycle-N's
            # allocations are identical to cycle-1's. Cache the buffers
            # dict and reuse across refits.
            #
            # Surfaced by a 16-receiver Llama 3.1 8B benchmark (2026-06-22)
            # where MX refit was ~6 s vs NCCL's 0.05 s. Cluster validation
            # on GB200 / Qwen3-4B-Thinking: register_tensors costs ~0.15 s
            # per cycle (paid by every refit). Caching drops warm-cycle
            # wall from 0.39 s to 0.21 s (-45%). Scales with buffer count
            # and total bytes.
            buffers = getattr(self, "_mx_megatron_buffers", None)
            if buffers is None:
                buffers = {}
                for spec in receive_specs.values():
                    dt = dt_map.get(spec.target_dtype, torch.bfloat16)
                    # Receiver's per-rank window — for matched-TP that's the
                    # source's natural shard along the role's shard axis.
                    full_shape = list(spec.target_shape)
                    if spec.role != "replicated" and target_tp > 1:
                        axis_extent = full_shape[spec.shard_axis]
                        per_rank = axis_extent // target_tp
                        full_shape[spec.shard_axis] = (
                            axis_extent if layout.tp_rank == target_tp - 1
                            else per_rank
                        )
                    if spec.role.startswith("expert_"):
                        # Grouped-MoE per-expert tensors are passthrough — the
                        # source-side per-expert shape IS the target.
                        pass
                    buffers[spec.megatron_name] = torch.empty(
                        full_shape, dtype=dt, device=device,
                    )
                self._mx_receiver._receiver._nixl.register_tensors(buffers)
                self._mx_megatron_buffers = buffers
                logger.info(
                    "[mx-megatron] matched-TP: ALLOCATED + registered %d buffers (%.2f GB) "
                    "[first cycle; cached for subsequent refits]",
                    len(buffers),
                    sum(b.numel() * b.element_size() for b in buffers.values()) / 1e9,
                )
            else:
                logger.info(
                    "[mx-megatron] matched-TP: reusing %d cached buffers (%.2f GB)",
                    len(buffers),
                    sum(b.numel() * b.element_size() for b in buffers.values()) / 1e9,
                )
            t0 = _time.perf_counter()
            for _name, _t in self._mx_receiver.receive_from(
                matched, timeout_seconds=mx_config.timeout_seconds,
            ):
                pass
            elapsed = _time.perf_counter() - t0
            logger.info(
                "[mx-megatron] matched-TP bulk receive_from: %.2fs", elapsed,
            )

            ctx = MegatronReceiverContext(
                target_tp_layout=layout,
                transformer_config=sidecar_cfg,
                hf_name_map=name_map,
                receive_specs=receive_specs,
            )
            for hf_name, hf_tensor in run_refit_cycle(
                self._mx_receiver,
                candidates=megatron_cands,
                context=ctx,
                pull=lambda src, dest: None,
                device=device,
                pre_assembled_buffers=buffers,
            ):
                weights.append((hf_name, hf_tensor))
        else:
            # Mixed-TP: v1 sliced-pull where dest narrow is contiguous,
            # v0 scratch+copy otherwise. Mirrors NeMo-RL's
            # vllm_backend.py::_update_weights_via_mx_megatron mixed-TP
            # branch.
            target_specs = {
                m_name: MegatronTensorSpec(
                    role=rs.role, target_shape=rs.target_shape,
                    target_dtype=rs.target_dtype, shard_axis=rs.shard_axis,
                    pp_rank=rs.pp_rank,
                    role_descriptor=dict(rs.role_descriptor or {}),
                )
                for m_name, rs in receive_specs.items()
            }
            plans = self._mx_receiver.pick_megatron_slice_plans(
                megatron_cands, target_tp_layout=layout,
                target_tensor_specs=target_specs,
            )

            # Cache plan_dests across refit cycles. Plan shapes are
            # deterministic for a fixed source TP layout + target TP
            # layout; re-allocating + re-registering NIXL buffers every
            # cycle is the bug surfaced by the 16-receiver Llama 3.1
            # benchmark (2026-06-22). v1 sliced-pull writes directly into
            # these dest views, so cached buffers stay live across pulls.
            cached_plan_dests: dict[str, torch.Tensor] | None = getattr(
                self, "_mx_megatron_plan_dests", None,
            )
            plan_dests: dict[str, torch.Tensor] = cached_plan_dests or {}
            v1_batches: dict[str, list] = {c.ref.mx_source_id: [] for c in megatron_cands}
            v0_plans: list = []
            newly_allocated_this_cycle = 0

            for plan in plans:
                if not plan.sources:
                    continue
                rs = receive_specs[plan.tensor_name]
                if plan.assembly == "per_expert":
                    v0_plans.append(plan)
                    continue
                dt = dt_map.get(rs.target_dtype, torch.bfloat16)
                if plan.tensor_name in plan_dests:
                    dest = plan_dests[plan.tensor_name]
                else:
                    dest = torch.empty(plan.target_shape, dtype=dt, device=device)
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
                    # Don't drop cached entries — they may be valid for
                    # other plans; just route this plan to v0.
                    if cached_plan_dests is None:
                        plan_dests.pop(plan.tensor_name, None)
                    for sid in v1_batches:
                        v1_batches[sid] = [
                            r for r in v1_batches[sid] if r[0] != plan.tensor_name
                        ]
                    v0_plans.append(plan)

            # Only register NIXL if we have NEW allocations. If everything
            # is cached, skip the register call entirely.
            if newly_allocated_this_cycle > 0 and plan_dests:
                self._mx_receiver._receiver._nixl.register_tensors(plan_dests)
                self._mx_megatron_plan_dests = plan_dests
                logger.info(
                    "[mx-megatron] mixed-TP: registered %d plan_dests "
                    "(%d newly allocated this cycle)",
                    len(plan_dests), newly_allocated_this_cycle,
                )
            n_v1_slices = sum(len(b) for b in v1_batches.values())
            logger.info(
                "[mx-megatron] mixed-TP: %d v1 slices across %d sources "
                "(plans: %d v1, %d v0)",
                n_v1_slices, sum(1 for b in v1_batches.values() if b),
                len(plan_dests), len(v0_plans),
            )

            for cand in megatron_cands:
                batch = v1_batches[cand.ref.mx_source_id]
                if not batch:
                    continue
                self._mx_receiver._receiver.pull_to(
                    cand.ref, batch,
                    timeout_seconds=mx_config.timeout_seconds,
                )

            scratch: dict[str, dict[str, torch.Tensor]] = {}
            if v0_plans:
                v0_source_ids = set()
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
                        plan, pull=_pull_factory(), device=device,
                    )
                for hf_name, hf_tensor in translate_megatron_to_hf(
                    plan, assembled,
                    transformer_config=ctx.transformer_config,
                    hf_names=list(rs.hf_names),
                ):
                    weights.append((hf_name, hf_tensor))

        if not weights:
            logger.warning("[mx-megatron] cycle yielded 0 tensors; refit aborted")
            return False
        logger.info(
            "[mx-megatron] yielded %d HF tensors; calling vLLM load_weights",
            len(weights),
        )

        self._mx_load_weights(weights)
        torch.cuda.current_stream().synchronize()
        self._mx_maybe_process_fp8_kv_cache()

        if mx_config.tree_scale_out:
            try:
                self._mx_receiver.publish_self_as_source(
                    version=version, model_name=model_name,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[mx-megatron] tree-scale-out republish failed: %s", exc,
                )

        gc.collect()
        torch.cuda.empty_cache()
        return True

    # ------------------------------------------------------------------ #
    # Receiver-side polling — the trainer publishes to MX without sending
    # any trigger RPC, so the worker watches the MX server itself and
    # refits whenever a newer version appears for its model_name.
    # ------------------------------------------------------------------ #
    def start_mx_refit_poller(
        self,
        *,
        mx_config: Any = None,
        poll_interval_s: float = 5.0,
    ) -> bool:
        """Spawn a background thread that watches MX for new versions.

        Called once per worker at startup (or on first publish-detection),
        from the dynamo handler. The thread:
          1. Builds an MxConfig (defaults are fine for the smoke).
          2. Loops on ``discover_v2_sources(min_version=last_seen+1)``.
          3. When a new version appears, calls ``update_weights_via_mx``
             with the new version. That method's lazy-init flow registers
             NIXL buffers on first call.
          4. Sleeps ``poll_interval_s`` between polls.

        Returns True if the thread was started (or was already running).
        Idempotent — repeated calls are no-ops.
        """
        if getattr(self, "_mx_poller_thread", None) is not None:
            return True

        import threading

        cfg = (
            mx_config
            if isinstance(mx_config, MxConfig)
            else MxConfig.from_dict(mx_config or {})
        )
        self._mx_poller_stop = threading.Event()
        self._mx_poller_last_version: int = 0
        self._mx_poller_cfg = cfg
        self._mx_poller_interval = float(poll_interval_s)

        def _poll_loop() -> None:
            from modelexpress import MxV2RefitReceiver

            model_name = getattr(
                self.model_runner.vllm_config.model_config, "model", "unknown"
            )
            rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_initialized()
                else 0
            )
            logger.info(
                "[mx-poller] started: rank=%d model=%s interval=%.1fs",
                rank,
                model_name,
                self._mx_poller_interval,
            )
            # Lazy receiver just for discovery — the refit path lazy-inits
            # its own receiver against the same MX server. We call
            # ``initialize(model_tensors=None)`` to wire the NIXL agent +
            # gRPC client without registering receive buffers (we don't
            # use this receiver to pull; only to poll for new versions).
            discover_only = MxV2RefitReceiver(
                agent_name=f"dynamo-vllm-poller-r{rank}",
                device_id=self.device.index,
                mx_server_url=cfg.mx_server_url,
                worker_rank=rank,
            )
            try:
                discover_only.initialize(model_tensors=None)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "[mx-poller] discover-receiver initialize() failed: %s; "
                    "polling thread will not start", exc,
                )
                return
            while not self._mx_poller_stop.is_set():
                try:
                    candidates = discover_only.discover_v2_sources(
                        model_name=model_name,
                        min_version=int(self._mx_poller_last_version) + 1,
                        same_rank_only=cfg.same_rank_only,
                        include_replicas=cfg.tree_scale_out,
                    )
                    if candidates:
                        latest = max(
                            int(c.ref.training_step) for c in candidates
                        )
                        logger.info(
                            "[mx-poller] new version detected: %d (last=%d)",
                            latest,
                            self._mx_poller_last_version,
                        )
                        ok = self.update_weights_via_mx(
                            version=latest, mx_config=cfg
                        )
                        if ok:
                            self._mx_poller_last_version = latest
                            logger.info(
                                "[mx-poller] refit OK to version %d", latest
                            )
                        else:
                            logger.warning(
                                "[mx-poller] refit failed for version %d; will retry",
                                latest,
                            )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[mx-poller] poll error: %s", exc)
                self._mx_poller_stop.wait(self._mx_poller_interval)
            logger.info("[mx-poller] stopped on rank=%d", rank)

        self._mx_poller_thread = threading.Thread(
            target=_poll_loop, name="mx-refit-poller", daemon=True
        )
        self._mx_poller_thread.start()
        return True
