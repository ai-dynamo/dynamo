# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Enumerate legal per-worker parallel shapes and the replica counts that fit a GPU budget.

The per-worker shape enumeration mirrors
``aiconfigurator.sdk.utils.enumerate_parallel_config`` + ``filter_real_silicon_configs``
(real-silicon profile): ``pp`` is pinned to 1; the MoE width constraint
``dp*tp == moe_tp*moe_ep`` holds; for MoE only the pure TEP / DEP / MoE-TP patterns are
kept (MoE-TP — moe_ep==1 under tensor- or DP-attention — gated by ``allow_moe_pure_tp``,
now enabled for every MoE model incl. MLA); dense models use plain TP.
The backend-specific MoE filters are mirrored too.

``enumerate_parallel_config`` stops at *one worker's* shape (GPUs/worker = tp*pp*dp).
On top of that, this module iterates the replica counts ``r`` such that
``gpus_per_worker * r`` fits the GPU budget — the replica/worker count AIC derives
separately in its sweep layer.

Kept standalone (no aiconfigurator import) so it is light and unit-testable;
parity with AIC's rules is covered by tests. ``is_moe`` is an input here
(resolved from the model via AIC's ``check_is_moe`` by the caller); the KV-cache
feasibility of each shape is applied separately by :mod:`dynamo.profiler.spica.model_hw`.
"""

from __future__ import annotations

from dataclasses import dataclass

# GPUs-per-worker ladder (matches AIC's default num_gpu_per_worker).
_DEFAULT_GPUS_PER_WORKER: tuple[int, ...] = (1, 2, 4, 8, 16)
# Ladder used to enumerate tp / dp / moe_tp / moe_ep candidates within a worker.
_DIM_LADDER: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)


@dataclass(frozen=True)
class ParallelShape:
    """One worker's parallel shape (``pp`` pinned to 1, real-silicon profile).

    ``dp`` is the attention data-parallel size (attention_dp_size).
    """

    tp: int
    dp: int
    moe_tp: int
    moe_ep: int
    pp: int = 1

    @property
    def gpus_per_worker(self) -> int:
        return self.tp * self.pp * self.dp

    @property
    def strategy(self) -> str:
        """Label per AIC's real-silicon patterns: ``tp`` (dense, or MoE tensor-parallel
        under tensor-attention), ``tep`` (attention-TP + expert-EP), ``dep``
        (attention-DP + expert-EP), ``dtp`` (attention-DP + MoE tensor-parallel —
        e.g. InferenceX GLM-5's EP=1 + DP-attention)."""
        if self.moe_tp == 1 and self.moe_ep == 1:
            return "tp"  # dense
        if self.tp > 1 and self.dp == 1 and self.moe_tp == 1 and self.moe_ep > 1:
            return "tep"
        if self.tp == 1 and self.dp > 1 and self.moe_tp == 1 and self.moe_ep > 1:
            return "dep"
        if self.tp > 1 and self.dp == 1 and self.moe_tp > 1 and self.moe_ep == 1:
            return "tp"  # MoE tensor-parallel, tensor-attention
        if self.tp == 1 and self.dp > 1 and self.moe_tp > 1 and self.moe_ep == 1:
            return "dtp"  # MoE tensor-parallel, DP-attention
        return "mixed"


@dataclass(frozen=True)
class ReplicaParallelConfig:
    """A worker shape plus how many replicas of it run, under a GPU budget."""

    shape: ParallelShape
    replicas: int

    @property
    def total_gpus(self) -> int:
        return self.shape.gpus_per_worker * self.replicas


@dataclass(frozen=True)
class DisaggParallelConfig:
    """A disaggregated candidate: a prefill worker config + a decode worker
    config sharing the GPU budget. prefill and decode are independent (shape and
    replica count may differ)."""

    prefill: ReplicaParallelConfig
    decode: ReplicaParallelConfig

    @property
    def total_gpus(self) -> int:
        return self.prefill.total_gpus + self.decode.total_gpus


def _ladder_upto(max_value: int, ladder: tuple[int, ...] = _DIM_LADDER) -> list[int]:
    return [v for v in ladder if v <= max_value]


def _backend_allows_moe_tp(
    backend: str, *, enable_wideep: bool, moe_backend: str | None
) -> bool:
    """sglang's EP-only MoE *kernels* (deepep_moe / megamoe) require moe_tp=1. wideEP
    (multinode wide expert-parallelism) does NOT force it on its own: real GLM-5 sglang
    deployments run MoE tensor-parallel multinode (InferenceX reports EP=1), so MoE-TP
    stays available. ``enable_wideep`` is accepted for call-site compatibility."""
    del enable_wideep  # no longer gates MoE-TP; kept in the signature for callers
    return not (backend == "sglang" and moe_backend in {"deepep_moe", "megamoe"})


def enumerate_worker_shapes(
    *,
    is_moe: bool,
    backend: str,
    gpus_per_worker: int,
    enable_wideep: bool = False,
    moe_backend: str | None = None,
    allow_moe_pure_tp: bool = True,
) -> list[ParallelShape]:
    """Legal per-worker shapes at exactly ``gpus_per_worker`` GPUs (``pp`` = 1).

    Mirrors ``enumerate_parallel_config`` (width + backend filters) followed by
    ``filter_real_silicon_configs``. For MoE, TEP / DEP are always scanned; MoE
    tensor-parallel (moe_ep == 1, under tensor- or DP-attention) is kept when
    ``allow_moe_pure_tp`` — now enabled for every MoE model, MLA included, since
    real deployments run it (InferenceX GLM-5 reports EP=1). Dense models scan
    plain TP and are unaffected. Backend EP-only filters (sglang wideep) still apply.
    """
    g = gpus_per_worker
    if not is_moe:
        # dense: plain TP, no attention-dp, no experts.
        return [ParallelShape(tp=g, dp=1, moe_tp=1, moe_ep=1)]

    shapes: list[ParallelShape] = []
    cand = _ladder_upto(g)
    for tp in cand:
        for dp in cand:
            if tp * dp != g:  # one worker spans tp*pp*dp = g GPUs (pp=1)
                continue
            width = tp * dp
            for moe_tp in cand:
                for moe_ep in cand:
                    if moe_tp * moe_ep != width:  # MoE width constraint
                        continue
                    # backend filters (from enumerate_parallel_config)
                    if backend == "trtllm" and dp > 1 and tp > 1:
                        continue
                    if (
                        backend == "sglang"
                        and moe_tp > 1
                        and not _backend_allows_moe_tp(
                            backend,
                            enable_wideep=enable_wideep,
                            moe_backend=moe_backend,
                        )
                    ):
                        continue
                    if backend == "vllm" and moe_tp > 1 and moe_ep > 1:
                        continue
                    # real-silicon pure-pattern filter
                    is_tep = tp > 1 and dp == 1 and moe_tp == 1 and moe_ep > 1
                    is_dep = tp == 1 and dp > 1 and moe_tp == 1 and moe_ep > 1
                    # MoE tensor-parallel (moe_ep==1) under tensor-attention (tp>1,dp==1,
                    # strategy "tp") OR DP-attention (tp==1,dp>1, strategy "dtp"). Gated by
                    # allow_moe_pure_tp, now enabled for every MoE model incl. MLA
                    # (InferenceX GLM-5 runs MoE-TP, reported as EP=1, with DPA on or off).
                    is_moe_tp = (
                        allow_moe_pure_tp
                        and moe_tp > 1
                        and moe_ep == 1
                        and ((tp > 1 and dp == 1) or (tp == 1 and dp > 1))
                    )
                    if not (is_tep or is_dep or is_moe_tp):
                        continue
                    shapes.append(
                        ParallelShape(tp=tp, dp=dp, moe_tp=moe_tp, moe_ep=moe_ep)
                    )
    return shapes


def enumerate_parallel_configs(
    *,
    is_moe: bool,
    backend: str,
    gpu_budget: int,
    min_gpu_budget: int | None = None,
    min_gpus_per_worker: int = 1,
    gpus_per_worker_candidates: tuple[int, ...] = _DEFAULT_GPUS_PER_WORKER,
    enable_wideep: bool = False,
    moe_backend: str | None = None,
    allow_moe_pure_tp: bool = True,
) -> list[ReplicaParallelConfig]:
    """Enumerate ``(worker shape, replica count)`` configs that fit ``gpu_budget``.

    For each candidate GPUs-per-worker ``g`` (in ``[min_gpus_per_worker, budget]``),
    enumerate the legal worker shapes, then iterate replica counts ``r`` in
    ``1..(gpu_budget // g)`` so the total ``g * r`` stays within
    ``[min_gpu_budget, gpu_budget]``.

    ``min_gpus_per_worker`` is an optional lower bound on a worker's GPU count
    (default 1). :func:`dynamo.profiler.spica.model_hw.parallel_configs_for` leaves it at 1 and
    applies the KV-cache feasibility filter instead of a static weight floor.

    This is branch-agnostic: call once for an ``agg`` worker, or once per role
    (prefill / decode) for ``disagg`` — the prefill/decode pairing under the
    shared budget is the downstream rate-matching step.
    """
    configs: list[ReplicaParallelConfig] = []
    for g in gpus_per_worker_candidates:
        if g > gpu_budget or g < min_gpus_per_worker:
            continue
        shapes = enumerate_worker_shapes(
            is_moe=is_moe,
            backend=backend,
            gpus_per_worker=g,
            enable_wideep=enable_wideep,
            moe_backend=moe_backend,
            allow_moe_pure_tp=allow_moe_pure_tp,
        )
        if not shapes:
            continue
        max_replicas = gpu_budget // g
        for shape in shapes:
            for r in range(1, max_replicas + 1):
                total = g * r
                if min_gpu_budget is not None and total < min_gpu_budget:
                    continue
                configs.append(ReplicaParallelConfig(shape=shape, replicas=r))
    return configs


def enumerate_disagg_configs(
    *,
    is_moe: bool,
    backend: str,
    gpu_budget: int,
    min_gpu_budget: int | None = None,
    min_gpus_per_worker: int = 1,
    gpus_per_worker_candidates: tuple[int, ...] = _DEFAULT_GPUS_PER_WORKER,
    enable_wideep: bool = False,
    moe_backend: str | None = None,
    allow_moe_pure_tp: bool = True,
) -> list[DisaggParallelConfig]:
    """Enumerate disagg ``(prefill, decode)`` configs that share the GPU budget.

    Both roles are enumerated from the same per-role candidate set (shared
    model / hardware / backend, first pass) and paired so that
    ``prefill.total_gpus + decode.total_gpus`` lies in
    ``[min_gpu_budget, gpu_budget]``. prefill and decode may differ in shape and
    replica count.

    Required for building the disagg sweep search space. The set grows quickly
    with the budget, so the smart sweep samples from it rather than
    grid-enumerating; prefill/decode throughput rate-matching is applied
    downstream when each candidate is evaluated.
    """
    per_role = enumerate_parallel_configs(
        is_moe=is_moe,
        backend=backend,
        gpu_budget=gpu_budget,
        min_gpus_per_worker=min_gpus_per_worker,
        gpus_per_worker_candidates=gpus_per_worker_candidates,
        enable_wideep=enable_wideep,
        moe_backend=moe_backend,
        allow_moe_pure_tp=allow_moe_pure_tp,
    )
    if not per_role:
        return []

    # Each role needs at least its smallest worker, so cap one role's footprint
    # at budget minus the other role's minimum (prunes pairs that can never fit).
    min_role = min(c.total_gpus for c in per_role)
    candidates = [c for c in per_role if c.total_gpus <= gpu_budget - min_role]

    configs: list[DisaggParallelConfig] = []
    for prefill in candidates:
        for decode in candidates:
            total = prefill.total_gpus + decode.total_gpus
            if total > gpu_budget:
                continue
            if min_gpu_budget is not None and total < min_gpu_budget:
                continue
            configs.append(DisaggParallelConfig(prefill=prefill, decode=decode))
    return configs
