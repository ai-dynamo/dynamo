# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal

from dynamo.llm import KvRouterConfig, MockEngineArgs

from .constants import AIC_BACKEND_VERSIONS


def _build_candidate_engine_args(
    *,
    base_args: MockEngineArgs,
    tp_size: int,
    worker_type: Literal["prefill", "decode", "aggregated"],
    backend: str,
    system: str,
    model: str,
) -> MockEngineArgs:
    args = base_args.copy()
    args.worker_type = worker_type
    args.enable_prefix_caching = worker_type != "decode"
    # Keep the base KV block capacity fixed across TP for now.
    #
    # TP does not have a simple, backend-agnostic relationship with
    # effective KV capacity. In particular, MLA-style attention and other
    # specialized cache layouts break the usual KV-head-sharding intuition.
    # A future version should derive a TP-aware capacity estimate from the
    # AIC SDK instead of applying a generic heuristic here.
    args.num_gpu_blocks = base_args.num_gpu_blocks
    args.aic_backend = backend
    args.aic_system = system
    args.aic_backend_version = AIC_BACKEND_VERSIONS[backend]
    args.aic_tp_size = tp_size
    args.aic_model_path = model
    return args


def _build_agg_candidate_engine_args(
    *,
    base_args: MockEngineArgs,
    tp_size: int,
    backend: str,
    system: str,
    model: str,
) -> MockEngineArgs:
    return _build_candidate_engine_args(
        base_args=base_args,
        tp_size=tp_size,
        worker_type="aggregated",
        backend=backend,
        system=system,
        model=model,
    )


def _build_router_config(
    base_router_config: KvRouterConfig | None,
    overlap_score_weight: float,
) -> KvRouterConfig:
    if base_router_config is None:
        return KvRouterConfig(overlap_score_weight=overlap_score_weight)
    router_config = base_router_config.copy()
    router_config.overlap_score_weight = overlap_score_weight
    return router_config
