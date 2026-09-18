# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Literal

from dynamo.llm import KvRouterConfig
from dynamo.mocker import MockEngineArgs

from .constants import AIS_BACKEND_VERSIONS


def _build_candidate_engine_args(
    *,
    base_args: Mapping[str, Any],
    tp_size: int,
    worker_type: Literal["prefill", "decode", "aggregated"],
    backend: str,
    system: str,
    model: str,
) -> MockEngineArgs:
    payload = dict(base_args)
    payload["worker_type"] = worker_type
    perf_config = dict(payload.get("ais_perf_config") or {})
    perf_config.update(
        model=model,
        system=system,
        backend=backend,
        worker_type=worker_type,
        tp=tp_size,
    )
    perf_config.setdefault("backend_version", AIS_BACKEND_VERSIONS[backend])
    if "block_size" in payload:
        perf_config.setdefault("kv_block_size", payload["block_size"])
    payload["ais_perf_config"] = perf_config
    # Keep engine args as user-intent data until this boundary. In particular,
    # do not synthesize base-only fields here; if num_gpu_blocks was omitted,
    # replay materialization will estimate capacity for the candidate TP shape.
    return MockEngineArgs.from_json(json.dumps(payload))


def _build_agg_candidate_engine_args(
    *,
    base_args: Mapping[str, Any],
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
    base_router_config: Mapping[str, Any] | None,
    overlap_score_credit: float,
    prefill_load_scale: float,
) -> KvRouterConfig:
    if not base_router_config:
        return KvRouterConfig(
            overlap_score_credit=overlap_score_credit,
            prefill_load_scale=prefill_load_scale,
        )
    # Non-empty router input is user intent; materialize it once, then apply
    # the two sweep dimensions owned by the optimizer.
    router_config = KvRouterConfig.from_json(json.dumps(dict(base_router_config)))
    router_config.overlap_score_credit = overlap_score_credit
    router_config.prefill_load_scale = prefill_load_scale
    return router_config
