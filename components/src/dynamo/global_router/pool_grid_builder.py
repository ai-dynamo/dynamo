# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""POC (Phase-3 M1): turn an explicit pool set into a multi-pool DGD + the EPP
pool-selector grid, with the routing table derived from AIConfigurator curves
instead of hand-written.

Pipeline:
    pool TPs ──► per-pool AIC prefill-latency curves (estimate_prefill_perf)
              ──► build_isl_latency_grid (cheapest pool meeting the TTFT SLA)
              ──► pool-selector eppConfig params + a v1alpha1 multi-pool DGD

The AIC step is the only part needing aiconfigurator (planner/profiler image, and
the system id must be e.g. "h200_sxm"). Everything else is pure and host-testable;
``build_pool_grid`` takes injectable ``curve_fns`` so the grid logic is exercised
without AIC. POC scope: the DGD is templated directly (one decode worker per pool)
rather than going through generate_backend_artifacts — see the scope doc's
"generator N-composition" open item.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional

from dynamo.global_router.pool_selection import build_isl_latency_grid

logger = logging.getLogger(__name__)

POOL_LABEL = "nvidia.com/dynamo-pool"


@dataclass
class PoolSpec:
    """One heterogeneous pool: a TP size and how many replicas of it."""

    tp: int
    replicas: int = 1


def aic_prefill_curve_fns(
    model: str, system: str, backend: str, pool_tps: List[int]
) -> List[Callable[[float], float]]:
    """One prefill TTFT(ISL) callable per pool, from AIConfigurator.

    Lazy-imports aiconfigurator + the dynamo AIC wrapper (planner image only).
    `system` must be an AIC system id, e.g. "h200_sxm" (NOT "h200").
    """
    from dynamo.planner.config.parallelization import (
        PickedParallelConfig,
        picked_to_aic_model_config_kwargs,
    )
    from dynamo.planner.monitoring.aic_estimator import AIConfiguratorPerfEstimator

    est = AIConfiguratorPerfEstimator(hf_id=model, system=system, backend=backend)
    fns: List[Callable[[float], float]] = []
    for tp in pool_tps:
        kwargs = picked_to_aic_model_config_kwargs(PickedParallelConfig(tp=tp))

        def f(isl: float, _kwargs=kwargs) -> float:
            perf = est.estimate_prefill_perf(int(isl), **_kwargs)
            return float(perf.get("context_latency", 0.0))

        fns.append(f)
    return fns


def build_pool_grid(
    pool_tps: List[int],
    curve_fns: List[Callable[[float], float]],
    *,
    isl_max: float,
    ttft_min_ms: float,
    ttft_max_ms: float,
    isl_resolution: int,
    ttft_resolution: int,
) -> List[List[int]]:
    """Cheapest-pool-meeting-SLA grid from per-pool curves. Cost = TP (GPUs)."""
    costs = [float(tp) for tp in pool_tps]
    return build_isl_latency_grid(
        curve_fns,
        costs,
        size_min=0,
        size_max=isl_max,
        size_resolution=isl_resolution,
        latency_min_ms=ttft_min_ms,
        latency_max_ms=ttft_max_ms,
        latency_resolution=ttft_resolution,
    )


def pool_selector_params(
    pool_tps: List[int],
    grid: List[List[int]],
    *,
    isl_max: float,
    ttft_min_ms: float,
    ttft_max_ms: float,
    isl_resolution: int,
    ttft_resolution: int,
    pool_label: str = POOL_LABEL,
) -> dict:
    """The eppConfig `pool-selector` plugin `parameters` block."""
    return {
        "poolLabel": pool_label,
        "poolLabels": [f"tp{tp}" for tp in pool_tps],
        "sizeMin": 0,
        "sizeMax": isl_max,
        "sizeResolution": isl_resolution,
        "latencyMinMs": ttft_min_ms,
        "latencyMaxMs": ttft_max_ms,
        "latencyResolution": ttft_resolution,
        "mapping": grid,
    }


def _epp_config(selector_params: dict) -> dict:
    """EndpointPickerConfig with pool-selector first (cross-pool), then the
    within-pool decode filter + dyn scorer + picker."""
    return {
        "plugins": [
            {"type": "disagg-profile-handler"},
            {
                "name": "pool-selector",
                "type": "pool-selector",
                "parameters": selector_params,
            },
            {
                "name": "decode-filter",
                "type": "label-filter",
                "parameters": {
                    "label": "nvidia.com/dynamo-sub-component-type",
                    "validValues": ["decode"],
                    "allowsNoLabel": True,
                },
            },
            {"name": "picker", "type": "max-score-picker"},
            {"name": "dyn-decode", "type": "dyn-decode-scorer"},
        ],
        "schedulingProfiles": [
            {
                "name": "decode",
                "plugins": [
                    {"pluginRef": "pool-selector", "weight": 1},
                    {"pluginRef": "decode-filter", "weight": 1},
                    {"pluginRef": "dyn-decode", "weight": 1},
                    {"pluginRef": "picker", "weight": 1},
                ],
            }
        ],
    }


def _decode_worker(
    model: str, runtime_image: str, tp: int, replicas: int, secret: str, block_size: int
) -> dict:
    return {
        "componentType": "worker",
        "subComponentType": "decode",
        "envFromSecret": secret,
        "replicas": replicas,
        "sharedMemory": {"size": "10Gi"},
        "extraPodMetadata": {"labels": {POOL_LABEL: f"tp{tp}"}},
        "frontendSidecar": {
            "image": runtime_image,
            "args": ["-m", "dynamo.frontend", "--router-mode", "direct"],
            "envFromSecret": secret,
        },
        "extraPodSpec": {
            "mainContainer": {
                "image": runtime_image,
                "command": ["/bin/sh", "-c"],
                "args": [
                    f"python3 -m dynamo.vllm --model {model} --served-model-name {model} "
                    f"--tensor-parallel-size {tp} --gpu-memory-utilization 0.90 "
                    f"--enable-prefix-caching --block-size {block_size}"
                ],
            }
        },
        "resources": {"limits": {"gpu": str(tp)}, "requests": {"gpu": str(tp)}},
    }


def build_multipool_dgd(
    name: str,
    model: str,
    runtime_image: str,
    epp_image: str,
    pools: List[PoolSpec],
    selector_params: dict,
    *,
    secret: str = "hf-token-secret",
    block_size: int = 128,
    gateway_name: Optional[str] = None,
    gateway_namespace: Optional[str] = None,
) -> dict:
    """Assemble the v1alpha1 multi-pool DGD: one pool-labeled decode worker per
    pool + an EPP carrying the pool-selector grid. Sets the inference-gateway
    annotation(s) when a gateway is named (so the operator emits the HTTPRoute)."""
    services: dict = {
        "Epp": {
            "envFromSecret": secret,
            "componentType": "epp",
            "replicas": 1,
            "extraPodSpec": {
                "mainContainer": {
                    "image": epp_image,
                    "env": [
                        {"name": "DYN_KV_CACHE_BLOCK_SIZE", "value": str(block_size)},
                        {"name": "DYN_MODEL_NAME", "value": model},
                        {"name": "DYN_ENFORCE_DISAGG", "value": "false"},
                    ],
                }
            },
            "eppConfig": {"config": _epp_config(selector_params)},
        }
    }
    for p in pools:
        services[f"VllmDecodeWorkerTp{p.tp}"] = _decode_worker(
            model, runtime_image, p.tp, p.replicas, secret, block_size
        )

    metadata: dict = {"name": name}
    if gateway_name:
        ann = {"nvidia.com/inference-gateway-name": gateway_name}
        if gateway_namespace:
            ann["nvidia.com/inference-gateway-namespace"] = gateway_namespace
        metadata["annotations"] = ann

    return {
        "apiVersion": "nvidia.com/v1alpha1",
        "kind": "DynamoGraphDeployment",
        "metadata": metadata,
        "spec": {"backendFramework": "vllm", "services": services},
    }
