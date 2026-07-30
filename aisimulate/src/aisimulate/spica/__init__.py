# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental, backend-neutral deployment simulation sweeper.

The package root eagerly exposes only schemas and ABI contracts. Search and
AIConfigurator-backed helpers are loaded lazily so an optional adapter can import
``aisimulate.spica.adapter`` without importing a simulation backend.
"""

from __future__ import annotations

import importlib
from typing import Any

from .adapter import (
    API_VERSION,
    AdapterReplaySpec,
    AdapterSearchPlan,
    CandidateContext,
    RuntimeHookSpec,
    SearchSpaceFragment,
    SimulationAdapter,
    SweepContext,
)
from .config import (
    AdapterRequest,
    Candidate,
    OptimizationGoal,
    OptimizationTarget,
    SearchSpace,
    SLATarget,
    SmartSearchConfig,
    SweepConfig,
    Workload,
)
from .discovery import (
    ADAPTER_ENTRY_POINT_GROUP,
    AdapterResolutionError,
    resolve_adapters,
)
from .replay import (
    REPLAY_SPEC_API_VERSION,
    BackendDeploymentSpec,
    HookCapability,
    ReplayReport,
    ReplaySpec,
    Runner,
    RunnerCapabilities,
    RunnerFactory,
    canonical_json,
)

_LAZY_EXPORTS = {
    "build_backend_deployment": (".deploy", "build_backend_deployment"),
    "NoPerfDatabase": (".kv_estimate", "NoPerfDatabase"),
    "estimate_kv_tokens": (".kv_estimate", "estimate_kv_tokens"),
    "feasible_shape_tokens": (".kv_estimate", "feasible_shape_tokens"),
    "ModelHardware": (".model_hw", "ModelHardware"),
    "NoViableParallelConfig": (".model_hw", "NoViableParallelConfig"),
    "parallel_configs_for": (".model_hw", "parallel_configs_for"),
    "resolve_model_hardware": (".model_hw", "resolve_model_hardware"),
    "DisaggParallelConfig": (".parallel_enum", "DisaggParallelConfig"),
    "ParallelShape": (".parallel_enum", "ParallelShape"),
    "ReplicaParallelConfig": (".parallel_enum", "ReplicaParallelConfig"),
    "enumerate_disagg_configs": (".parallel_enum", "enumerate_disagg_configs"),
    "enumerate_parallel_configs": (".parallel_enum", "enumerate_parallel_configs"),
    "enumerate_worker_shapes": (".parallel_enum", "enumerate_worker_shapes"),
    "unroll_sample": (".sample", "unroll_sample"),
    "BranchSampler": (".sampler", "BranchSampler"),
    "Suggestion": (".sampler", "Suggestion"),
    "make_branch_sampler": (".sampler", "make_branch_sampler"),
    "objective_value": (".score", "objective_value"),
    "score_report": (".score", "score_report"),
    "is_feasible": (".score", "is_feasible"),
    "make_candidate": (".score", "make_candidate"),
    "rank": (".score", "rank"),
    "run_smart_search": (".search", "run_smart_search"),
    "BranchSpace": (".search_space", "BranchSpace"),
    "enumerate_branches": (".search_space", "enumerate_branches"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(importlib.import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


__all__ = [
    "API_VERSION",
    "REPLAY_SPEC_API_VERSION",
    "AdapterReplaySpec",
    "AdapterRequest",
    "AdapterSearchPlan",
    "Candidate",
    "CandidateContext",
    "OptimizationGoal",
    "OptimizationTarget",
    "RuntimeHookSpec",
    "SearchSpace",
    "SearchSpaceFragment",
    "SimulationAdapter",
    "SLATarget",
    "SmartSearchConfig",
    "SweepConfig",
    "SweepContext",
    "Workload",
    "ADAPTER_ENTRY_POINT_GROUP",
    "AdapterResolutionError",
    "resolve_adapters",
    "BackendDeploymentSpec",
    "HookCapability",
    "ReplayReport",
    "ReplaySpec",
    "Runner",
    "RunnerCapabilities",
    "RunnerFactory",
    "canonical_json",
    *_LAZY_EXPORTS,
]
