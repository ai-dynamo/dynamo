# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental replay-backed smart sweeper for Dynamo deployments.

Spica is an experimental feature. Its Python API, configuration schema, output
format, and optimization behavior may change without the compatibility guarantees
provided for stable Dynamo APIs.
"""

from __future__ import annotations

# First, before anything (lazily) imports vizier/jax: pin JAX to CPU + quiet the
# jax/equinox/jaxopt import noise the Vizier GP-bandit pulls in. See
# dynamo.profiler.spica._quiet.
from . import _quiet as _quiet
from .config import (
    Candidate,
    OptimizationGoal,
    OptimizationTarget,
    SearchSpace,
    SLATarget,
    SmartSearchConfig,
    SweepConfig,
    Workload,
)
from .deploy import DeploymentPlan, build_deployment
from .evaluator import ReplayEvaluator
from .kv_estimate import NoPerfDatabase, estimate_kv_tokens, feasible_shape_tokens
from .load_predictor_sweep import (
    LoadPredictorResult,
    predictor_fields,
    sweep_load_predictor,
    window_loss,
)
from .model_hw import (
    ModelHardware,
    NoViableParallelConfig,
    parallel_configs_for,
    resolve_model_hardware,
)
from .parallel_enum import (
    DisaggParallelConfig,
    ParallelShape,
    ReplicaParallelConfig,
    enumerate_disagg_configs,
    enumerate_parallel_configs,
    enumerate_worker_shapes,
)
from .planner import (
    FPM_SAMPLING,
    LOAD_SENSITIVITY,
    SCALING_POLICIES,
    ScalingPolicy,
    throughput_intervals,
)
from .sample import unroll_sample
from .sampler import BranchSampler, Suggestion, make_branch_sampler
from .score import is_feasible, make_candidate, objective_value, rank, score_report
from .search import run_smart_search
from .search_space import BranchSpace, enumerate_branches

__all__ = [
    "Candidate",
    "OptimizationGoal",
    "OptimizationTarget",
    "SearchSpace",
    "SLATarget",
    "SmartSearchConfig",
    "SweepConfig",
    "Workload",
    "run_smart_search",
    # parallel-config enumeration
    "ParallelShape",
    "ReplicaParallelConfig",
    "DisaggParallelConfig",
    "enumerate_parallel_configs",
    "enumerate_worker_shapes",
    "enumerate_disagg_configs",
    # planner preset decode
    "SCALING_POLICIES",
    "ScalingPolicy",
    "throughput_intervals",
    "FPM_SAMPLING",
    "LOAD_SENSITIVITY",
    # load-predictor sweep
    "LoadPredictorResult",
    "sweep_load_predictor",
    "window_loss",
    "predictor_fields",
    # KV-cache feasibility
    "NoPerfDatabase",
    "estimate_kv_tokens",
    "feasible_shape_tokens",
    # sample unroll
    "unroll_sample",
    # model/hardware resolution + parallel-config enumeration
    "ModelHardware",
    "NoViableParallelConfig",
    "resolve_model_hardware",
    "parallel_configs_for",
    # candidate space + sampler
    "BranchSpace",
    "enumerate_branches",
    "BranchSampler",
    "Suggestion",
    "make_branch_sampler",
    # deployment translation + evaluation
    "DeploymentPlan",
    "build_deployment",
    "ReplayEvaluator",
    # scoring
    "objective_value",
    "score_report",
    "is_feasible",
    "make_candidate",
    "rank",
]
