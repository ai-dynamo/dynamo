# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The authoritative Python API and native extension are attested as one source
# pair.  Expose the immutable build revision on the Python layer so consumers
# can fail closed on a missing or mixed installation.
from dynamo._core import __source_revision__ as __dynamo_source_commit__

from dynamo.replay.api import (
    OfflineReplaySession,
    PoolSpec,
    ReplayAgenticRequest,
    ReplayAgenticWorkflow,
    ReplayEvent,
    ReplayEventData,
    ReplayPendingPlacement,
    ReplayPlacementCandidate,
    ReplayRequestSpec,
    ReplayRoutingConstraints,
    ReplaySnapshot,
    ReplayStepStatus,
    ReplayWorkerSnapshot,
    ReplayWorkerTargetData,
    WorkerLifecycleStatus,
    WorkerTarget,
    WorkerSpec,
    run_synthetic_trace_replay,
    run_trace_replay,
)
from dynamo.replay.report import PlannerReplayDetails, ReplayReport

__all__ = [
    "PlannerReplayDetails",
    "OfflineReplaySession",
    "PoolSpec",
    "ReplayAgenticRequest",
    "ReplayAgenticWorkflow",
    "ReplayEvent",
    "ReplayEventData",
    "ReplayPendingPlacement",
    "ReplayPlacementCandidate",
    "ReplayReport",
    "ReplayRequestSpec",
    "ReplayRoutingConstraints",
    "ReplaySnapshot",
    "ReplayStepStatus",
    "ReplayWorkerSnapshot",
    "ReplayWorkerTargetData",
    "WorkerLifecycleStatus",
    "WorkerTarget",
    "WorkerSpec",
    "run_synthetic_trace_replay",
    "run_trace_replay",
]
