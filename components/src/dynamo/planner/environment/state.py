# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

from dynamo.planner.monitoring.worker_info import WorkerInfo


@dataclass
class ReplicaState:
    active: int = 0
    expected: Optional[int] = None
    scaling: bool = False


@dataclass
class ComponentState:
    info: Optional[WorkerInfo] = None
    replicas: ReplicaState = field(default_factory=ReplicaState)
    num_gpus: Optional[int] = None
    # DGD-owned per-GPU power cap (watts) parsed from this component's worker
    # podTemplate annotation, and the already-multiplied per-replica draw
    # (cap × get_total_gpu_count()). Both stay None when power awareness is off.
    # ``num_gpus`` deliberately stays on the per-pod ``get_gpu_count()`` — power
    # awareness must not change GPU-budget math — while ``power_watts_per_replica``
    # uses the replica-wide (nodeCount × per-pod) GPU total.
    power_gpu_limit_watts: Optional[int] = None
    power_watts_per_replica: Optional[int] = None


@dataclass
class DeploymentState:
    prefill: ComponentState = field(default_factory=ComponentState)
    decode: ComponentState = field(default_factory=ComponentState)
    model_name: Optional[str] = None
    # Deployment-scoped power scale-up suppression. Set when a runtime refresh
    # observes a changed or malformed per-GPU cap: the planner keeps the
    # conservative last-good watts, blocks scale-up, and surfaces a
    # restart-required diagnostic. Lives on DeploymentState (not ComponentState)
    # so a malformed refresh that leaves the per-component watts unchanged still
    # flips ``deployment_state_changed`` and re-publishes capabilities.
    power_scale_up_blocked: bool = False
    power_scale_up_blocked_reason: str = ""

    def clone(self) -> "DeploymentState":
        return deepcopy(self)
