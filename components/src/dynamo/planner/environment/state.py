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


@dataclass
class DeploymentState:
    prefill: ComponentState = field(default_factory=ComponentState)
    decode: ComponentState = field(default_factory=ComponentState)
    model_name: Optional[str] = None

    def clone(self) -> "DeploymentState":
        return deepcopy(self)
