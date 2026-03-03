# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data structures for scale request/response protocol between delegating and centralized planners."""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel

from dynamo.planner.kubernetes_connector import TargetReplica


class ScaleStatus(str, Enum):
    """Status values for scaling operations"""

    SUCCESS = "success"
    ERROR = "error"
    SCALING = "scaling"


class ScaleRequest(BaseModel):
    """Request to scale a deployment"""

    # Caller identification
    caller_namespace: str

    # Target deployment
    graph_deployment_name: str  # K8s DynamoGraphDeployment name
    k8s_namespace: str  # K8s namespace

    # Scaling targets
    target_replicas: List[TargetReplica]

    # Execution options
    blocking: bool = False

    # Optional context (for debugging/logging)
    timestamp: Optional[float] = None
    predicted_load: Optional[dict] = None


class ScaleResponse(BaseModel):
    """Response from scaling operation"""

    status: ScaleStatus
    message: str
    current_replicas: dict  # {"prefill": 3, "decode": 5}
