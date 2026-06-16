# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data structures for scale request/response protocol between delegating and centralized planners."""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel

from dynamo.planner.config.defaults import TargetReplica


class ScaleStatus(str, Enum):
    """Status values for scaling operations"""

    SUCCESS = "success"
    ERROR = "error"
    # Soft denial: the request was well-formed and the GlobalPlanner is
    # functioning correctly, but the requested change would breach the
    # cluster-wide GPU budget (floor or ceiling) and could not be paired with
    # an opposite-direction intent. Local planners should treat this as
    # "hold current allocation this tick" and re-decide next tick — it is
    # NOT a fatal error.
    REJECTED = "rejected"
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
