# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data structures for scale request/response protocol between delegating and centralized planners."""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, StrictInt

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
    # Identifies a submitted demand update, including retries of that update.
    # Optional so existing Local Planners can still use the legacy endpoint.
    request_id: Optional[str] = None


class AllocationPhase(str, Enum):
    """Managed allocation lifecycle, in reconciliation order."""

    # Controller-wide phases.
    RECOVERING = "recovering"  # waiting for a fenced, fully settled baseline
    READY = "ready"  # baseline established; reported per pool as below
    RECOVERY_REQUIRED = "recovery_required"  # mutations frozen; manual repair
    # Per-pool phases while READY.
    SETTLED = "settled"
    PENDING = "pending"  # retained demand differs from the grant
    REDUCING = "reducing"  # donor reductions being written
    WAITING_RELEASE = "waiting_release"  # donor GPUs not yet released
    STARTING = "starting"  # requester growth being written
    SETTLING = "settling"  # waiting for the complete plan to settle


class AllocationStatusRequest(BaseModel):
    """Identify a managed pool without changing its retained demand."""

    caller_namespace: str
    graph_deployment_name: str
    k8s_namespace: str


class ManagedAllocationStatus(BaseModel):
    """Demand, grant, and observed capacity reported by a managed controller.

    A grant is an allocation target, not a ready worker count. Nullable counts
    mean the controller has no valid observation for that field yet.
    """

    pool_id: str
    controller_id: str
    # A plain string keeps older readers compatible with newly added phases.
    phase: str
    reason: str
    desired_replicas: Optional[StrictInt] = Field(default=None, ge=0)
    granted_replicas: Optional[StrictInt] = Field(default=None, ge=0)
    observed_replicas: Optional[StrictInt] = Field(default=None, ge=0)
    ready_replicas: Optional[StrictInt] = Field(default=None, ge=0)
    terminating_replicas: Optional[StrictInt] = Field(default=None, ge=0)
    request_id: Optional[str] = None
    transfer_id: Optional[str] = None


class ScaleResponse(BaseModel):
    """Response from scaling operation"""

    status: ScaleStatus
    message: str
    current_replicas: dict  # {"prefill": 3, "decode": 5}
    allocation: Optional[ManagedAllocationStatus] = None
