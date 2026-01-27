"""Data structures for scale request/response protocol between delegating and centralized planners."""

from typing import TYPE_CHECKING, List, Optional

from pydantic import BaseModel

# Import SubComponentType only for type checking to avoid runtime dependency
if TYPE_CHECKING:
    pass


class TargetReplicaRequest(BaseModel):
    """Replica target for scaling request"""

    sub_component_type: str  # SubComponentType: "prefill" or "decode"
    component_name: Optional[str] = None
    desired_replicas: int


class ScaleRequest(BaseModel):
    """Request to scale a deployment"""

    # Caller identification
    caller_namespace: str

    # Target deployment
    graph_deployment_name: str  # K8s DynamoGraphDeployment name
    k8s_namespace: str  # K8s namespace

    # Scaling targets
    target_replicas: List[TargetReplicaRequest]

    # Execution options
    blocking: bool = False

    # Optional context (for debugging/logging)
    timestamp: Optional[float] = None
    predicted_load: Optional[dict] = None


class ScaleResponse(BaseModel):
    """Response from scaling operation"""

    status: str  # "success", "error", "scaling"
    message: str
    current_replicas: dict  # {"prefill": 3, "decode": 5}
