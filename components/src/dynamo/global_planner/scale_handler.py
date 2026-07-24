# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""API layer for the GlobalPlanner ``scale_request`` endpoint.

Thin *driving adapter* over the capacity control loop. It owns the
dynamo-runtime endpoint contract, caller authorization, no-operation mode, and
translation between the wire DTOs (``ScaleRequest`` / ``ScaleResponse``) and the
orchestrator. All budget arbitration and execution live below it:

- :class:`~dynamo.global_planner.orchestrator.Orchestrator` — participants,
  authorization, GPU-budget arbitration, and observe -> decide -> scale
- :class:`~dynamo.global_planner.capacity_manager.CapacityManager` — Kubernetes
  observe/scale backend

Behavior is unchanged from the pre-refactor monolithic handler.
"""

import logging

from dynamo.global_planner.kubernetes_capacity_manager import KubernetesCapacityManager
from dynamo.global_planner.orchestrator import Orchestrator
from dynamo.planner.connectors.protocol import ScaleRequest, ScaleResponse, ScaleStatus
from dynamo.planner.errors import DynamoGraphDeploymentNotReadyError
from dynamo.runtime import DistributedRuntime, dynamo_endpoint

logger = logging.getLogger(__name__)


class ScaleRequestHandler:
    """Handles incoming scale requests in GlobalPlanner (API layer).

    Receives scale requests from Planners, validates caller authorization, and
    delegates budget arbitration and Kubernetes execution to an
    :class:`Orchestrator`. Returns current replica counts.

    Management modes:
    - **Explicit** (``managed_namespaces`` set): only listed Dynamo namespaces
      are authorized, and only their DGDs count toward the GPU budget.
    - **Implicit** (no ``managed_namespaces``): any caller is accepted and every
      DGD in the namespace counts toward the budget.

    Budget enforcement (``max_total_gpus`` ceiling, ``min_total_gpus`` floor with
    cross-pool pairing) is performed by the :class:`Orchestrator`; see it for the
    full semantics.
    """

    def __init__(
        self,
        runtime: DistributedRuntime,
        managed_namespaces: list,
        k8s_namespace: str,
        no_operation: bool = False,
        max_total_gpus: int = -1,
        min_total_gpus: int = -1,
        intent_cache_ttl_seconds: float = 360.0,
    ):
        """Initialize the scale request handler.

        Args:
            runtime: Dynamo runtime instance
            managed_namespaces: List of authorized namespaces (None = accept all)
            k8s_namespace: Kubernetes namespace where GlobalPlanner is running
            no_operation: If True, log scale requests without executing scaling
            max_total_gpus: Maximum total GPUs across all managed pools (-1 = unlimited)
            min_total_gpus: Minimum total GPUs across all managed pools (-1 = no floor)
            intent_cache_ttl_seconds: How long a cached scale intent from a pool
                is considered fresh for pairing
        """
        self.runtime = runtime
        self.no_operation = no_operation

        # TODO(global-planner): Separate the caller authorization allowlist from
        # the Kubernetes backend's discovery scope instead of deriving both from
        # managed_namespaces and a namespace-prefix convention.
        # The wire vocabulary (K8s namespace / DGD name) is mapped here onto the
        # orchestrator's neutral vocabulary; the orchestrator is a no-K8s zone.
        capacity_manager = KubernetesCapacityManager(namespace=k8s_namespace)
        self.orchestrator = Orchestrator(
            capacity_manager=capacity_manager,
            managed_deployments=managed_namespaces,
            max_total_gpus=max_total_gpus,
            min_total_gpus=min_total_gpus,
            intent_cache_ttl_seconds=intent_cache_ttl_seconds,
            use_lock=True,
        )

        if managed_namespaces:
            logger.info(
                f"ScaleRequestHandler initialized for namespaces: {managed_namespaces}"
            )
        else:
            logger.info("ScaleRequestHandler initialized (accepting all namespaces)")

        if self.no_operation:
            logger.info(
                "ScaleRequestHandler running in NO-OPERATION mode: "
                "scale requests will be logged but not executed"
            )

        if max_total_gpus >= 0:
            logger.info(f"GPU budget ceiling ENABLED: max {max_total_gpus} total GPUs")
        else:
            logger.info("GPU budget ceiling DISABLED (unlimited)")

        if min_total_gpus >= 0:
            logger.info(
                f"GPU budget floor ENABLED: min {min_total_gpus} total GPUs, "
                f"intent cache TTL {intent_cache_ttl_seconds}s"
            )
        else:
            logger.info("GPU budget floor DISABLED")

        # Discover existing DGDs (and warn if below floor) when a budget is
        # active, so the initial GPU total accounts for pre-existing pools.
        self.orchestrator.startup()

    # ------------------------------------------------------------------ #
    # Compatibility accessors (handler-level config)                     #
    # ------------------------------------------------------------------ #

    # TODO(global-planner): Preserve the pre-refactor writable handler API
    # with forwarding setters for max_total_gpus, min_total_gpus, and
    # managed_namespaces (including its previous shape), or document the break.
    @property
    def max_total_gpus(self) -> int:
        return self.orchestrator.max_total_gpus

    @property
    def min_total_gpus(self) -> int:
        return self.orchestrator.min_total_gpus

    @property
    def managed_namespaces(self):
        return self.orchestrator.managed_deployments

    @dynamo_endpoint(ScaleRequest, ScaleResponse)
    async def scale_request(self, request: ScaleRequest):
        """Process scaling request from a Planner.

        Args:
            request: ScaleRequest with target replicas and DGD info

        Yields:
            ScaleResponse with status and current replica counts
        """
        try:
            # Validate caller namespace (if authorization is enabled)
            # TODO(global-planner): Authenticate a trusted runtime principal
            # instead of relying on payload-claimed caller_namespace, and bind
            # authorization to the requested Kubernetes namespace and DGD.
            if not self.orchestrator.is_authorized(request.caller_namespace):
                yield {
                    "status": ScaleStatus.ERROR.value,
                    "message": f"Namespace {request.caller_namespace} not authorized",
                    "current_replicas": {},
                }
                return

            # No-operation mode: log and return success without touching the backend
            if self.no_operation:
                replicas_summary = {
                    r.sub_component_type.value: r.desired_replicas
                    for r in request.target_replicas
                }
                logger.info(
                    f"[NO-OP] Scale request from {request.caller_namespace} "
                    f"for DGD {request.graph_deployment_name} "
                    f"in K8s namespace {request.k8s_namespace}: {replicas_summary}"
                )
                yield {
                    "status": ScaleStatus.SUCCESS.value,
                    "message": "[no-operation] Scale request received and logged (not executed)",
                    "current_replicas": {},
                }
                return

            logger.info(
                f"Processing scale request from {request.caller_namespace} "
                f"for DGD {request.graph_deployment_name} "
                f"in K8s namespace {request.k8s_namespace}"
            )

            # Register the participant (idempotent) so it can be observed and
            # scaled and counts toward the budget. Map the wire fields onto the
            # orchestrator's neutral vocabulary.
            participant_id = f"{request.k8s_namespace}/{request.graph_deployment_name}"
            self.orchestrator.register(
                participant_id,
                caller_name=request.caller_namespace,
                namespace=request.k8s_namespace,
                deployment_name=request.graph_deployment_name,
            )

            outcome = await self.orchestrator.submit(
                participant_id,
                request.target_replicas,
                blocking=request.blocking,
                deployment_name=request.graph_deployment_name,
                caller_name=request.caller_namespace,
            )
            yield {
                "status": outcome.status.value,
                "message": outcome.message,
                "current_replicas": outcome.current_replicas,
            }

        except DynamoGraphDeploymentNotReadyError as e:
            logger.warning("Rejected scale request: %s", e)
            yield {
                "status": ScaleStatus.REJECTED.value,
                "message": str(e),
                "current_replicas": {},
            }
        except Exception as e:
            logger.exception(f"Error processing scale request: {e}")
            yield {
                "status": ScaleStatus.ERROR.value,
                "message": str(e),
                "current_replicas": {},
            }
