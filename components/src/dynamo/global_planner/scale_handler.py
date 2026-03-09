# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Handler for scale_request endpoint in GlobalPlanner."""

import logging

from dynamo.planner import KubernetesConnector
from dynamo.planner.scale_protocol import ScaleRequest, ScaleResponse, ScaleStatus
from dynamo.runtime import DistributedRuntime, dynamo_endpoint

logger = logging.getLogger(__name__)

# Model name used for KubernetesConnector in remote execution mode
MANAGED_MODEL_NAME = "managed"


class ScaleRequestHandler:
    """Handles incoming scale requests in GlobalPlanner.

    This handler:
    1. Receives scale requests from Planners
    2. Validates caller authorization (optional)
    3. Caches KubernetesConnector per DGD for efficiency
    4. Executes scaling via Kubernetes API
    5. Returns current replica counts
    """

    def __init__(
        self,
        runtime: DistributedRuntime,
        managed_namespaces: list,
        k8s_namespace: str,
        no_operation: bool = False,
        max_total_gpus: int = -1,
    ):
        """Initialize the scale request handler.

        Args:
            runtime: Dynamo runtime instance
            managed_namespaces: List of authorized namespaces (None = accept all)
            k8s_namespace: Kubernetes namespace where GlobalPlanner is running
            no_operation: If True, log scale requests without executing K8s scaling
            max_total_gpus: Maximum total GPUs across all managed pools (-1 = unlimited)
        """
        self.runtime = runtime
        # If managed_namespaces is None, accept all namespaces
        self.managed_namespaces = (
            set(managed_namespaces) if managed_namespaces else None
        )
        self.k8s_namespace = k8s_namespace
        self.no_operation = no_operation
        self.max_total_gpus = max_total_gpus
        self.connectors = {}  # Cache of KubernetesConnector per DGD

        if self.managed_namespaces:
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

        if self.max_total_gpus > 0:
            logger.info(
                f"GPU budget enforcement ENABLED: max {self.max_total_gpus} total GPUs"
            )
        else:
            logger.info("GPU budget enforcement DISABLED (unlimited)")

    def _calculate_total_gpus_after_request(self, request: ScaleRequest) -> int:
        """Calculate total GPUs across all managed DGDs if this request is granted.

        For the requesting DGD, uses the desired replica counts from the request.
        For all other known DGDs, uses their current replica counts.
        """
        total_gpus = 0
        requesting_key = f"{request.k8s_namespace}/{request.graph_deployment_name}"

        for key, connector in self.connectors.items():
            try:
                deployment = connector.kube_api.get_graph_deployment(
                    connector.parent_dgd_name
                )
            except Exception as e:
                logger.warning(f"Failed to read DGD for {key}: {e}")
                continue

            services = deployment.get("spec", {}).get("services", {})

            for svc_spec in services.values():
                sub_type = svc_spec.get("subComponentType", "")
                if not sub_type:
                    continue

                gpu_per_replica = int(
                    svc_spec.get("resources", {}).get("limits", {}).get("gpu", 0)
                )
                if gpu_per_replica == 0:
                    continue

                replicas = svc_spec.get("replicas", 0)

                # For the requesting DGD, use desired replicas from the request
                if key == requesting_key:
                    for target in request.target_replicas:
                        if target.sub_component_type.value == sub_type:
                            replicas = target.desired_replicas
                            break

                total_gpus += replicas * gpu_per_replica

        return total_gpus

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
            if (
                self.managed_namespaces is not None
                and request.caller_namespace not in self.managed_namespaces
            ):
                yield {
                    "status": ScaleStatus.ERROR.value,
                    "message": f"Namespace {request.caller_namespace} not authorized",
                    "current_replicas": {},
                }
                return

            # No-operation mode: log and return success without touching K8s
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

            # Get or create connector for this DGD
            connector_key = f"{request.k8s_namespace}/{request.graph_deployment_name}"
            if connector_key not in self.connectors:
                connector = KubernetesConnector(
                    dynamo_namespace=request.caller_namespace,
                    model_name=MANAGED_MODEL_NAME,  # Not used for remote execution
                    k8s_namespace=request.k8s_namespace,
                    parent_dgd_name=request.graph_deployment_name,
                )
                self.connectors[connector_key] = connector
                logger.debug(f"Created new connector for {connector_key}")
            else:
                connector = self.connectors[connector_key]
                logger.debug(f"Reusing cached connector for {connector_key}")

            # Check GPU budget before scaling
            if self.max_total_gpus > 0:
                total_gpus = self._calculate_total_gpus_after_request(request)
                if total_gpus > self.max_total_gpus:
                    logger.warning(
                        f"Rejecting scale request from {request.caller_namespace}: "
                        f"would use {total_gpus} GPUs, exceeding max of {self.max_total_gpus}"
                    )
                    yield {
                        "status": ScaleStatus.ERROR.value,
                        "message": (
                            f"GPU budget exceeded: request would use {total_gpus} total GPUs, "
                            f"max allowed is {self.max_total_gpus}"
                        ),
                        "current_replicas": {},
                    }
                    return
                logger.info(
                    f"GPU budget check passed: {total_gpus}/{self.max_total_gpus} GPUs"
                )

            # Execute scaling (request.target_replicas is already List[TargetReplica])
            await connector.set_component_replicas(
                request.target_replicas, blocking=request.blocking
            )

            # Get current replica counts
            current_replicas = {}
            deployment = connector.kube_api.get_graph_deployment(
                connector.parent_dgd_name
            )
            for service_name, service_spec in deployment["spec"]["services"].items():
                sub_type = service_spec.get("subComponentType", "")
                if sub_type:
                    current_replicas[sub_type] = service_spec.get("replicas", 0)

            logger.info(
                f"Successfully scaled {request.graph_deployment_name}: {current_replicas}"
            )
            yield {
                "status": ScaleStatus.SUCCESS.value,
                "message": f"Scaled {request.graph_deployment_name} successfully",
                "current_replicas": current_replicas,
            }

        except Exception as e:
            logger.exception(f"Error processing scale request: {e}")
            yield {
                "status": ScaleStatus.ERROR.value,
                "message": str(e),
                "current_replicas": {},
            }
