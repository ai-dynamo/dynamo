# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Connector for delegating scaling decisions to a centralized GlobalPlanner."""

import logging
import os
import time
from typing import Optional

from kubernetes.client import ApiException
from kubernetes.config.config_exception import ConfigException

from dynamo.planner.config.defaults import SubComponentType, TargetReplica
from dynamo.planner.connectors.base import PlannerConnector
from dynamo.planner.connectors.kubernetes import KubernetesConnector
from dynamo.planner.connectors.protocol import ScaleRequest, ScaleStatus
from dynamo.planner.connectors.remote_client import RemotePlannerClient
from dynamo.planner.errors import (
    DeploymentModelNameMismatchError,
    DeploymentValidationError,
    EmptyTargetReplicasError,
    ModelNameNotFoundError,
    UserProvidedModelNameMismatchError,
)
from dynamo.planner.monitoring.worker_info import (
    WorkerInfo,
    build_worker_info_from_defaults,
)
from dynamo.runtime import DistributedRuntime
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)


class GlobalPlannerConnector(PlannerConnector):
    """
    Connector that delegates scaling decisions to a centralized GlobalPlanner.

    This connector wraps RemotePlannerClient and implements the PlannerConnector
    interface, allowing planner_core.py to treat global-planner environment mode
    consistently with kubernetes and virtual modes.
    """

    def __init__(
        self,
        runtime: DistributedRuntime,
        dynamo_namespace: str,
        global_planner_namespace: str,
        global_planner_component: str = "GlobalPlanner",
        model_name: Optional[str] = None,
    ):
        """
        Initialize GlobalPlannerConnector.

        Args:
            runtime: Distributed runtime for communication
            dynamo_namespace: Local dynamo namespace (caller identification)
            global_planner_namespace: Namespace where GlobalPlanner is deployed
            global_planner_component: Component name of GlobalPlanner (default: "GlobalPlanner")
            model_name: Optional model name (will be managed remotely if not provided)
        """
        self.runtime = runtime
        self.dynamo_namespace = dynamo_namespace
        self.global_planner_namespace = global_planner_namespace
        self.global_planner_component = global_planner_component
        self.model_name = model_name
        self.remote_client: Optional[RemotePlannerClient] = None

        # Cache for predicted load (will be set by planner before scaling)
        self.last_predicted_load: Optional[dict] = None

        # Lazily-initialized KubernetesConnector scoped to the pool's own DGD.
        # Used only to read pool-local MDC / DGD args for capability discovery
        # (get_worker_info, get_model_name). Scaling actions still go through
        # the RemotePlannerClient.
        self._local_k8s_connector: Optional[KubernetesConnector] = None
        self._local_k8s_init_attempted: bool = False

    async def _async_init(self):
        """Async initialization - creates RemotePlannerClient"""
        self.remote_client = RemotePlannerClient(
            self.runtime,
            self.global_planner_namespace,
            self.global_planner_component,
        )
        logger.info(
            f"GlobalPlannerConnector initialized: will delegate to {self.global_planner_namespace}.{self.global_planner_component}"
        )

    def set_predicted_load(
        self, num_requests: Optional[float], isl: Optional[float], osl: Optional[float]
    ):
        """
        Set predicted load for inclusion in next scale request.

        This is called by planner_core.py before calling set_component_replicas.
        """
        self.last_predicted_load = {
            "num_requests": num_requests,
            "isl": isl,
            "osl": osl,
        }

    async def set_component_replicas(
        self, target_replicas: list[TargetReplica], blocking: bool = True
    ):
        """
        Set component replicas by delegating to GlobalPlanner.

        Sends a ScaleRequest to the GlobalPlanner with the target replica configurations.

        Args:
            target_replicas: List of target replica configurations
            blocking: Whether to wait for scaling completion (passed to GlobalPlanner)

        Raises:
            EmptyTargetReplicasError: If target_replicas is empty
            RuntimeError: If remote_client is not initialized or response indicates error
        """
        if not target_replicas:
            raise EmptyTargetReplicasError()

        if self.remote_client is None:
            raise RuntimeError(
                "GlobalPlannerConnector not initialized. Call _async_init() first."
            )

        # Get DGD info from environment variables
        graph_deployment_name = os.environ.get("DYN_PARENT_DGD_K8S_NAME")
        if not graph_deployment_name:
            raise ValueError(
                "DYN_PARENT_DGD_K8S_NAME environment variable is required but not set. "
                "Please set DYN_PARENT_DGD_K8S_NAME to specify the parent DGD name."
            )
        k8s_namespace = os.environ.get("POD_NAMESPACE")
        if not k8s_namespace:
            raise ValueError(
                "POD_NAMESPACE environment variable is required but not set. "
                "Please set POD_NAMESPACE to specify the Kubernetes namespace."
            )

        # Create scale request
        request = ScaleRequest(
            caller_namespace=self.dynamo_namespace,
            graph_deployment_name=graph_deployment_name,
            k8s_namespace=k8s_namespace,
            target_replicas=target_replicas,
            blocking=blocking,
            timestamp=time.time(),
            predicted_load=self.last_predicted_load,
        )

        logger.info(
            f"Delegating scale request to GlobalPlanner: "
            f"DGD={graph_deployment_name}, "
            f"prefill={[r.desired_replicas for r in target_replicas if r.sub_component_type == SubComponentType.PREFILL]}, "
            f"decode={[r.desired_replicas for r in target_replicas if r.sub_component_type == SubComponentType.DECODE]}"
        )

        # Send request to GlobalPlanner
        response = await self.remote_client.send_scale_request(request)

        # Check response status
        if response.status == ScaleStatus.SUCCESS:
            logger.info(f"GlobalPlanner scaling successful: {response.message}")
        elif response.status == ScaleStatus.REJECTED:
            # Over-budget rejection is a legitimate business outcome — keep running.
            logger.warning(f"GlobalPlanner rejected scale request: {response.message}")
        elif response.status == ScaleStatus.ERROR:
            logger.error(f"GlobalPlanner scaling failed: {response.message}")
            raise RuntimeError(f"GlobalPlanner scaling failed: {response.message}")
        else:
            logger.warning(
                f"GlobalPlanner returned status '{response.status.value}': {response.message}"
            )

    async def add_component(
        self, sub_component_type: SubComponentType, blocking: bool = True
    ):
        """
        Add a component (not supported for GlobalPlanner).

        GlobalPlanner only supports batch operations via set_component_replicas.
        """
        raise NotImplementedError(
            "GlobalPlannerConnector only supports batch operations via set_component_replicas(). "
            "Use set_component_replicas() to scale components."
        )

    async def remove_component(
        self, sub_component_type: SubComponentType, blocking: bool = True
    ):
        """
        Remove a component (not supported for GlobalPlanner).

        GlobalPlanner only supports batch operations via set_component_replicas.
        """
        raise NotImplementedError(
            "GlobalPlannerConnector only supports batch operations via set_component_replicas(). "
            "Use set_component_replicas() to scale components."
        )

    async def validate_deployment(
        self,
        prefill_component_name: Optional[str] = None,
        decode_component_name: Optional[str] = None,
        **kwargs,
    ):
        """
        Validate deployment (no-op for GlobalPlanner).

        The GlobalPlanner validates the deployment on its side, so local
        validation is not needed in delegating mode.
        """
        logger.info(
            "GlobalPlannerConnector: Skipping local deployment validation "
            "(GlobalPlanner will validate on its side)"
        )

    async def wait_for_deployment_ready(self, include_planner: bool = True):
        """
        Wait for the pool's own workers to be ready.

        Even though GlobalPlanner handles cluster-wide orchestration, the
        pool Planner still reads its own workers' DynamoWorkerMetadata CRs
        for capability discovery (``get_worker_info``). Without a local
        wait, ``_async_init`` runs within milliseconds of pod entry — long
        before workers register MDC — so ``get_worker_info`` falls back to
        defaults with ``context_length`` / ``max_kv_tokens`` unset and
        load-scaling silently disables itself for the pod's lifetime.

        Mirror the standalone path by delegating to the pool-local
        KubernetesConnector. If no local connector is available (e.g.
        running outside a cluster), fall back to the previous no-op so
        out-of-cluster callers are not blocked.
        """
        local = self._get_local_k8s_connector()
        if local is None:
            logger.info(
                "GlobalPlannerConnector: no local KubernetesConnector available, "
                "skipping deployment ready check"
            )
            return
        logger.info(
            "GlobalPlannerConnector: waiting for pool-local workers to be ready"
        )
        await local.wait_for_deployment_ready(include_planner=include_planner)

    def _get_local_k8s_connector(self) -> Optional[KubernetesConnector]:
        """Lazily build a KubernetesConnector scoped to the pool's own DGD.

        The pool's Planner pod has access to its own DGD and the
        DynamoWorkerMetadata CRs of its own workers, even under
        ``environment: global-planner``. Querying them directly is the
        simplest way to populate per-engine capabilities (context_length,
        max_kv_tokens, ...) that load-scaling needs. Returns ``None`` if
        the connector can't be created (e.g. running outside a cluster).
        """
        if self._local_k8s_init_attempted:
            return self._local_k8s_connector
        self._local_k8s_init_attempted = True
        try:
            self._local_k8s_connector = KubernetesConnector(
                dynamo_namespace=self.dynamo_namespace,
                model_name=self.model_name,
            )
        except (DeploymentValidationError, ConfigException, ApiException) as e:
            logger.warning(
                "GlobalPlannerConnector: could not initialize local "
                f"KubernetesConnector for MDC capability lookup: {e}. "
                "Falling back to hard-coded worker defaults; easy-mode "
                "load scaling will be disabled."
            )
        return self._local_k8s_connector

    def get_worker_info(
        self,
        sub_component_type: SubComponentType,
        backend: str = "vllm",
    ) -> WorkerInfo:
        """Resolve per-worker capabilities from the pool's own MDC/DGD.

        Without this, ``resolve_worker_info`` falls through to
        ``build_worker_info_from_defaults`` which leaves ``context_length``
        and ``max_kv_tokens`` unset, and load_scaling's easy-mode decisions
        bail out every tick — so the pool Planner silently sends no
        ScaleRequests.
        """
        local = self._get_local_k8s_connector()
        if local is not None:
            return local.get_worker_info(sub_component_type, backend)
        return build_worker_info_from_defaults(backend, sub_component_type)

    def get_model_name(self, **kwargs) -> str:
        """
        Get model name.

        Prefers the value provided at init time, then the pool's own DGD
        container args (via the local KubernetesConnector), and finally
        falls back to a placeholder indicating the model is managed
        remotely.
        """
        if self.model_name:
            return self.model_name
        local = self._get_local_k8s_connector()
        if local is not None:
            try:
                return local.get_model_name(**kwargs)
            except (
                ModelNameNotFoundError,
                DeploymentModelNameMismatchError,
                UserProvidedModelNameMismatchError,
                ApiException,
            ) as e:
                logger.warning(f"Could not resolve model name from local DGD args: {e}")
        return "managed-remotely"
