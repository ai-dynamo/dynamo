# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import json
import logging
import os
from typing import Optional

from dynamo.planner.config.defaults import SubComponentType, TargetReplica
from dynamo.planner.connectors.base import (
    DynamoGraphPowerBudgetComponentSnapshot,
    DynamoGraphPowerBudgetSnapshot,
    PlannerConnector,
)
from dynamo.planner.connectors.clients.kubernetes_api import (
    DYNAMO_WORKER_METADATA_API_VERSION,
    NVIDIA_API_GROUP,
    KubernetesAPI,
)
from dynamo.planner.connectors.mdc import (
    MdcEntry,
    is_model_card,
    select_entry,
    worker_info_from_mdc,
)
from dynamo.planner.errors import (
    DeploymentModelNameMismatchError,
    DeploymentValidationError,
    DynamoGraphDeploymentNotReadyError,
    EmptyTargetReplicasError,
    ModelNameNotFoundError,
    PlannerError,
    UserProvidedModelNameMismatchError,
)
from dynamo.planner.monitoring.dgd_services import (
    ComponentPowerConfig,
    get_component_from_type_or_name,
    get_component_type,
    get_components_by_name,
    resolve_component_power_configs,
)
from dynamo.planner.monitoring.worker_info import (
    WorkerInfo,
    build_worker_info_from_defaults,
)
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)

CURRENT_WORKER_HASH_ANNOTATION = "nvidia.com/current-worker-hash"
CURRENT_WORKER_HASH_V2_ANNOTATION = "nvidia.com/current-worker-hash-v2"
WORKER_COMPONENT_TYPES = {"worker", "prefill", "decode"}
WORKER_SUFFIX_COMPONENT_KINDS = {"Deployment", "LeaderWorkerSet"}
POWER_CONTROL_MODE_ANNOTATION = "dynamo.nvidia.com/power-control-mode"
TRANSACTIONAL_REPLICA_FENCE = "transactional-replica-fence"


class KubernetesConnector(PlannerConnector):
    def __init__(
        self,
        dynamo_namespace: str,
        model_name: Optional[str] = None,
        k8s_namespace: Optional[str] = None,
        parent_dgd_name: Optional[str] = None,
        raise_not_ready: bool = False,
    ):
        self.kube_api = KubernetesAPI(k8s_namespace)

        self.user_provided_model_name: Optional[str] = None
        if model_name:
            self.user_provided_model_name = (
                model_name.lower()
            )  # normalize model name to lowercase (MDC)

        # Allow overriding parent DGD name for centralized planner
        if parent_dgd_name:
            self.parent_dgd_name = parent_dgd_name
        else:
            graph_deployment_name = os.getenv("DYN_PARENT_DGD_K8S_NAME")
            if not graph_deployment_name:
                raise DeploymentValidationError(
                    ["DYN_PARENT_DGD_K8S_NAME environment variable is not set"]
                )
            self.parent_dgd_name = graph_deployment_name

        # For backwards compatibility
        self.graph_deployment_name = self.parent_dgd_name
        self.raise_not_ready = raise_not_ready
        self._power_budget_snapshot: Optional[DynamoGraphPowerBudgetSnapshot] = None
        self._power_budget_snapshot_observed = False
        self._transactional_power_control = False
        self._transactional_policy_logged = False

    async def async_init(self):
        """No-op asynchronous lifecycle hook."""
        return

    def _deployment_uses_transactional_power(self, deployment: dict) -> bool:
        annotations = deployment.get("metadata", {}).get("annotations", {}) or {}
        transactional = self._transactional_power_control or (
            annotations.get(POWER_CONTROL_MODE_ANNOTATION)
            == TRANSACTIONAL_REPLICA_FENCE
        )
        if transactional:
            self._transactional_power_control = True
        return transactional

    def _request_component_replicas(
        self,
        service_name: str,
        replicas: int,
        *,
        transactional: bool,
    ) -> None:
        if transactional:
            self.kube_api.update_service_replicas(
                self.graph_deployment_name,
                service_name,
                replicas,
                transactional=True,
            )
        else:
            self.kube_api.update_graph_replicas(
                self.graph_deployment_name,
                service_name,
                replicas,
            )

    def get_worker_runtime_namespace(self, base_dynamo_namespace: str) -> str:
        """Return the Dynamo namespace used by the current worker generation.

        Newer operators publish the effective runtime namespace on the worker
        component status. Older operators expose only the active worker hash, so
        the planner falls back to appending that hash only for Deployment-backed
        and LeaderWorkerSet-backed workers.
        """
        deployment = self.kube_api.get_graph_deployment(self.graph_deployment_name)
        worker_status = self._get_first_worker_component_status(deployment)
        if worker_status:
            runtime_namespace = worker_status.get("runtimeNamespace")
            if runtime_namespace:
                # Newer operators report the effective namespace directly.
                return runtime_namespace

        worker_hash = self._get_current_worker_hash(deployment)
        if not worker_hash:
            # No active managed worker hash means workers use the base namespace.
            return base_dynamo_namespace
        if worker_status is None and self._has_worker_component(deployment):
            # A hash with no worker status leaves the backing kind unknown.
            raise PlannerError(
                "Worker component status is not available yet; runtime namespace is indeterminate"
            )
        if not self._worker_status_uses_namespace_suffix(worker_status):
            # Only old Deployment/LWS-backed workers used the hash as a namespace suffix.
            return base_dynamo_namespace
        return f"{base_dynamo_namespace}-{worker_hash}"

    def _get_current_worker_hash(self, deployment: dict) -> Optional[str]:
        annotations = deployment.get("metadata", {}).get("annotations", {}) or {}
        worker_hash = annotations.get(CURRENT_WORKER_HASH_ANNOTATION)
        if worker_hash:
            return worker_hash
        return annotations.get(CURRENT_WORKER_HASH_V2_ANNOTATION)

    def _is_worker_component(self, component_name: str, component: dict) -> bool:
        component_type = get_component_type(component)
        if component_type:
            return component_type in WORKER_COMPONENT_TYPES
        return component_name in WORKER_COMPONENT_TYPES

    def _has_worker_component(self, deployment: dict) -> bool:
        return any(
            self._is_worker_component(component_name, component)
            for component_name, component in get_components_by_name(deployment).items()
        )

    def _get_first_worker_component_status(self, deployment: dict) -> Optional[dict]:
        """Return the first worker-class component status in DGD spec order."""
        status_components = deployment.get("status", {}).get("components", {}) or {}
        components_by_name = get_components_by_name(deployment)
        for component_name, component in components_by_name.items():
            if not self._is_worker_component(component_name, component):
                continue
            worker_status = status_components.get(component_name)
            if worker_status:
                return worker_status
        return None

    def _worker_status_uses_namespace_suffix(
        self, worker_status: Optional[dict]
    ) -> bool:
        if not worker_status:
            return False
        component_kind = worker_status.get("componentKind", "")
        return component_kind in WORKER_SUFFIX_COMPONENT_KINDS

    async def add_component(
        self, sub_component_type: SubComponentType, blocking: bool = True
    ):
        """Add a component by increasing its replica count by 1"""

        deployment = self.kube_api.get_graph_deployment(self.graph_deployment_name)

        service = get_component_from_type_or_name(deployment, sub_component_type)
        self._request_component_replicas(
            service.name,
            service.number_replicas() + 1,
            transactional=self._deployment_uses_transactional_power(deployment),
        )
        if blocking:
            await self.kube_api.wait_for_graph_deployment_ready(
                self.graph_deployment_name,
            )

    async def remove_component(
        self, sub_component_type: SubComponentType, blocking: bool = True
    ):
        """Remove a component by decreasing its replica count by 1"""

        deployment = self.kube_api.get_graph_deployment(self.graph_deployment_name)

        service = get_component_from_type_or_name(deployment, sub_component_type)
        if service.number_replicas() > 0:
            self._request_component_replicas(
                service.name,
                service.number_replicas() - 1,
                transactional=self._deployment_uses_transactional_power(deployment),
            )
            if blocking:
                await self.kube_api.wait_for_graph_deployment_ready(
                    self.graph_deployment_name,
                )

    async def validate_deployment(
        self,
        prefill_component_name: Optional[str] = None,
        decode_component_name: Optional[str] = None,
        require_prefill: bool = True,
        require_decode: bool = True,
    ):
        """
        Verify that the deployment contains prefill/decode components and the model name exists.
        Allows explicit component-name overrides when the caller provides them.

        Raises:
            DynamoGraphDeploymentNotFoundError: If the deployment is not found
            DeploymentValidationError: If the deployment does not contain required prefill/decode components
        """
        deployment = self.kube_api.get_graph_deployment(self.graph_deployment_name)

        errors = []

        if require_prefill:
            try:
                get_component_from_type_or_name(
                    deployment,
                    SubComponentType.PREFILL,
                    component_name=prefill_component_name,
                )
            except PlannerError as e:
                errors.append(str(e))

        if require_decode:
            try:
                get_component_from_type_or_name(
                    deployment,
                    SubComponentType.DECODE,
                    component_name=decode_component_name,
                )
            except PlannerError as e:
                errors.append(str(e))

        try:
            self._get_model_name_from_deployment(
                deployment,
                prefill_component_name=prefill_component_name,
                decode_component_name=decode_component_name,
                require_prefill=require_prefill,
                require_decode=require_decode,
            )
        except PlannerError as e:
            errors.append(str(e))

        # Raise combined error if any issues found
        if errors:
            raise DeploymentValidationError(errors)

    def get_model_name(
        self,
        require_prefill: bool = True,
        require_decode: bool = True,
    ) -> str:
        """Get the model name from the current deployment."""
        try:
            deployment = self.kube_api.get_graph_deployment(self.graph_deployment_name)
        except PlannerError as e:
            if self.user_provided_model_name:
                logger.warning(
                    f"Failed to get model name from deployment with error: {e}, using provided model name: {self.user_provided_model_name}"
                )
                return self.user_provided_model_name
            raise

        return self._get_model_name_from_deployment(
            deployment,
            require_prefill=require_prefill,
            require_decode=require_decode,
        )

    def _get_model_name_from_deployment(
        self,
        deployment: dict,
        require_prefill: bool = True,
        require_decode: bool = True,
        prefill_component_name: Optional[str] = None,
        decode_component_name: Optional[str] = None,
    ) -> str:
        """Get the model name from an already-fetched deployment."""
        try:
            # TODO: dynamo/profiler/utils/config.py already contains DGD config parsing
            # and model name logic, should consolidate
            prefill_model_name = None
            decode_model_name = None
            if require_prefill:
                prefill_service = get_component_from_type_or_name(
                    deployment,
                    SubComponentType.PREFILL,
                    component_name=prefill_component_name,
                )
                prefill_model_name = prefill_service.get_model_name()
            if require_decode:
                decode_service = get_component_from_type_or_name(
                    deployment,
                    SubComponentType.DECODE,
                    component_name=decode_component_name,
                )
                decode_model_name = decode_service.get_model_name()

            if prefill_model_name is None and decode_model_name is None:
                raise ModelNameNotFoundError()

            # Check model name between prefill and decode
            if prefill_model_name is None:
                model_name = decode_model_name
            elif decode_model_name is None:
                model_name = prefill_model_name
            elif prefill_model_name.lower() != decode_model_name.lower():
                raise DeploymentModelNameMismatchError(
                    prefill_model_name, decode_model_name
                )
            else:
                model_name = prefill_model_name

        except PlannerError as e:
            if self.user_provided_model_name:
                logger.warning(
                    f"Failed to get model name from deployment with error: {e}, using provided model name: {self.user_provided_model_name}"
                )
                model_name = self.user_provided_model_name
            else:
                raise e

        if not model_name:
            raise ModelNameNotFoundError()

        # If user provided a model name and it doesn't match the model name from the deployment, raise an error
        if self.user_provided_model_name:
            if model_name.lower() != self.user_provided_model_name:
                raise UserProvidedModelNameMismatchError(
                    model_name, self.user_provided_model_name
                )

        return model_name

    def get_graph_deployment(self) -> dict:
        """Fetch the DGD once for callers that share it across GPU/power reads.

        Not on the base ``PlannerConnector`` protocol — power awareness is
        Kubernetes-local and must not expand that ABC. The environment checks
        ``is_power_aware_connector(controller)`` (all four methods present)
        rather than duck-typing via ``getattr``.
        """
        return self.kube_api.get_graph_deployment(self.graph_deployment_name)

    def get_power_budget_snapshot(
        self,
    ) -> Optional[DynamoGraphPowerBudgetSnapshot]:
        """Return the current tick's cached DGPB observation without API I/O."""

        return self._power_budget_snapshot

    def consume_power_budget_snapshot(self) -> None:
        """Allow the next Planner refresh cycle to observe a new DGPB snapshot."""

        self._power_budget_snapshot_observed = False

    @staticmethod
    def _snapshot_int(value: object, field: str, *, minimum: int = 0) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise DeploymentValidationError(
                [f"DGPB {field} must be an integer >= {minimum}"]
            )
        return value

    @classmethod
    def _parse_power_budget_snapshot(
        cls, resource: dict
    ) -> DynamoGraphPowerBudgetSnapshot:
        """Validate the narrow DGPB policy/inventory surface Planner consumes."""

        spec = resource.get("spec")
        status = resource.get("status")
        if not isinstance(spec, dict) or not isinstance(status, dict):
            raise DeploymentValidationError(
                ["DGPB must contain spec and status objects"]
            )

        policy = spec.get("policy")
        if not isinstance(policy, dict):
            raise DeploymentValidationError(["DGPB spec.policy must be an object"])
        budget_watts = cls._snapshot_int(
            spec.get("budgetWatts"), "spec.budgetWatts", minimum=1
        )
        min_endpoint = cls._snapshot_int(
            policy.get("minEndpoint"), "spec.policy.minEndpoint", minimum=1
        )
        inventory_epoch = cls._snapshot_int(
            status.get("inventoryEpoch", 0), "status.inventoryEpoch"
        )
        phase = status.get("phase", "")
        if not isinstance(phase, str):
            raise DeploymentValidationError(["DGPB status.phase must be a string"])
        rollout_in_progress = status.get("rolloutInProgress", False)
        if not isinstance(rollout_in_progress, bool):
            raise DeploymentValidationError(
                ["DGPB status.rolloutInProgress must be a boolean"]
            )

        committed = status.get("committedReplicaTargets", {})
        if not isinstance(committed, dict):
            raise DeploymentValidationError(
                ["DGPB status.committedReplicaTargets must be an object"]
            )
        committed_targets = {
            name: cls._snapshot_int(
                replicas, f"status.committedReplicaTargets[{name!r}]"
            )
            for name, replicas in committed.items()
            if isinstance(name, str) and name
        }
        if len(committed_targets) != len(committed):
            raise DeploymentValidationError(
                ["DGPB committed replica target names must be non-empty strings"]
            )

        raw_components = status.get("components", [])
        if not isinstance(raw_components, list):
            raise DeploymentValidationError(["DGPB status.components must be a list"])
        components: list[DynamoGraphPowerBudgetComponentSnapshot] = []
        seen_names: set[str] = set()
        for index, row in enumerate(raw_components):
            if not isinstance(row, dict):
                raise DeploymentValidationError(
                    [f"DGPB status.components[{index}] must be an object"]
                )
            name = row.get("name")
            if not isinstance(name, str) or not name or name in seen_names:
                raise DeploymentValidationError(
                    ["DGPB component names must be unique non-empty strings"]
                )
            seen_names.add(name)
            replica_status = row.get("replicaStatus")
            if not isinstance(replica_status, dict):
                raise DeploymentValidationError(
                    [f"DGPB component {name!r} replicaStatus must be an object"]
                )
            available = replica_status.get("availableReplicas")
            serving = (
                available
                if available is not None
                else replica_status.get("readyReplicas", 0)
            )
            components.append(
                DynamoGraphPowerBudgetComponentSnapshot(
                    name=name,
                    ready_replicas=cls._snapshot_int(
                        serving, f"component {name!r} serving replicas"
                    ),
                    committed_replicas=committed_targets.get(name, 0),
                    updated_replicas=cls._snapshot_int(
                        replica_status.get("updatedReplicas", 0),
                        f"component {name!r} updatedReplicas",
                    ),
                    terminating_replicas=cls._snapshot_int(
                        row.get("terminatingReplicas", 0),
                        f"component {name!r} terminatingReplicas",
                    ),
                )
            )

        raw_conditions = status.get("conditions", [])
        if not isinstance(raw_conditions, list) or not all(
            isinstance(condition, dict) for condition in raw_conditions
        ):
            raise DeploymentValidationError(
                ["DGPB status.conditions must be a list of objects"]
            )
        return DynamoGraphPowerBudgetSnapshot(
            budget_watts=budget_watts,
            min_endpoint=min_endpoint,
            phase=phase,
            inventory_epoch=inventory_epoch,
            rollout_in_progress=rollout_in_progress,
            components=tuple(components),
            conditions=tuple(dict(condition) for condition in raw_conditions),
        )

    def _observe_power_budget_snapshot(
        self,
    ) -> Optional[DynamoGraphPowerBudgetSnapshot]:
        if self._power_budget_snapshot_observed:
            return self._power_budget_snapshot

        resource = self.kube_api.get_graph_power_budget(self.graph_deployment_name)
        if resource is None:
            if self._transactional_power_control:
                raise DeploymentValidationError(
                    [
                        "Transactional power control was already observed, but its "
                        f"DGPB {self.graph_deployment_name!r} is now missing"
                    ]
                )
            self._power_budget_snapshot = None
            self._power_budget_snapshot_observed = True
            return None

        snapshot = self._parse_power_budget_snapshot(resource)
        self._transactional_power_control = True
        self._power_budget_snapshot = snapshot
        self._power_budget_snapshot_observed = True
        if not self._transactional_policy_logged:
            logger.info(
                "Transactional power control detected for %s: PlannerConfig."
                "total_gpu_power_limit and PlannerConfig.min_endpoint are inactive; "
                "using DGPB policy budgetWatts=%s minEndpoint=%s",
                self.graph_deployment_name,
                snapshot.budget_watts,
                snapshot.min_endpoint,
            )
            self._transactional_policy_logged = True
        logger.debug(
            "Observed DGPB %s inventoryEpoch=%s phase=%s conditions=%s",
            self.graph_deployment_name,
            snapshot.inventory_epoch,
            snapshot.phase,
            [condition.get("type") for condition in snapshot.conditions],
        )
        return snapshot

    def get_gpu_counts(
        self,
        require_prefill: bool = True,
        require_decode: bool = True,
        deployment: Optional[dict] = None,
    ) -> tuple[int, int]:
        """Get the GPU counts for prefill and decode components.

        Pass ``deployment`` to reuse an already-fetched DGD (avoids a second
        GET when the environment also resolves power configs on the same tick).
        """
        if deployment is None:
            deployment = self.get_graph_deployment()
        return self._get_gpu_counts_from_deployment(
            deployment,
            require_prefill=require_prefill,
            require_decode=require_decode,
        )

    def _get_gpu_counts_from_deployment(
        self,
        deployment: dict,
        require_prefill: bool = True,
        require_decode: bool = True,
    ) -> tuple[int, int]:
        """Get GPU counts from an already-fetched deployment.

        Args:
            deployment: Deployment dict to inspect
            require_prefill: Whether to require a prefill component
            require_decode: Whether to require a decode component

        Returns:
            Tuple of (prefill_gpu_count, decode_gpu_count)

        Raises:
            DeploymentValidationError: If GPU counts cannot be determined from DGD
        """
        prefill_gpu_count = 0
        decode_gpu_count = 0
        errors = []

        if require_prefill:
            try:
                prefill_service = get_component_from_type_or_name(
                    deployment,
                    SubComponentType.PREFILL,
                )
                prefill_gpu_count = prefill_service.get_gpu_count()
            except (PlannerError, ValueError) as e:
                errors.append(f"Failed to get prefill GPU count: {e}")

        if require_decode:
            try:
                decode_service = get_component_from_type_or_name(
                    deployment,
                    SubComponentType.DECODE,
                )
                decode_gpu_count = decode_service.get_gpu_count()
            except (PlannerError, ValueError) as e:
                errors.append(f"Failed to get decode GPU count: {e}")

        if errors:
            raise DeploymentValidationError(errors)

        return prefill_gpu_count, decode_gpu_count

    def get_component_power_configs(
        self,
        require_prefill: bool = True,
        require_decode: bool = True,
        prefill_component_name: Optional[str] = None,
        decode_component_name: Optional[str] = None,
        deployment: Optional[dict] = None,
    ) -> tuple[Optional[ComponentPowerConfig], Optional[ComponentPowerConfig]]:
        """Resolve DGD-owned per-role power configs from worker podTemplate annotations.

        One DGD GET unless ``deployment`` is provided (shared with
        ``get_gpu_counts`` on the same tick). ``watts_per_replica`` on each
        config uses the replica-wide GPU total (nodeCount × per-pod) via
        ``Service.get_total_gpu_count()``, independent of the per-pod
        ``num_gpus`` the GPU-budget math consumes.

        The typed parser errors (``PowerAnnotationMissingError`` /
        ``PowerAnnotationInvalidError`` / ``SubComponentNotFoundError`` /
        ``DuplicateSubComponentError`` / ``ValueError`` for a bad GPU count)
        propagate so the environment can apply the startup-fail vs
        runtime-conservative policy rather than the planner guessing a cap.
        """
        if deployment is None:
            deployment = self.get_graph_deployment()
        return resolve_component_power_configs(
            deployment,
            require_prefill=require_prefill,
            require_decode=require_decode,
            prefill_name=prefill_component_name,
            decode_name=decode_component_name,
        )

    def get_frontend_metrics_url(self, port: int = 8000) -> Optional[str]:
        """Auto-discover the frontend component's metrics URL from the DGD.

        Iterates DGD components to find the component with type "frontend",
        then constructs the in-cluster URL using the operator's naming convention:
        http://{dgd_name}-{component_name_lowercase}:{port}/metrics

        Returns:
            The metrics URL string, or None if no frontend component is found.
        """
        deployment = self.kube_api.get_graph_deployment(self.graph_deployment_name)
        components = get_components_by_name(deployment)

        for component_name, component_spec in components.items():
            if get_component_type(component_spec) == "frontend":
                service_name = f"{self.graph_deployment_name}-{component_name.lower()}"
                url = f"http://{service_name}:{port}/metrics"
                logger.info(f"Auto-discovered frontend metrics URL: {url}")
                return url

        return None

    async def wait_for_deployment_ready(self, include_planner: bool = True):
        """Wait for the deployment to be ready (legacy replica-stability path).

        Does **not** check pod annotation convergence or require
        ``observedGeneration`` catch-up. Power-aware callers that permanently
        cache DGD fields must use :meth:`wait_for_settled_graph_deployment`
        instead.

        Args:
            include_planner: If False, skip the planner component when checking
                readiness. This lets the planner read MDC from worker pods
                without waiting for itself to be marked ready in the DGD.
        """
        await self.kube_api.wait_for_graph_deployment_ready(
            self.graph_deployment_name,
            include_planner=include_planner,
            require_backing_settled=False,
        )

    async def wait_for_settled_graph_deployment(
        self,
        include_planner: bool = False,
        *,
        require_prefill: bool = True,
        require_decode: bool = True,
        prefill_component_name: Optional[str] = None,
        decode_component_name: Optional[str] = None,
    ) -> dict:
        """Wait for a settled DGD snapshot and return that same object.

        When ``include_planner`` is False, the snapshot has:
        - non-planner worker replica counts stable (desired == updated == ready)
        - ``status.observedGeneration >= metadata.generation``
        - every non-terminal worker Pod carries the expected
          ``dynamo.nvidia.com/gpu-power-limit`` annotation from the current
          DGD snapshot, confirming the operator has propagated the DGD intent
          to running Pods (hardware enforcement by the Power Agent/NVML is
          separate and not verified here)

        Power-relevant workers are selected with the same role/name resolution
        as :meth:`get_component_power_configs` (typed roles, explicit-name
        fallback for untyped workers, unique generic ``type: worker`` for agg).

        Callers that permanently cache fields from the DGD (power caps) must
        use this snapshot rather than issuing a later GET, so an
        annotation-only generation bump cannot be adopted before workers
        have rolled onto that generation. Active rolling updates
        (``status.rollingUpdate.phase`` Pending/InProgress/Failed) also
        block settlement because old Pods still carry the previous cap.
        """
        return await self.kube_api.wait_for_graph_deployment_ready(
            self.graph_deployment_name,
            include_planner=include_planner,
            require_backing_settled=True,
            require_prefill=require_prefill,
            require_decode=require_decode,
            prefill_component_name=prefill_component_name,
            decode_component_name=decode_component_name,
        )

    def _list_worker_metadata_crs(self) -> list[dict]:
        """List all DynamoWorkerMetadata CRs in the current namespace.

        Returns an empty list only when the CRD is not yet installed (404).
        Other API errors (RBAC, connectivity) are re-raised so callers can
        handle them explicitly.
        """
        from kubernetes.client import ApiException

        try:
            result = self.kube_api.custom_api.list_namespaced_custom_object(
                group=NVIDIA_API_GROUP,
                version=DYNAMO_WORKER_METADATA_API_VERSION,
                namespace=self.kube_api.current_namespace,
                plural="dynamoworkermetadatas",
            )
            return result.get("items", [])
        except ApiException as e:
            if e.status == 404:
                logger.info("DynamoWorkerMetadata CRD not found, skipping MDC")
                return []
            raise

    def _get_dgd_component_names(self) -> list[str]:
        """Return the Kubernetes component names reported in DGD status.

        Grove may truncate and hash long DGD names when it creates component
        resources. These status names reflect the names actually used by the
        worker pods and their DynamoWorkerMetadata CRs.
        """
        deployment = self.kube_api.get_graph_deployment(self.graph_deployment_name)
        component_statuses = deployment.get("status", {}).get("components", {}) or {}
        return [
            component_name
            for component_status in component_statuses.values()
            for component_name in component_status.get("componentNames", []) or []
        ]

    def _extract_mdc_entries(self) -> list[MdcEntry]:
        """Extract MDC entries belonging to this DGD.

        CRs are named after the worker pod. Match the DGD name prefix and the
        actual component names from DGD status because Grove may truncate and
        hash a long DGD name. LoRA-adapter wrappers are dropped via
        :func:`is_model_card`.
        """
        crs = self._list_worker_metadata_crs()
        component_names = self._get_dgd_component_names()
        dgd_prefix = f"{self.graph_deployment_name}-"

        entries: list[MdcEntry] = []
        for cr in crs:
            cr_name = cr.get("metadata", {}).get("name", "")
            belongs_to_dgd = cr_name.startswith(dgd_prefix) or any(
                cr_name == component_name or cr_name.startswith(f"{component_name}-")
                for component_name in component_names
            )
            if not belongs_to_dgd:
                continue

            data = cr.get("spec", {}).get("data", {})
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    continue
            model_cards = data.get("model_cards", {})
            for _key, wrapper in model_cards.items():
                if not is_model_card(wrapper):
                    continue
                entries.append(
                    MdcEntry(
                        card_json=wrapper.get("card_json") or {},
                        component=wrapper.get("component"),
                        endpoint=wrapper.get("endpoint"),
                        instance_id=wrapper.get("instance_id"),
                    )
                )
        return entries

    def _resolve_dgd_service(
        self, sub_component_type: SubComponentType, backend: str
    ) -> tuple[Optional[str], str]:
        """Return (dgd_service_name, component_name_for_filter).

        ``dgd_service_name`` is the stable DGD component name (typically
        PascalCase, e.g. ``"VllmPrefillWorker"``) and is used for Kubernetes
        operations like patching replica counts.

        ``component_name_for_filter`` is the component name that the Rust
        runtime registers via ``Endpoint`` and writes into the MDC
        ``component`` field. Source of truth, in priority order:

        1. The user's ``--endpoint <ns>.<component>.<ep>`` override in the
           worker's container args (supported by all backends --
           see vllm/args.py:171-176, sglang/args.py:428, trtllm/args.py:137).
        2. The backend-specific default from
           :func:`build_worker_info_from_defaults` (e.g. ``"prefill"`` /
           ``"backend"``).

        Note: the DGD component name (``service.name``) must NOT be used for
        filtering -- it is typically PascalCase (``"VllmPrefillWorker"``) and
        would never match the lowercase value the worker writes to MDC.
        """
        defaults = build_worker_info_from_defaults(backend, sub_component_type)
        expected_component = defaults.component_name or ""
        try:
            deployment = self.kube_api.get_graph_deployment(self.graph_deployment_name)
            service = get_component_from_type_or_name(deployment, sub_component_type)
            user_component = service.get_component_name_from_endpoint_arg()
            if user_component:
                expected_component = user_component
            return service.name, expected_component
        except PlannerError:
            return None, expected_component

    def get_worker_info(
        self,
        sub_component_type: SubComponentType,
        backend: str = "vllm",
    ) -> WorkerInfo:
        """Get WorkerInfo for a sub-component, trying MDC first, then fallbacks.

        Args:
            sub_component_type: PREFILL or DECODE
            backend: Backend framework name (for default fallback)
        """
        entries = self._extract_mdc_entries()
        dgd_service_name, expected_component = self._resolve_dgd_service(
            sub_component_type, backend
        )

        def _dgd_model_name() -> Optional[str]:
            try:
                deployment = self.kube_api.get_graph_deployment(
                    self.graph_deployment_name
                )
                service = get_component_from_type_or_name(
                    deployment, sub_component_type
                )
                return service.get_model_name()
            except PlannerError:
                return None

        entry = select_entry(entries, sub_component_type, expected_component)
        if entry is not None:
            info = worker_info_from_mdc(
                entry,
                sub_component_type,
                backend=backend,
                model_name_fallback=_dgd_model_name,
                k8s_name_override=dgd_service_name,
            )
            if not info.model_name:
                logger.warning(
                    f"Could not determine model name for {sub_component_type.value} "
                    f"from MDC or DGD container args"
                )
            logger.info(
                f"Built {sub_component_type.value} WorkerInfo from MDC: "
                f"{info.summary()}"
            )
            return info

        # No MDC entry found -- fall back entirely to defaults + DGD arg parsing.
        logger.warning(
            f"No DynamoWorkerMetadata CR found for {sub_component_type.value}. "
            f"Workers may not be registered yet. Falling back to defaults."
        )
        info = build_worker_info_from_defaults(backend, sub_component_type)
        if dgd_service_name is not None:
            info.k8s_name = dgd_service_name
        arg_model = _dgd_model_name()
        if arg_model:
            info.model_name = arg_model
            logger.info(
                f"Enriched {sub_component_type.value} WorkerInfo model name "
                f"from DGD args: {arg_model}"
            )

        logger.info(
            f"Using fallback WorkerInfo for {sub_component_type.value}: {info.summary()}"
        )
        return info

    # todo -> how are we handling 3 active 2 more new workers pending?
    async def get_actual_worker_counts(
        self,
        prefill_component_name: Optional[str] = None,
        decode_component_name: Optional[str] = None,
    ) -> tuple[int, int, bool]:
        """Get ready worker counts from DGD status without listing Pods."""
        deployment = self.kube_api.get_graph_deployment(self.graph_deployment_name)
        return self._worker_counts_from_snapshot(
            deployment,
            prefill_component_name=prefill_component_name,
            decode_component_name=decode_component_name,
        )

    async def get_power_aware_worker_counts(
        self,
        prefill_component_name: Optional[str] = None,
        decode_component_name: Optional[str] = None,
    ) -> tuple[int, int, bool]:
        """Get power-safe worker counts without blocking the Planner event loop.

        One thread dispatch contains the synchronous DGD GET and the single
        DGD-scoped Pod LIST. The returned Pod snapshot is partitioned locally by
        component before terminating-Pod checks run.
        """
        return await asyncio.to_thread(
            self._get_power_aware_worker_counts_sync,
            prefill_component_name,
            decode_component_name,
        )

    def _get_power_aware_worker_counts_sync(
        self,
        prefill_component_name: Optional[str],
        decode_component_name: Optional[str],
    ) -> tuple[int, int, bool]:
        power_budget = self._observe_power_budget_snapshot()
        if power_budget is not None:
            return self._transactional_worker_counts_from_snapshot(
                power_budget,
                prefill_component_name=prefill_component_name,
                decode_component_name=decode_component_name,
            )

        deployment = self.kube_api.get_graph_deployment(self.graph_deployment_name)
        annotations = deployment.get("metadata", {}).get("annotations", {}) or {}
        if (
            annotations.get(POWER_CONTROL_MODE_ANNOTATION)
            == TRANSACTIONAL_REPLICA_FENCE
        ):
            self._transactional_power_control = True
            raise DeploymentValidationError(
                [
                    f"Transactional DGD {self.graph_deployment_name!r} has no DGPB; "
                    "refusing to fall back to DGD and Pod inventory"
                ]
            )
        dgd_name = deployment.get("metadata", {}).get("name", "")
        pods = self.kube_api.list_pods_for_graph(dgd_name) if dgd_name else []
        pods_by_component = self.kube_api.partition_pods_by_component(pods)
        return self._worker_counts_from_snapshot(
            deployment,
            prefill_component_name=prefill_component_name,
            decode_component_name=decode_component_name,
            pods_by_component=pods_by_component,
            power_aware=True,
        )

    def _transactional_worker_counts_from_snapshot(
        self,
        snapshot: DynamoGraphPowerBudgetSnapshot,
        *,
        prefill_component_name: Optional[str],
        decode_component_name: Optional[str],
    ) -> tuple[int, int, bool]:
        """Resolve serving counts and the global hold from one DGPB snapshot."""

        rows = {component.name: component for component in snapshot.components}
        missing = [
            name
            for name in (prefill_component_name, decode_component_name)
            if name is not None and name not in rows
        ]
        if missing:
            raise DeploymentValidationError(
                [
                    "DGPB status is missing required component row(s): "
                    + ", ".join(sorted(missing))
                ]
            )

        prefill = (
            rows[prefill_component_name].ready_replicas if prefill_component_name else 0
        )
        decode = (
            rows[decode_component_name].ready_replicas if decode_component_name else 0
        )
        relevant = [
            rows[name]
            for name in (prefill_component_name, decode_component_name)
            if name is not None
        ]
        all_stable = not snapshot.rollout_in_progress and all(
            component.settled for component in relevant
        )
        return prefill, decode, all_stable

    def _worker_counts_from_snapshot(
        self,
        deployment: dict,
        *,
        prefill_component_name: Optional[str],
        decode_component_name: Optional[str],
        pods_by_component: Optional[dict[str, list]] = None,
        power_aware: bool = False,
    ) -> tuple[int, int, bool]:
        prefill_count = 0
        decode_count = 0
        all_stable = True

        if prefill_component_name:
            service = get_component_from_type_or_name(
                deployment,
                SubComponentType.PREFILL,
                component_name=prefill_component_name,
            )
            ready_replicas, is_stable = self.kube_api.get_service_replica_status(
                deployment, service.name
            )
            if (
                is_stable
                and power_aware
                and self.kube_api.has_terminating_pods(
                    (pods_by_component or {}).get(service.name, [])
                )
            ):
                is_stable = False
            if not is_stable:
                all_stable = False
            prefill_count = ready_replicas

        if decode_component_name:
            service = get_component_from_type_or_name(
                deployment,
                SubComponentType.DECODE,
                component_name=decode_component_name,
            )
            ready_replicas, is_stable = self.kube_api.get_service_replica_status(
                deployment, service.name
            )
            if (
                is_stable
                and power_aware
                and self.kube_api.has_terminating_pods(
                    (pods_by_component or {}).get(service.name, [])
                )
            ):
                is_stable = False
            if not is_stable:
                all_stable = False
            decode_count = ready_replicas

        if power_aware:
            is_blocking, reason = self.kube_api.is_rolling_update_blocking_settlement(
                deployment
            )
            if not is_blocking:
                # Failed is not in ROLLING_UPDATE_BLOCKING_PHASES because at
                # startup it raises immediately rather than blocking. At runtime
                # there is no raise, so the dedicated power path treats Failed as
                # fail-closed (unstable) and does not admit scale-ups.
                rolling = deployment.get("status", {}).get("rollingUpdate") or {}
                if rolling.get("phase") == "Failed":
                    is_blocking = True
                    reason = "rollingUpdate.phase=Failed"
            if is_blocking:
                logger.info(
                    "%s: treating runtime counts as unstable: %s",
                    self.graph_deployment_name,
                    reason,
                )
                all_stable = False

        return prefill_count, decode_count, all_stable

    async def set_component_replicas(
        self, target_replicas: list[TargetReplica], blocking: bool = True
    ):
        """Set the replicas for multiple components at once"""
        if not target_replicas:
            raise EmptyTargetReplicasError()

        deployment = self.kube_api.get_graph_deployment(self.graph_deployment_name)
        transactional = self._deployment_uses_transactional_power(deployment)

        # The DGD is the committed mirror in transactional mode and is often
        # intentionally unready while an admitted rollout is applying. Safe
        # reductions and cancellation requests must still reach DGDSA during
        # that window. Preserve the legacy Ready guard only for direct-DGD
        # (static) scaling.
        if not transactional and not self.kube_api.is_deployment_ready(deployment):
            if self.raise_not_ready:
                logger.warning(
                    "Deployment %s is not ready, rejecting this scaling",
                    self.graph_deployment_name,
                )
                raise DynamoGraphDeploymentNotReadyError(
                    deployment_name=self.graph_deployment_name,
                    namespace=getattr(self.kube_api, "current_namespace", None),
                )
            logger.warning(
                "Deployment %s is not ready, ignoring this scaling",
                self.graph_deployment_name,
            )
            return

        for target_replica in target_replicas:
            service = get_component_from_type_or_name(
                deployment,
                target_replica.sub_component_type,
                component_name=target_replica.component_name,
            )
            current_replicas = service.number_replicas()
            # In transactional mode DGD spec is the committed mirror, not the
            # current request. Always write an explicit DGDSA request so a
            # proposal equal to the commitment can cancel a different pending
            # request. Static mode retains its existing no-op suppression.
            if transactional or current_replicas != target_replica.desired_replicas:
                logger.info(
                    f"Updating {target_replica.sub_component_type.value} component {service.name} to desired replica count {target_replica.desired_replicas}"
                )
                self._request_component_replicas(
                    service.name,
                    target_replica.desired_replicas,
                    transactional=transactional,
                )
            else:
                logger.info(
                    f"{target_replica.sub_component_type.value} component {service.name} already at desired replica count {target_replica.desired_replicas}, skipping"
                )

        if blocking:
            await self.kube_api.wait_for_graph_deployment_ready(
                self.graph_deployment_name,
            )


if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser()
    parser.add_argument("--dynamo_namespace", type=str, default="dynamo")
    parser.add_argument("--k8s_namespace", type=str, default="default")
    parser.add_argument("--action", type=str, choices=["add", "remove"])
    parser.add_argument(
        "--component",
        type=str,
        choices=[t.value for t in SubComponentType],
        default=SubComponentType.PREFILL.value,
        help="Target sub-component to scale",
    )
    parser.add_argument("--blocking", action="store_true")
    args = parser.parse_args()
    connector = KubernetesConnector(
        args.dynamo_namespace, k8s_namespace=args.k8s_namespace
    )

    if args.action == "add":
        task = connector.add_component(SubComponentType(args.component), args.blocking)
    elif args.action == "remove":
        task = connector.remove_component(
            SubComponentType(args.component), args.blocking
        )
    asyncio.run(task)
