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
import copy
import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from threading import Lock
from typing import Optional

from kubernetes.client import ApiException

from dynamo.planner.config.defaults import SubComponentType, TargetReplica
from dynamo.planner.connectors.base import PlannerConnector
from dynamo.planner.connectors.clients.kubernetes_api import (
    DYNAMO_WORKER_METADATA_API_VERSION,
    NVIDIA_API_GROUP,
    PLANNER_WRITER_FENCE_ANNOTATION,
    KubernetesAPI,
)
from dynamo.planner.connectors.mdc import (
    MdcEntry,
    is_model_card,
    select_entry,
    worker_info_from_mdc,
)
from dynamo.planner.core.types import WorkerCounts
from dynamo.planner.errors import (
    DeploymentModelNameMismatchError,
    DeploymentValidationError,
    DynamoGraphDeploymentNotReadyError,
    EmptyTargetReplicasError,
    GPUShapeUnavailableError,
    ModelNameNotFoundError,
    PlannerError,
    UserProvidedModelNameMismatchError,
)
from dynamo.planner.monitoring.dgd_services import (
    ComponentGPUShape,
    ComponentPowerConfig,
    Service,
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


@dataclass(frozen=True)
class _ReplicaUpdatePlan:
    """One fully resolved replica target from a single DGD snapshot."""

    target: TargetReplica
    service: Service
    current_replicas: int
    scaling_adapter: dict | None


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
        # DGDSA application and DGD status are asynchronous. Hold subsequent
        # writes until a startup reversal has actually finished, not just until
        # the pre-write Ready condition is observed again.
        self._startup_scale_down_lock = Lock()
        self._startup_scale_down_targets: dict[str, int] = {}
        self._startup_read_warnings: set[str] = set()
        self._batch_writer_id: str | None = None

    async def async_init(self):
        """No-op asynchronous lifecycle hook."""
        return

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
        current_replicas, scaling_adapter = self._authoritative_replica_state(
            deployment, service
        )
        self._write_preflighted_replica_target(
            service, current_replicas + 1, scaling_adapter
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
        current_replicas, scaling_adapter = self._authoritative_replica_state(
            deployment, service
        )
        if current_replicas > 0:
            self._write_preflighted_replica_target(
                service, current_replicas - 1, scaling_adapter
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
                prefill_model_name = prefill_service.get_model_name(deployment)
            if require_decode:
                decode_service = get_component_from_type_or_name(
                    deployment,
                    SubComponentType.DECODE,
                    component_name=decode_component_name,
                )
                decode_model_name = decode_service.get_model_name(deployment)

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

    def get_gpu_counts(
        self,
        require_prefill: bool = True,
        require_decode: bool = True,
        deployment: Optional[dict] = None,
    ) -> tuple[int, int]:
        """Get per-engine GPU counts for prefill and decode components."""
        prefill_shape, decode_shape = self.get_gpu_shapes(
            require_prefill=require_prefill,
            require_decode=require_decode,
            deployment=deployment,
        )
        errors = []
        if require_prefill and prefill_shape is None:
            errors.append("Prefill mocker requires a configured logical GPU count")
        if require_decode and decode_shape is None:
            errors.append("Decode mocker requires a configured logical GPU count")
        if errors:
            raise DeploymentValidationError(errors)
        return (
            prefill_shape.gpus_per_engine if prefill_shape is not None else 0,
            decode_shape.gpus_per_engine if decode_shape is not None else 0,
        )

    def get_gpu_shapes(
        self,
        require_prefill: bool = True,
        require_decode: bool = True,
        deployment: Optional[dict] = None,
    ) -> tuple[Optional[ComponentGPUShape], Optional[ComponentGPUShape]]:
        """Get per-engine performance width and per-replica GPU cost.

        Pass ``deployment`` to reuse an already-fetched DGD (avoids a second
        GET when the environment also resolves power configs on the same tick).
        """
        if deployment is None:
            deployment = self.get_graph_deployment()
        return self._get_gpu_shapes_from_deployment(
            deployment,
            require_prefill=require_prefill,
            require_decode=require_decode,
        )

    def _get_gpu_shapes_from_deployment(
        self,
        deployment: dict,
        require_prefill: bool = True,
        require_decode: bool = True,
    ) -> tuple[Optional[ComponentGPUShape], Optional[ComponentGPUShape]]:
        """Get GPU shapes from an already-fetched deployment.

        Args:
            deployment: Deployment dict to inspect
            require_prefill: Whether to require a prefill component
            require_decode: Whether to require a decode component

        Returns:
            Tuple of (prefill_gpu_shape, decode_gpu_shape)

        Raises:
            DeploymentValidationError: If GPU shapes cannot be determined from DGD
            GPUShapeUnavailableError: If an authoritative shape is stale, missing,
                for a required Planner worker
        """
        prefill_gpu_shape = None
        decode_gpu_shape = None
        errors = []

        if require_prefill:
            try:
                prefill_service = get_component_from_type_or_name(
                    deployment,
                    SubComponentType.PREFILL,
                )
                prefill_gpu_shape = self._planner_gpu_shape(
                    prefill_service.get_gpu_shape(deployment)
                )
            except GPUShapeUnavailableError:
                raise
            except (PlannerError, ValueError) as e:
                errors.append(f"Failed to get prefill GPU shape: {e}")

        if require_decode:
            try:
                decode_service = get_component_from_type_or_name(
                    deployment,
                    SubComponentType.DECODE,
                )
                decode_gpu_shape = self._planner_gpu_shape(
                    decode_service.get_gpu_shape(deployment)
                )
            except GPUShapeUnavailableError:
                raise
            except (PlannerError, ValueError) as e:
                errors.append(f"Failed to get decode GPU shape: {e}")

        if errors:
            raise DeploymentValidationError(errors)

        return prefill_gpu_shape, decode_gpu_shape

    @staticmethod
    def _planner_gpu_shape(
        shape: ComponentGPUShape,
    ) -> Optional[ComponentGPUShape]:
        """Use configured logical width when the operator reports no physical GPUs."""
        if shape == ComponentGPUShape(0, 0):
            return None
        return shape

    def get_component_power_configs(
        self,
        require_prefill: bool = True,
        require_decode: bool = True,
        prefill_component_name: Optional[str] = None,
        decode_component_name: Optional[str] = None,
        deployment: Optional[dict] = None,
    ) -> tuple[Optional[ComponentPowerConfig], Optional[ComponentPowerConfig]]:
        """Resolve operator-projected per-role power configs from DGD status.

        One DGD GET unless ``deployment`` is provided (shared with
        ``get_gpu_shapes`` on the same tick). ``watts_per_replica`` on each
        config uses the operator-projected ``gpusPerReplica``.

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

        ``dgd_service_name`` is the DGD ``spec.services`` dict key (typically
        PascalCase, e.g. ``"prefill"``) and is used for Kubernetes
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

        Note: the DGD services dict key (``service.name``) must NOT be used
        here -- it is typically PascalCase (``"prefill"``) and
        would never match the lowercase value the worker writes to MDC.
        """
        defaults = build_worker_info_from_defaults(backend, sub_component_type)
        expected_component = defaults.component_name or ""
        try:
            deployment = self.kube_api.get_graph_deployment(self.graph_deployment_name)
            service = get_component_from_type_or_name(deployment, sub_component_type)
            user_component = service.get_runtime_component_name(deployment)
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
                return service.get_model_name(deployment)
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

    def _list_startup_pods(self) -> Optional[list]:
        """Return lifecycle Pods, or None when legacy RBAC forbids listing them."""
        try:
            pods = self.kube_api.list_pods_for_graph(self.graph_deployment_name)
        except ApiException as exc:
            if exc.status != 403:
                raise
            with self._startup_scale_down_lock:
                warn = "pods" not in self._startup_read_warnings
                self._startup_read_warnings.add("pods")
            if warn:
                logger.warning(
                    "Pod list forbidden for %s; startup cancellation is disabled "
                    "until pods/list is granted",
                    self.graph_deployment_name,
                )
            return None
        with self._startup_scale_down_lock:
            self._startup_read_warnings.discard("pods")
        return self.kube_api.exclude_checkpoint_capture_pods(pods)

    def _startup_scale_down_in_progress(self, deployment: dict, pods: list) -> bool:
        # Snapshot under the lock, then do Kubernetes I/O without holding it.
        # Writers replace the dictionary so identity detects even a concurrent
        # request for the same target, not only a different replica count.
        with self._startup_scale_down_lock:
            previous_targets = self._startup_scale_down_targets
            targets = previous_targets.copy()
        if not targets:
            return False
        if not self.kube_api.is_spec_generation_observed(deployment):
            return True
        if self.kube_api.has_terminating_pods(pods):
            return True
        if not self.kube_api.pcsg_pods_within_desired_replicas(deployment, pods):
            return True
        components = get_components_by_name(deployment)
        startup = self.kube_api.pending_startup_replicas(deployment, pods)
        remaining: dict[str, int] = {}
        for name, target in targets.items():
            service = Service(name=name, service=components.get(name, {}))
            desired = service.number_replicas()
            _, stable = self.kube_api.get_service_replica_status(deployment, name)
            if desired != target:
                if "scalingAdapter" not in service.service:
                    # A direct DGD write is authoritative for an unmarked
                    # component. Ignore any convention-named stray DGDSA.
                    authoritative_target = desired
                else:
                    try:
                        authoritative_target, _ = self._authoritative_replica_state(
                            deployment, service
                        )
                    except ApiException as exc:
                        if exc.status != 403:
                            raise
                        # GET may be revoked after admission. Retain this request
                        # until its DGD target arrives; do not guess that an
                        # unobserved write was superseded or drop the latch.
                        with self._startup_scale_down_lock:
                            warn = "scale" not in self._startup_read_warnings
                            self._startup_read_warnings.add("scale")
                        if warn:
                            logger.warning(
                                "DGDSA get forbidden for %s; holding startup "
                                "scale-down until its DGD target is observed",
                                self.graph_deployment_name,
                            )
                        remaining[name] = target
                        continue
                    with self._startup_scale_down_lock:
                        self._startup_read_warnings.discard("scale")
                if authoritative_target == target or desired != authoritative_target:
                    remaining[name] = target
                # A superseding DGDSA target has reached the observed DGD spec.
                # Retire our old request; ordinary inventory guards its state.
            elif not stable and name not in startup:
                remaining[name] = target
            # Once excess/terminating replicas are gone, a survivor becoming
            # unready is startup capacity again, not an unfinished drain.
        with self._startup_scale_down_lock:
            if self._startup_scale_down_targets is previous_targets:
                self._startup_scale_down_targets = remaining
            return bool(self._startup_scale_down_targets)

    async def get_worker_inventory(
        self,
        prefill_component_name: Optional[str] = None,
        decode_component_name: Optional[str] = None,
    ) -> Optional[WorkerCounts]:
        """Read selected roles' serving/pending inventory off the event loop.

        DGD status and Pods supply startup, drain, and rollout checks, including
        the power-aware readiness guarantees. An omitted component name reports
        zero counts for that role; pending counts never include draining workers.
        Return None on a Pod-list 403 so callers can retain legacy behavior.
        """
        return await asyncio.to_thread(
            self._get_worker_inventory_sync,
            prefill_component_name,
            decode_component_name,
        )

    def _get_worker_inventory_sync(
        self,
        prefill_component_name: Optional[str],
        decode_component_name: Optional[str],
    ) -> Optional[WorkerCounts]:
        observed_deployment = self.kube_api.get_graph_deployment(
            self.graph_deployment_name
        )
        deployment = self.kube_api.fetch_authoritative_replica_targets(
            observed_deployment
        )
        pods = self._list_startup_pods()
        if pods is None:
            return None
        p, d, stable = self._worker_counts_from_snapshot(
            deployment,
            prefill_component_name=prefill_component_name,
            decode_component_name=decode_component_name,
            pods_by_component=self.kube_api.partition_pods_by_component(pods),
            power_aware=True,
        )
        stable = (
            stable
            and self.kube_api.is_spec_generation_observed(deployment)
            and self.kube_api.pcsg_pods_within_desired_replicas(deployment, pods)
        )
        pending = self.kube_api.pending_startup_replicas(deployment, pods)
        # Latch settlement needs both snapshots: authoritative targets drive
        # counts, while the unmodified DGD shows whether a superseding adapter
        # target has propagated into the operator-observed spec yet.
        if self._startup_scale_down_in_progress(observed_deployment, pods):
            stable, pending = False, {}
        return WorkerCounts(
            ready_num_prefill=p,
            ready_num_decode=d,
            expected_num_prefill=p if stable else None,
            expected_num_decode=d if stable else None,
            prefill_scaling_in_progress=not stable,
            decode_scaling_in_progress=not stable,
            pending_num_prefill=pending.get(prefill_component_name or "", 0),
            pending_num_decode=pending.get(decode_component_name or "", 0),
        )

    async def get_actual_worker_counts(
        self,
        prefill_component_name: Optional[str] = None,
        decode_component_name: Optional[str] = None,
    ) -> tuple[int, int, bool]:
        """Get worker counts without blocking the Planner event loop."""
        return await asyncio.to_thread(
            self._get_actual_worker_counts_sync,
            prefill_component_name,
            decode_component_name,
        )

    def _get_actual_worker_counts_sync(
        self,
        prefill_component_name: Optional[str],
        decode_component_name: Optional[str],
    ) -> tuple[int, int, bool]:
        deployment = self.kube_api.get_graph_deployment(self.graph_deployment_name)
        deployment = self.kube_api.fetch_authoritative_replica_targets(deployment)
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
        deployment = self.kube_api.get_graph_deployment(self.graph_deployment_name)
        dgd_name = deployment.get("metadata", {}).get("name", "")
        pods = self.kube_api.list_pods_for_graph(dgd_name) if dgd_name else []
        pods_by_component = self.kube_api.partition_pods_by_component(pods)
        deployment = self.kube_api.fetch_authoritative_replica_targets(deployment)
        return self._worker_counts_from_snapshot(
            deployment,
            prefill_component_name=prefill_component_name,
            decode_component_name=decode_component_name,
            pods_by_component=pods_by_component,
            power_aware=True,
        )

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
        deployment_ready = self.kube_api.is_deployment_ready(deployment)
        ready = deployment_ready

        # Validate the DGD snapshot before reading an authoritative adapter on
        # either narrow unready path (startup cancellation or batch bootstrap).
        if not deployment_ready:
            rejection = self._unready_recovery_dgd_rejection(deployment)
            if rejection is not None:
                return self._reject_not_ready_scaling(rejection)

        # Resolve every component and authoritative desired state before the
        # first mutation. Besides avoiding partial multi-component writes on a
        # bad target, this is required for the narrow unready-DGD recovery path.
        try:
            plans = self._preflight_replica_updates(deployment, target_replicas)
        except Exception as exc:
            if not deployment_ready:
                return self._reject_not_ready_scaling(
                    f"replica-target preflight failed: {type(exc).__name__}: {exc}"
                )
            raise

        mutations = [
            plan
            for plan in plans
            if plan.current_replicas != plan.target.desired_replicas
        ]

        startup_reduction = False
        reducing = any(
            plan.target.desired_replicas < plan.current_replicas for plan in plans
        )
        startup_pods: Optional[list] = None
        with self._startup_scale_down_lock:
            scale_down_pending = bool(self._startup_scale_down_targets)
        if not ready or scale_down_pending or reducing:
            startup_pods = self._list_startup_pods()
            if startup_pods is None:
                # Keep ordinary Ready-deployment scaling working with old RBAC.
                # Never release an accepted startup reversal without observing
                # its drain, or use the startup exception to scale an unready DGD.
                ready = (
                    ready
                    and not scale_down_pending
                    and self.kube_api.non_planner_components_stable(deployment)[0]
                )
            else:
                if self._startup_scale_down_in_progress(deployment, startup_pods):
                    logger.info("Startup scale-down still converging, ignoring scaling")
                    return
                pending = self.kube_api.pending_startup_replicas(
                    deployment, startup_pods
                )
                # Ready can still describe the state before a Pod deletion or the
                # latest spec change. Recheck lifecycle before issuing a reduction.
                phase = (deployment.get("status", {}).get("rollingUpdate") or {}).get(
                    "phase"
                )
                ready = (
                    ready
                    and self.kube_api.is_spec_generation_observed(deployment)
                    and not self.kube_api.has_terminating_pods(startup_pods)
                    and self.kube_api.pcsg_pods_within_desired_replicas(
                        deployment, startup_pods
                    )
                    and self.kube_api.non_planner_components_stable(deployment)[0]
                    and phase in (None, "", "Completed")
                    and not pending
                )
                startup_reduction = bool(pending) and bool(mutations)
                for plan in mutations:
                    if not startup_reduction:
                        break
                    serving, _ = self.kube_api.get_service_replica_status(
                        deployment, plan.service.name
                    )
                    startup_reduction &= (
                        0 <= plan.target.desired_replicas <= serving
                        and plan.target.desired_replicas < plan.current_replicas
                    )

        if startup_reduction:
            # Reconciliation needs Scale GET to recognize superseding writers.
            # Check every declared adapter before the first PATCH, so partial
            # RBAC cannot admit an asynchronous write we cannot track. Direct
            # DGD writes have no separate replica authority to read.
            try:
                for plan in mutations:
                    if plan.scaling_adapter is not None:
                        self.kube_api.get_service_replica_target(
                            self.graph_deployment_name, plan.service.name
                        )
            except ApiException as exc:
                if exc.status != 403:
                    raise
                startup_reduction = False
                with self._startup_scale_down_lock:
                    warn = "scale" not in self._startup_read_warnings
                    self._startup_read_warnings.add("scale")
                if warn:
                    logger.warning(
                        "Scale get forbidden for %s; startup cancellation is "
                        "disabled until Scale get is granted",
                        self.graph_deployment_name,
                    )
            else:
                with self._startup_scale_down_lock:
                    self._startup_read_warnings.discard("scale")

        if not ready and not startup_reduction:
            if deployment_ready:
                return self._reject_not_ready_scaling(
                    "deployment lifecycle is not settled"
                )
            rejection = self._unready_recovery_target_rejection(deployment, plans)
            if rejection is not None:
                return self._reject_not_ready_scaling(rejection)

        if not ready and not startup_reduction and mutations:
            if len(mutations) != 1:
                return self._reject_not_ready_scaling(
                    "unready recovery requires exactly one replica mutation"
                )
            rejection = self._unready_recovery_pod_rejection(
                deployment, mutations[0], startup_pods
            )
            if rejection is not None:
                return self._reject_not_ready_scaling(rejection)
            rejection = self._unready_recovery_settlement_rejection(
                deployment, mutations[0]
            )
            if rejection is not None:
                return self._reject_not_ready_scaling(rejection)
            try:
                plans, rejection = self._refresh_unready_recovery_plans(
                    deployment, target_replicas
                )
            except Exception as exc:
                return self._reject_not_ready_scaling(
                    f"replica-target revalidation failed: {type(exc).__name__}: {exc}"
                )
            if rejection is not None:
                return self._reject_not_ready_scaling(rejection)
            mutations = [
                plan
                for plan in plans
                if plan.current_replicas != plan.target.desired_replicas
            ]
            if len(mutations) > 1:
                return self._reject_not_ready_scaling(
                    "unready recovery requires exactly one replica mutation"
                )

        for plan in mutations:
            logger.info(
                "Updating %s component %s from %s to desired replica count %s",
                plan.target.sub_component_type.value,
                plan.service.name,
                plan.current_replicas,
                plan.target.desired_replicas,
            )
            self._write_preflighted_replica_target(
                plan.service,
                plan.target.desired_replicas,
                plan.scaling_adapter,
            )
            if startup_reduction:
                with self._startup_scale_down_lock:
                    self._startup_scale_down_targets = {
                        **self._startup_scale_down_targets,
                        plan.service.name: plan.target.desired_replicas,
                    }

        for plan in plans:
            if plan.current_replicas == plan.target.desired_replicas:
                logger.info(
                    "%s component %s already at desired replica count %s, skipping",
                    plan.target.sub_component_type.value,
                    plan.service.name,
                    plan.target.desired_replicas,
                )

        # An unready no-op is safe but cannot make the graph become Ready; keep
        # the historical nonblocking skip behavior instead of waiting forever.
        if not deployment_ready and not mutations:
            return

        if blocking:
            await self.kube_api.wait_for_graph_deployment_ready(
                self.graph_deployment_name,
            )

    async def acquire_batch_writer_fence(self, writer_id: str) -> None:
        """Fence native batch actuation against an overlapping old Planner.

        The POC supports one aggregate worker with an owned DGDSA. Its separate
        Planner deployment uses ``Recreate`` so a replacement cannot overlap an
        old process. This CAS annotation then orders any Scale PATCH already
        fetched by that old process: it either commits before the fence (and is
        visible to the refresh below) or conflicts afterward.
        """
        if not isinstance(writer_id, str) or not writer_id:
            raise ValueError("batch writer_id must be a non-empty string")
        self._batch_writer_id = None

        expected_dgd_uid: str | None = None
        component_name: str | None = None
        for attempt in range(3):
            deployment = self.kube_api.get_graph_deployment(self.graph_deployment_name)
            rejection = self._unready_recovery_dgd_rejection(deployment)
            if rejection is not None:
                raise RuntimeError(f"cannot acquire batch writer fence: {rejection}")
            dgd_uid = deployment["metadata"]["uid"]
            if expected_dgd_uid is None:
                expected_dgd_uid = dgd_uid
            elif dgd_uid != expected_dgd_uid:
                raise RuntimeError(
                    "cannot acquire batch writer fence: DGD identity changed"
                )

            workers = [
                Service(name=name, service=component)
                for name, component in get_components_by_name(deployment).items()
                if self._is_worker_component(name, component)
            ]
            if len(workers) != 1:
                raise RuntimeError(
                    "native batch writer fencing requires exactly one aggregate "
                    f"worker component; found {len(workers)}"
                )
            service = workers[0]
            component_name = service.name
            if "scalingAdapter" not in service.service:
                raise RuntimeError(
                    "native batch writer fencing requires a declared scalingAdapter"
                )
            adapter = self.kube_api.get_service_scaling_adapter(
                self.graph_deployment_name, component_name
            )
            adapter_rejection = self._scaling_adapter_rejection(
                deployment, component_name, adapter
            )
            if adapter_rejection is not None:
                raise RuntimeError(
                    f"cannot acquire batch writer fence: {adapter_rejection}"
                )

            try:
                self.kube_api.patch_scaling_adapter_writer_fence(
                    self.graph_deployment_name,
                    component_name,
                    writer_id,
                    adapter=adapter,
                )
            except ApiException as error:
                if error.status == 409 and attempt < 2:
                    continue
                raise
            break
        else:
            raise RuntimeError("failed to acquire batch writer fence after retries")

        refreshed_deployment = self.kube_api.get_graph_deployment(
            self.graph_deployment_name
        )
        if refreshed_deployment.get("metadata", {}).get("uid") != expected_dgd_uid:
            raise RuntimeError(
                "cannot acquire batch writer fence: DGD identity changed after CAS"
            )
        refreshed_adapter = self.kube_api.get_service_scaling_adapter(
            self.graph_deployment_name, component_name
        )
        adapter_rejection = self._scaling_adapter_rejection(
            refreshed_deployment, component_name, refreshed_adapter
        )
        if adapter_rejection is not None:
            raise RuntimeError(f"cannot verify batch writer fence: {adapter_rejection}")
        annotations = refreshed_adapter.get("metadata", {}).get("annotations") or {}
        if annotations.get(PLANNER_WRITER_FENCE_ANNOTATION) != writer_id:
            raise RuntimeError("batch writer fence annotation was not retained")

        # Refresh the authoritative target and rollout state after the CAS.
        # An in-progress result is allowed, but the next policy tick will see it
        # as unknown capacity and keep the already-published zero lease.
        await self.get_actual_worker_counts(
            prefill_component_name=None,
            decode_component_name=component_name,
        )
        self._batch_writer_id = writer_id

    def _preflight_replica_updates(
        self, deployment: dict, target_replicas: list[TargetReplica]
    ) -> list[_ReplicaUpdatePlan]:
        """Resolve all targets and authoritative desired counts without writes."""
        plans: list[_ReplicaUpdatePlan] = []
        seen_components: set[str] = set()
        for target in target_replicas:
            service = get_component_from_type_or_name(
                deployment,
                target.sub_component_type,
                component_name=target.component_name,
            )
            if service.name in seen_components:
                raise ValueError(
                    f"duplicate replica target for component {service.name!r}"
                )
            seen_components.add(service.name)
            current_replicas, scaling_adapter = self._authoritative_replica_state(
                deployment, service
            )
            plans.append(
                _ReplicaUpdatePlan(
                    target=target,
                    service=service,
                    current_replicas=current_replicas,
                    scaling_adapter=scaling_adapter,
                )
            )
        return plans

    def _authoritative_replica_state(
        self, deployment: dict, service: Service
    ) -> tuple[int, dict | None]:
        """Resolve desired replicas from the authority declared by the DGD."""
        if "scalingAdapter" not in service.service:
            return service.number_replicas(), None
        scaling_adapter = self.kube_api.get_service_scaling_adapter(
            self.graph_deployment_name, service.name
        )
        adapter_rejection = self._scaling_adapter_rejection(
            deployment, service.name, scaling_adapter
        )
        if adapter_rejection is not None:
            raise ValueError(adapter_rejection)
        writer_rejection = self._batch_writer_annotation_rejection(scaling_adapter)
        if writer_rejection is not None:
            raise ValueError(writer_rejection)
        current_replicas = self.kube_api.get_scaling_adapter_desired_replicas(
            scaling_adapter
        )
        return current_replicas, scaling_adapter

    def _batch_writer_annotation_rejection(self, adapter: dict) -> str | None:
        """Require this process's writer claim before a fenced Scale PATCH."""
        if self._batch_writer_id is None:
            return None
        metadata = adapter.get("metadata", {}) or {}
        annotations = metadata.get("annotations") or {}
        if not isinstance(annotations, Mapping):
            return "DGDSA annotations are malformed"
        if annotations.get(PLANNER_WRITER_FENCE_ANNOTATION) != self._batch_writer_id:
            return "DGDSA Planner writer fence is missing or owned by another writer"
        return None

    def _write_preflighted_replica_target(
        self, service: Service, replicas: int, scaling_adapter: dict | None
    ) -> None:
        """Write through the same replica authority selected during preflight."""
        if scaling_adapter is None:
            self.kube_api.update_dgd_replicas_directly(
                self.graph_deployment_name, service.name, replicas
            )
            return
        self.kube_api.update_scaling_adapter_replicas(
            self.graph_deployment_name,
            service.name,
            replicas,
            resource_version=scaling_adapter["metadata"]["resourceVersion"],
        )

    def _unready_recovery_dgd_rejection(self, deployment: dict) -> str | None:
        """Reject unsafe parent snapshots before reading any DGDSA target."""
        metadata = deployment.get("metadata", {}) or {}
        if not isinstance(metadata, Mapping):
            return "DGD metadata is malformed"
        if metadata.get("name") != self.graph_deployment_name:
            return "DGD metadata.name does not match the connector target"
        if metadata.get("deletionTimestamp") is not None:
            return "DGD is being deleted"
        dgd_uid = metadata.get("uid")
        if not isinstance(dgd_uid, str) or not dgd_uid:
            return "DGD metadata.uid is unavailable"
        if not self.kube_api.is_spec_generation_observed(deployment):
            return "DGD status has not observed the current spec generation"

        status = deployment.get("status", {}) or {}
        if not isinstance(status, Mapping):
            return "DGD status is malformed"
        state = status.get("state")
        if isinstance(state, str) and state.lower() == "failed":
            return "DGD status.state is failed"
        rolling = status.get("rollingUpdate") or {}
        if not isinstance(rolling, Mapping):
            return "DGD rolling-update status is malformed"
        phase = rolling.get("phase") or ""
        if phase not in ("", "Completed"):
            return f"DGD rolling update phase is {phase!r}"
        return None

    def _unready_recovery_target_rejection(
        self, deployment: dict, plans: list[_ReplicaUpdatePlan]
    ) -> str | None:
        """Allow only owned DGDSA no-ops or strict zero-to-positive recovery."""
        for plan in plans:
            if plan.scaling_adapter is None:
                return f"component {plan.service.name!r} has no declared DGDSA"

            desired = plan.target.desired_replicas
            current = plan.current_replicas
            if current == desired:
                continue
            if current != 0 or desired <= 0:
                return (
                    f"component {plan.service.name!r} is not a zero-to-positive "
                    f"recovery (current={current}, target={desired})"
                )
        return None

    def _refresh_unready_recovery_plans(
        self,
        initial_deployment: dict,
        target_replicas: list[TargetReplica],
    ) -> tuple[list[_ReplicaUpdatePlan], str | None]:
        """Fence an unready recovery against DGD replacement before writing."""
        current_deployment = self.kube_api.get_graph_deployment(
            self.graph_deployment_name
        )
        rejection = self._unready_recovery_dgd_rejection(current_deployment)
        if rejection is not None:
            return [], rejection

        initial_uid = initial_deployment.get("metadata", {}).get("uid")
        current_uid = current_deployment.get("metadata", {}).get("uid")
        if current_uid != initial_uid:
            return [], "DGD metadata.uid changed during unready recovery"

        plans = self._preflight_replica_updates(current_deployment, target_replicas)
        rejection = self._unready_recovery_target_rejection(current_deployment, plans)
        if rejection is not None:
            return plans, rejection
        mutations = [
            plan
            for plan in plans
            if plan.current_replicas != plan.target.desired_replicas
        ]
        if len(mutations) == 1:
            pods = self._list_startup_pods()
            rejection = self._unready_recovery_pod_rejection(
                current_deployment, mutations[0], pods
            )
            if rejection is not None:
                return plans, rejection
            return plans, self._unready_recovery_settlement_rejection(
                current_deployment, mutations[0]
            )
        return plans, None

    def _unready_recovery_pod_rejection(
        self,
        deployment: dict,
        mutation: _ReplicaUpdatePlan,
        pods: Optional[list],
    ) -> str | None:
        """Require a trustworthy zero-Pod lifecycle snapshot for bootstrap."""
        if pods is None:
            return "Pod snapshot is unavailable before unready recovery"
        if not self.kube_api.pcsg_pods_within_desired_replicas(deployment, pods):
            return "PodCliqueScalingGroup Pods exceed the desired replica range"

        target_pods = self.kube_api.partition_pods_by_component(pods).get(
            mutation.service.name, []
        )
        for pod in target_pods:
            metadata = getattr(pod, "metadata", None)
            status = getattr(pod, "status", None)
            phase = getattr(status, "phase", None)
            deletion_timestamp = getattr(metadata, "deletion_timestamp", None)
            if deletion_timestamp is not None or phase not in ("Succeeded", "Failed"):
                pod_name = getattr(metadata, "name", "<unknown>")
                return (
                    f"component {mutation.service.name!r} still has terminating or "
                    f"nonterminal Pod {pod_name!r}"
                )
        return None

    def _unready_recovery_settlement_rejection(
        self,
        deployment: dict,
        mutation: _ReplicaUpdatePlan,
    ) -> str | None:
        """Require a settled-zero target and stable peers before bootstrap."""
        status = deployment.get("status")
        if not isinstance(status, Mapping):
            return "DGD status is unavailable before unready recovery"
        component_statuses = status.get("components")
        if not isinstance(component_statuses, Mapping):
            return "DGD component status is unavailable before unready recovery"
        target_status = component_statuses.get(mutation.service.name)
        if not isinstance(target_status, Mapping):
            return (
                f"component {mutation.service.name!r} status is unavailable "
                "before unready recovery"
            )

        for field_name in ("replicas", "updatedReplicas"):
            value = target_status.get(field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                return (
                    f"component {mutation.service.name!r} status.{field_name} "
                    "is unavailable before unready recovery"
                )
            if value != 0:
                return (
                    f"component {mutation.service.name!r} is not settled at zero "
                    f"({field_name}={value})"
                )
        serving_fields = ("readyReplicas", "availableReplicas")
        present_serving_fields = [
            field_name for field_name in serving_fields if field_name in target_status
        ]
        if not present_serving_fields:
            return (
                f"component {mutation.service.name!r} serving status is "
                "unavailable before unready recovery"
            )
        for field_name in present_serving_fields:
            value = target_status[field_name]
            if isinstance(value, bool) or not isinstance(value, int):
                return (
                    f"component {mutation.service.name!r} "
                    f"status.{field_name} is invalid"
                )
            if value != 0:
                return (
                    f"component {mutation.service.name!r} is not settled at zero "
                    f"({field_name}={value})"
                )

        # DGD spec replicas are only a seed when a component declares a
        # scalingAdapter. Resolve every peer's authoritative DGDSA desired
        # value before judging settlement, otherwise stale DGD propagation can
        # make a genuinely stable peer look unstable (or an in-flight peer look
        # settled).
        settlement_deployment = copy.deepcopy(deployment)
        settlement_components = get_components_by_name(settlement_deployment)
        for component_name, component_spec in get_components_by_name(
            deployment
        ).items():
            if (
                component_name == mutation.service.name
                or get_component_type(component_spec) == "planner"
                or "scalingAdapter" not in component_spec
            ):
                continue
            peer_adapter = self.kube_api.get_service_scaling_adapter(
                self.graph_deployment_name, component_name
            )
            adapter_rejection = self._scaling_adapter_rejection(
                deployment, component_name, peer_adapter
            )
            if adapter_rejection is not None:
                return adapter_rejection
            settlement_components[component_name][
                "replicas"
            ] = self.kube_api.get_scaling_adapter_desired_replicas(peer_adapter)

        _, unstable_names = self.kube_api.non_planner_components_stable(
            settlement_deployment
        )
        unstable_peers = sorted(
            name for name in unstable_names if name != mutation.service.name
        )
        if unstable_peers:
            return "non-target components are unstable: " + ", ".join(unstable_peers)
        return None

    def _scaling_adapter_rejection(
        self, deployment: dict, component_name: str, adapter: dict
    ) -> str | None:
        """Delegate canonical DGDSA validation to the Kubernetes API layer."""
        return self.kube_api.scaling_adapter_identity_rejection(
            deployment,
            component_name,
            adapter,
            require_resource_version=True,
        )

    def _reject_not_ready_scaling(self, reason: str) -> None:
        """Preserve the connector's existing skip-versus-raise contract."""
        if self.raise_not_ready:
            logger.warning(
                "Deployment %s is not ready, rejecting this scaling: %s",
                self.graph_deployment_name,
                reason,
            )
            raise DynamoGraphDeploymentNotReadyError(
                deployment_name=self.graph_deployment_name,
                namespace=self.kube_api.current_namespace,
            )
        logger.warning(
            "Deployment %s is not ready, ignoring this scaling: %s",
            self.graph_deployment_name,
            reason,
        )


if __name__ == "__main__":
    import argparse

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
