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

import json
import logging
import os
from typing import Optional

from dynamo.planner.config.defaults import SubComponentType, TargetReplica
from dynamo.planner.connectors.base import PlannerConnector
from dynamo.planner.connectors.kubernetes_api import (
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
    get_component_from_type_or_name,
    get_component_type,
    get_components_by_name,
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
        self.kube_api.update_graph_replicas(
            self.graph_deployment_name,
            service.name,
            service.number_replicas() + 1,
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
            self.kube_api.update_graph_replicas(
                self.graph_deployment_name,
                service.name,
                service.number_replicas() - 1,
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
            self.get_model_name(
                deployment,
                require_prefill=require_prefill,
                require_decode=require_decode,
                prefill_component_name=prefill_component_name,
                decode_component_name=decode_component_name,
            )
        except PlannerError as e:
            errors.append(str(e))

        # Raise combined error if any issues found
        if errors:
            raise DeploymentValidationError(errors)

    def get_model_name(
        self,
        deployment: Optional[dict] = None,
        require_prefill: bool = True,
        require_decode: bool = True,
        prefill_component_name: Optional[str] = None,
        decode_component_name: Optional[str] = None,
    ) -> str:
        """Get the model name from the deployment"""
        try:
            if deployment is None:
                deployment = self.kube_api.get_graph_deployment(
                    self.graph_deployment_name
                )

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

    def get_gpu_counts(
        self,
        deployment: Optional[dict] = None,
        require_prefill: bool = True,
        require_decode: bool = True,
        prefill_component_name: Optional[str] = None,
        decode_component_name: Optional[str] = None,
    ) -> tuple[int, int]:
        """Get the GPU counts for prefill and decode components from the deployment.

        Args:
            deployment: Optional deployment dict, fetched if not provided
            require_prefill: Whether to require a prefill component
            require_decode: Whether to require a decode component
            prefill_component_name: Explicit DGD component name for prefill lookup
            decode_component_name: Explicit DGD component name for decode lookup
                (required for agg mode where the worker is not typed as decode)

        Returns:
            Tuple of (prefill_gpu_count, decode_gpu_count)

        Raises:
            DeploymentValidationError: If GPU counts cannot be determined from DGD
        """
        if deployment is None:
            deployment = self.kube_api.get_graph_deployment(self.graph_deployment_name)

        prefill_gpu_count = 0
        decode_gpu_count = 0
        errors = []

        if require_prefill:
            try:
                prefill_service = get_component_from_type_or_name(
                    deployment,
                    SubComponentType.PREFILL,
                    component_name=prefill_component_name,
                )
                prefill_gpu_count = prefill_service.get_total_gpu_count()
            except (PlannerError, ValueError) as e:
                errors.append(f"Failed to get prefill GPU count: {e}")

        if require_decode:
            try:
                decode_service = get_component_from_type_or_name(
                    deployment,
                    SubComponentType.DECODE,
                    component_name=decode_component_name,
                )
                decode_gpu_count = decode_service.get_total_gpu_count()
            except (PlannerError, ValueError) as e:
                errors.append(f"Failed to get decode GPU count: {e}")

        if errors:
            raise DeploymentValidationError(errors)

        return prefill_gpu_count, decode_gpu_count

    def get_component_pods(
        self,
        sub_component_type: SubComponentType,
        deployment: Optional[dict] = None,
        component_name: Optional[str] = None,
    ) -> list:
        """Return the list of Pod objects for the given sub-component (prefill/decode).

        Uses the operator's canonical pod labels (see
        deploy/operator/internal/consts/consts.go):
          - nvidia.com/dynamo-graph-deployment-name=<dgd>
          - nvidia.com/dynamo-component=<service-key> (e.g. VllmPrefillWorker)
        Returns an empty list when the service cannot be resolved (e.g. the DGD
        has no matching component — handled gracefully by the caller).

        ``deployment`` lets a caller that already fetched the DGD this tick
        (e.g. a power-annotation sweep resolving both prefill and decode) reuse
        that snapshot, avoiding a redundant apiserver GET per role. When None,
        the current DGD is fetched.

        ``component_name`` is the fallback DGD component name used when the
        role cannot be matched by ``type`` alone. Aggregated (``mode=agg``)
        deployments label their single worker ``type: worker`` (the generic
        v1beta1 worker type), which does not map to the DECODE role; when that
        generic worker is unique, the resolver carries the actual DGD component
        name rather than depending on a backend hard-coded agg name. Ignored
        when the role matches by type (disagg).
        """
        if deployment is None:
            deployment = self.kube_api.get_graph_deployment(self.graph_deployment_name)
        try:
            service = get_component_from_type_or_name(
                deployment, sub_component_type, component_name=component_name
            )
        except (PlannerError, ValueError) as e:
            # A role legitimately absent from the DGD and a genuine resolution
            # failure (invalid/renamed component, schema drift) both land here.
            # Returning [] keeps the sweep non-fatal, but log so an unexpected
            # "0 pods annotated" is diagnosable instead of silent.
            logger.warning(
                "Could not resolve %s component for DGD %s; annotating no pods "
                "for this role this sweep: %s",
                sub_component_type,
                self.graph_deployment_name,
                e,
            )
            return []
        label_selector = (
            f"nvidia.com/dynamo-graph-deployment-name={self.graph_deployment_name},"
            f"nvidia.com/dynamo-component={service.name}"
        )
        return self.kube_api.list_pods_by_label(label_selector)

    def list_frontend_pods(self) -> list:
        """Return all frontend pods for this DGD.

        Frontends are identified by nvidia.com/dynamo-component-type=frontend
        (semantic role, robust to service-key naming variation across DGDs).
        Used by the planner's admission-control fanout.
        Returns an empty list when no frontend service is running.
        Raises ApiException on transient apiserver errors (callers must handle).
        """
        label_selector = (
            f"nvidia.com/dynamo-graph-deployment-name={self.graph_deployment_name},"
            f"nvidia.com/dynamo-component-type=frontend"
        )
        return self.kube_api.list_pods_by_label(label_selector)

    @staticmethod
    def resolve_frontend_http_port(pod, fallback: int) -> int:
        """Resolve the frontend HTTP port from the live pod spec.

        The operator emits a named port ``http`` on the frontend container
        (see ``deploy/operator/internal/dynamo/component_frontend.go``; the
        constant ``DynamoContainerPortName = "http"`` lives in
        ``deploy/operator/internal/consts/consts.go``).  Reading it from
        ``V1Pod`` is the deterministic path: it follows DGD-spec port
        overrides automatically and doesn't require the planner to keep a
        config mirror of the frontend port in sync with the DGD spec.

        Falls back to ``fallback`` (typically the deprecated
        ``frontend_http_port`` config field) when the named port is absent —
        covers legacy manifests authored before the operator started
        emitting the named port, hand-rolled DGDs that omit ``ports``, and
        unit-test fixtures that build minimal ``V1Pod`` mocks.

        Args:
            pod: A ``V1Pod`` (or a duck-typed mock).  The first container
                with a ``ports[].name == "http"`` entry wins.
            fallback: Port to return when no named port is found on any
                container.  Must be a positive ``int``.

        Returns:
            The container port whose name is ``"http"``, or ``fallback``.
        """
        try:
            containers = (pod.spec.containers if pod.spec else None) or []
            for container in containers:
                for port in container.ports or []:
                    if getattr(port, "name", None) == "http":
                        return int(port.container_port)
        except (AttributeError, TypeError, ValueError):
            # Defensive: malformed pod spec shouldn't take down the
            # admission-control loop; we always have a fallback.
            pass
        return fallback

    async def post_busy_threshold(
        self,
        pod,
        model: str,
        port: int,
        active_decode_blocks_threshold: Optional[float],
        active_prefill_tokens_threshold: Optional[int],
        active_prefill_tokens_threshold_frac: Optional[float],
    ) -> None:
        """POST /busy_threshold to a single frontend pod (admission control).

        Raises on HTTP error or network failure — callers should gather with
        ``return_exceptions=True`` and inspect the results.
        Per-pod timeout is 5 s (frontend is in-cluster; long timeouts mask
        saturation events).
        """
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError(
                "httpx is required for admission-control POST support. "
                "Install it with: pip install httpx"
            ) from exc

        if not pod.status or not pod.status.pod_ip:
            raise ValueError(
                f"Pod {pod.metadata.name} has no pod IP; cannot POST /busy_threshold"
            )
        pod_ip = pod.status.pod_ip

        # IPv6 literals must be bracketed in a URL authority (RFC 3986);
        # an IPv6 pod IP always contains ':', IPv4 and hostnames never do.
        host = f"[{pod_ip}]" if ":" in pod_ip else pod_ip
        url = f"http://{host}:{port}/busy_threshold"
        body: dict = {"model": model}
        if active_decode_blocks_threshold is not None:
            body["active_decode_blocks_threshold"] = active_decode_blocks_threshold
        if active_prefill_tokens_threshold is not None:
            body["active_prefill_tokens_threshold"] = active_prefill_tokens_threshold
        if active_prefill_tokens_threshold_frac is not None:
            body[
                "active_prefill_tokens_threshold_frac"
            ] = active_prefill_tokens_threshold_frac

        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.post(url, json=body)
            r.raise_for_status()

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
        """Wait for the deployment to be ready.

        Args:
            include_planner: If False, skip the planner component when checking
                readiness. This lets the planner read MDC from worker pods
                without waiting for itself to be marked ready in the DGD.
        """
        await self.kube_api.wait_for_graph_deployment_ready(
            self.graph_deployment_name,
            include_planner=include_planner,
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

    def _extract_mdc_entries(self) -> list[MdcEntry]:
        """Extract MDC entries belonging to this DGD.

        CRs are named after the worker pod (e.g. ``<dgd>-0-<service>-<hash>``),
        so we filter by the DGD name prefix to avoid picking up entries from
        other deployments sharing the namespace.  LoRA-adapter wrappers are
        dropped via :func:`is_model_card`.
        """
        crs = self._list_worker_metadata_crs()
        dgd_prefix = f"{self.graph_deployment_name}-"

        entries: list[MdcEntry] = []
        for cr in crs:
            cr_name = cr.get("metadata", {}).get("name", "")
            if not cr_name.startswith(dgd_prefix):
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
        self,
        sub_component_type: SubComponentType,
        backend: str,
        component_name_override: Optional[str] = None,
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

        ``component_name_override`` is the DGD component name to use as the
        explicit-name fallback when type-only resolution fails (e.g. agg mode
        where the single worker is typed as ``worker``, not ``decode``).

        Note: the DGD component name (``service.name``) must NOT be used for
        filtering -- it is typically PascalCase (``"VllmPrefillWorker"``) and
        would never match the lowercase value the worker writes to MDC.
        """
        defaults = build_worker_info_from_defaults(backend, sub_component_type)
        expected_component = defaults.component_name or ""
        try:
            deployment = self.kube_api.get_graph_deployment(self.graph_deployment_name)
            service = get_component_from_type_or_name(
                deployment, sub_component_type, component_name=component_name_override
            )
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
        component_name: Optional[str] = None,
    ) -> WorkerInfo:
        """Get WorkerInfo for a sub-component, trying MDC first, then fallbacks.

        Args:
            sub_component_type: PREFILL or DECODE
            backend: Backend framework name (for default fallback)
            component_name: Explicit DGD component name to use as fallback when
                type-only resolution fails (e.g. agg mode — see
                ``_resolve_dgd_service``).
        """
        entries = self._extract_mdc_entries()
        dgd_service_name, expected_component = self._resolve_dgd_service(
            sub_component_type, backend, component_name_override=component_name
        )

        def _dgd_model_name() -> Optional[str]:
            try:
                deployment = self.kube_api.get_graph_deployment(
                    self.graph_deployment_name
                )
                service = get_component_from_type_or_name(
                    deployment, sub_component_type, component_name=component_name
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
                k8s_name_override=dgd_service_name or component_name,
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
        info = build_worker_info_from_defaults(
            backend,
            sub_component_type,
            k8s_name_override=dgd_service_name or component_name,
        )
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

    def get_actual_worker_counts(
        self,
        prefill_component_name: Optional[str] = None,
        decode_component_name: Optional[str] = None,
    ) -> tuple[int, int, bool]:
        """
        Get actual ready worker counts for prefill and decode from DGD status.

        Returns:
            tuple[int, int, bool]: (prefill_count, decode_count, is_stable)
            - is_stable: False if any component is in a rollout (scaling should be skipped)
        """
        deployment = self.kube_api.get_graph_deployment(self.graph_deployment_name)

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
            if not is_stable:
                all_stable = False
            decode_count = ready_replicas

        return prefill_count, decode_count, all_stable

    async def set_component_replicas(
        self, target_replicas: list[TargetReplica], blocking: bool = True
    ):
        """Set the replicas for multiple components at once"""
        if not target_replicas:
            raise EmptyTargetReplicasError()

        deployment = self.kube_api.get_graph_deployment(self.graph_deployment_name)

        if not self.kube_api.is_deployment_ready(deployment):
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
            if current_replicas != target_replica.desired_replicas:
                logger.info(
                    f"Updating {target_replica.sub_component_type.value} component {service.name} to desired replica count {target_replica.desired_replicas}"
                )
                self.kube_api.update_graph_replicas(
                    self.graph_deployment_name,
                    service.name,
                    target_replica.desired_replicas,
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
