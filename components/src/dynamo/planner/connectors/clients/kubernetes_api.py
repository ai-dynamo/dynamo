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
import logging
from collections.abc import Mapping
from typing import Optional

from kubernetes import client, config
from kubernetes.config.config_exception import ConfigException

from dynamo.planner.errors import DynamoGraphDeploymentNotFoundError, RolloutFailedError
from dynamo.planner.monitoring.dgd_services import (
    POWER_ANNOTATION_KEY,
    Service,
    get_component_type,
    get_components_by_name,
    resolve_power_component_names,
)
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)

NVIDIA_API_GROUP = "nvidia.com"
DYNAMO_API_VERSION = "v1beta1"
DYNAMO_WORKER_METADATA_API_VERSION = "v1alpha1"
DGD_PLURAL = "dynamographdeployments"
DGDSA_PLURAL = "dynamographdeploymentscalingadapters"
# During a rollout old Pods are still active and carry the previous cap, so
# progressing phases must block pod-annotation settlement.
# Failed is terminal (the operator sets endTime) and must NOT be treated as a
# retryable blocking phase — wait_for_graph_deployment_ready raises
# RolloutFailedError immediately when require_backing_settled=True (the power
# settlement path) so callers get an actionable error rather than timing out.
# The legacy replica-stability path (require_backing_settled=False) does not
# raise on Failed because it predates the rolling-update contract.
ROLLING_UPDATE_BLOCKING_PHASES = frozenset({"Pending", "InProgress"})
JSON_PATCH_CONTENT_TYPE = "application/json-patch+json"
# Stable labels the operator stamps on every worker Pod.
DYNAMO_DGD_NAME_LABEL = "nvidia.com/dynamo-graph-deployment-name"
DYNAMO_COMPONENT_LABEL = "nvidia.com/dynamo-component"


def get_current_k8s_namespace() -> str:
    """Get the current namespace if running inside a k8s cluster"""
    try:
        with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        # Fallback to 'default' if not running in k8s
        return "default"


class KubernetesAPI:
    def __init__(self, k8s_namespace: Optional[str] = None):
        # Load kubernetes configuration
        try:
            config.load_incluster_config()  # for in-cluster deployment
        except ConfigException:
            config.load_kube_config()  # for out-of-cluster deployment

        self.custom_api = client.CustomObjectsApi()
        self.core_api = client.CoreV1Api()
        self.current_namespace = k8s_namespace or get_current_k8s_namespace()

    def _get_graph_deployment_from_name(self, graph_deployment_name: str) -> dict:
        """Get the graph deployment from the dynamo graph deployment name"""
        return self.custom_api.get_namespaced_custom_object(
            group=NVIDIA_API_GROUP,
            version=DYNAMO_API_VERSION,
            namespace=self.current_namespace,
            plural=DGD_PLURAL,
            name=graph_deployment_name,
        )

    def list_graph_deployments(self) -> list[dict]:
        """List all DynamoGraphDeployments in the current namespace."""
        result = self.custom_api.list_namespaced_custom_object(
            group=NVIDIA_API_GROUP,
            version=DYNAMO_API_VERSION,
            namespace=self.current_namespace,
            plural=DGD_PLURAL,
        )
        return result.get("items", [])

    def get_graph_deployment(self, graph_deployment_name: str) -> dict:
        """
        Get the parent DynamoGraphDeployment

        Returns:
            The DynamoGraphDeployment object

        Raises:
            DynamoGraphDeploymentNotFoundError: If the parent graph deployment is not found
        """
        try:
            return self._get_graph_deployment_from_name(graph_deployment_name)
        except client.ApiException as e:
            if e.status == 404:
                raise DynamoGraphDeploymentNotFoundError(
                    deployment_name=graph_deployment_name,
                    namespace=self.current_namespace,
                )
            raise

    def update_service_replicas(
        self, graph_deployment_name: str, service_name: str, replicas: int
    ) -> None:
        """
        Update replicas for a component using Scale subresource when DGDSA exists.
        Falls back to a direct DGD patch when the component does not have a DGDSA.

        Args:
            graph_deployment_name: Name of the DynamoGraphDeployment
            service_name: Name of the component in DGD.spec.components
            replicas: Desired number of replicas
        """
        # DGDSA naming convention: <dgd-name>-<lowercase-service-name>
        adapter_name = f"{graph_deployment_name}-{service_name.lower()}"

        try:
            # Try to scale via DGDSA Scale subresource
            self.custom_api.patch_namespaced_custom_object_scale(
                group=NVIDIA_API_GROUP,
                version=DYNAMO_API_VERSION,
                namespace=self.current_namespace,
                plural=DGDSA_PLURAL,
                name=adapter_name,
                body={"spec": {"replicas": replicas}},
            )
            logger.info(f"Scaled DGDSA {adapter_name} to {replicas} replicas")

        except client.ApiException as e:
            if e.status == 404:
                # DGDSA doesn't exist - fall back to a direct DGD patch.
                logger.info(
                    f"DGDSA {adapter_name} not found, falling back to DGD update"
                )
                self._update_dgd_replicas(graph_deployment_name, service_name, replicas)
            else:
                raise

    def _update_dgd_replicas(
        self, graph_deployment_name: str, service_name: str, replicas: int
    ) -> None:
        """Update replicas directly in DGD when no DGDSA is available."""
        deployment = self.get_graph_deployment(graph_deployment_name)
        components = self._dgd_components(deployment, graph_deployment_name)
        self._patch_component_replicas(
            graph_deployment_name, components, service_name, replicas
        )
        logger.info(
            f"Updated DGD {graph_deployment_name} component {service_name} to {replicas} replicas"
        )

    @staticmethod
    def _dgd_components(deployment: dict, graph_deployment_name: str) -> list[dict]:
        components = deployment.get("spec", {}).get("components")
        if components is None:
            raise KeyError(
                f"DGD {graph_deployment_name!r} has no v1beta1 spec.components"
            )
        if not isinstance(components, list):
            raise TypeError(
                f"DGD {graph_deployment_name!r} spec.components must be a list"
            )
        return components

    def _patch_component_replicas(
        self,
        graph_deployment_name: str,
        components: list[dict],
        component_name: str,
        replicas: int,
    ) -> None:
        index = self._find_component_index(
            graph_deployment_name, components, component_name
        )
        patch = self._component_replicas_json_patch(index, component_name, replicas)
        self._patch_dgd_with_json_patch(graph_deployment_name, patch)

    @staticmethod
    def _find_component_index(
        graph_deployment_name: str, components: list[dict], component_name: str
    ) -> int:
        for index, component in enumerate(components):
            if component.get("name") == component_name:
                return index
        raise KeyError(
            f"component {component_name!r} not found in DGD {graph_deployment_name!r}"
        )

    @staticmethod
    def _component_replicas_json_patch(
        index: int, component_name: str, replicas: int
    ) -> list[dict]:
        return [
            {
                "op": "test",
                "path": f"/spec/components/{index}/name",
                "value": component_name,
            },
            {
                "op": "add",
                "path": f"/spec/components/{index}/replicas",
                "value": replicas,
            },
        ]

    def _patch_dgd_with_json_patch(
        self, graph_deployment_name: str, patch: list[dict]
    ) -> None:
        """Patch a v1beta1 DGD with RFC 6902 JSON Patch operations."""
        self.custom_api.api_client.call_api(
            "/apis/{group}/{version}/namespaces/{namespace}/{plural}/{name}",
            "PATCH",
            {
                "group": NVIDIA_API_GROUP,
                "version": DYNAMO_API_VERSION,
                "namespace": self.current_namespace,
                "plural": DGD_PLURAL,
                "name": graph_deployment_name,
            },
            [],
            {
                "Accept": "application/json",
                "Content-Type": JSON_PATCH_CONTENT_TYPE,
            },
            body=patch,
            response_type="object",
            auth_settings=["BearerToken"],
            _return_http_data_only=True,
            collection_formats={},
        )

    def update_graph_replicas(
        self, graph_deployment_name: str, component_name: str, replicas: int
    ) -> None:
        """
        Update replicas for a component. Now uses DGDSA when available.

        Deprecated: Use update_service_replicas() instead for clarity.
        This method is kept for backward compatibility.
        """
        self.update_service_replicas(graph_deployment_name, component_name, replicas)

    def is_deployment_ready(self, deployment: dict) -> bool:
        """Check if a graph deployment is ready"""

        conditions = deployment.get("status", {}).get("conditions", [])
        ready_condition = next(
            (c for c in conditions if c.get("type") == "Ready"), None
        )

        return ready_condition is not None and ready_condition.get("status") == "True"

    @staticmethod
    def is_spec_generation_observed(deployment: dict) -> bool:
        """True when status has caught up to ``metadata.generation``.

        Annotation-only DGD edits bump generation without changing desired
        replica counts. Replica-count stability alone can look "ready" while
        ``observedGeneration`` still describes the previous generation, so
        callers that cache startup-static fields (power caps) must require
        this catch-up on the same snapshot they read.
        """
        generation = deployment.get("metadata", {}).get("generation")
        if generation is None:
            return False
        observed = deployment.get("status", {}).get("observedGeneration")
        if observed is None:
            return False
        try:
            return int(observed) >= int(generation)
        except (TypeError, ValueError):
            return False

    def get_service_replica_status(
        self, deployment: dict, service_name: str
    ) -> tuple[int, bool]:
        """
        Get the actual ready replica count for a component from DGD status.

        Returns:
            tuple[int, bool]: (replica_count, is_stable)
            - replica_count: number of replicas serving traffic (availableReplicas if present, else readyReplicas)
            - is_stable: no rollout is in progress (desired == updated == ready/available)
        """
        # Get desired replicas from spec
        service_spec = get_components_by_name(deployment).get(service_name, {})
        desired_replicas = Service(
            name=service_name, service=service_spec
        ).number_replicas()

        # Get status fields
        status = deployment.get("status", {})
        service_status = status.get("components", {}).get(service_name, {})
        available = service_status.get("availableReplicas")
        ready = service_status.get("readyReplicas", 0)
        updated = service_status.get("updatedReplicas", 0)

        # availableReplicas takes precedence over readyReplicas for the count
        # refer to ComponentReplicaStatus in deploy/operator/api/v1beta1/common.go
        if available is not None:
            traffic_serving_replicas = available
        else:
            traffic_serving_replicas = ready

        # Stable means: desired == updated == ready/available
        # This ensures we're not in a scale-up, scale-down, or rollout
        is_stable = desired_replicas == updated == traffic_serving_replicas

        return traffic_serving_replicas, is_stable

    def non_planner_components_stable(self, deployment: dict) -> tuple[bool, list[str]]:
        """Return ``(all_stable, unstable_names)`` for non-planner components."""
        components = get_components_by_name(deployment)
        not_ready: list[str] = []
        for component_name, component_spec in components.items():
            if get_component_type(component_spec) == "planner":
                continue
            _, is_stable = self.get_service_replica_status(deployment, component_name)
            if not is_stable:
                not_ready.append(component_name)
        return not not_ready, not_ready

    @staticmethod
    def is_rolling_update_blocking_settlement(deployment: dict) -> tuple[bool, str]:
        """True while an operator-managed worker rollout is not yet cut over."""
        rolling = deployment.get("status", {}).get("rollingUpdate") or {}
        phase = rolling.get("phase") or ""
        if phase in ROLLING_UPDATE_BLOCKING_PHASES:
            return True, f"rollingUpdate.phase={phase}"
        return False, ""

    @staticmethod
    def has_terminating_pods(pods: list) -> bool:
        """True if any non-terminal pod in ``pods`` has a deletionTimestamp."""
        return any(
            p.metadata.deletion_timestamp is not None
            and (p.status is None or p.status.phase not in ("Succeeded", "Failed"))
            for p in pods
        )

    def list_pods_for_graph(self, dgd_name: str) -> list:
        """List all Pods for a DGD using its stable graph label."""
        return (
            self.core_api.list_namespaced_pod(
                namespace=self.current_namespace,
                label_selector=f"{DYNAMO_DGD_NAME_LABEL}={dgd_name}",
            ).items
            or []
        )

    @staticmethod
    def partition_pods_by_component(pods: list) -> dict[str, list]:
        """Partition Pods by the stable Dynamo component label."""
        by_component: dict[str, list] = {}
        for pod in pods:
            component_name = (pod.metadata.labels or {}).get(DYNAMO_COMPONENT_LABEL)
            if component_name:
                by_component.setdefault(component_name, []).append(pod)
        return by_component

    def worker_pods_settled(
        self,
        deployment: dict,
        expected_power_by_component: Mapping[str, str],
    ) -> tuple[bool, list[str]]:
        """True when all non-terminal pods carry the expected per-GPU annotation.

        ``expected_power_by_component`` maps each power-relevant component name
        to the raw ``dynamo.nvidia.com/gpu-power-limit`` string from the DGD
        snapshot (verbatim, for exact propagation verification).

        Terminal means phase Succeeded or Failed. Terminating pods
        (DeletionTimestamp set) in Running/Pending/Unknown are non-terminal:
        they still consume GPU power and block annotation convergence.

        A component with zero desired replicas and no pods is settled; there is
        nothing currently enforcing a stale cap, and future pods will be created
        from the current DGD intent.
        """
        blocking, reason = self.is_rolling_update_blocking_settlement(deployment)
        if blocking:
            return False, [reason]

        dgd_name = deployment.get("metadata", {}).get("name", "")
        components_by_name = get_components_by_name(deployment)
        pending: list[str] = []
        pods_by_component = self.partition_pods_by_component(
            self.list_pods_for_graph(dgd_name)
        )

        for component_name, expected_raw in expected_power_by_component.items():
            all_pods = pods_by_component.get(component_name, [])
            non_terminal = [
                p
                for p in all_pods
                if p.status is None or p.status.phase not in ("Succeeded", "Failed")
            ]
            if not non_terminal:
                desired = Service(
                    name=component_name,
                    service=components_by_name.get(component_name, {}),
                ).number_replicas()
                if desired > 0:
                    pending.append(
                        f"{component_name}: no non-terminal pods (desired={desired})"
                    )
                continue

            for pod in non_terminal:
                # Terminating pods still consume GPU power and will soon be
                # replaced. Even when they carry the expected annotation they
                # must block settlement so that the planner never counts them
                # as "done" and admits new pods that could push total power
                # past the ceiling before the old ones fully disappear.
                if pod.metadata.deletion_timestamp is not None:
                    phase = (pod.status.phase if pod.status else None) or "?"
                    pending.append(
                        f"{component_name}/{pod.metadata.name}"
                        f" (phase={phase}, terminating): waiting for pod to disappear"
                    )
                    continue
                actual = (pod.metadata.annotations or {}).get(POWER_ANNOTATION_KEY)
                if actual != expected_raw:
                    phase = (pod.status.phase if pod.status else None) or "?"
                    pending.append(
                        f"{component_name}/{pod.metadata.name}"
                        f" (phase={phase}):"
                        f" annotation {actual!r} != {expected_raw!r}"
                    )

        return not pending, pending

    async def wait_for_graph_deployment_ready(
        self,
        graph_deployment_name: str,
        include_planner: bool = True,
        max_attempts: int = 180,  # default: 30 minutes total
        delay_seconds: int = 10,  # default: check every 10 seconds
        *,
        require_backing_settled: bool = False,
        require_prefill: bool = True,
        require_decode: bool = True,
        prefill_component_name: Optional[str] = None,
        decode_component_name: Optional[str] = None,
    ) -> dict:
        """Wait for a graph deployment to be ready; return the ready snapshot.

        Args:
            graph_deployment_name: Name of the DGD to wait for.
            include_planner: If False, skip components with type "planner"
                and check per-component readiness instead of the global DGD Ready
                condition. This avoids a circular wait when the planner itself
                is one of the services in the DGD.
            max_attempts: Maximum polling iterations.
            delay_seconds: Seconds between polls.
            require_backing_settled: Power-only gate. When True, raises
                immediately on ``status.rollingUpdate.phase == "Failed"``
                regardless of ``include_planner`` (the check runs before the
                ``include_planner`` branch so ``include_planner=True`` callers
                are not exempt; after fix-4 no production caller combines
                ``require_backing_settled=True`` with ``include_planner=True``).
                When True and ``include_planner`` is False, additionally
                requires ``status.observedGeneration >= metadata.generation``
                and every non-terminal worker Pod carrying the expected per-GPU
                annotation from the current DGD snapshot, and blocks while
                ``status.rollingUpdate.phase`` is Pending or InProgress.
                Must stay False for the legacy ``wait_for_deployment_ready``
                path so power-disabled planners keep working with older
                operators or custom SAs that lack pods/list permission.
            require_prefill / require_decode / ``*_component_name``:
                Forwarded to :func:`resolve_power_component_names` when
                ``require_backing_settled`` is True so settlement covers the
                same workers the power resolver will read (including untyped
                named workers).

        Returns:
            The DGD dict that satisfied the readiness criteria (same object
            callers should use for startup-static reads such as power caps).
        """
        for attempt in range(max_attempts):
            await asyncio.sleep(delay_seconds)

            graph_deployment = self.get_graph_deployment(graph_deployment_name)

            # Failed is a terminal rollout state; retrying until timeout (up to
            # 30 min) serves no purpose. Raise immediately so the operator can
            # investigate. Checked before include_planner so this fires on both
            # code paths when require_backing_settled is True.
            if require_backing_settled:
                _rolling = graph_deployment.get("status", {}).get("rollingUpdate") or {}
                if _rolling.get("phase") == "Failed":
                    raise RolloutFailedError(
                        deployment_name=graph_deployment_name,
                        reason=_rolling.get("message", ""),
                    )

            if include_planner:
                conditions = graph_deployment.get("status", {}).get("conditions", [])
                ready_condition = next(
                    (c for c in conditions if c.get("type") == "Ready"), None
                )
                if ready_condition and ready_condition.get("status") == "True":
                    return graph_deployment

                logger.info(
                    f"[Attempt {attempt + 1}/{max_attempts}] "
                    f"(status: {ready_condition.get('status') if ready_condition else 'N/A'}, "
                    f"message: {ready_condition.get('message') if ready_condition else 'no condition found'})"
                )
                continue

            # Legacy exclude-planner path: replica-count stability only.
            all_stable, not_ready = self.non_planner_components_stable(graph_deployment)
            if not all_stable:
                logger.info(
                    f"[Attempt {attempt + 1}/{max_attempts}] "
                    f"Waiting for components (excluding planner): "
                    f"not ready: {not_ready}"
                )
                continue

            if not require_backing_settled:
                return graph_deployment

            # Power settlement: generation catch-up + resolved workers' backing.
            # The legacy path above does not gate on rollingUpdate because it
            # predates the rolling-update contract and may run without pods/list
            # RBAC. Failed was caught before the include_planner branch above.
            if not self.is_spec_generation_observed(graph_deployment):
                generation = graph_deployment.get("metadata", {}).get("generation")
                observed = graph_deployment.get("status", {}).get("observedGeneration")
                logger.info(
                    "[Attempt %d/%d] Waiting for DGD generation to be observed: "
                    "generation=%s, observedGeneration=%s",
                    attempt + 1,
                    max_attempts,
                    generation,
                    observed,
                )
                continue

            # Role resolution runs after is_spec_generation_observed has
            # confirmed the DGD spec is stable. Missing or duplicate roles at
            # this point are configuration errors, not operator rollout lag.
            power_names = resolve_power_component_names(
                graph_deployment,
                require_prefill=require_prefill,
                require_decode=require_decode,
                prefill_name=prefill_component_name,
                decode_name=decode_component_name,
            )

            # Build per-component expected annotation strings from the same
            # DGD snapshot. A missing or malformed annotation is invalid
            # configuration — raise immediately rather than retrying.
            # Use the raw DGD annotation string: the operator copies it verbatim
            # onto Pod annotations, so the settlement comparison must use the
            # same raw value to get an exact match.
            components_map = get_components_by_name(graph_deployment)
            expected_power: dict[str, str] = {}
            for name in power_names:
                svc = Service(name=name, service=components_map.get(name, {}))
                # Raises PowerAnnotationMissingError / PowerAnnotationInvalidError
                # on bad config; let those propagate as a fail-fast startup error.
                expected_power[name] = svc.get_gpu_power_limit_annotation()

            pods_ok, pods_pending = self.worker_pods_settled(
                graph_deployment, expected_power
            )
            if not pods_ok:
                logger.info(
                    "[Attempt %d/%d] Waiting for pods to reflect current power"
                    " annotation: %s",
                    attempt + 1,
                    max_attempts,
                    pods_pending,
                )
                continue

            return graph_deployment

        raise TimeoutError(
            f"Graph deployment '{graph_deployment_name}' "
            f"is not ready after {max_attempts * delay_seconds} seconds"
        )
