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
from typing import Optional

from kubernetes import client, config
from kubernetes.client import ApiException
from kubernetes.config.config_exception import ConfigException

from dynamo.planner.errors import DynamoGraphDeploymentNotFoundError
from dynamo.planner.monitoring.dgd_services import (
    Service,
    get_component_type,
    get_components_by_name,
)
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)

NVIDIA_API_GROUP = "nvidia.com"
DYNAMO_API_VERSION = "v1beta1"
DYNAMO_WORKER_METADATA_API_VERSION = "v1alpha1"
DGD_PLURAL = "dynamographdeployments"
DGDSA_PLURAL = "dynamographdeploymentscalingadapters"
DCD_PLURAL = "dynamocomponentdeployments"
GROVE_API_GROUP = "grove.io"
GROVE_API_VERSION = "v1alpha1"
POD_CLIQUE_PLURAL = "podcliques"
PCSG_PLURAL = "podcliquescalinggroups"
WORKER_HASH_V2_ANNOTATION = "nvidia.com/current-worker-hash-v2"
WORKER_HASH_V1_ANNOTATION = "nvidia.com/current-worker-hash"
WORKER_COMPONENT_TYPES = frozenset({"prefill", "decode", "worker"})
DCD_AVAILABLE_CONDITION = "Available"
# current-worker-hash annotations stay on the old revision until cutover, so
# an InProgress/Pending/Failed rollout must block settlement even when the old
# DCD still looks generation-ready.
ROLLING_UPDATE_BLOCKING_PHASES = frozenset({"Pending", "InProgress", "Failed"})
JSON_PATCH_CONTENT_TYPE = "application/json-patch+json"


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
    def _worker_hash_candidates(deployment: dict) -> list[str]:
        """Active worker-hash suffixes to probe, mirroring operator precedence."""
        annotations = deployment.get("metadata", {}).get("annotations") or {}
        candidates: list[str] = []
        for key in (WORKER_HASH_V2_ANNOTATION, WORKER_HASH_V1_ANNOTATION):
            value = annotations.get(key)
            if value:
                candidates.append(value)
        candidates.append("")
        seen: set[str] = set()
        ordered: list[str] = []
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                ordered.append(candidate)
        return ordered

    @staticmethod
    def _dcd_resource_name(
        dgd_name: str,
        component_name: str,
        worker_suffix: str,
        component_spec: dict,
    ) -> str:
        base = f"{dgd_name}-{component_name.lower()}"
        if (
            worker_suffix
            and get_component_type(component_spec) in WORKER_COMPONENT_TYPES
        ):
            return f"{base}-{worker_suffix}"
        return base

    def _get_namespaced_custom_object(
        self, *, group: str, version: str, plural: str, name: str
    ) -> dict:
        return self.custom_api.get_namespaced_custom_object(
            group=group,
            version=version,
            namespace=self.current_namespace,
            plural=plural,
            name=name,
        )

    @staticmethod
    def is_dcd_ready(dcd: dict) -> bool:
        """Mirror ``checkDCDReady``: observed generation caught up + Available."""
        generation = dcd.get("metadata", {}).get("generation")
        observed = dcd.get("status", {}).get("observedGeneration", 0)
        if generation is None:
            return False
        try:
            if int(observed) < int(generation):
                return False
        except (TypeError, ValueError):
            return False
        for condition in dcd.get("status", {}).get("conditions") or []:
            if (
                condition.get("type") == DCD_AVAILABLE_CONDITION
                and condition.get("status") == "True"
            ):
                return True
        return False

    @staticmethod
    def is_pod_clique_ready(pod_clique: dict) -> bool:
        """Mirror ``CheckPodCliqueReady`` generation + replica convergence."""
        generation = pod_clique.get("metadata", {}).get("generation")
        observed = pod_clique.get("status", {}).get("observedGeneration")
        if generation is None or observed is None:
            return False
        try:
            if int(observed) < int(generation):
                return False
        except (TypeError, ValueError):
            return False
        desired = pod_clique.get("spec", {}).get("replicas", 0)
        if desired == 0:
            return True
        status = pod_clique.get("status", {})
        replicas = status.get("replicas", 0)
        updated = status.get("updatedReplicas", 0)
        ready = status.get("readyReplicas", 0)
        return replicas == desired == updated == ready

    @staticmethod
    def is_pod_clique_scaling_group_ready(pcsg: dict) -> bool:
        """Mirror ``CheckPCSGReady`` generation + available convergence."""
        generation = pcsg.get("metadata", {}).get("generation")
        observed = pcsg.get("status", {}).get("observedGeneration")
        if generation is None or observed is None:
            return False
        try:
            if int(observed) < int(generation):
                return False
        except (TypeError, ValueError):
            return False
        desired = pcsg.get("spec", {}).get("replicas", 0)
        if desired == 0:
            return True
        status = pcsg.get("status", {})
        replicas = status.get("replicas", 0)
        updated = status.get("updatedReplicas", 0)
        available = status.get("availableReplicas", 0)
        return replicas == desired == updated == available

    def _is_grove_backing_ready(self, component_kind: str, resource_name: str) -> bool:
        if component_kind == "PodCliqueScalingGroup":
            resource = self._get_namespaced_custom_object(
                group=GROVE_API_GROUP,
                version=GROVE_API_VERSION,
                plural=PCSG_PLURAL,
                name=resource_name,
            )
            return self.is_pod_clique_scaling_group_ready(resource)
        resource = self._get_namespaced_custom_object(
            group=GROVE_API_GROUP,
            version=GROVE_API_VERSION,
            plural=POD_CLIQUE_PLURAL,
            name=resource_name,
        )
        return self.is_pod_clique_ready(resource)

    @staticmethod
    def is_rolling_update_blocking_settlement(deployment: dict) -> tuple[bool, str]:
        """True while an operator-managed worker rollout is not yet cut over."""
        rolling = deployment.get("status", {}).get("rollingUpdate") or {}
        phase = rolling.get("phase") or ""
        if phase in ROLLING_UPDATE_BLOCKING_PHASES:
            return True, f"rollingUpdate.phase={phase}"
        return False, ""

    def _dcd_ready_or_reason(self, dcd_name: str) -> tuple[bool, str]:
        try:
            dcd = self._get_namespaced_custom_object(
                group=NVIDIA_API_GROUP,
                version=DYNAMO_API_VERSION,
                plural=DCD_PLURAL,
                name=dcd_name,
            )
        except ApiException as exc:
            if exc.status == 404:
                return False, f"DCD {dcd_name} not found"
            raise
        if self.is_dcd_ready(dcd):
            return True, ""
        generation = dcd.get("metadata", {}).get("generation")
        observed = dcd.get("status", {}).get("observedGeneration")
        return (
            False,
            f"DCD {dcd_name} not ready "
            f"(generation={generation}, observedGeneration={observed})",
        )

    def _is_dcd_backing_ready(
        self,
        deployment: dict,
        component_name: str,
        component_spec: dict,
        component_names: list[str],
    ) -> tuple[bool, str]:
        # Prefer status.components[*].componentNames (same source Grove uses).
        # During a rolling update that list includes both old and new DCDs, so
        # accepting "first ready hash from current-worker-hash" would wrongly
        # settle on the still-annotated old revision.
        if component_names:
            for dcd_name in component_names:
                ready, reason = self._dcd_ready_or_reason(dcd_name)
                if not ready:
                    return False, reason
            return True, ""

        dgd_name = deployment.get("metadata", {}).get("name", "")
        last_reason = "resource not found"
        for suffix in self._worker_hash_candidates(deployment):
            dcd_name = self._dcd_resource_name(
                dgd_name, component_name, suffix, component_spec
            )
            ready, reason = self._dcd_ready_or_reason(dcd_name)
            if ready:
                return True, ""
            last_reason = reason
            if reason.endswith("not found"):
                continue
            # Found the annotated revision but it is not ready yet.
            return False, reason
        return False, last_reason

    def is_worker_backing_settled(
        self, deployment: dict, component_name: str
    ) -> tuple[bool, str]:
        """True when the worker's backing CR has adopted the current template."""
        components = get_components_by_name(deployment)
        component_spec = components.get(component_name)
        if component_spec is None:
            return False, "component missing from DGD spec"

        status_entry = (
            deployment.get("status", {}).get("components", {}).get(component_name) or {}
        )
        component_kind = status_entry.get("componentKind", "")
        component_names = status_entry.get("componentNames") or []

        if component_kind in ("PodClique", "PodCliqueScalingGroup"):
            if not component_names:
                return False, "backing resource name missing from DGD status"
            for resource_name in component_names:
                try:
                    if not self._is_grove_backing_ready(component_kind, resource_name):
                        return (
                            False,
                            f"{component_kind} {resource_name} not generation-ready",
                        )
                except ApiException as exc:
                    if exc.status == 404:
                        return False, f"{component_kind} {resource_name} not found"
                    raise
            return True, ""

        return self._is_dcd_backing_ready(
            deployment, component_name, component_spec, component_names
        )

    def worker_backing_resources_settled(
        self, deployment: dict
    ) -> tuple[bool, list[str]]:
        """Return ``(all_settled, pending_worker_components)`` for power workers."""
        blocking, reason = self.is_rolling_update_blocking_settlement(deployment)
        if blocking:
            return False, [reason]

        pending: list[str] = []
        for component_name, component_spec in get_components_by_name(
            deployment
        ).items():
            if get_component_type(component_spec) not in WORKER_COMPONENT_TYPES:
                continue
            settled, reason = self.is_worker_backing_settled(deployment, component_name)
            if not settled:
                pending.append(f"{component_name}: {reason}")
        return not pending, pending

    async def wait_for_graph_deployment_ready(
        self,
        graph_deployment_name: str,
        include_planner: bool = True,
        max_attempts: int = 180,  # default: 30 minutes total
        delay_seconds: int = 10,  # default: check every 10 seconds
    ) -> dict:
        """Wait for a graph deployment to be ready; return the settled snapshot.

        Args:
            graph_deployment_name: Name of the DGD to wait for.
            include_planner: If False, skip components with type "planner"
                and check per-component readiness instead of the global DGD Ready
                condition. This avoids a circular wait when the planner itself
                is one of the services in the DGD. Also requires
                ``status.observedGeneration >= metadata.generation`` on the
                same snapshot so annotation-only updates are not treated as
                settled before the controller observes them, and requires each
                power-relevant worker's backing CR (DCD / Grove PodClique) to
                have caught up its own generation before Pods are treated as
                having adopted the template. Active rolling updates
                (Pending/InProgress/Failed) block settlement so a still-
                annotated old DCD cannot satisfy the gate.
            max_attempts: Maximum polling iterations.
            delay_seconds: Seconds between polls.

        Returns:
            The DGD dict that satisfied the readiness criteria (same object
            callers should use for startup-static reads such as power caps).
        """
        for attempt in range(max_attempts):
            await asyncio.sleep(delay_seconds)

            graph_deployment = self.get_graph_deployment(graph_deployment_name)

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
            else:
                if not self.is_spec_generation_observed(graph_deployment):
                    generation = graph_deployment.get("metadata", {}).get("generation")
                    observed = graph_deployment.get("status", {}).get(
                        "observedGeneration"
                    )
                    logger.info(
                        f"[Attempt {attempt + 1}/{max_attempts}] "
                        f"Waiting for DGD generation to be observed: "
                        f"generation={generation}, observedGeneration={observed}"
                    )
                    continue

                all_stable, not_ready = self.non_planner_components_stable(
                    graph_deployment
                )
                if not all_stable:
                    logger.info(
                        f"[Attempt {attempt + 1}/{max_attempts}] "
                        f"Waiting for components (excluding planner): "
                        f"not ready: {not_ready}"
                    )
                    continue

                backing_ok, backing_pending = self.worker_backing_resources_settled(
                    graph_deployment
                )
                if not backing_ok:
                    logger.info(
                        f"[Attempt {attempt + 1}/{max_attempts}] "
                        f"Waiting for worker backing resources: {backing_pending}"
                    )
                    continue

                return graph_deployment

        raise TimeoutError(
            f"Graph deployment '{graph_deployment_name}' "
            f"is not ready after {max_attempts * delay_seconds} seconds"
        )
