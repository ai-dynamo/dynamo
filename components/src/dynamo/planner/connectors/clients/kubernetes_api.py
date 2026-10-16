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
import logging
from collections.abc import Callable, Collection, Mapping
from typing import Optional

from kubernetes import client, config
from kubernetes.config.config_exception import ConfigException
from kubernetes.utils.retry import DEFAULT_BACKOFF, on_error

from dynamo.common.kubernetes import (
    AmbiguousKubernetesWriteError,
    client_go_api_client,
    retry_kubernetes_request,
    retry_kubernetes_write,
)
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
# Stable labels the operator stamps on every worker Pod.
DYNAMO_DGD_NAME_LABEL = "nvidia.com/dynamo-graph-deployment-name"
DYNAMO_DGD_UID_LABEL = "nvidia.com/dynamo-graph-deployment-uid"
DYNAMO_COMPONENT_LABEL = "nvidia.com/dynamo-component"
GROVE_PCSG_REPLICA_INDEX_LABEL = "grove.io/podcliquescalinggroup-replica-index"
PLANNER_WRITER_FENCE_ANNOTATION = "dynamo.nvidia.com/planner-writer-fence"
REPLICA_WRITE_FENCE_ANNOTATION = "dynamo.nvidia.com/replica-write-fence"
KUBERNETES_LIST_PAGE_SIZE = 500
NONTERMINAL_POD_FIELD_SELECTOR = "status.phase!=Succeeded,status.phase!=Failed"
PARTIAL_OBJECT_METADATA_ACCEPT = (
    "application/json;as=PartialObjectMetadataList;v=v1;g=meta.k8s.io"
)


def _as_custom_object(value: object, resource: str) -> dict:
    """Validate the generated CustomObjectsApi's untyped response boundary."""
    if not isinstance(value, dict):
        raise TypeError(
            f"Kubernetes API returned {type(value).__name__} instead of an object for {resource}"
        )
    return value


def _is_resource_expired(error: Exception) -> bool:
    return isinstance(error, client.ApiException) and error.status == 410


def _object_field(value: object, key: str, attribute: str | None = None) -> object:
    """Read a field from an untyped Kubernetes object or generated model."""
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, attribute or key, None)


def _pod_metadata(pod: object) -> object:
    return _object_field(pod, "metadata") or {}


def _pod_labels(pod: object) -> Mapping:
    labels = _object_field(_pod_metadata(pod), "labels") or {}
    return labels if isinstance(labels, Mapping) else {}


def _pod_phase(pod: object) -> object:
    status = _object_field(pod, "status") or {}
    return _object_field(status, "phase")


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

        api_client = client_go_api_client()
        self.api_client = api_client
        self.custom_api = client.CustomObjectsApi(api_client)
        self.core_api = client.CoreV1Api(api_client)
        self.current_namespace = k8s_namespace or get_current_k8s_namespace()

    def _read_graph_deployment_once(self, graph_deployment_name: str) -> dict:
        return _as_custom_object(
            self.custom_api.get_namespaced_custom_object(
                group=NVIDIA_API_GROUP,
                version=DYNAMO_API_VERSION,
                namespace=self.current_namespace,
                plural=DGD_PLURAL,
                name=graph_deployment_name,
            ),
            f"DynamoGraphDeployment {graph_deployment_name}",
        )

    def _get_graph_deployment_from_name(self, graph_deployment_name: str) -> dict:
        """Get the graph deployment from the dynamo graph deployment name"""
        return self._read_graph_deployment_once(graph_deployment_name)

    def list_graph_deployments(self) -> list[dict]:
        """List all DynamoGraphDeployments in the current namespace."""
        return self._list_namespaced_custom_objects(
            DGD_PLURAL,
            "DynamoGraphDeployment",
        )

    def list_scaling_adapters(self) -> list[dict]:
        """List all DGDSAs in the current namespace."""
        return self._list_namespaced_custom_objects(
            DGDSA_PLURAL,
            "DynamoGraphDeploymentScalingAdapter",
        )

    def list_graph_deployment_metadata(self) -> list[dict]:
        """List DGD identity metadata without downloading specs or status."""
        return self._list_namespaced_custom_object_metadata(
            DGD_PLURAL,
            "DynamoGraphDeployment",
        )

    def list_scaling_adapter_metadata(self) -> list[dict]:
        """List DGDSA identity metadata without downloading specs or status."""
        return self._list_namespaced_custom_object_metadata(
            DGDSA_PLURAL,
            "DynamoGraphDeploymentScalingAdapter",
        )

    def _list_namespaced_custom_objects(self, plural: str, resource: str) -> list[dict]:
        return on_error(
            DEFAULT_BACKOFF,
            _is_resource_expired,
            lambda: self._list_namespaced_custom_objects_once(plural, resource),
        )

    def _list_namespaced_custom_objects_once(
        self, plural: str, resource: str
    ) -> list[dict]:
        """Read one internally consistent paginated custom-resource snapshot."""
        return self._collect_paginated_custom_objects(
            resource,
            lambda continue_token: _as_custom_object(
                self.custom_api.list_namespaced_custom_object(
                    group=NVIDIA_API_GROUP,
                    version=DYNAMO_API_VERSION,
                    namespace=self.current_namespace,
                    plural=plural,
                    limit=KUBERNETES_LIST_PAGE_SIZE,
                    _continue=continue_token,
                ),
                f"{resource} list",
            ),
        )

    @staticmethod
    def _collect_paginated_custom_objects(
        resource: str,
        read_page: Callable[[str | None], dict],
    ) -> list[dict]:
        """Collect and validate one paginated custom-resource snapshot."""
        objects: list[dict] = []
        continue_token: str | None = None
        seen_tokens: set[str] = set()
        while True:
            result = read_page(continue_token)
            items = result.get("items")
            if not isinstance(items, list):
                raise TypeError(
                    f"Kubernetes API returned a malformed {resource} list: "
                    "items is not a list"
                )
            for item in items:
                if not isinstance(item, dict):
                    raise TypeError(
                        f"Kubernetes API returned a malformed {resource} list: "
                        f"items[{len(objects)}] is not an object"
                    )
                objects.append(item)

            metadata = result.get("metadata") or {}
            if not isinstance(metadata, Mapping):
                raise TypeError(
                    f"Kubernetes API returned a malformed {resource} list: "
                    "metadata is not an object"
                )
            raw_token = metadata.get("continue")
            if raw_token in (None, ""):
                return objects
            if not isinstance(raw_token, str):
                raise TypeError(
                    f"Kubernetes API returned a malformed {resource} list: "
                    "continue is not a string"
                )
            if raw_token in seen_tokens:
                raise RuntimeError(
                    f"Kubernetes API repeated a {resource} list continue token"
                )
            seen_tokens.add(raw_token)
            continue_token = raw_token

    def _list_namespaced_custom_object_metadata(
        self,
        plural: str,
        resource: str,
    ) -> list[dict]:
        return on_error(
            DEFAULT_BACKOFF,
            _is_resource_expired,
            lambda: self._list_namespaced_custom_object_metadata_once(
                plural,
                resource,
            ),
        )

    def _list_namespaced_custom_object_metadata_once(
        self,
        plural: str,
        resource: str,
    ) -> list[dict]:
        """Read one paginated PartialObjectMetadata snapshot for a CRD."""
        return self._collect_paginated_custom_objects(
            f"{resource} metadata",
            lambda continue_token: self._read_custom_object_metadata_page(
                plural=plural,
                continue_token=continue_token,
            ),
        )

    def _read_custom_object_metadata_page(
        self,
        *,
        plural: str,
        continue_token: str | None,
    ) -> dict:
        """Read one untyped PartialObjectMetadata page for a namespaced CRD."""
        query_params = [("limit", KUBERNETES_LIST_PAGE_SIZE)]
        if continue_token is not None:
            query_params.append(("continue", continue_token))
        request = self.api_client.param_serialize(
            method="GET",
            resource_path="/apis/{group}/{version}/namespaces/{namespace}/{plural}",
            path_params={
                "group": NVIDIA_API_GROUP,
                "version": DYNAMO_API_VERSION,
                "namespace": self.current_namespace,
                "plural": plural,
            },
            query_params=query_params,
            header_params={"Accept": PARTIAL_OBJECT_METADATA_ACCEPT},
            auth_settings=["BearerToken"],
            collection_formats={},
        )
        response = self.api_client.call_api(*request)
        response.read()
        result = self.api_client.response_deserialize(
            response,
            {"200": "object"},
        ).data
        return _as_custom_object(result, f"{plural} metadata list")

    def list_worker_metadata(self) -> list[dict]:
        """List all DynamoWorkerMetadata resources in the current namespace."""

        result = _as_custom_object(
            self.custom_api.list_namespaced_custom_object(
                group=NVIDIA_API_GROUP,
                version=DYNAMO_WORKER_METADATA_API_VERSION,
                namespace=self.current_namespace,
                plural="dynamoworkermetadatas",
            ),
            "DynamoWorkerMetadata list",
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
        try:
            # Try to scale via DGDSA Scale subresource
            self.update_scaling_adapter_replicas(
                graph_deployment_name, service_name, replicas
            )

        except AmbiguousKubernetesWriteError:
            raise
        except client.ApiException as e:
            if e.status == 404:
                # DGDSA doesn't exist - fall back to a direct DGD patch.
                adapter_name = self.scaling_adapter_name(
                    graph_deployment_name, service_name
                )
                logger.info(
                    "DGDSA %s not found, falling back to DGD update", adapter_name
                )
                self._update_dgd_replicas(graph_deployment_name, service_name, replicas)
            else:
                raise

    def get_service_replica_target(
        self, graph_deployment_name: str, service_name: str
    ) -> int:
        """Read the authoritative scale target, including an unapplied DGDSA write."""
        try:
            adapter_name = self.scaling_adapter_name(
                graph_deployment_name, service_name
            )
            scale = _as_custom_object(
                self.custom_api.get_namespaced_custom_object_scale(
                    group=NVIDIA_API_GROUP,
                    version=DYNAMO_API_VERSION,
                    namespace=self.current_namespace,
                    plural=DGDSA_PLURAL,
                    name=adapter_name,
                ),
                f"DynamoGraphDeploymentScalingAdapter scale {adapter_name}",
            )
            return int(scale["spec"]["replicas"])
        except client.ApiException as e:
            if e.status != 404:
                raise
        deployment = self.get_graph_deployment(graph_deployment_name)
        component = get_components_by_name(deployment)[service_name]
        return Service(name=service_name, service=component).number_replicas()

    @staticmethod
    def scaling_adapter_name(graph_deployment_name: str, service_name: str) -> str:
        """Return the operator-defined DGDSA name for one DGD component."""
        return f"{graph_deployment_name}-{service_name.lower()}"

    def get_service_scaling_adapter(
        self, graph_deployment_name: str, service_name: str
    ) -> dict:
        """Read the DGDSA that owns a component's desired replica count.

        Callers use this only when the DGD component explicitly declares the
        ``scalingAdapter`` key. A missing adapter is therefore an operator
        reconciliation error and is intentionally returned as a 404 rather
        than being hidden behind the legacy DGD fallback.
        """
        adapter_name = self.scaling_adapter_name(graph_deployment_name, service_name)
        return _as_custom_object(
            self.custom_api.get_namespaced_custom_object(
                group=NVIDIA_API_GROUP,
                version=DYNAMO_API_VERSION,
                namespace=self.current_namespace,
                plural=DGDSA_PLURAL,
                name=adapter_name,
            ),
            f"DynamoGraphDeploymentScalingAdapter {adapter_name}",
        )

    @staticmethod
    def get_scaling_adapter_desired_replicas(adapter: dict) -> int:
        """Return a validated authoritative replica target from a DGDSA."""
        replicas = adapter.get("spec", {}).get("replicas")
        if isinstance(replicas, bool) or not isinstance(replicas, int) or replicas < 0:
            adapter_name = adapter.get("metadata", {}).get("name", "<unknown>")
            raise ValueError(
                f"DGDSA {adapter_name!r} spec.replicas must be a non-negative integer"
            )
        return replicas

    def update_scaling_adapter_replicas(
        self,
        graph_deployment_name: str,
        service_name: str,
        replicas: int,
        *,
        resource_version: str | None = None,
    ) -> None:
        """Strictly patch a declared DGDSA through its Scale subresource.

        Unlike :meth:`update_service_replicas`, this method never falls back to
        mutating the DGD. Components that declare ``scalingAdapter`` have made
        the DGDSA desired state authoritative, so a missing adapter must fail
        loudly instead of creating a second replica owner.
        """
        adapter_name = self.scaling_adapter_name(graph_deployment_name, service_name)

        def patch_scale(version: str) -> None:
            retry_kubernetes_request(
                lambda: self.custom_api.patch_namespaced_custom_object_scale(
                    group=NVIDIA_API_GROUP,
                    version=DYNAMO_API_VERSION,
                    namespace=self.current_namespace,
                    plural=DGDSA_PLURAL,
                    name=adapter_name,
                    body={
                        "metadata": {"resourceVersion": version},
                        "spec": {"replicas": replicas},
                    },
                )
            )

        if resource_version is not None:
            # The caller owns retries because it must revalidate the DGDSA
            # writer fence before using a new resource version.
            patch_scale(resource_version)
        else:
            expected_uid: str | None = None

            def update_scale() -> None:
                nonlocal expected_uid
                scale = _as_custom_object(
                    self.custom_api.get_namespaced_custom_object_scale(
                        group=NVIDIA_API_GROUP,
                        version=DYNAMO_API_VERSION,
                        namespace=self.current_namespace,
                        plural=DGDSA_PLURAL,
                        name=adapter_name,
                    ),
                    f"DynamoGraphDeploymentScalingAdapter scale {adapter_name}",
                )
                current_uid = self._resource_uid(scale, adapter_name)
                if expected_uid is None:
                    expected_uid = current_uid
                elif current_uid != expected_uid:
                    raise RuntimeError(
                        f"DGDSA {adapter_name!r} identity changed during retry"
                    )
                patch_scale(self._resource_version(scale, adapter_name))

            retry_kubernetes_write(update_scale)
        logger.info("Scaled DGDSA %s to %s replicas", adapter_name, replicas)

    def patch_scaling_adapter_writer_fence(
        self,
        graph_deployment_name: str,
        service_name: str,
        writer_id: str,
        *,
        adapter: Mapping,
    ) -> None:
        """CAS-patch a Planner writer nonce onto an owned DGDSA."""
        adapter_name = self.scaling_adapter_name(graph_deployment_name, service_name)
        metadata = adapter.get("metadata", {}) or {}
        if not isinstance(metadata, Mapping):
            raise ValueError("DGDSA metadata is malformed")
        resource_version = metadata.get("resourceVersion")
        if not isinstance(resource_version, str) or not resource_version:
            raise ValueError("DGDSA resourceVersion is unavailable")
        annotations = metadata.get("annotations") or {}
        if not isinstance(annotations, Mapping):
            raise ValueError("DGDSA annotations are malformed")
        updated_annotations = dict(annotations)
        updated_annotations[PLANNER_WRITER_FENCE_ANNOTATION] = writer_id
        retry_kubernetes_request(
            lambda: self.custom_api.patch_namespaced_custom_object(
                group=NVIDIA_API_GROUP,
                version=DYNAMO_API_VERSION,
                namespace=self.current_namespace,
                plural=DGDSA_PLURAL,
                name=adapter_name,
                body={
                    "metadata": {
                        "resourceVersion": resource_version,
                        "annotations": updated_annotations,
                    }
                },
            )
        )
        logger.info("Claimed DGDSA %s for Planner writer %s", adapter_name, writer_id)

    def patch_scaling_adapter_replica_write_fence(
        self,
        graph_deployment_name: str,
        service_name: str,
        fence_value: str,
        *,
        adapter: Mapping,
    ) -> None:
        """Advance a DGDSA resource version after an ambiguous Scale PATCH.

        The caller owns the outer conflict retry so every 409 re-reads and
        reclassifies the original ambiguous target before writing another fence.
        """
        adapter_name = self.scaling_adapter_name(graph_deployment_name, service_name)
        metadata = adapter.get("metadata", {}) or {}
        if not isinstance(metadata, Mapping):
            raise ValueError("DGDSA metadata is malformed")
        resource_version = metadata.get("resourceVersion")
        if not isinstance(resource_version, str) or not resource_version:
            raise ValueError("DGDSA resourceVersion is unavailable")
        retry_kubernetes_request(
            lambda: self.custom_api.patch_namespaced_custom_object(
                group=NVIDIA_API_GROUP,
                version=DYNAMO_API_VERSION,
                namespace=self.current_namespace,
                plural=DGDSA_PLURAL,
                name=adapter_name,
                body={
                    "metadata": {
                        "resourceVersion": resource_version,
                        "annotations": {
                            REPLICA_WRITE_FENCE_ANNOTATION: fence_value,
                        },
                    }
                },
            )
        )

    def patch_dgd_replica_write_fence(
        self,
        graph_deployment_name: str,
        fence_value: str,
        *,
        deployment: Mapping,
    ) -> None:
        """Advance a DGD resource version after an ambiguous replica PATCH.

        The caller owns the outer conflict retry so every 409 re-reads and
        reclassifies the original ambiguous target before writing another fence.
        """
        metadata = deployment.get("metadata", {}) or {}
        if not isinstance(metadata, Mapping):
            raise ValueError("DGD metadata is malformed")
        resource_version = metadata.get("resourceVersion")
        if not isinstance(resource_version, str) or not resource_version:
            raise ValueError("DGD resourceVersion is unavailable")
        self._patch_dgd_with_merge_patch(
            graph_deployment_name,
            {
                "metadata": {
                    "resourceVersion": resource_version,
                    "annotations": {
                        REPLICA_WRITE_FENCE_ANNOTATION: fence_value,
                    },
                }
            },
        )

    def _update_dgd_replicas(
        self, graph_deployment_name: str, service_name: str, replicas: int
    ) -> None:
        """Update replicas directly in DGD when no DGDSA is available."""
        self.update_dgd_replica_targets(graph_deployment_name, {service_name: replicas})

    def update_dgd_replicas_directly(
        self,
        graph_deployment_name: str,
        service_name: str,
        replicas: int,
        *,
        deployment: dict | None = None,
    ) -> None:
        """Update the DGD replica field when the DGD still owns scaling.

        Callers use this after proving the component does not declare a
        ``scalingAdapter``. It deliberately ignores any convention-named stray
        DGDSA so preflight and execution use the same replica authority. A
        supplied deployment is the caller's validated snapshot. The merge
        patch replaces its complete component list, and its resource version
        makes the write fail with 409 if the DGD changes first. Standalone
        callers get the same behavior through a complete read-modify-write
        retry.
        """
        self.update_dgd_replica_targets(
            graph_deployment_name,
            {service_name: replicas},
            deployment=deployment,
        )

    def update_dgd_replica_targets(
        self,
        graph_deployment_name: str,
        replica_targets: Mapping[str, int],
        *,
        deployment: dict | None = None,
    ) -> None:
        """CAS-patch all direct-DGD replica targets in one object update."""
        if not replica_targets:
            return

        def update(snapshot: dict) -> None:
            resource_version = self._resource_version(snapshot, graph_deployment_name)
            self._resource_uid(snapshot, graph_deployment_name)
            components = copy.deepcopy(
                self._dgd_components(snapshot, graph_deployment_name)
            )
            for service_name, replicas in replica_targets.items():
                index = self._find_component_index(
                    graph_deployment_name, components, service_name
                )
                if "scalingAdapter" in components[index]:
                    raise RuntimeError(
                        f"component {service_name!r} now declares a scalingAdapter"
                    )
                components[index]["replicas"] = replicas
            self._patch_dgd_with_merge_patch(
                graph_deployment_name,
                {
                    "metadata": {"resourceVersion": resource_version},
                    "spec": {"components": components},
                },
            )

        if deployment is None:
            expected_uid: str | None = None

            def update_current() -> None:
                nonlocal expected_uid
                current = self.get_graph_deployment(graph_deployment_name)
                current_uid = self._resource_uid(current, graph_deployment_name)
                if expected_uid is None:
                    expected_uid = current_uid
                elif current_uid != expected_uid:
                    raise RuntimeError(
                        f"DGD {graph_deployment_name!r} identity changed during retry"
                    )
                update(current)

            retry_kubernetes_write(update_current)
        else:
            update(deployment)
        logger.info(
            "Updated DGD %s replica targets: %s",
            graph_deployment_name,
            dict(replica_targets),
        )

    @staticmethod
    def _resource_version(resource: dict, resource_name: str) -> str:
        resource_version = resource.get("metadata", {}).get("resourceVersion")
        if not resource_version:
            raise KeyError(
                f"resource {resource_name!r} has no metadata.resourceVersion"
            )
        return resource_version

    @staticmethod
    def _resource_uid(resource: dict, resource_name: str) -> str:
        uid = resource.get("metadata", {}).get("uid")
        if not isinstance(uid, str) or not uid:
            raise KeyError(f"resource {resource_name!r} has no metadata.uid")
        return uid

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

    def _patch_dgd_with_merge_patch(
        self, graph_deployment_name: str, patch: dict
    ) -> None:
        """Apply a DGD merge patch guarded by metadata.resourceVersion."""

        def request() -> None:
            self.custom_api.patch_namespaced_custom_object(
                group=NVIDIA_API_GROUP,
                version=DYNAMO_API_VERSION,
                namespace=self.current_namespace,
                plural=DGD_PLURAL,
                name=graph_deployment_name,
                body=patch,
            )

        retry_kubernetes_request(request)

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

    def pending_startup_replicas(self, deployment: dict, pods: list) -> dict[str, int]:
        """Identify startup-only scaling; an empty result never authorizes a write.

        A Ready deficit alone is ambiguous: it also occurs during drain, rollout,
        and stale status. Require observed spec, current worker revisions, and
        no terminating/failed Pods before exposing pending startup capacity.
        """
        if not self.is_spec_generation_observed(deployment):
            return {}
        phase = (deployment.get("status", {}).get("rollingUpdate") or {}).get("phase")
        if phase not in (None, "", "Completed"):
            return {}
        pods = self.exclude_checkpoint_capture_pods(pods)
        if not pods or any(
            pod.metadata.deletion_timestamp is not None
            or pod.status is None
            or pod.status.phase not in ("Pending", "Running")
            for pod in pods
        ):
            return {}
        if not self.pcsg_pods_within_desired_replicas(deployment, pods):
            return {}
        pending: dict[str, int] = {}
        statuses = deployment.get("status", {}).get("components", {})
        for name, spec in get_components_by_name(deployment).items():
            if get_component_type(spec) == "planner":
                continue
            status = statuses.get(name, {})
            desired = Service(name=name, service=spec).number_replicas()
            ready, stable = self.get_service_replica_status(deployment, name)
            replicas = status.get("replicas")
            updated = status.get("updatedReplicas")
            if (
                replicas is None
                or updated is None
                or not (0 <= ready <= replicas <= desired)
            ):
                return {}
            # Grove PCSG counts a replica as updated only once it is available;
            # PodClique/Deployment updated counts include unready new Pods.
            is_pcsg = status.get("componentKind") == "PodCliqueScalingGroup"
            if updated != (ready if is_pcsg else replicas):
                return {}
            if not stable:
                if get_component_type(spec) not in ("prefill", "decode", "worker"):
                    return {}
                if desired <= ready:
                    return {}
                pending[name] = desired - ready
        return pending

    @staticmethod
    def exclude_checkpoint_capture_pods(pods: list) -> list:
        """Capture Jobs inherit worker labels but do not serve inference traffic."""
        serving_pods = []
        for pod in pods:
            metadata = _pod_metadata(pod)
            labels = _pod_labels(pod)
            owners = (
                _object_field(metadata, "ownerReferences", "owner_references") or []
            )
            is_capture_pod = labels.get("nvidia.com/snapshot-job-uid") and any(
                _object_field(owner, "controller")
                and _object_field(owner, "apiVersion", "api_version") == "batch/v1"
                and _object_field(owner, "kind") == "Job"
                and _object_field(owner, "name")
                == labels.get("nvidia.com/snapshot-job")
                for owner in owners
            )
            if not is_capture_pod:
                serving_pods.append(pod)
        return serving_pods

    def pcsg_pods_within_desired_replicas(self, deployment: dict, pods: list) -> bool:
        """Reject excess PCSG groups hidden by its spec-derived replica count."""
        components = get_components_by_name(deployment)
        pods_by_component = self.partition_pods_by_component(pods)
        for name, status in deployment.get("status", {}).get("components", {}).items():
            if status.get("componentKind") != "PodCliqueScalingGroup":
                continue
            desired = Service(
                name=name, service=components.get(name, {})
            ).number_replicas()
            for pod in pods_by_component.get(name, []):
                # Terminal Pods no longer belong to the active replica range.
                if _pod_phase(pod) in ("Succeeded", "Failed"):
                    continue
                replica_index = _pod_labels(pod).get(GROVE_PCSG_REPLICA_INDEX_LABEL)
                if replica_index is None:
                    return False
                try:
                    if not 0 <= int(replica_index) < desired:
                        return False
                except (TypeError, ValueError):
                    return False
        return True

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

    def fetch_authoritative_replica_targets(
        self,
        deployment: dict,
        expected_authorities: Mapping[str, tuple[bool, str | None]] | None = None,
        scaling_adapters: Mapping[str, dict] | None = None,
    ) -> dict:
        """Overlay declared DGDSA targets onto a DGD settlement snapshot.

        ``spec.components[].replicas`` is only an initial seed for a component
        that declares ``scalingAdapter``. Startup must compare status with the
        live, owned DGDSA target or a restart can wait forever after that target
        has legitimately reached zero.
        """
        components = get_components_by_name(deployment)
        expected_authorities = expected_authorities or {}
        for component_name, (expects_adapter, _) in expected_authorities.items():
            component = components.get(component_name)
            if component is None:
                raise RuntimeError(
                    f"component {component_name!r} disappeared during replica settlement"
                )
            has_adapter = "scalingAdapter" in component
            if has_adapter != expects_adapter:
                raise RuntimeError(
                    f"component {component_name!r} changed replica authority "
                    "during settlement"
                )
        adapter_components = [
            name
            for name, component in components.items()
            if "scalingAdapter" in component
        ]
        if not adapter_components:
            return deployment

        metadata = deployment.get("metadata", {}) or {}
        if not isinstance(metadata, Mapping):
            raise ValueError("DGD metadata is malformed")
        dgd_name = metadata.get("name")
        dgd_uid = metadata.get("uid")
        if not isinstance(dgd_name, str) or not dgd_name:
            raise ValueError("DGD metadata.name is unavailable")
        if not isinstance(dgd_uid, str) or not dgd_uid:
            raise ValueError("DGD metadata.uid is unavailable")

        authoritative = copy.deepcopy(deployment)
        authoritative_components = get_components_by_name(authoritative)
        for component_name in adapter_components:
            adapter_name = self.scaling_adapter_name(dgd_name, component_name)
            if scaling_adapters is None:
                adapter = self.get_service_scaling_adapter(dgd_name, component_name)
            else:
                adapter = scaling_adapters.get(adapter_name)
                if adapter is None:
                    raise ValueError(
                        f"DGDSA {adapter_name!r} is absent from the namespace inventory"
                    )
            rejection = self.scaling_adapter_identity_rejection(
                deployment, component_name, adapter
            )
            if rejection is not None:
                raise ValueError(rejection)
            expected_authority = expected_authorities.get(component_name)
            if expected_authority is not None:
                expected_uid = expected_authority[1]
                adapter_uid = (adapter.get("metadata", {}) or {}).get("uid")
                if adapter_uid != expected_uid:
                    raise RuntimeError(
                        f"DGDSA identity for component {component_name!r} changed "
                        "during replica settlement"
                    )
            authoritative_components[component_name][
                "replicas"
            ] = self.get_scaling_adapter_desired_replicas(adapter)
        return authoritative

    def replica_targets_stable(
        self,
        deployment: dict,
        expected_targets: Mapping[str, int],
    ) -> tuple[bool, list[str]]:
        """Verify exact accepted targets and fully settled component status."""
        components = get_components_by_name(deployment)
        for component_name, expected in expected_targets.items():
            component = components.get(component_name)
            if component is None:
                raise RuntimeError(
                    f"component {component_name!r} disappeared during replica settlement"
                )
            actual = Service(name=component_name, service=component).number_replicas()
            if actual != expected:
                raise RuntimeError(
                    f"replica target for component {component_name!r} changed "
                    f"from {expected} to {actual} during settlement"
                )

        unsettled: list[str] = []
        statuses = deployment.get("status", {}).get("components", {})
        for component_name, component in components.items():
            if get_component_type(component) == "planner":
                continue
            desired = Service(name=component_name, service=component).number_replicas()
            status = statuses.get(component_name, {})
            replicas = status.get("replicas")
            updated = status.get("updatedReplicas")
            available = status.get("availableReplicas")
            serving = (
                available if available is not None else status.get("readyReplicas")
            )
            if replicas != desired or updated != desired or serving != desired:
                unsettled.append(component_name)
        return not unsettled, unsettled

    @staticmethod
    def adapter_targets_projected(
        deployment: dict,
        expected_targets: Mapping[str, int],
        expected_authorities: Mapping[str, tuple[bool, str | None]],
    ) -> tuple[bool, list[str]]:
        """Check that accepted DGDSA targets reached the raw DGD spec."""
        components = get_components_by_name(deployment)
        pending: list[str] = []
        for component_name, expected in expected_targets.items():
            authority = expected_authorities.get(component_name)
            if authority is None:
                raise RuntimeError(
                    f"component {component_name!r} has no expected replica authority"
                )
            if not authority[0]:
                continue
            component = components.get(component_name)
            if component is None:
                raise RuntimeError(
                    f"component {component_name!r} disappeared during replica settlement"
                )
            actual = Service(name=component_name, service=component).number_replicas()
            if actual != expected:
                pending.append(component_name)
        return not pending, pending

    def scaling_adapter_identity_rejection(
        self,
        deployment: dict,
        component_name: str,
        adapter: dict,
        *,
        require_resource_version: bool = False,
    ) -> str | None:
        """Validate the identity fields needed to trust a DGDSA target."""
        if not isinstance(adapter, Mapping):
            return f"DGDSA for component {component_name!r} is malformed"
        dgd_metadata = deployment.get("metadata", {}) or {}
        if not isinstance(dgd_metadata, Mapping):
            return "DGD metadata is malformed"
        dgd_name = dgd_metadata.get("name")
        dgd_uid = dgd_metadata.get("uid")
        if not isinstance(dgd_name, str) or not dgd_name:
            return "DGD metadata.name is unavailable"
        if not isinstance(dgd_uid, str) or not dgd_uid:
            return "DGD metadata.uid is unavailable"
        adapter_metadata = adapter.get("metadata", {}) or {}
        if not isinstance(adapter_metadata, Mapping):
            return f"DGDSA metadata for component {component_name!r} is malformed"
        expected_name = self.scaling_adapter_name(str(dgd_name), component_name)
        if adapter_metadata.get("name") != expected_name:
            return f"DGDSA metadata.name does not match component {component_name!r}"
        if adapter_metadata.get("deletionTimestamp") is not None:
            return f"DGDSA {expected_name!r} is being deleted"
        resource_version = adapter_metadata.get("resourceVersion")
        if require_resource_version and (
            not isinstance(resource_version, str) or not resource_version.strip()
        ):
            return f"DGDSA {expected_name!r} has no resourceVersion"

        expected_api_version = f"{NVIDIA_API_GROUP}/{DYNAMO_API_VERSION}"
        owners = adapter_metadata.get("ownerReferences", []) or []
        if not isinstance(owners, list):
            return f"DGDSA {expected_name!r} ownerReferences are malformed"
        if not any(
            isinstance(owner, Mapping)
            and owner.get("apiVersion") == expected_api_version
            and owner.get("controller") is True
            and owner.get("kind") == "DynamoGraphDeployment"
            and owner.get("name") == dgd_name
            and owner.get("uid") == dgd_uid
            for owner in owners
        ):
            return f"DGDSA {expected_name!r} is not controller-owned by the DGD"

        adapter_spec = adapter.get("spec", {}) or {}
        if not isinstance(adapter_spec, Mapping):
            return f"DGDSA {expected_name!r} spec is malformed"
        dgd_ref = adapter_spec.get("dgdRef") or {}
        if not isinstance(dgd_ref, Mapping):
            return f"DGDSA {expected_name!r} dgdRef is malformed"
        if dgd_ref.get("name") != dgd_name:
            return f"DGDSA {expected_name!r} references a different DGD"
        if dgd_ref.get("componentName") != component_name:
            return f"DGDSA {expected_name!r} references a different component"

        labels = adapter_metadata.get("labels") or {}
        if not isinstance(labels, Mapping):
            return f"DGDSA {expected_name!r} labels are malformed"
        if (
            DYNAMO_DGD_NAME_LABEL in labels
            and labels[DYNAMO_DGD_NAME_LABEL] != dgd_name
        ):
            return f"DGDSA {expected_name!r} has a mismatched DGD label"
        if (
            DYNAMO_COMPONENT_LABEL in labels
            and labels[DYNAMO_COMPONENT_LABEL] != component_name
        ):
            return f"DGDSA {expected_name!r} has a mismatched component label"
        return None

    def verify_scaling_adapter_replica_target(
        self,
        deployment: dict,
        component_name: str,
        *,
        expected_uid: str,
        expected_replicas: int,
        writer_id: str | None,
    ) -> None:
        """Verify that a completed Scale PATCH still owns its replica target."""
        adapter = self.get_service_scaling_adapter(
            deployment["metadata"]["name"], component_name
        )
        rejection = self.scaling_adapter_identity_rejection(
            deployment, component_name, adapter
        )
        if rejection is not None:
            raise ValueError(rejection)

        metadata = adapter.get("metadata", {}) or {}
        if metadata.get("uid") != expected_uid:
            raise ValueError("DGDSA identity changed after its Scale PATCH")
        annotations = metadata.get("annotations") or {}
        if not isinstance(annotations, Mapping):
            raise ValueError("DGDSA annotations are malformed")
        if (
            writer_id is not None
            and annotations.get(PLANNER_WRITER_FENCE_ANNOTATION) != writer_id
        ):
            raise ValueError("DGDSA Planner writer fence changed after its Scale PATCH")
        if self.get_scaling_adapter_desired_replicas(adapter) != expected_replicas:
            raise ValueError("DGDSA replica target changed after its Scale PATCH")

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
            _object_field(_pod_metadata(pod), "deletionTimestamp", "deletion_timestamp")
            is not None
            and _pod_phase(pod) not in ("Succeeded", "Failed")
            for pod in pods
        )

    def list_pods_for_graph(self, dgd_name: str) -> list:
        """List all Pods for a DGD using its stable graph label."""
        return self._list_graph_pods(f"{DYNAMO_DGD_NAME_LABEL}={dgd_name}")

    def list_nonterminal_pods_for_graph(self, dgd_name: str) -> list:
        """List active Pods for DGD settlement using server-side filtering."""
        return self._list_graph_pods(
            f"{DYNAMO_DGD_NAME_LABEL}={dgd_name}",
            field_selector=NONTERMINAL_POD_FIELD_SELECTOR,
            metadata_only=True,
        )

    def list_graph_pods(self, dgd_names: Collection[str] | None = None) -> list:
        """List nonterminal DGD Pods, optionally scoped to selected graphs."""
        if dgd_names is None:
            label_selector = DYNAMO_DGD_NAME_LABEL
        else:
            names = sorted(set(dgd_names))
            if not names:
                return []
            label_selector = f"{DYNAMO_DGD_NAME_LABEL} in ({','.join(names)})"
        return self._list_graph_pods(
            label_selector,
            field_selector=NONTERMINAL_POD_FIELD_SELECTOR,
            metadata_only=True,
        )

    def _list_graph_pods(
        self,
        label_selector: str,
        field_selector: str | None = None,
        *,
        metadata_only: bool = False,
    ) -> list:
        return on_error(
            DEFAULT_BACKOFF,
            _is_resource_expired,
            lambda: self._list_graph_pods_once(
                label_selector,
                field_selector,
                metadata_only=metadata_only,
            ),
        )

    def _list_graph_pods_once(
        self,
        label_selector: str,
        field_selector: str | None = None,
        *,
        metadata_only: bool = False,
    ) -> list:
        """Read one internally consistent paginated Pod snapshot."""
        pods: list = []
        continue_token: str | None = None
        seen_tokens: set[str] = set()
        while True:
            if metadata_only:
                result = self._read_core_pod_metadata_page(
                    label_selector=label_selector,
                    field_selector=field_selector,
                    continue_token=continue_token,
                )
                page_items = result.get("items")
                metadata = result.get("metadata") or {}
                if not isinstance(page_items, list):
                    raise TypeError(
                        "Kubernetes API returned a malformed Pod list: "
                        "items is not a list"
                    )
                if not isinstance(metadata, Mapping):
                    raise TypeError(
                        "Kubernetes API returned a malformed Pod list: "
                        "metadata is not an object"
                    )
                pods.extend(page_items)
                raw_token = metadata.get("continue")
            else:
                page = self.core_api.list_namespaced_pod(
                    namespace=self.current_namespace,
                    label_selector=label_selector,
                    field_selector=field_selector,
                    limit=KUBERNETES_LIST_PAGE_SIZE,
                    _continue=continue_token,
                )
                pods.extend(page.items or [])
                metadata = getattr(page, "metadata", None)
                raw_token = getattr(metadata, "_continue", None)
            if raw_token in (None, ""):
                return pods
            if not isinstance(raw_token, str):
                raise TypeError(
                    "Kubernetes API returned a malformed Pod list: "
                    "continue is not a string"
                )
            if raw_token in seen_tokens:
                raise RuntimeError("Kubernetes API repeated a Pod list continue token")
            seen_tokens.add(raw_token)
            continue_token = raw_token

    def _read_core_pod_metadata_page(
        self,
        *,
        label_selector: str,
        field_selector: str | None,
        continue_token: str | None,
    ) -> dict:
        """Read one untyped PartialObjectMetadata Pod page.

        Generated CoreV1Api methods always deserialize list items as V1Pod,
        whose required spec rejects PartialObjectMetadata. The public ApiClient
        request/response primitives preserve the lightweight payload without
        depending on generated private serializers.
        """
        query_params = [
            ("labelSelector", label_selector),
            ("limit", KUBERNETES_LIST_PAGE_SIZE),
        ]
        if field_selector is not None:
            query_params.append(("fieldSelector", field_selector))
        if continue_token is not None:
            query_params.append(("continue", continue_token))
        request = self.api_client.param_serialize(
            method="GET",
            resource_path="/api/v1/namespaces/{namespace}/pods",
            path_params={"namespace": self.current_namespace},
            query_params=query_params,
            header_params={"Accept": PARTIAL_OBJECT_METADATA_ACCEPT},
            auth_settings=["BearerToken"],
            collection_formats={},
        )
        response = self.api_client.call_api(*request)
        response.read()
        result = self.api_client.response_deserialize(
            response,
            {"200": "object"},
        ).data
        return _as_custom_object(result, "Pod list")

    @staticmethod
    def partition_pods_by_component(pods: list) -> dict[str, list]:
        """Partition Pods by the stable Dynamo component label."""
        by_component: dict[str, list] = {}
        for pod in pods:
            component_name = _pod_labels(pod).get(DYNAMO_COMPONENT_LABEL)
            if component_name:
                by_component.setdefault(component_name, []).append(pod)
        return by_component

    @staticmethod
    def partition_pods_by_graph(pods: list) -> dict[str, list]:
        """Partition Pods by the stable Dynamo graph label."""
        by_graph: dict[str, list] = {}
        for pod in pods:
            graph_name = _pod_labels(pod).get(DYNAMO_DGD_NAME_LABEL)
            if graph_name:
                by_graph.setdefault(graph_name, []).append(pod)
        return by_graph

    @staticmethod
    def pod_graph_uid_mismatches(pods: list, expected_uid: str) -> list[str]:
        """Return Pods that do not carry the current DGD identity label."""
        mismatches: list[str] = []
        for pod in pods:
            if _pod_labels(pod).get(DYNAMO_DGD_UID_LABEL) == expected_uid:
                continue
            name = _object_field(_pod_metadata(pod), "name")
            mismatches.append(str(name or "<unknown>"))
        return mismatches

    @staticmethod
    def pod_component_mismatches(pods: list, deployment: dict) -> list[str]:
        """Return Pods whose component label is absent from the current DGD."""
        component_names = set(get_components_by_name(deployment))
        mismatches: list[str] = []
        for pod in pods:
            if _pod_labels(pod).get(DYNAMO_COMPONENT_LABEL) in component_names:
                continue
            name = _object_field(_pod_metadata(pod), "name")
            mismatches.append(str(name or "<unknown>"))
        return mismatches

    def worker_pods_settled(
        self,
        deployment: dict,
        expected_power_by_component: Mapping[str, int],
    ) -> tuple[bool, list[str]]:
        """True when all non-terminal pods carry the expected per-GPU annotation.

        ``expected_power_by_component`` maps each power-relevant component name
        to the operator-projected per-GPU power limit.

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

        dgd_metadata = deployment.get("metadata", {}) or {}
        dgd_name = dgd_metadata.get("name", "")
        dgd_uid = dgd_metadata.get("uid", "")
        if not isinstance(dgd_uid, str) or not dgd_uid:
            return False, ["DGD metadata.uid is unavailable"]
        components_by_name = get_components_by_name(deployment)
        pending: list[str] = []
        graph_pods = self.exclude_checkpoint_capture_pods(
            self.list_pods_for_graph(dgd_name)
        )
        mismatches = self.pod_graph_uid_mismatches(graph_pods, dgd_uid)
        if mismatches:
            return False, [f"Pods have stale DGD identity: {mismatches}"]
        pods_by_component = self.partition_pods_by_component(graph_pods)

        for component_name, expected_watts in expected_power_by_component.items():
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
                actual_raw = (pod.metadata.annotations or {}).get(POWER_ANNOTATION_KEY)
                try:
                    actual_watts = int(str(actual_raw).strip())
                except (TypeError, ValueError):
                    actual_watts = None
                if actual_watts != expected_watts:
                    phase = (pod.status.phase if pod.status else None) or "?"
                    pending.append(
                        f"{component_name}/{pod.metadata.name}"
                        f" (phase={phase}):"
                        f" annotation {actual_raw!r} != {expected_watts!r}"
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
        expected_dgd_uid: str | None = None,
        expected_replica_targets: Mapping[str, int] | None = None,
        expected_replica_authorities: Mapping[str, tuple[bool, str | None]]
        | None = None,
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
            expected_dgd_uid / expected_replica_targets /
                expected_replica_authorities: When targets are supplied, wait
                for that exact accepted replica transaction instead of trusting
                the DGD Ready condition. Identity or authority changes are
                terminal; rollout, status, and Pod convergence are polled.

        Returns:
            The DGD dict that satisfied the readiness criteria (same object
            callers should use for startup-static reads such as power caps).
        """
        for attempt in range(max_attempts):
            if attempt:
                await asyncio.sleep(delay_seconds)

            graph_deployment = await asyncio.to_thread(
                self.get_graph_deployment, graph_deployment_name
            )

            if expected_replica_targets is not None:
                if not isinstance(expected_dgd_uid, str) or not expected_dgd_uid:
                    raise ValueError(
                        "expected DGD UID is required for replica settlement"
                    )
                current_uid = graph_deployment.get("metadata", {}).get("uid")
                if current_uid != expected_dgd_uid:
                    raise RuntimeError("DGD identity changed during replica settlement")
                if expected_replica_authorities is None:
                    raise ValueError(
                        "expected replica authorities are required for settlement"
                    )

                rolling = graph_deployment.get("status", {}).get("rollingUpdate") or {}
                if rolling.get("phase") == "Failed":
                    raise RolloutFailedError(
                        deployment_name=graph_deployment_name,
                        reason=rolling.get("message", ""),
                    )
                settlement_deployment = await asyncio.to_thread(
                    self.fetch_authoritative_replica_targets,
                    graph_deployment,
                    expected_replica_authorities,
                )
                targets_stable, unsettled = self.replica_targets_stable(
                    settlement_deployment, expected_replica_targets
                )
                targets_projected, unprojected = self.adapter_targets_projected(
                    graph_deployment,
                    expected_replica_targets,
                    expected_replica_authorities,
                )
                generation_observed = self.is_spec_generation_observed(graph_deployment)
                deployment_ready = not include_planner or self.is_deployment_ready(
                    graph_deployment
                )
                (
                    blocking_rollout,
                    rollout_reason,
                ) = self.is_rolling_update_blocking_settlement(graph_deployment)
                if (
                    not targets_stable
                    or not targets_projected
                    or not generation_observed
                    or not deployment_ready
                    or blocking_rollout
                ):
                    logger.info(
                        "[Attempt %d/%d] Waiting for accepted replica targets: "
                        "unsettled=%s, unprojected=%s, observed=%s, ready=%s, "
                        "rollout=%s",
                        attempt + 1,
                        max_attempts,
                        unsettled,
                        unprojected,
                        generation_observed,
                        deployment_ready,
                        rollout_reason or "settled",
                    )
                    continue

                pods = await asyncio.to_thread(
                    self.list_nonterminal_pods_for_graph, graph_deployment_name
                )
                pods = self.exclude_checkpoint_capture_pods(pods)
                identity_mismatches = self.pod_graph_uid_mismatches(
                    pods, expected_dgd_uid
                )
                component_mismatches = self.pod_component_mismatches(
                    pods,
                    settlement_deployment,
                )
                if (
                    identity_mismatches
                    or component_mismatches
                    or self.has_terminating_pods(pods)
                    or not self.pcsg_pods_within_desired_replicas(
                        settlement_deployment, pods
                    )
                ):
                    logger.info(
                        "[Attempt %d/%d] Waiting for replica Pods to drain: "
                        "identity_mismatches=%s, component_mismatches=%s",
                        attempt + 1,
                        max_attempts,
                        identity_mismatches,
                        component_mismatches,
                    )
                    continue

                current_deployment = await asyncio.to_thread(
                    self.get_graph_deployment, graph_deployment_name
                )
                current_metadata = current_deployment.get("metadata", {}) or {}
                if current_metadata.get("uid") != expected_dgd_uid:
                    raise RuntimeError("DGD identity changed during replica settlement")
                observed_version = (graph_deployment.get("metadata", {}) or {}).get(
                    "resourceVersion"
                )
                current_version = current_metadata.get("resourceVersion")
                if not isinstance(observed_version, str) or not observed_version:
                    raise ValueError(
                        "DGD resourceVersion is required for replica settlement"
                    )
                if not isinstance(current_version, str) or not current_version:
                    raise ValueError(
                        "DGD resourceVersion is required for replica settlement"
                    )
                if observed_version != current_version:
                    logger.info(
                        "[Attempt %d/%d] DGD changed during Pod inventory",
                        attempt + 1,
                        max_attempts,
                    )
                    continue

                current_settlement = await asyncio.to_thread(
                    self.fetch_authoritative_replica_targets,
                    current_deployment,
                    expected_replica_authorities,
                )
                current_targets_stable, _ = self.replica_targets_stable(
                    current_settlement, expected_replica_targets
                )
                current_targets_projected, _ = self.adapter_targets_projected(
                    current_deployment,
                    expected_replica_targets,
                    expected_replica_authorities,
                )
                current_generation_observed = self.is_spec_generation_observed(
                    current_deployment
                )
                current_deployment_ready = (
                    not include_planner or self.is_deployment_ready(current_deployment)
                )
                (
                    current_rollout_blocking,
                    _,
                ) = self.is_rolling_update_blocking_settlement(current_deployment)
                if (
                    not current_targets_stable
                    or not current_targets_projected
                    or not current_generation_observed
                    or not current_deployment_ready
                    or current_rollout_blocking
                ):
                    logger.info(
                        "[Attempt %d/%d] DGD settlement changed after Pod inventory",
                        attempt + 1,
                        max_attempts,
                    )
                    continue
                return current_deployment

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
            settlement_deployment = await asyncio.to_thread(
                self.fetch_authoritative_replica_targets, graph_deployment
            )
            all_stable, not_ready = self.non_planner_components_stable(
                settlement_deployment
            )
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

            # Build per-component expected limits from operator-owned status.
            # A missing or malformed value is invalid configuration — raise
            # immediately rather than retrying.
            components_map = get_components_by_name(graph_deployment)
            expected_power: dict[str, int] = {}
            for name in power_names:
                svc = Service(name=name, service=components_map.get(name, {}))
                expected_power[name] = svc.get_gpu_power_limit_watts(graph_deployment)

            pods_ok, pods_pending = await asyncio.to_thread(
                self.worker_pods_settled, graph_deployment, expected_power
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
