# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Narrow Kubernetes capacity backend for serialized managed allocations.

Only fixed-cost, single-node Deployment workers with a positive minimum are
supported. All present worker Pods remain charged until they disappear,
including draining and failed Pods. This assumes normal graceful deletion,
healthy nodes, and exclusive scale ownership; force deletion and concurrent
rollouts are outside this contract.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from http.client import HTTPException
from typing import TypeVar

from kubernetes import client
from kubernetes.config.config_exception import ConfigException
from urllib3.exceptions import HTTPError

from dynamo.global_planner.allocation_policy import (
    AllocationPolicy,
    PoolAllocationPolicy,
)
from dynamo.global_planner.managed_capacity import (
    ManagedCapacityError,
    ManagedPoolSnapshot,
)
from dynamo.planner.connectors.clients.kubernetes_api import (
    DGD_PLURAL,
    DGDSA_PLURAL,
    DYNAMO_API_VERSION,
    DYNAMO_COMPONENT_LABEL,
    DYNAMO_DGD_NAME_LABEL,
    NVIDIA_API_GROUP,
    PLANNER_WRITER_FENCE_ANNOTATION,
    KubernetesAPI,
)

_HTTP_TIMEOUT = (3, 10)
_MAX_PODS_PER_GRAPH = 1000
_WORKER_HASH = "nvidia.com/dynamo-worker-hash"
_CURRENT_HASHES = (
    "nvidia.com/current-worker-hash",
    "nvidia.com/current-worker-hash-v2",
)
_PROVIDER = "nvidia.com/workload-provider"
_GPU_RESOURCE = "nvidia.com/gpu"
_IDLE_PHASES = (None, "", "Completed")
_Result = TypeVar("_Result")


def _integer(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ManagedCapacityError(f"{label} must be a nonnegative integer")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ManagedCapacityError(f"{label} is unavailable")
    return value


def _gpu_cost(spec: dict) -> int:
    """Conservative supported shape: scalar GPUs in application containers only."""
    if spec.get("resourceClaims") or spec.get("resource_claims"):
        raise ManagedCapacityError("DRA resource claims are unsupported")
    total = 0
    for container in spec.get("containers") or []:
        resources = container.get("resources") or {}
        if resources.get("claims"):
            raise ManagedCapacityError("DRA resource claims are unsupported")
        requests = resources.get("requests") or {}
        limits = resources.get("limits") or {}
        for name in requests.keys() | limits.keys():
            vendor, _, _ = name.partition("/")
            if vendor == "nvidia.com" and name != _GPU_RESOURCE:
                raise ManagedCapacityError("only scalar nvidia.com/gpu is supported")
        raw = requests.get(_GPU_RESOURCE, limits.get(_GPU_RESOURCE, 0))
        if isinstance(raw, bool) or not str(raw).isdigit():
            raise ManagedCapacityError("GPU requests must be whole numbers")
        count = int(raw)
        if _GPU_RESOURCE in limits and str(limits[_GPU_RESOURCE]) != str(count):
            raise ManagedCapacityError("GPU requests and limits must agree")
        total += count
    # Reject GPU init containers (including native sidecars) instead of trying
    # to approximate Kubernetes' effective scheduling footprint here.
    for container in spec.get("initContainers", spec.get("init_containers")) or []:
        if _gpu_cost({"containers": [container]}):
            raise ManagedCapacityError("GPU init containers are unsupported")
    if spec.get("ephemeralContainers") or spec.get("ephemeral_containers"):
        raise ManagedCapacityError("ephemeral containers are unsupported")
    return total


@dataclass(frozen=True)
class _Read:
    snapshot: ManagedPoolSnapshot
    adapter: dict


class ManagedKubernetesCapacity:
    """Use existing DGD, DGDSA and Pod permissions; never directly patch a DGD."""

    def __init__(
        self,
        policy: AllocationPolicy,
        namespace: str,
        *,
        kube_api: KubernetesAPI | None = None,
    ) -> None:
        self._policy = policy
        self._namespace = _text(namespace, "namespace")
        self._api = kube_api
        self._pools = {pool.pool_id: pool for pool in policy.pools}
        self._names: dict[str, str] = {}
        for pool in policy.pools:
            # The operator considers zero-replica Deployments ready before
            # their controller has observed the scale-down. Existing Planner
            # RBAC cannot read Deployments directly, so retain one warm worker
            # until zero-target release can be proved authoritatively.
            if pool.min_replicas < 1:
                raise ValueError(
                    "managed Kubernetes pools require a minimum of one replica"
                )
            parts = pool.participant_id.split("/")
            if len(parts) != 2 or parts[0] != namespace:
                raise ValueError(
                    "managed pools must use exact namespace/DGD identities in one namespace"
                )
            self._names[pool.pool_id] = parts[1]
        self._writer_id: str | None = None
        self._identities: dict[str, str] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, writer_id: str) -> None:
        """CAS-claim every adapter, then verify the claims before exposing them."""
        _text(writer_id, "writer_id")
        async with self._lock:
            await self._run(self._acquire, writer_id)

    async def observe(self) -> dict[str, ManagedPoolSnapshot]:
        async with self._lock:
            return await self._run(self._observe)

    async def scale(
        self,
        pool_id: str,
        replicas: int,
        *,
        expected_identity: str,
        expected_target: int,
    ) -> None:
        _integer(replicas, "replicas")
        _integer(expected_target, "expected_target")
        async with self._lock:
            await self._run(
                self._scale, pool_id, replicas, expected_identity, expected_target
            )

    async def _run(self, operation: Callable[..., _Result], *args: object) -> _Result:
        task = asyncio.create_task(asyncio.to_thread(operation, *args))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # Cancelling to_thread does not stop its HTTP request. Keep the
            # backend lock until it finishes, so a subsequent read or writer
            # claim cannot overlap an abandoned mutation in this instance.
            await asyncio.gather(task, return_exceptions=True)
            raise
        except (
            client.ApiException,
            ConfigException,
            HTTPError,
            HTTPException,
            OSError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise ManagedCapacityError(
                f"Kubernetes capacity is uncertain: {exc}"
            ) from exc

    def _get(self, plural: str, name: str) -> dict:
        assert self._api is not None
        return self._api.custom_api.get_namespaced_custom_object(
            group=NVIDIA_API_GROUP,
            version=DYNAMO_API_VERSION,
            namespace=self._namespace,
            plural=plural,
            name=name,
            _request_timeout=_HTTP_TIMEOUT,
        )

    def _acquire(self, writer_id: str) -> None:
        self._writer_id = None
        if self._api is None:
            self._api = KubernetesAPI(self._namespace)
        reads = {
            pool.pool_id: self._read(pool, check_fence=False)
            for pool in self._policy.pools
        }
        for pool in self._policy.pools:
            adapter = reads[pool.pool_id].adapter
            metadata = adapter["metadata"]
            # A metadata merge patch with resourceVersion is an atomic CAS,
            # including when the annotation map did not previously exist.
            self._api.custom_api.patch_namespaced_custom_object(
                group=NVIDIA_API_GROUP,
                version=DYNAMO_API_VERSION,
                namespace=self._namespace,
                plural=DGDSA_PLURAL,
                name=metadata["name"],
                body={
                    "metadata": {
                        "resourceVersion": metadata["resourceVersion"],
                        "uid": metadata["uid"],
                        "annotations": {PLANNER_WRITER_FENCE_ANNOTATION: writer_id},
                    }
                },
                _request_timeout=_HTTP_TIMEOUT,
            )
        self._writer_id = writer_id
        try:
            self._observe()
        except Exception:
            self._writer_id = None
            raise

    def _observe(self) -> dict[str, ManagedPoolSnapshot]:
        if self._writer_id is None:
            raise ManagedCapacityError("writer fences have not been acquired")
        return {pool.pool_id: self._read(pool).snapshot for pool in self._policy.pools}

    def _scale(self, pool_id: str, replicas: int, identity: str, target: int) -> None:
        if self._writer_id is None:
            raise ManagedCapacityError("writer fences have not been acquired")
        if pool_id not in self._pools:
            raise ManagedCapacityError(f"unknown pool {pool_id!r}")
        pool = self._pools[pool_id]
        if not pool.min_replicas <= replicas <= pool.max_replicas:
            raise ManagedCapacityError("scale target is outside pool policy bounds")
        # Recheck the full enrolled set, including ownership of unrelated pools.
        reads = {pool.pool_id: self._read(pool) for pool in self._policy.pools}
        read = reads[pool_id]
        if (
            read.snapshot.identity != identity
            or read.snapshot.target_replicas != target
        ):
            raise ManagedCapacityError(
                "pool identity or replica target changed before scale"
            )
        if replicas > target:
            if not all(item.snapshot.settled for item in reads.values()):
                raise ManagedCapacityError("pool became unsettled before growth")
            held = sum(
                item.snapshot.observed_replicas * self._pools[key].gpu_per_replica
                for key, item in reads.items()
            )
            if (
                held + (replicas - target) * pool.gpu_per_replica
                > self._policy.max_total_gpus
            ):
                raise ManagedCapacityError("fresh capacity does not permit growth")
        metadata = read.adapter["metadata"]
        assert self._api is not None
        self._api.custom_api.patch_namespaced_custom_object_scale(
            group=NVIDIA_API_GROUP,
            version=DYNAMO_API_VERSION,
            namespace=self._namespace,
            plural=DGDSA_PLURAL,
            name=metadata["name"],
            body={
                "metadata": {
                    "resourceVersion": metadata["resourceVersion"],
                    "uid": metadata["uid"],
                },
                "spec": {"replicas": replicas},
            },
            _request_timeout=_HTTP_TIMEOUT,
        )

    def _read(self, pool: PoolAllocationPolicy, *, check_fence: bool = True) -> _Read:
        name = self._names[pool.pool_id]
        deployment = self._get(DGD_PLURAL, name)
        graph = self._read_graph(pool, name, deployment)
        adapter = self._read_adapter(pool, name, deployment, check_fence=check_fence)
        identity = self._check_identity(pool, graph, adapter)
        target = _integer(adapter["spec"].get("replicas"), "DGDSA replicas")
        dgd_target = _integer(graph.component.get("replicas"), "DGD replicas")
        observed, ready, terminating = self._count_workers(pool, name, graph)
        status_current = all(
            graph.component_status.get(field) == target
            for field in ("replicas", "updatedReplicas", "readyReplicas")
        )
        backing_ready = graph.status.get("state") == "successful" and any(
            condition.get("type") == "Ready"
            and condition.get("status") == "True"
            and condition.get("observedGeneration") == graph.generation
            for condition in graph.status.get("conditions") or []
        )
        settled = (
            graph.generation_current
            and status_current
            and backing_ready
            and target == dgd_target == observed == ready
            and terminating == 0
        )
        snapshot = ManagedPoolSnapshot(
            pool.pool_id, identity, target, observed, ready, terminating, settled
        )
        return _Read(snapshot, adapter)

    def _read_graph(
        self, pool: PoolAllocationPolicy, name: str, deployment: object
    ) -> _Graph:
        """Validate that the DGD has exactly the supported single-worker shape."""
        if not isinstance(deployment, dict) or not all(
            isinstance(deployment.get(field), dict) for field in ("metadata", "spec")
        ):
            raise ManagedCapacityError("malformed DGD metadata or specification")
        metadata = deployment["metadata"]
        uid = _text(metadata.get("uid"), "DGD UID")
        if (
            metadata.get("name") != name
            or metadata.get("namespace") != self._namespace
            or metadata.get("deletionTimestamp")
        ):
            raise ManagedCapacityError("DGD identity changed or is being deleted")
        spec = deployment["spec"]
        components = spec.get("components") or []
        if not isinstance(components, list) or not all(
            isinstance(component, dict) for component in components
        ):
            raise ManagedCapacityError("malformed DGD components")
        by_name = {component["name"]: component for component in components}
        if len(by_name) != len(components) or pool.component_name not in by_name:
            raise ManagedCapacityError("ambiguous or missing worker component")
        component = by_name[pool.component_name]
        if component.get("type") != pool.role:
            raise ManagedCapacityError("worker role does not match pool policy")
        for candidate in components:
            _check_component_shape(pool, candidate, is_worker=candidate is component)
        if not isinstance(component.get("scalingAdapter"), dict):
            raise ManagedCapacityError("worker must declare scalingAdapter")

        annotations = metadata.get("annotations") or {}
        if annotations.get(_PROVIDER) != "component" or spec.get("providerOverride"):
            raise ManagedCapacityError(
                "only the component workload provider is supported"
            )
        status = deployment.get("status") or {}
        if (status.get("rollingUpdate") or {}).get("phase") not in _IDLE_PHASES:
            raise ManagedCapacityError("concurrent worker rollouts are unsupported")
        if (status.get("restart") or {}).get("phase") not in _IDLE_PHASES:
            raise ManagedCapacityError("concurrent graph restarts are unsupported")
        if str(status.get("state", "")).lower() == "failed":
            raise ManagedCapacityError("DGD is failed")
        component_status = (status.get("components") or {}).get(
            pool.component_name
        ) or {}
        resource_names = component_status.get("componentNames") or []
        if (
            component_status.get("componentKind") != "Deployment"
            or len(resource_names) != 1
        ):
            raise ManagedCapacityError(
                "worker must have one standard Deployment backing resource"
            )
        generation = _integer(metadata.get("generation"), "DGD generation")
        observed = _integer(status.get("observedGeneration"), "DGD observedGeneration")
        generation_current = observed == generation
        # After enrollment, generation lag is expected while a scale write is
        # reconciled; it only blocks settlement.
        if not generation_current and pool.pool_id not in self._identities:
            raise ManagedCapacityError("initial GPU shape is not current")
        if (
            _integer(component_status.get("gpusPerReplica"), "projected GPU cost")
            != pool.gpu_per_replica
        ):
            raise ManagedCapacityError("projected GPU cost does not match pool policy")
        return _Graph(
            uid=uid,
            spec=spec,
            component_names=frozenset(by_name),
            component=component,
            annotations=annotations,
            status=status,
            component_status=component_status,
            resource_name=resource_names[0],
            generation=generation,
            generation_current=generation_current,
        )

    def _read_adapter(
        self,
        pool: PoolAllocationPolicy,
        name: str,
        deployment: dict,
        *,
        check_fence: bool,
    ) -> dict:
        """Read the worker's scaling adapter and verify ownership and fencing."""
        assert self._api is not None
        adapter = self._get(
            DGDSA_PLURAL, KubernetesAPI.scaling_adapter_name(name, pool.component_name)
        )
        rejection = self._api.scaling_adapter_identity_rejection(
            deployment, pool.component_name, adapter, require_resource_version=True
        )
        if rejection:
            raise ManagedCapacityError(rejection)
        metadata = adapter["metadata"]
        _text(metadata.get("uid"), "DGDSA UID")
        if metadata.get("namespace") != self._namespace:
            raise ManagedCapacityError("DGDSA namespace differs from pool policy")
        fence = (metadata.get("annotations") or {}).get(PLANNER_WRITER_FENCE_ANNOTATION)
        if check_fence and fence != self._writer_id:
            raise ManagedCapacityError(
                "DGDSA writer fence belongs to another controller"
            )
        return adapter

    def _check_identity(
        self, pool: PoolAllocationPolicy, graph: _Graph, adapter: dict
    ) -> str:
        """Fingerprint everything but the replica count; it must never change."""
        if not graph.worker_hashes:
            raise ManagedCapacityError("current worker generation is unavailable")
        spec = copy.deepcopy(graph.spec)
        for item in spec["components"]:
            if item["name"] == pool.component_name:
                item.pop("replicas", None)
        fingerprint = hashlib.sha256(
            json.dumps(
                [
                    spec,
                    sorted(graph.worker_hashes),
                    [graph.resource_name],
                    graph.annotations.get(_PROVIDER),
                ],
                sort_keys=True,
            ).encode()
        ).hexdigest()
        identity = f"{graph.uid}:{adapter['metadata']['uid']}:{fingerprint}"
        previous = self._identities.setdefault(pool.pool_id, identity)
        if identity != previous:
            raise ManagedCapacityError(
                "pool UID, worker generation, or non-replica configuration changed"
            )
        return identity

    def _count_workers(
        self, pool: PoolAllocationPolicy, name: str, graph: _Graph
    ) -> tuple[int, int, int]:
        """Return (present, ready current-generation, terminating) worker Pods.

        Every present worker Pod is charged, including old generations and
        terminating or failed Pods, so release is never inferred early.
        """
        assert self._api is not None
        pod_list = self._api.core_api.list_namespaced_pod(
            namespace=self._namespace,
            label_selector=f"{DYNAMO_DGD_NAME_LABEL}={name}",
            limit=_MAX_PODS_PER_GRAPH,
            _request_timeout=_HTTP_TIMEOUT,
        )
        if pod_list.metadata is None or pod_list.items is None:
            raise ManagedCapacityError("incomplete worker Pod list")
        if pod_list.metadata._continue:
            raise ManagedCapacityError("worker Pod list is truncated")
        present = ready = terminating = 0
        seen_uids: set[str] = set()
        for pod in (item.to_dict() for item in pod_list.items):
            if not all(
                isinstance(pod.get(field), dict) for field in ("metadata", "spec")
            ):
                raise ManagedCapacityError(
                    "malformed worker Pod metadata or specification"
                )
            metadata = pod["metadata"]
            uid = _text(metadata.get("uid"), "Pod UID")
            labels = metadata.get("labels") or {}
            if (
                uid in seen_uids
                or metadata.get("namespace") != self._namespace
                or labels.get(DYNAMO_DGD_NAME_LABEL) != name
            ):
                raise ManagedCapacityError("ambiguous Pod identity")
            seen_uids.add(uid)
            pod_component = labels.get(DYNAMO_COMPONENT_LABEL)
            if pod_component not in graph.component_names:
                raise ManagedCapacityError("Pod has unknown component ownership")
            cost = _gpu_cost(pod["spec"])
            if pod_component != pool.component_name:
                if cost:
                    raise ManagedCapacityError("unmanaged component has GPU holdings")
                continue
            if cost != pool.gpu_per_replica:
                raise ManagedCapacityError(
                    "actual Pod GPU cost differs from pool policy"
                )
            owner = _replica_set_owner(metadata)
            current = labels.get(_WORKER_HASH) in graph.worker_hashes and owner.get(
                "name", ""
            ).startswith(graph.resource_name + "-")
            deleting = bool(metadata.get("deletion_timestamp"))
            present += 1
            terminating += deleting
            ready += current and not deleting and _pod_ready(pod)
        return present, ready, terminating


@dataclass(frozen=True)
class _Graph:
    """A validated DGD view of one pool's worker component."""

    uid: str
    spec: dict
    component_names: frozenset[str]
    component: dict
    annotations: dict
    status: dict
    component_status: dict
    resource_name: str
    generation: int
    generation_current: bool

    @property
    def worker_hashes(self) -> frozenset[str]:
        return frozenset(
            self.annotations[key]
            for key in _CURRENT_HASHES
            if self.annotations.get(key)
        )


def _check_component_shape(
    pool: PoolAllocationPolicy, component: dict, *, is_worker: bool
) -> None:
    if not is_worker and component.get("type") not in ("frontend", "planner"):
        raise ManagedCapacityError("only one worker component per DGD is supported")
    if any(
        component.get(field)
        for field in ("multinode", "roles", "experimental", "providerOverride")
    ):
        raise ManagedCapacityError(
            "multinode, multi-role and experimental providers are unsupported"
        )
    cost = _gpu_cost((component.get("podTemplate") or {}).get("spec") or {})
    if cost != (pool.gpu_per_replica if is_worker else 0):
        raise ManagedCapacityError(
            "declared worker GPU cost does not match pool policy"
        )


def _replica_set_owner(metadata: dict) -> dict:
    owners = metadata.get("owner_references") or []
    if (
        len(owners) != 1
        or owners[0].get("kind") != "ReplicaSet"
        or owners[0].get("api_version") != "apps/v1"
        or owners[0].get("controller") is not True
        or not owners[0].get("uid")
    ):
        raise ManagedCapacityError("worker Pod has unknown workload ownership")
    return owners[0]


def _pod_ready(pod: dict) -> bool:
    status = pod.get("status") or {}
    return status.get("phase") == "Running" and any(
        condition.get("type") == "Ready" and condition.get("status") == "True"
        for condition in status.get("conditions") or []
    )
