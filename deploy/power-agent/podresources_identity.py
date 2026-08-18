# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kubelet PodResources identity for transactionally managed GPU Pods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

try:
    import grpc
    import podresources_api
    import podresources_api_grpc
except ImportError:
    grpc = None  # type: ignore
    podresources_api = None  # type: ignore
    podresources_api_grpc = None  # type: ignore


POD_RESOURCES_SOCKET = "/var/lib/kubelet/pod-resources/kubelet.sock"
NVIDIA_GPU_RESOURCE = "nvidia.com/gpu"
POD_RESOURCES_TIMEOUT_S = 5


@dataclass(frozen=True)
class PodGPUAllocation:
    namespace: str
    pod_name: str
    container_name: str
    gpu_uuids: tuple[str, ...]


def canonical_gpu_uuids(device_ids: Iterable[str]) -> tuple[str, ...]:
    """Return a nonempty, sorted, duplicate-free physical GPU UUID set."""
    uuids = tuple(sorted(device_id.strip() for device_id in device_ids))
    if not uuids or any(not uuid for uuid in uuids):
        raise ValueError("PodResources GPU UUID set must be nonempty")
    if len(set(uuids)) != len(uuids):
        raise ValueError("PodResources GPU UUID set contains duplicates")
    return uuids


def allocation_id(
    pod_uid: str,
    container_name: str,
    gpu_uuids: Iterable[str],
) -> str:
    """Bind one Pod/container to its ordered PodResources GPU UUID set."""
    if not pod_uid or not container_name:
        raise ValueError("pod UID and container name are required")
    ordered = canonical_gpu_uuids(gpu_uuids)
    return f"{pod_uid}/{container_name}/{','.join(ordered)}"


def find_pod_gpu_allocation(
    response,
    namespace: str,
    pod_name: str,
    container_name: str,
) -> Optional[PodGPUAllocation]:
    """Extract exactly one classic device-plugin GPU allocation."""
    matches: list[PodGPUAllocation] = []
    for pod in response.pod_resources:
        if pod.namespace != namespace or pod.name != pod_name:
            continue
        for container in pod.containers:
            if container.name != container_name:
                continue
            device_ids: list[str] = []
            for device in container.devices:
                if device.resource_name == NVIDIA_GPU_RESOURCE:
                    device_ids.extend(device.device_ids)
            if device_ids:
                matches.append(
                    PodGPUAllocation(
                        namespace=namespace,
                        pod_name=pod_name,
                        container_name=container_name,
                        gpu_uuids=canonical_gpu_uuids(device_ids),
                    )
                )
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError("PodResources returned duplicate Pod/container allocations")
    return matches[0]


class PodResourcesClient:
    """Small synchronous client for kubelet's node-local PodResources API."""

    def __init__(
        self,
        socket_path: str = POD_RESOURCES_SOCKET,
        timeout_s: int = POD_RESOURCES_TIMEOUT_S,
    ) -> None:
        if grpc is None or podresources_api is None or podresources_api_grpc is None:
            raise RuntimeError(
                "grpcio and generated PodResources bindings are required"
            )
        self._timeout_s = timeout_s
        self._channel = grpc.insecure_channel(f"unix://{socket_path}")
        self._stub = podresources_api_grpc.PodResourcesListerStub(self._channel)

    def list(self):
        return self._stub.List(
            podresources_api.ListPodResourcesRequest(), timeout=self._timeout_s
        )

    def close(self) -> None:
        self._channel.close()
