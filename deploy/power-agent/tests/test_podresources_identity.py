# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import podresources_api
import pytest
from podresources_identity import allocation_id, find_pod_gpu_allocation


def _response(device_ids):
    return podresources_api.ListPodResourcesResponse(
        pod_resources=[
            podresources_api.PodResources(
                namespace="dynamo",
                name="worker-0",
                containers=[
                    podresources_api.ContainerResources(
                        name="main",
                        devices=[
                            podresources_api.ContainerDevices(
                                resource_name="example.com/other",
                                device_ids=["ignored"],
                            ),
                            podresources_api.ContainerDevices(
                                resource_name="nvidia.com/gpu",
                                device_ids=device_ids,
                            ),
                        ],
                    )
                ],
            )
        ]
    )


def test_allocation_id_uses_pod_container_and_ordered_uuid_set():
    allocation = find_pod_gpu_allocation(
        _response(["GPU-b", "GPU-a"]), "dynamo", "worker-0", "main"
    )

    assert allocation is not None
    assert allocation.gpu_uuids == ("GPU-a", "GPU-b")
    assert (
        allocation_id("pod-uid", allocation.container_name, allocation.gpu_uuids)
        == "pod-uid/main/GPU-a,GPU-b"
    )


def test_uuid_set_change_produces_new_allocation_id():
    first = allocation_id("pod-uid", "main", ["GPU-a", "GPU-b"])
    second = allocation_id("pod-uid", "main", ["GPU-a", "GPU-c"])

    assert first != second


def test_only_exact_pod_namespace_container_and_nvidia_resource_match():
    response = _response(["GPU-a"])

    assert find_pod_gpu_allocation(response, "other", "worker-0", "main") is None
    assert find_pod_gpu_allocation(response, "dynamo", "other", "main") is None
    assert find_pod_gpu_allocation(response, "dynamo", "worker-0", "sidecar") is None


def test_duplicate_or_empty_uuid_sets_are_rejected():
    with pytest.raises(ValueError, match="duplicates"):
        find_pod_gpu_allocation(
            _response(["GPU-a", "GPU-a"]), "dynamo", "worker-0", "main"
        )
    with pytest.raises(ValueError, match="nonempty"):
        allocation_id("pod-uid", "main", [])
