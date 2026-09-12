# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise provider status contracts and asynchronous startup cancellation."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from threading import Event, get_ident
from unittest.mock import Mock

import pytest
from kubernetes import client

from dynamo.planner.config.defaults import SubComponentType, TargetReplica
from dynamo.planner.connectors.clients.kubernetes_api import (
    GROVE_PCSG_REPLICA_INDEX_LABEL,
    KubernetesAPI,
)
from dynamo.planner.tests.unit.test_pending_startup_scaling import (
    _connector,
    _deployment,
    _pods,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def _decode_target(replicas):
    return [
        TargetReplica(
            sub_component_type=SubComponentType.DECODE, desired_replicas=replicas
        )
    ]


def _pcsg(deployment, pods):
    status = deployment["status"]["components"]["d"]
    status["componentKind"] = "PodCliqueScalingGroup"
    status["availableReplicas"] = status.pop("readyReplicas")
    status["updatedReplicas"] = status["availableReplicas"]
    for index, pod in enumerate(pods[1:]):
        pod.metadata.labels[GROVE_PCSG_REPLICA_INDEX_LABEL] = str(index)


def test_scale_target_reads_dgdsa_before_dgd_propagation():
    api = KubernetesAPI.__new__(KubernetesAPI)
    api.current_namespace = "test"
    api.custom_api = Mock()
    api.custom_api.get_namespaced_custom_object_scale.return_value = {
        "spec": {"replicas": 3}
    }
    assert api.get_service_replica_target("qwen", "Decode") == 3
    assert (
        api.custom_api.get_namespaced_custom_object_scale.call_args.kwargs["name"]
        == "qwen-decode"
    )


@pytest.mark.parametrize("status", [404, 403, 500])
def test_scale_target_only_falls_back_on_missing_adapter(status):
    api = KubernetesAPI.__new__(KubernetesAPI)
    api.current_namespace = "test"
    api.custom_api = Mock()
    api.custom_api.get_namespaced_custom_object_scale.side_effect = client.ApiException(
        status=status
    )
    api.get_graph_deployment = Mock(return_value=_deployment())
    if status == 404:
        assert api.get_service_replica_target("qwen", "d") == 2
    else:
        with pytest.raises(client.ApiException):
            api.get_service_replica_target("qwen", "d")
        api.get_graph_deployment.assert_not_called()


@pytest.mark.asyncio
async def test_pcsg_unavailable_replica_can_be_cancelled():
    deployment, pods = _deployment(), _pods()
    _pcsg(deployment, pods)
    connector = _connector(deployment, pods)
    inventory = await connector.get_worker_inventory("p", "d")
    assert inventory.ready_num_decode == 1
    assert inventory.pending_num_decode == 1
    await connector.set_component_replicas(_decode_target(1), blocking=False)
    connector.kube_api.update_graph_replicas.assert_called_once_with("qwen", "d", 1)


@pytest.mark.parametrize("unsafe", ["old_revision", "surplus_group", "missing_index"])
def test_pcsg_startup_does_not_hide_old_revisions_or_extra_groups(unsafe):
    deployment, pods = _deployment(), _pods()
    _pcsg(deployment, pods)
    if unsafe == "old_revision":
        deployment["status"]["components"]["d"]["updatedReplicas"] = 0
    elif unsafe == "surplus_group":
        pods[2].metadata.labels[GROVE_PCSG_REPLICA_INDEX_LABEL] = "2"
    else:
        pods[2].metadata.labels.pop(GROVE_PCSG_REPLICA_INDEX_LABEL)
    connector = _connector(deployment, pods)
    assert not connector._get_worker_inventory_sync("p", "d").startup_in_progress


@pytest.mark.asyncio
async def test_superseding_adapter_target_releases_old_latch_after_propagation():
    deployment, pods = _deployment(), _pods()
    connector = _connector(deployment, pods)
    await connector.set_component_replicas(_decode_target(1), blocking=False)
    connector.kube_api.get_service_replica_target = Mock(return_value=3)

    # The new DGDSA target has not reached the DGD. Even an unrelated observed
    # generation and a newly Ready second worker cannot release the latch yet.
    deployment["metadata"]["generation"] = 3
    deployment["status"]["observedGeneration"] = 3
    deployment["status"]["components"]["d"]["readyReplicas"] = 2
    deployment["status"]["conditions"][0]["status"] = "True"
    pods[2].status.phase = "Running"
    assert (await connector.get_worker_inventory("p", "d")).decode_scaling_in_progress
    await connector.set_component_replicas(_decode_target(1), blocking=False)
    connector.kube_api.update_graph_replicas.assert_called_once()

    # The replacement target reaches both spec and status; no old request may
    # keep a completely stable deployment in observing-only mode.
    deployment["spec"]["components"][1]["replicas"] = 3
    deployment["status"]["components"]["d"] = {
        "replicas": 3,
        "updatedReplicas": 3,
        "readyReplicas": 3,
    }
    pods.append(deepcopy(pods[2]))
    pods[3].metadata.name = "d2"
    assert not (
        await connector.get_worker_inventory("p", "d")
    ).decode_scaling_in_progress
    assert not connector._startup_scale_down_targets


@pytest.mark.asyncio
async def test_unrelated_generation_does_not_release_unapplied_scale_request():
    deployment, pods = _deployment(), _pods()
    connector = _connector(deployment, pods)
    await connector.set_component_replicas(_decode_target(1), blocking=False)
    deployment["metadata"]["generation"] = 3
    deployment["status"]["observedGeneration"] = 3
    deployment["status"]["conditions"][0]["status"] = "True"
    deployment["status"]["components"]["d"]["readyReplicas"] = 2
    pods[2].status.phase = "Running"
    assert (await connector.get_worker_inventory("p", "d")).decode_scaling_in_progress
    assert connector._startup_scale_down_targets == {"d": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize("pcsg", [False, True])
async def test_survivor_readiness_loss_after_drain_can_be_cancelled(pcsg):
    deployment, pods = _deployment(), _pods()
    deployment["spec"]["components"][1]["replicas"] = 3
    deployment["status"]["components"]["d"] = {
        "replicas": 3,
        "updatedReplicas": 3,
        "readyReplicas": 2,
    }
    pods.append(deepcopy(pods[2]))
    pods[3].metadata.name = "d2"
    pods[2].status.phase = "Running"
    if pcsg:
        _pcsg(deployment, pods)
    connector = _connector(deployment, pods)
    await connector.set_component_replicas(_decode_target(2), blocking=False)

    deployment["spec"]["components"][1]["replicas"] = 2
    deployment["status"]["components"]["d"] = {
        "replicas": 2,
        "updatedReplicas": 2,
        "readyReplicas": 1,
    }
    if pcsg:
        _pcsg(deployment, pods)
    # PCSG status.replicas alone must not hide the surplus group's live Pod.
    if pcsg:
        assert not (await connector.get_worker_inventory("p", "d")).startup_in_progress
    pods[3].metadata.deletion_timestamp = datetime.now(timezone.utc)
    assert not (await connector.get_worker_inventory("p", "d")).startup_in_progress
    pods.pop()
    inventory = await connector.get_worker_inventory("p", "d")
    assert inventory.pending_num_decode == 1
    assert not connector._startup_scale_down_targets
    await connector.set_component_replicas(_decode_target(1), blocking=False)
    assert connector.kube_api.update_graph_replicas.call_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "unsafe", ["drain", "unobserved", "rollout", "old_revision", "surplus_pcsg"]
)
async def test_stale_ready_condition_cannot_bypass_reduction_guards(unsafe):
    deployment, pods = _deployment(), _pods()
    deployment["status"]["conditions"][0]["status"] = "True"
    if unsafe == "drain":
        pods[2].metadata.deletion_timestamp = datetime.now(timezone.utc)
    elif unsafe == "unobserved":
        deployment["metadata"]["generation"] += 1
    elif unsafe == "rollout":
        deployment["status"]["rollingUpdate"] = {"phase": "InProgress"}
    elif unsafe == "old_revision":
        deployment["status"]["components"]["d"]["updatedReplicas"] = 1
    else:
        _pcsg(deployment, pods)
        pods[2].metadata.labels[GROVE_PCSG_REPLICA_INDEX_LABEL] = "2"
    connector = _connector(deployment, pods)
    await connector.set_component_replicas(_decode_target(1), blocking=False)
    connector.kube_api.update_graph_replicas.assert_not_called()


@pytest.mark.asyncio
async def test_startup_reduction_is_latched_even_with_stale_ready_condition():
    deployment, pods = _deployment(), _pods()
    deployment["status"]["conditions"][0]["status"] = "True"
    connector = _connector(deployment, pods)
    await connector.set_component_replicas(_decode_target(1), blocking=False)
    assert connector._startup_scale_down_targets == {"d": 1}
    await connector.set_component_replicas(_decode_target(1), blocking=False)
    connector.kube_api.update_graph_replicas.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["Failed", "Succeeded", "Running"])
async def test_checkpoint_capture_pod_does_not_block_serving_worker_cancellation(phase):
    deployment, pods = _deployment(), _pods()
    capture = deepcopy(pods[2])
    capture.metadata.name = "capture-abc"
    capture.metadata.labels.update(
        {"nvidia.com/snapshot-job": "capture", "nvidia.com/snapshot-job-uid": "uid"}
    )
    capture.metadata.owner_references = [
        client.V1OwnerReference(
            api_version="batch/v1",
            kind="Job",
            name="capture",
            uid="job",
            controller=True,
        )
    ]
    capture.status.phase = phase
    capture.metadata.deletion_timestamp = datetime.now(timezone.utc)
    pods.append(capture)
    connector = _connector(deployment, pods)
    assert (await connector.get_worker_inventory("p", "d")).pending_num_decode == 1
    await connector.set_component_replicas(_decode_target(1), blocking=False)
    connector.kube_api.update_graph_replicas.assert_called_once_with("qwen", "d", 1)


def test_checkpoint_filter_does_not_hide_failed_worker_with_inherited_labels():
    deployment, pods = _deployment(), _pods()
    pods[2].status.phase = "Failed"
    pods[2].metadata.labels.update(
        {"nvidia.com/snapshot-job": "capture", "nvidia.com/snapshot-job-uid": "uid"}
    )
    pods[2].metadata.owner_references = [
        client.V1OwnerReference(
            api_version="grove.io/v1alpha1",
            kind="PodClique",
            name="decode",
            uid="clique",
            controller=True,
        )
    ]
    connector = _connector(deployment, pods)
    assert not connector._get_worker_inventory_sync("p", "d").startup_in_progress


@pytest.mark.asyncio
async def test_inventory_retirement_cannot_erase_concurrent_scale_down():
    deployment, pods = _deployment(), _pods()
    connector = _connector(deployment, pods)
    connector._startup_scale_down_targets = {"p": 1}
    status = connector.kube_api.get_service_replica_status
    inspecting_old_target, resume = Event(), Event()
    main_thread = get_ident()
    worker_calls = 0

    def interleaved_status(snapshot, name):
        nonlocal worker_calls
        if get_ident() != main_thread:
            worker_calls += 1
            # Two status reads classify startup; the third inspects the old
            # completed p=1 target, just before it is retired.
            if worker_calls == 3:
                inspecting_old_target.set()
                assert resume.wait(5)
        return status(snapshot, name)

    connector.kube_api.get_service_replica_status = interleaved_status
    with ThreadPoolExecutor(max_workers=1) as executor:
        retirement = executor.submit(
            connector._startup_scale_down_in_progress, deployment, pods
        )
        try:
            assert await asyncio.to_thread(inspecting_old_target.wait, 5)
            await connector.set_component_replicas(
                [
                    TargetReplica(
                        sub_component_type=SubComponentType.DECODE, desired_replicas=1
                    )
                ],
                blocking=False,
            )
        finally:
            resume.set()
        assert await asyncio.wrap_future(retirement)

    # The stale inventory observed p=1 settled, but d=1 was accepted meanwhile.
    # It must remain held until the new reduction actually converges.
    assert connector._startup_scale_down_targets == {"d": 1}
    connector.kube_api.update_graph_replicas.assert_called_once_with("qwen", "d", 1)
