# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kubernetes HTTP-boundary tests of fenced scale and conservative release."""

import asyncio
import copy
import threading
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from kubernetes import client
from urllib3.exceptions import HTTPError

from dynamo.global_planner.allocation_policy import AllocationPolicy
from dynamo.global_planner.managed_capacity import ManagedCapacityError
from dynamo.global_planner.managed_kubernetes import ManagedKubernetesCapacity
from dynamo.planner.connectors.clients.kubernetes_api import (
    DGD_PLURAL,
    DGDSA_PLURAL,
    DYNAMO_COMPONENT_LABEL,
    DYNAMO_DGD_NAME_LABEL,
    PLANNER_WRITER_FENCE_ANNOTATION,
    KubernetesAPI,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.planner,
]


def _pod(graph: str, index: int) -> client.V1Pod:
    return client.V1Pod(
        metadata=client.V1ObjectMeta(
            name=f"{graph}-worker-generation-rs-{index}",
            namespace="prod",
            uid=f"{graph}-pod-{index}",
            labels={
                DYNAMO_DGD_NAME_LABEL: graph,
                DYNAMO_COMPONENT_LABEL: "Worker",
                "nvidia.com/dynamo-worker-hash": "generation",
            },
            owner_references=[
                client.V1OwnerReference(
                    api_version="apps/v1",
                    kind="ReplicaSet",
                    name=f"{graph}-worker-generation-rs",
                    uid=f"{graph}-rs",
                    controller=True,
                )
            ],
        ),
        spec=client.V1PodSpec(
            containers=[
                client.V1Container(
                    name="main",
                    resources=client.V1ResourceRequirements(
                        requests={"nvidia.com/gpu": "4"}, limits={"nvidia.com/gpu": "4"}
                    ),
                )
            ]
        ),
        status=client.V1PodStatus(
            phase="Running",
            conditions=[client.V1PodCondition(type="Ready", status="True")],
        ),
    )


class FakeKubernetes(KubernetesAPI):
    """Fake only generated HTTP methods; retain production identity helpers."""

    def __init__(self) -> None:
        self.deployments: dict[str, dict] = {}
        self.adapters: dict[str, dict] = {}
        self.pods: dict[str, list[client.V1Pod]] = {}
        self.calls: list[tuple[str, dict, int]] = []
        self.truncated = False
        self.before_scale = None
        for graph in ("interactive", "batch"):
            self.deployments[graph] = {
                "metadata": {
                    "name": graph,
                    "namespace": "prod",
                    "uid": f"{graph}-uid",
                    "generation": 1,
                    "annotations": {
                        "nvidia.com/workload-provider": "component",
                        "nvidia.com/current-worker-hash": "generation",
                    },
                },
                "spec": {
                    "components": [
                        {
                            "name": "Worker",
                            "type": "decode",
                            "replicas": 2,
                            "scalingAdapter": {},
                            "podTemplate": {
                                "spec": {
                                    "containers": [
                                        {
                                            "name": "main",
                                            "resources": {
                                                "requests": {"nvidia.com/gpu": "4"},
                                                "limits": {"nvidia.com/gpu": "4"},
                                            },
                                        }
                                    ]
                                }
                            },
                        }
                    ]
                },
                "status": {
                    "state": "successful",
                    "observedGeneration": 1,
                    "conditions": [
                        {"type": "Ready", "status": "True", "observedGeneration": 1}
                    ],
                    "components": {
                        "Worker": {
                            "componentKind": "Deployment",
                            "componentNames": [f"{graph}-worker-generation"],
                            "gpusPerReplica": 4,
                            "gpusPerEngine": 4,
                            "replicas": 2,
                            "updatedReplicas": 2,
                            "readyReplicas": 2,
                        }
                    },
                },
            }
            self.adapters[f"{graph}-worker"] = {
                "metadata": {
                    "name": f"{graph}-worker",
                    "namespace": "prod",
                    "uid": f"{graph}-adapter",
                    "resourceVersion": "1",
                    "ownerReferences": [
                        {
                            "apiVersion": "nvidia.com/v1beta1",
                            "kind": "DynamoGraphDeployment",
                            "name": graph,
                            "uid": f"{graph}-uid",
                            "controller": True,
                        }
                    ],
                },
                "spec": {
                    "dgdRef": {"name": graph, "componentName": "Worker"},
                    "replicas": 2,
                },
            }
            self.pods[graph] = [_pod(graph, 0), _pod(graph, 1)]
        self.custom_api = MagicMock()
        self.custom_api.get_namespaced_custom_object.side_effect = self._get
        self.custom_api.patch_namespaced_custom_object.side_effect = self._claim
        self.custom_api.patch_namespaced_custom_object_scale.side_effect = self._scale
        self.core_api = MagicMock()
        self.core_api.list_namespaced_pod.side_effect = self._list

    def _record(self, operation: str, kwargs: dict) -> None:
        assert kwargs["namespace"] == "prod"
        assert kwargs["_request_timeout"] == (3, 10)
        self.calls.append((operation, copy.deepcopy(kwargs), threading.get_ident()))

    def _get(self, **kwargs) -> dict:
        self._record("get", kwargs)
        objects = self.deployments if kwargs["plural"] == DGD_PLURAL else self.adapters
        return copy.deepcopy(objects[kwargs["name"]])

    def _cas(self, kwargs: dict) -> dict:
        assert kwargs["plural"] == DGDSA_PLURAL
        adapter = self.adapters[kwargs["name"]]
        metadata = kwargs["body"]["metadata"]
        if (
            metadata["resourceVersion"] != adapter["metadata"]["resourceVersion"]
            or metadata["uid"] != adapter["metadata"]["uid"]
        ):
            raise client.ApiException(status=409, reason="CAS conflict")
        adapter["metadata"]["resourceVersion"] = str(
            int(adapter["metadata"]["resourceVersion"]) + 1
        )
        return adapter

    def _claim(self, **kwargs) -> dict:
        self._record("claim", kwargs)
        adapter = self._cas(kwargs)
        adapter["metadata"].setdefault("annotations", {}).update(
            kwargs["body"]["metadata"]["annotations"]
        )
        return copy.deepcopy(adapter)

    def _scale(self, **kwargs) -> dict:
        self._record("scale", kwargs)
        if self.before_scale:
            self.before_scale()
        adapter = self._cas(kwargs)
        adapter["spec"]["replicas"] = kwargs["body"]["spec"]["replicas"]
        return copy.deepcopy(adapter)

    def _list(self, **kwargs) -> client.V1PodList:
        self._record("list", kwargs)
        assert kwargs["limit"] == 1000
        graph = kwargs["label_selector"].split("=", 1)[1]
        return client.V1PodList(
            metadata=client.V1ListMeta(_continue="next" if self.truncated else None),
            items=copy.deepcopy(self.pods[graph]),
        )

    def reconcile(self, graph: str, replicas: int, *, ready: bool = True) -> None:
        deployment = self.deployments[graph]
        deployment["spec"]["components"][0]["replicas"] = replicas
        deployment["metadata"]["generation"] += 1
        deployment["status"]["observedGeneration"] = deployment["metadata"][
            "generation"
        ]
        deployment["status"]["conditions"][0]["observedGeneration"] = deployment[
            "metadata"
        ]["generation"]
        for field in ("replicas", "updatedReplicas", "readyReplicas"):
            deployment["status"]["components"]["Worker"][field] = (
                replicas if ready else 0
            )


@pytest.fixture
def backend():
    policy = AllocationPolicy.model_validate(
        {
            "min_total_gpus": 4,
            "max_total_gpus": 16,
            "pools": [
                {
                    "pool_id": name,
                    "participant_id": f"prod/{name}",
                    "component_name": "Worker",
                    "role": "decode",
                    "gpu_per_replica": 4,
                    "min_gpus": 4,
                    "max_gpus": 16,
                    "priority": priority,
                }
                for name, priority in (("interactive", 100), ("batch", 10))
            ],
        }
    )
    api = FakeKubernetes()
    return ManagedKubernetesCapacity(policy, "prod", kube_api=api), api


@pytest.mark.asyncio
async def test_release_is_not_inferred_from_a_lower_target_or_ready_count(backend):
    capacity, api = backend
    await capacity.acquire("writer")
    initial = await capacity.observe()
    assert {key: value.observed_replicas * 4 for key, value in initial.items()} == {
        "interactive": 8,
        "batch": 8,
    }
    await capacity.scale(
        "batch", 1, expected_identity=initial["batch"].identity, expected_target=2
    )
    snapshot = (await capacity.observe())["batch"]
    assert snapshot.target_replicas == 1
    assert snapshot.observed_replicas == 2
    assert not snapshot.settled

    api.reconcile("batch", 1)
    draining = api.pods["batch"][1]
    draining.metadata.deletion_timestamp = datetime.now(timezone.utc)
    draining.status.conditions[0].status = "False"
    snapshot = (await capacity.observe())["batch"]
    assert (
        snapshot.observed_replicas,
        snapshot.ready_replicas,
        snapshot.terminating_replicas,
    ) == (2, 1, 1)
    assert not snapshot.settled

    api.pods["batch"].pop()
    snapshot = (await capacity.observe())["batch"]
    assert snapshot.settled
    await capacity.scale(
        "interactive",
        3,
        expected_identity=initial["interactive"].identity,
        expected_target=2,
    )
    api.reconcile("interactive", 3)
    api.pods["interactive"].append(_pod("interactive", 2))
    final = await capacity.observe()
    assert all(item.settled for item in final.values())
    assert {key: value.observed_replicas * 4 for key, value in final.items()} == {
        "interactive": 12,
        "batch": 4,
    }
    assert all(thread_id != threading.get_ident() for _, _, thread_id in api.calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["Pending", "Unknown", "Failed", "Succeeded", None])
async def test_present_nonready_pods_remain_charged(backend, phase):
    capacity, api = backend
    await capacity.acquire("writer")
    api.pods["batch"][1].status.phase = phase
    snapshot = (await capacity.observe())["batch"]
    assert snapshot.observed_replicas == 2
    assert snapshot.ready_replicas == 1
    assert not snapshot.settled


@pytest.mark.asyncio
async def test_old_generation_pods_are_included_and_block_settlement(backend):
    capacity, api = backend
    await capacity.acquire("writer")
    api.pods["batch"][1].metadata.labels["nvidia.com/dynamo-worker-hash"] = "previous"
    snapshot = (await capacity.observe())["batch"]
    assert snapshot.observed_replicas == 2
    assert snapshot.ready_replicas == 1
    assert not snapshot.settled


@pytest.mark.asyncio
async def test_expected_scaling_generation_lag_waits_without_losing_identity(backend):
    capacity, api = backend
    await capacity.acquire("writer")
    before = (await capacity.observe())["batch"]
    api.deployments["batch"]["metadata"]["generation"] += 1
    after = (await capacity.observe())["batch"]
    assert after.identity == before.identity
    assert not after.settled
    api.deployments["batch"]["status"]["observedGeneration"] += 1
    api.deployments["batch"]["status"]["conditions"][0]["observedGeneration"] += 1
    assert (await capacity.observe())["batch"].settled


@pytest.mark.asyncio
async def test_component_status_must_catch_up_even_when_pod_count_matches(backend):
    capacity, api = backend
    await capacity.acquire("writer")
    api.deployments["batch"]["status"]["components"]["Worker"]["updatedReplicas"] = 1
    assert not (await capacity.observe())["batch"].settled


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", ["condition_generation", "condition_false", "condition_missing", "state"]
)
async def test_projected_readiness_must_prove_backing_controller_convergence(
    backend, change
):
    capacity, api = backend
    await capacity.acquire("writer")
    status = api.deployments["batch"]["status"]
    if change == "condition_generation":
        status["conditions"][0]["observedGeneration"] = 0
    elif change == "condition_false":
        status["conditions"][0]["status"] = "False"
    elif change == "condition_missing":
        status["conditions"] = []
    else:
        status["state"] = "pending"
    assert not (await capacity.observe())["batch"].settled


@pytest.mark.asyncio
async def test_stale_target_and_identity_cannot_write(backend):
    capacity, api = backend
    await capacity.acquire("writer")
    initial = (await capacity.observe())["batch"]
    for identity, target in (("old", 2), (initial.identity, 3)):
        with pytest.raises(ManagedCapacityError, match="identity or replica target"):
            await capacity.scale(
                "batch", 1, expected_identity=identity, expected_target=target
            )
    api.custom_api.patch_namespaced_custom_object_scale.assert_not_called()


@pytest.mark.asyncio
async def test_new_writer_fences_a_late_old_scale_patch(backend):
    capacity, api = backend
    await capacity.acquire("old")
    initial = (await capacity.observe())["batch"]

    def replace_writer():
        adapter = api.adapters["batch-worker"]
        adapter["metadata"]["resourceVersion"] = str(
            int(adapter["metadata"]["resourceVersion"]) + 1
        )
        adapter["metadata"]["annotations"][PLANNER_WRITER_FENCE_ANNOTATION] = "new"

    api.before_scale = replace_writer
    with pytest.raises(ManagedCapacityError, match="409"):
        await capacity.scale(
            "batch", 1, expected_identity=initial.identity, expected_target=2
        )
    assert api.adapters["batch-worker"]["spec"]["replicas"] == 2
    with pytest.raises(ManagedCapacityError, match="writer fence"):
        await capacity.observe()


@pytest.mark.asyncio
async def test_acquisition_cas_conflict_does_not_allow_observation_or_scaling(backend):
    capacity, api = backend
    api.custom_api.patch_namespaced_custom_object.side_effect = client.ApiException(
        status=409
    )
    with pytest.raises(ManagedCapacityError):
        await capacity.acquire("writer")
    with pytest.raises(ManagedCapacityError, match="not been acquired"):
        await capacity.observe()
    api.custom_api.patch_namespaced_custom_object_scale.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", ["dgd_uid", "adapter_uid", "template", "backing", "worker_hash", "fence"]
)
async def test_changed_enrollment_fails_closed(backend, change):
    capacity, api = backend
    await capacity.acquire("writer")
    graph = api.deployments["batch"]
    adapter = api.adapters["batch-worker"]
    if change == "dgd_uid":
        graph["metadata"]["uid"] = "replacement"
        adapter["metadata"]["ownerReferences"][0]["uid"] = "replacement"
    elif change == "adapter_uid":
        adapter["metadata"]["uid"] = "replacement"
    elif change == "template":
        graph["spec"]["components"][0]["podTemplate"]["spec"]["containers"][0][
            "image"
        ] = "different"
    elif change == "backing":
        graph["status"]["components"]["Worker"]["componentNames"] = ["different"]
    elif change == "worker_hash":
        graph["metadata"]["annotations"]["nvidia.com/current-worker-hash"] = "different"
    else:
        adapter["metadata"]["annotations"][
            PLANNER_WRITER_FENCE_ANNOTATION
        ] = "different"
    with pytest.raises(ManagedCapacityError):
        await capacity.observe()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        "role",
        "cost",
        "projected_cost",
        "multinode",
        "grove",
        "lws",
        "dra",
        "gpu_init",
        "rollout",
        "restart",
        "extra_worker",
        "missing_adapter",
        "initial_generation",
        "missing_hash",
    ],
)
async def test_unsupported_deployments_are_rejected_before_claiming(backend, change):
    capacity, api = backend
    graph = api.deployments["batch"]
    component = graph["spec"]["components"][0]
    pod_spec = component["podTemplate"]["spec"]
    if change == "role":
        component["type"] = "prefill"
    elif change == "cost":
        pod_spec["containers"][0]["resources"]["requests"]["nvidia.com/gpu"] = "8"
    elif change == "projected_cost":
        graph["status"]["components"]["Worker"]["gpusPerReplica"] = 8
    elif change == "multinode":
        component["multinode"] = {"nodeCount": 2}
    elif change == "grove":
        graph["metadata"]["annotations"]["nvidia.com/workload-provider"] = "grove"
    elif change == "lws":
        graph["status"]["components"]["Worker"]["componentKind"] = "LeaderWorkerSet"
    elif change == "dra":
        pod_spec["resourceClaims"] = [{"name": "gpu"}]
    elif change == "gpu_init":
        pod_spec["initContainers"] = [pod_spec["containers"][0]]
    elif change == "rollout":
        graph["status"]["rollingUpdate"] = {"phase": "InProgress"}
    elif change == "restart":
        graph["status"]["restart"] = {"phase": "Restarting"}
    elif change == "extra_worker":
        other = copy.deepcopy(component)
        other["name"] = "Other"
        graph["spec"]["components"].append(other)
    elif change == "missing_adapter":
        del component["scalingAdapter"]
    elif change == "initial_generation":
        graph["metadata"]["generation"] = 2
    else:
        graph["metadata"]["annotations"].pop("nvidia.com/current-worker-hash")
    with pytest.raises(ManagedCapacityError):
        await capacity.acquire("writer")
    api.custom_api.patch_namespaced_custom_object.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    ["uid", "component", "cost", "owner", "namespace", "duplicate", "missing_metadata"],
)
async def test_unknown_or_ambiguous_pods_are_not_ignored(backend, change):
    capacity, api = backend
    await capacity.acquire("writer")
    pod = api.pods["batch"][0]
    if change == "uid":
        pod.metadata.uid = None
    elif change == "component":
        pod.metadata.labels[DYNAMO_COMPONENT_LABEL] = "Unknown"
    elif change == "cost":
        pod.spec.containers[0].resources.requests["nvidia.com/gpu"] = "8"
    elif change == "owner":
        pod.metadata.owner_references = []
    elif change == "namespace":
        pod.metadata.namespace = "other"
    elif change == "missing_metadata":
        pod.metadata = None
    else:
        api.pods["batch"].append(copy.deepcopy(pod))
    with pytest.raises(ManagedCapacityError):
        await capacity.observe()


@pytest.mark.asyncio
async def test_read_errors_and_truncated_lists_never_report_release(backend):
    capacity, api = backend
    await capacity.acquire("writer")
    api.truncated = True
    with pytest.raises(ManagedCapacityError, match="truncated"):
        await capacity.observe()
    api.truncated = False
    api.core_api.list_namespaced_pod.side_effect = client.ApiException(status=503)
    with pytest.raises(ManagedCapacityError):
        await capacity.observe()
    api.custom_api.patch_namespaced_custom_object_scale.assert_not_called()


@pytest.mark.asyncio
async def test_slow_reads_do_not_block_the_event_loop(backend):
    capacity, api = backend
    await capacity.acquire("writer")
    entered = threading.Event()
    release = threading.Event()

    def blocking_list(**kwargs):
        entered.set()
        assert release.wait(timeout=5)
        return api._list(**kwargs)

    api.core_api.list_namespaced_pod.side_effect = blocking_list
    task = asyncio.create_task(capacity.observe())
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        assert not task.done()
        await asyncio.wait_for(asyncio.sleep(0), timeout=1)
    finally:
        release.set()
        await task


@pytest.mark.asyncio
async def test_cancelled_http_call_keeps_lock_until_the_worker_finishes(backend):
    capacity, api = backend
    await capacity.acquire("writer")
    entered = threading.Event()
    release = threading.Event()

    def blocking_list(**kwargs):
        entered.set()
        assert release.wait(timeout=5)
        return api._list(**kwargs)

    api.core_api.list_namespaced_pod.side_effect = blocking_list
    task = asyncio.create_task(capacity.observe())
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        assert capacity._lock.locked()
        assert not task.done()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert not capacity._lock.locked()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        OSError("network unavailable"),
        HTTPError("transport unavailable"),
        KeyError("metadata"),
        ValueError("malformed response"),
    ],
)
async def test_transport_and_malformed_response_failures_use_capacity_error(
    backend, failure
):
    capacity, api = backend
    await capacity.acquire("writer")
    api.custom_api.get_namespaced_custom_object.side_effect = failure
    with pytest.raises(ManagedCapacityError) as caught:
        await capacity.observe()
    assert caught.value.__cause__ is failure


def test_policy_requires_exact_same_namespace_identities(backend):
    capacity, _ = backend
    with pytest.raises(ValueError, match="one namespace"):
        ManagedKubernetesCapacity(capacity._policy, "other")


def test_zero_floor_requires_stronger_backing_observation_than_existing_rbac(backend):
    capacity, _ = backend
    document = capacity._policy.model_dump()
    document["pools"][0]["min_gpus"] = 0
    with pytest.raises(ValueError, match="minimum of one replica"):
        ManagedKubernetesCapacity(AllocationPolicy.model_validate(document), "prod")


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", [False, True])
async def test_growth_rechecks_fresh_settlement_and_headroom(backend, changed):
    capacity, api = backend
    await capacity.acquire("writer")
    if changed:
        api.adapters["batch-worker"]["spec"]["replicas"] = 1
        api.reconcile("batch", 1)
        api.pods["batch"].pop()
    initial = (await capacity.observe())["interactive"]
    if changed:
        # Readiness can change after the controller's last full observation.
        api.pods["batch"][0].status.conditions[0].status = "False"
    with pytest.raises(ManagedCapacityError, match="unsettled|fresh capacity"):
        await capacity.scale(
            "interactive", 3, expected_identity=initial.identity, expected_target=2
        )
    api.custom_api.patch_namespaced_custom_object_scale.assert_not_called()
