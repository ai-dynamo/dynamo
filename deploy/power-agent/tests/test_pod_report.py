# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import stat
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import managed_state
import podresources_api
import power_agent
import pytest
from actuator import ApplyResult
from pod_report import (
    COMPONENT_ENV,
    DGD_UID_ENV,
    EXPECTED_GPU_COUNT_ENV,
    IN_GATE_BOUND_WATTS_ENV,
    REPORT_ANNOTATION_KEY,
    PodReportPatcher,
    PowerGateContext,
    build_report,
    encode_report,
    report_patch_required,
)
from power_agent import POWER_ANNOTATION_KEY, PowerAgent


def _result(uuid="GPU-a", observed_at=None):
    return ApplyResult(
        gpu_uuid=uuid,
        requested_watts=350,
        target_watts=350,
        constraint_min_watts=100,
        constraint_max_watts=700,
        policy_outcome="annotated",
        write_outcome="succeeded",
        readback_outcome="succeeded",
        enforced_cap_watts=350,
        actuator="nvml",
        observed_at=observed_at or datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
    )


def _report(observed_at=None):
    return build_report(
        context=PowerGateContext("dgd-uid", "decode", 1, 350),
        pod_uid="pod-uid",
        node_name="node-a",
        allocation_id="pod-uid/main/GPU-a",
        allocation_gpu_uuids=["GPU-a"],
        results=[_result(observed_at=observed_at)],
    )


def _pod(resource_version="10", report=None):
    annotations = {} if report is None else {REPORT_ANNOTATION_KEY: report}
    return SimpleNamespace(
        metadata=SimpleNamespace(
            name="worker-0",
            namespace="dynamo",
            resource_version=resource_version,
            uid="pod-uid",
            annotations=annotations,
        )
    )


def test_report_fields_match_p20_schema_and_are_bounded():
    report = _report()
    assert set(report) == {
        "version",
        "dgdUID",
        "component",
        "podUID",
        "node",
        "allocationID",
        "gpus",
    }
    assert set(report["gpus"][0]) == {
        "uuid",
        "requestedWatts",
        "targetWatts",
        "constraintMinWatts",
        "constraintMaxWatts",
        "policyOutcome",
        "writeOutcome",
        "readbackOutcome",
        "enforcedCapWatts",
        "actuator",
        "observedAt",
    }
    encoded = encode_report(report)
    assert len(encoded.encode()) < 64 * 1024
    assert json.loads(encoded) == report


def test_atomic_report_requires_every_allocated_gpu_exactly_once():
    context = PowerGateContext("dgd-uid", "decode", 2, 350)
    with pytest.raises(ValueError, match="exactly cover"):
        build_report(
            context=context,
            pod_uid="pod-uid",
            node_name="node-a",
            allocation_id="allocation",
            allocation_gpu_uuids=["GPU-a", "GPU-b"],
            results=[_result("GPU-a")],
        )


def test_conflict_retries_with_latest_resource_version():
    api = MagicMock()
    conflict = RuntimeError("conflict")
    conflict.status = 409
    updated = _pod(resource_version="12", report=encode_report(_report()))
    api.patch_namespaced_pod.side_effect = [conflict, updated]
    api.read_namespaced_pod.return_value = _pod(resource_version="11")

    returned, patched = PodReportPatcher(api).publish(_pod(), _report())

    assert patched
    assert returned is updated
    assert api.patch_namespaced_pod.call_count == 2
    second_body = api.patch_namespaced_pod.call_args_list[1].kwargs["body"]
    assert second_body["metadata"]["resourceVersion"] == "11"
    assert set(second_body["metadata"]["annotations"]) == {REPORT_ANNOTATION_KEY}


def test_fresh_semantic_noop_skips_patch_but_refreshes_on_cadence():
    observed = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    old_encoded = encode_report(_report(observed))
    new_report = _report(observed + timedelta(seconds=10))
    api = MagicMock()

    _, patched = PodReportPatcher(api).publish(
        _pod(report=old_encoded),
        new_report,
        now=observed + timedelta(seconds=20),
    )
    assert not patched
    api.patch_namespaced_pod.assert_not_called()

    assert report_patch_required(
        old_encoded,
        new_report,
        now=observed + timedelta(seconds=30),
    )
    assert all("acknowledg" not in key.lower() for key in json.loads(old_encoded))


def test_malformed_or_future_existing_report_is_repaired():
    observed = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    malformed = json.dumps({**_report(observed), "gpus": ["not-an-object"]})
    assert report_patch_required(malformed, _report(observed), now=observed)

    future = encode_report(_report(observed + timedelta(minutes=5)))
    assert report_patch_required(future, _report(observed), now=observed)


def test_conflict_retry_refuses_same_name_new_uid():
    api = MagicMock()
    conflict = RuntimeError("conflict")
    conflict.status = 409
    api.patch_namespaced_pod.side_effect = conflict
    replacement = _pod(resource_version="11")
    replacement.metadata.uid = "replacement-uid"
    api.read_namespaced_pod.return_value = replacement

    with pytest.raises(RuntimeError, match="UID changed"):
        PodReportPatcher(api).publish(_pod(), _report())

    assert api.patch_namespaced_pod.call_count == 1


def test_transactional_reconcile_uses_podresources_before_any_gpu_process(monkeypatch):
    env = [
        SimpleNamespace(name=DGD_UID_ENV, value="dgd-uid"),
        SimpleNamespace(name=COMPONENT_ENV, value="decode"),
        SimpleNamespace(name=EXPECTED_GPU_COUNT_ENV, value="1"),
        SimpleNamespace(name=IN_GATE_BOUND_WATTS_ENV, value="350"),
    ]
    pod = SimpleNamespace(
        metadata=SimpleNamespace(
            namespace="dynamo",
            name="worker-0",
            uid="pod-uid",
            labels={"nvidia.com/dynamo-component": "decode"},
            annotations={POWER_ANNOTATION_KEY: "350"},
        ),
        spec=SimpleNamespace(containers=[SimpleNamespace(name="main", env=env)]),
        status=SimpleNamespace(
            container_statuses=[
                SimpleNamespace(name="main", container_id=f"containerd://{'a' * 64}")
            ]
        ),
    )
    response = podresources_api.ListPodResourcesResponse(
        pod_resources=[
            podresources_api.PodResources(
                namespace="dynamo",
                name="worker-0",
                containers=[
                    podresources_api.ContainerResources(
                        name="main",
                        devices=[
                            podresources_api.ContainerDevices(
                                resource_name="nvidia.com/gpu",
                                device_ids=["GPU-a"],
                            )
                        ],
                    )
                ],
            )
        ]
    )
    agent = object.__new__(PowerAgent)
    agent.safe_default_watts = 500
    agent.node_name = "node-a"
    agent.device_count = 1
    agent.metrics = MagicMock()
    agent._pod_cache = {}
    agent._pod_resources = MagicMock()
    agent._pod_resources.list.return_value = response
    agent._actuator = MagicMock()
    agent._actuator.get_uuid.return_value = "GPU-a"
    agent._actuator.apply_cap.return_value = _result()
    agent._report_patcher = MagicMock()
    agent._report_patcher.publish.return_value = (pod, False)
    agent._transactional_enrollment_authorized = MagicMock(return_value=True)
    agent._transactional_ownership_is_durable = MagicMock(return_value=True)
    monkeypatch.setattr(
        power_agent,
        "_pod_runtime_gpu_uuids",
        lambda uid, container_id: ("GPU-a",),
    )

    states, held_uids = agent._transactional_pods([pod])
    agent._reconcile_transactional_pods(states)

    assert held_uids == {"pod-uid"}
    agent._actuator.list_running_pids.assert_not_called()
    ownership = agent._actuator.apply_cap.call_args.kwargs["ownership"]
    assert ownership == {
        "controlMode": "transactional-replica-fence",
        "dgdUID": "dgd-uid",
        "component": "decode",
        "podUID": "pod-uid",
        "allocationID": "pod-uid/main/GPU-a",
        "targetWatts": 350,
    }
    published_report = agent._report_patcher.publish.call_args.args[1]
    assert published_report["allocationID"] == "pod-uid/main/GPU-a"
    assert published_report["gpus"][0]["enforcedCapWatts"] == 350


def test_fresh_agent_replacement_correlates_current_uid_runtime_devices(
    monkeypatch,
):
    env = [
        SimpleNamespace(name=DGD_UID_ENV, value="dgd-uid"),
        SimpleNamespace(name=COMPONENT_ENV, value="decode"),
        SimpleNamespace(name=EXPECTED_GPU_COUNT_ENV, value="1"),
        SimpleNamespace(name=IN_GATE_BOUND_WATTS_ENV, value="350"),
    ]
    replacement = SimpleNamespace(
        metadata=SimpleNamespace(
            namespace="dynamo",
            name="worker-0",
            uid="pod-new",
            labels={"nvidia.com/dynamo-component": "decode"},
            annotations={POWER_ANNOTATION_KEY: "350"},
        ),
        spec=SimpleNamespace(containers=[SimpleNamespace(name="main", env=env)]),
        status=SimpleNamespace(
            container_statuses=[
                SimpleNamespace(name="main", container_id=f"containerd://{'b' * 64}")
            ]
        ),
    )
    stale = podresources_api.ListPodResourcesResponse(
        pod_resources=[
            podresources_api.PodResources(
                namespace="dynamo",
                name="worker-0",
                containers=[
                    podresources_api.ContainerResources(
                        name="main",
                        devices=[
                            podresources_api.ContainerDevices(
                                resource_name="nvidia.com/gpu",
                                device_ids=["GPU-old"],
                            )
                        ],
                    )
                ],
            )
        ]
    )
    current = podresources_api.ListPodResourcesResponse(
        pod_resources=[
            podresources_api.PodResources(
                namespace="dynamo",
                name="worker-0",
                containers=[
                    podresources_api.ContainerResources(
                        name="main",
                        devices=[
                            podresources_api.ContainerDevices(
                                resource_name="nvidia.com/gpu",
                                device_ids=["GPU-new"],
                            )
                        ],
                    )
                ],
            )
        ]
    )
    agent = object.__new__(PowerAgent)
    agent._pod_resources = MagicMock()
    agent._pod_replacement_barriers = set()
    agent._pod_cache = {}
    agent._pod_cache_initialized = True
    agent._pod_resource_version = "10"
    monkeypatch.setattr(
        power_agent,
        "_pod_runtime_gpu_uuids",
        lambda uid, container_id: ("GPU-new",),
    )

    agent._pod_resources.list.return_value = stale
    states, held = agent._transactional_pods([replacement])
    assert states == []
    assert held == {"pod-new"}
    assert agent._pod_replacement_barriers == {("dynamo", "worker-0")}

    agent._pod_resources.list.return_value = current
    states, _ = agent._transactional_pods([replacement])
    assert len(states) == 1
    assert states[0].allocation_id == "pod-new/main/GPU-new"
    assert agent._pod_replacement_barriers == set()


def test_runtime_device_correlation_binds_cgroup_uid_and_device_minor(monkeypatch):
    main_id = "a" * 64
    sidecar_id = "b" * 64
    handles = [object(), object()]
    monkeypatch.setattr(power_agent.pynvml, "nvmlDeviceGetCount", lambda: 2)
    monkeypatch.setattr(
        power_agent.pynvml, "nvmlDeviceGetHandleByIndex", handles.__getitem__
    )
    monkeypatch.setattr(
        power_agent.pynvml,
        "nvmlDeviceGetMinorNumber",
        lambda handle: handles.index(handle),
    )
    monkeypatch.setattr(
        power_agent.pynvml,
        "nvmlDeviceGetUUID",
        lambda handle: ("GPU-a", "GPU-b")[handles.index(handle)],
    )
    monkeypatch.setattr(
        power_agent.os,
        "scandir",
        lambda path: [SimpleNamespace(name="101"), SimpleNamespace(name="102")],
    )
    monkeypatch.setattr(
        power_agent,
        "_extract_pod_uid_from_cgroup",
        lambda pid: "pod-current",
    )
    monkeypatch.setattr(
        power_agent,
        "_extract_container_id_from_cgroup",
        lambda pid: main_id if pid == 101 else sidecar_id,
    )
    monkeypatch.setattr(
        power_agent.os,
        "listdir",
        lambda path: [
            "nvidia1" if "/101/" in path else "nvidia0",
            "nvidiactl",
            "nvidia-uvm",
        ],
    )
    monkeypatch.setattr(
        power_agent.os,
        "stat",
        lambda path, follow_symlinks=False: SimpleNamespace(
            st_mode=stat.S_IFCHR,
            st_rdev=os.makedev(195, int(os.path.basename(path).removeprefix("nvidia"))),
        ),
    )

    assert power_agent._pod_runtime_gpu_uuids("pod-current", main_id) == ("GPU-b",)


def test_runtime_device_correlation_rejects_sidecar_only(monkeypatch):
    main_id = "a" * 64
    sidecar_id = "b" * 64
    handle = object()
    monkeypatch.setattr(power_agent.pynvml, "nvmlDeviceGetCount", lambda: 1)
    monkeypatch.setattr(
        power_agent.pynvml, "nvmlDeviceGetHandleByIndex", lambda index: handle
    )
    monkeypatch.setattr(
        power_agent.pynvml, "nvmlDeviceGetMinorNumber", lambda candidate: 0
    )
    monkeypatch.setattr(
        power_agent.pynvml, "nvmlDeviceGetUUID", lambda candidate: "GPU-a"
    )
    monkeypatch.setattr(
        power_agent.os, "scandir", lambda path: [SimpleNamespace(name="102")]
    )
    monkeypatch.setattr(
        power_agent, "_extract_pod_uid_from_cgroup", lambda pid: "pod-current"
    )
    monkeypatch.setattr(
        power_agent, "_extract_container_id_from_cgroup", lambda pid: sidecar_id
    )

    assert power_agent._pod_runtime_gpu_uuids("pod-current", main_id) is None


def test_static_dev_mode_holds_reserved_context_without_transaction_modules(
    monkeypatch,
):
    static_pod = SimpleNamespace(
        metadata=SimpleNamespace(namespace="dynamo", name="static", uid="static-uid"),
        spec=SimpleNamespace(containers=[SimpleNamespace(name="main", env=[])]),
    )
    transaction_pod = SimpleNamespace(
        metadata=SimpleNamespace(
            namespace="dynamo", name="transaction", uid="transaction-uid"
        ),
        spec=SimpleNamespace(
            containers=[
                SimpleNamespace(
                    name="main",
                    env=[SimpleNamespace(name=DGD_UID_ENV, value="dgd-uid")],
                )
            ]
        ),
    )
    agent = object.__new__(PowerAgent)
    monkeypatch.setattr(power_agent, "_TRANSACTIONAL_MODULES_AVAILABLE", False)

    states, held = agent._transactional_pods([static_pod, transaction_pod])

    assert states == []
    assert held == {"transaction-uid"}


def _old_transactional_owner(gpu_uuid="GPU-a"):
    return {
        "controlMode": managed_state.TRANSACTIONAL_CONTROL_MODE,
        "dgdUID": "dgd-old",
        "component": "decode",
        "podUID": "pod-old",
        "allocationID": f"pod-old/main/{gpu_uuid}",
        "targetWatts": 350,
    }


def test_cross_dgd_reuse_explicitly_releases_absent_old_owner(monkeypatch):
    durable = {
        "version": managed_state.STATE_VERSION,
        "managed": {"GPU-a": _old_transactional_owner()},
    }
    agent = object.__new__(PowerAgent)
    agent._actuator = MagicMock()
    agent._actuator.restore_default_by_uuid.return_value = True
    agent._transactional_enrollment_authorized = MagicMock(side_effect=[False, True])
    power_agent._previously_managed.clear()
    monkeypatch.setattr(power_agent, "_read_managed_state", lambda: (durable, True))
    persist = MagicMock()
    monkeypatch.setattr(power_agent, "_persist_managed_gpus", persist)

    assert agent._prepare_transactional_enrollment("GPU-a", 0, "dgd-new", {"pod-new"})

    agent._actuator.restore_default_by_uuid.assert_called_once_with("GPU-a")
    assert "GPU-a" not in power_agent._previously_managed
    persist.assert_called_once_with(set())


def test_cross_dgd_reuse_refuses_while_old_owner_pod_is_live(monkeypatch):
    durable = {
        "version": managed_state.STATE_VERSION,
        "managed": {"GPU-a": _old_transactional_owner()},
    }
    agent = object.__new__(PowerAgent)
    agent._actuator = MagicMock()
    agent._transactional_enrollment_authorized = MagicMock(return_value=False)
    monkeypatch.setattr(power_agent, "_read_managed_state", lambda: (durable, True))

    assert not agent._prepare_transactional_enrollment(
        "GPU-a", 0, "dgd-new", {"pod-old", "pod-new"}
    )

    agent._actuator.restore_default_by_uuid.assert_not_called()


def test_idle_deleted_transactional_pod_restores_and_prunes_ownership(monkeypatch):
    owner = _old_transactional_owner()
    durable = {
        "version": managed_state.STATE_VERSION,
        "managed": {"GPU-a": owner},
    }
    agent = object.__new__(PowerAgent)
    agent.device_count = 1
    agent._actuator = MagicMock()
    # The reconcile UUID snapshot transiently misses GPU-a, but the independent
    # UUID-addressed restore succeeds. Retirement must still remove its old
    # NVML index rather than leaving a shutdown-clobber hazard.
    agent._actuator.get_uuid.return_value = "GPU-other"
    agent._actuator.restore_default_by_uuid.return_value = True
    power_agent._managed_gpu_indices.clear()
    power_agent._managed_gpu_indices.add(7)
    power_agent._managed_gpu_uuid_by_index.clear()
    power_agent._managed_gpu_uuid_by_index[7] = "GPU-a"
    power_agent._previously_managed.clear()
    power_agent._previously_managed.add("GPU-a")
    monkeypatch.setattr(power_agent, "_read_managed_state", lambda: (durable, True))
    persist = MagicMock()
    monkeypatch.setattr(power_agent, "_persist_managed_gpus", persist)

    agent._reconcile_transactional_pods([], live_pod_uids=set())

    agent._actuator.restore_default_by_uuid.assert_called_once_with("GPU-a")
    assert power_agent._previously_managed == set()
    assert power_agent._managed_gpu_indices == set()
    assert power_agent._managed_gpu_uuid_by_index == {}
    persist.assert_called_once_with(set())


def test_live_pod_allocation_change_retires_removed_uuid(monkeypatch):
    owner = _old_transactional_owner()
    durable = {
        "version": managed_state.STATE_VERSION,
        "managed": {"GPU-a": owner},
    }
    state = SimpleNamespace(
        pod=SimpleNamespace(
            metadata=SimpleNamespace(uid="pod-old", namespace="dynamo", name="worker-0")
        ),
        allocation=SimpleNamespace(gpu_uuids=("GPU-b",)),
        context=SimpleNamespace(dgd_uid="dgd-old", component="decode"),
        allocation_id="pod-old/main/GPU-b",
        requested_watts="350",
    )
    agent = object.__new__(PowerAgent)
    agent.device_count = 2
    agent.safe_default_watts = 500
    agent.metrics = MagicMock()
    agent._report_patcher = MagicMock()
    agent._pod_cache = {}
    agent._actuator = MagicMock()
    agent._actuator.get_uuid.side_effect = ["GPU-a", "GPU-b"]
    agent._actuator.restore_default_by_uuid.return_value = True
    agent._prepare_transactional_enrollment = MagicMock(return_value=False)
    power_agent._previously_managed.clear()
    power_agent._previously_managed.add("GPU-a")
    monkeypatch.setattr(power_agent, "_read_managed_state", lambda: (durable, True))
    persist = MagicMock()
    monkeypatch.setattr(power_agent, "_persist_managed_gpus", persist)

    agent._reconcile_transactional_pods([state], live_pod_uids={"pod-old"})

    agent._actuator.restore_default_by_uuid.assert_called_once_with("GPU-a")
    assert power_agent._previously_managed == set()
    persist.assert_called_once_with(set())


def test_live_pod_with_missing_allocation_retains_transactional_cap(monkeypatch):
    durable = {
        "version": managed_state.STATE_VERSION,
        "managed": {"GPU-a": _old_transactional_owner()},
    }
    agent = object.__new__(PowerAgent)
    agent.device_count = 1
    agent._actuator = MagicMock()
    agent._actuator.get_uuid.return_value = "GPU-a"
    monkeypatch.setattr(power_agent, "_read_managed_state", lambda: (durable, True))

    agent._reconcile_transactional_pods([], live_pod_uids={"pod-old"})

    agent._actuator.restore_default_by_uuid.assert_not_called()
