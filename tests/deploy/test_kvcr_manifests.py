# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST_DIR = _REPO_ROOT / "examples/backends/vllm/deploy/kvcr"
_MANIFESTS = ("agg.yaml", "agg-memory-service.yaml")


def _worker(manifest_name: str) -> dict:
    manifest = yaml.safe_load((_MANIFEST_DIR / manifest_name).read_text())
    return next(
        component
        for component in manifest["spec"]["components"]
        if component["name"] == "worker"
    )


@pytest.mark.parametrize("manifest_name", _MANIFESTS)
def test_kvcr_variants_require_two_gpu_rdma_nodes(manifest_name: str) -> None:
    worker = _worker(manifest_name)
    pod_spec = worker["podTemplate"]["spec"]
    main = pod_spec["containers"][0]

    assert worker["replicas"] == 2
    anti_affinity = pod_spec["affinity"]["podAntiAffinity"]
    assert anti_affinity["requiredDuringSchedulingIgnoredDuringExecution"]
    assert "preferredDuringSchedulingIgnoredDuringExecution" not in anti_affinity

    for resource_class in ("requests", "limits"):
        resources = main["resources"][resource_class]
        assert resources["nvidia.com/gpu"] == "1"
        assert resources["${DYNAMO_RDMA_RESOURCE}"] == "1"

    env = {item["name"]: item.get("value") for item in main["env"]}
    assert env["UCX_TLS"] == "rc_x,cuda"
    assert env["UCX_NET_DEVICES"] == "${DYNAMO_UCX_NET_DEVICES}"
    assert env["UCX_PROTO_INFO"] == "y"
    assert "No RDMA userspace device is mounted" in main["args"][0]
    assert "IPC_LOCK" in main["securityContext"]["capabilities"]["add"]


def test_process_local_variant_couples_state_agent_and_vllm() -> None:
    manifest = yaml.safe_load((_MANIFEST_DIR / "agg.yaml").read_text())
    assert (
        manifest["metadata"]["annotations"]["nvidia.com/dynamo-discovery-backend"]
        == "etcd"
    )
    pod_spec = _worker("agg.yaml")["podTemplate"]["spec"]
    main = pod_spec["containers"][0]
    command = main["args"][0]

    assert len(pod_spec["containers"]) == 1
    assert "initContainers" not in pod_spec
    assert "python3 -m dynamo.kv_state_agent" in command
    assert "python3 -m dynamo.vllm" in command
    assert 'if [ "$POD_INDEX" = "0" ]' in command
    assert "DYN_DISCOVERY_BACKEND=etcd" in command
    assert "DYN_SYSTEM_PORT=9091" in command
    assert "--max-slots 2" in command
    assert 'wait -n "$state_agent_pid" "$vllm_pid"' in command
    assert "kvcr.kvcr_service" not in command
    assert "owner_slot=00000000000000000000000000000000" in command
    assert "owner_slot=00000000000000000000000000000001" in command
    assert all(item["name"] != "POD_UID" for item in main["env"])

    for probe_name in ("startupProbe", "livenessProbe", "readinessProbe"):
        probe = main[probe_name]["exec"]["command"][-1]
        assert "9090" in probe
        assert "9091" in probe


def test_memory_service_variant_keeps_guard_in_sidecar() -> None:
    manifest = yaml.safe_load((_MANIFEST_DIR / "agg-memory-service.yaml").read_text())
    assert (
        manifest["metadata"]["annotations"]["nvidia.com/dynamo-kube-discovery-mode"]
        == "container"
    )
    assert (
        manifest["metadata"]["annotations"]["nvidia.com/dynamo-discovery-backend"]
        == "kubernetes"
    )
    pod_spec = _worker("agg-memory-service.yaml")["podTemplate"]["spec"]
    main, sidecar = pod_spec["containers"]
    main_command = main["args"][0]
    sidecar_command = sidecar["args"][0]

    assert sidecar["name"] == "kvcr-services"
    assert "python3 -m kvcr.kvcr_service" in sidecar_command
    assert "python3 -m dynamo.kv_state_agent" in sidecar_command
    assert "--guard-count 1" in sidecar_command
    assert "--pool-sizes-gb 2" in sidecar_command
    assert 'if [ "$POD_INDEX" = "0" ]' in sidecar_command
    assert "--max-slots 2" in sidecar_command
    assert 'wait -n "$kvcr_pid" "$state_agent_pid"' in sidecar_command
    assert '"kvcr_service_socket_path": "/run/kvcr/memory.sock"' in main_command
    assert "/run/kvcr/hold-engine-start" in main_command
    assert "owner_slot=00000000000000000000000000000000" in main_command
    assert "owner_slot=00000000000000000000000000000001" in main_command
    assert all(item["name"] != "POD_UID" for item in main["env"])
    pod_uid = next(item for item in sidecar["env"] if item["name"] == "POD_UID")
    assert pod_uid["valueFrom"]["fieldRef"]["fieldPath"] == "metadata.uid"

    for resource_class in ("requests", "limits"):
        assert sidecar["resources"][resource_class]["${DYNAMO_RDMA_RESOURCE}"] == "1"
    assert "IPC_LOCK" in sidecar["securityContext"]["capabilities"]["add"]

    for container in (main, sidecar):
        mounts = {mount["name"] for mount in container["volumeMounts"]}
        assert mounts == {"kvcr-memory", "kvcr-socket"}
    memory = next(
        volume for volume in pod_spec["volumes"] if volume["name"] == "kvcr-memory"
    )
    assert memory["emptyDir"]["medium"] == "Memory"

    for probe_name in ("startupProbe", "livenessProbe", "readinessProbe"):
        probe = sidecar[probe_name]["exec"]["command"][-1]
        assert "/run/kvcr/memory.sock" in probe
        assert "9091/live" in probe
        assert "os.environ['POD_INDEX'] != '0' or" in probe


@pytest.mark.skipif(shutil.which("envsubst") is None, reason="envsubst is unavailable")
@pytest.mark.parametrize("memory_service", ("false", "true"))
def test_deploy_script_renders_selected_variant(memory_service: str) -> None:
    env = os.environ.copy()
    env.update(
        {
            "DYNAMO_RDMA_RESOURCE": "rdma/test",
            "DYNAMO_UCX_NET_DEVICES": "mlx5_test:1",
            "DYNAMO_KVCR_COMPATIBILITY_DIGEST": "qwen3-0.6b-example-v1",
            "DYNAMO_VLLM_IMAGE": "runtime:1.4.0@sha256:test",
            "KVCR_MEMORY_SERVICE_ENABLED": memory_service,
        }
    )

    result = subprocess.run(
        [str(_MANIFEST_DIR / "deploy.sh"), "--render-only"],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    rendered = yaml.safe_load(result.stdout)
    assert rendered["kind"] == "DynamoGraphDeployment"
    containers = [
        container
        for component in rendered["spec"]["components"]
        for container in component["podTemplate"]["spec"]["containers"]
    ]
    assert all(
        container["image"] == "runtime:1.4.0@sha256:test" for container in containers
    )
    worker = next(
        component
        for component in rendered["spec"]["components"]
        if component["name"] == "worker"
    )
    for container in worker["podTemplate"]["spec"]["containers"]:
        env_by_name = {item["name"]: item.get("value") for item in container["env"]}
        assert env_by_name["UCX_NET_DEVICES"] == "mlx5_test:1"
        assert container["resources"]["limits"]["rdma/test"] == "1"
    if memory_service == "true":
        assert result.stdout.count("qwen3-0.6b-example-v1") == 2
    expected_name = (
        "vllm-agg-kvcr-memory-service" if memory_service == "true" else "vllm-agg-kvcr"
    )
    assert f"name: {expected_name}" in result.stdout


def test_deploy_script_rejects_unknown_argument() -> None:
    env = os.environ.copy()
    env.update(
        {
            "DYNAMO_UCX_NET_DEVICES": "mlx5_test:1",
            "DYNAMO_VLLM_IMAGE": "runtime:1.4.0@sha256:test",
        }
    )

    result = subprocess.run(
        [str(_MANIFEST_DIR / "deploy.sh"), "--render-onyl"],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 2
    assert "usage:" in result.stderr


def test_deploy_script_rejects_unsafe_compatibility_digest() -> None:
    env = os.environ.copy()
    env.update(
        {
            "DYNAMO_KVCR_COMPATIBILITY_DIGEST": "unsafe value",
            "DYNAMO_UCX_NET_DEVICES": "mlx5_test:1",
            "DYNAMO_VLLM_IMAGE": "runtime:1.4.0@sha256:test",
            "KVCR_MEMORY_SERVICE_ENABLED": "true",
        }
    )

    result = subprocess.run(
        [str(_MANIFEST_DIR / "deploy.sh"), "--render-only"],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert "unsupported character" in result.stderr
