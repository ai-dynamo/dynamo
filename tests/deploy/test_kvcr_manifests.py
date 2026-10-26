# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import json
import os
import select
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.timeout(90),
]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST_DIR = _REPO_ROOT / "examples/backends/vllm/deploy/kvcr"
_MANIFESTS = ("agg.yaml", "agg-memory-service.yaml")


@pytest.fixture(autouse=True)
def _clear_deployment_options(monkeypatch):
    for name in (
        "KVCR_POOL_SIZE_GIB",
        "KVCR_STARTUP_TIMEOUT_SECONDS",
        "KVCR_GUARD_STARTUP_TIMEOUT_SECONDS",
        "KVCR_ENGINE_STARTUP_TIMEOUT_SECONDS",
        "KVCR_TERMINATION_GRACE_PERIOD_SECONDS",
        "KVCR_WHEEL",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    "worker,exit_module,exit_code,guard_timeout",
    [
        ("0", "kvcr.kvcr_service", 7, None),
        ("0", "dynamo.kv_state_agent", 0, "600"),
        ("0", None, 0, None),
        ("1", "kvcr.kvcr_service", 7, None),
    ],
)
def test_services_supervises_children(
    tmp_path, worker, exit_module, exit_code, guard_timeout
):
    """Exercise the actual wrapper with controllable children, without engines."""
    for source in (
        _MANIFEST_DIR / "services.sh",
        _REPO_ROOT / "examples/common/launch_utils.sh",
    ):
        shutil.copy(source, tmp_path / source.name)
    child = tmp_path / "python3"
    child.write_text(
        f"#!{sys.executable}\n"
        + """
import json, os, signal, sys
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
signal.signal(signal.SIGUSR1, lambda *_: sys.exit(int(os.environ["TEST_EXIT_CODE"])))
print(json.dumps({"module": sys.argv[2], "pid": os.getpid(), "args": sys.argv[3:]}), flush=True)
signal.pause()
"""
    )
    child.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "POD_INDEX": worker,
        "DYN_NAMESPACE": "test",
        "KVCR_CACHE_SLOT_COUNT": "2",
        "DYN_NAMESPACE_WORKER_SUFFIX": "",
        "DYNAMO_KVCR_COMPATIBILITY_DIGEST": "test",
        "TEST_EXIT_CODE": str(exit_code),
        "KVCR_POOL_SIZE_GIB": "1024",
    }
    if guard_timeout is not None:
        env["KVCR_GUARD_STARTUP_TIMEOUT_SECONDS"] = guard_timeout
    process = subprocess.Popen(
        ["bash", str(tmp_path / "services.sh")],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        bufsize=0,
    )
    children = {}
    try:
        for _ in range(2 if worker == "0" else 1):
            assert select.select([process.stdout], [], [], 5)[0], "child did not start"
            info = json.loads(process.stdout.readline())
            children[info["module"]] = info
        guard_args = children["kvcr.kvcr_service"]["args"]
        assert guard_args[guard_args.index("--pool-sizes-gb") + 1] == "1024"
        assert guard_args[guard_args.index("--startup-timeout-seconds") + 1] == (
            guard_timeout or "30"
        )
        assert ("--health-dependency-url" in guard_args) == (worker == "0")
        if exit_module is None:
            process.terminate()
        else:
            os.kill(children[exit_module]["pid"], signal.SIGUSR1)
        output, _ = process.communicate(timeout=5)
        assert process.returncode == exit_code, output.decode()
        for info in children.values():
            with pytest.raises(ProcessLookupError):
                os.kill(info["pid"], 0)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def _worker(manifest_name: str) -> dict:
    manifest = yaml.safe_load((_MANIFEST_DIR / manifest_name).read_text())
    return next(
        component
        for component in manifest["spec"]["components"]
        if component["name"] == "worker"
    )


def _kv_transfer_config_python(command: str) -> str:
    marker = "python3 - <<'PY'\n"
    start = command.index(marker) + len(marker)
    end = command.index("\nPY\n", start)
    return command[start:end]


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
    assert all(
        item["name"] != "UCX_PROTO_INFO"
        for container in pod_spec["containers"]
        for item in container["env"]
    )
    assert env["KVCR_CACHE_SLOT_COUNT"] == "2"
    slot = next(item for item in main["env"] if item["name"] == "KVCR_CACHE_SLOT")
    assert slot["valueFrom"]["fieldRef"]["fieldPath"] == (
        "metadata.labels['grove.io/podclique-pod-index']"
    )
    assert "IPC_LOCK" in main["securityContext"]["capabilities"]["add"]


@pytest.mark.parametrize("manifest_name", _MANIFESTS)
def test_kvcr_cache_slot_is_validated_and_formatted(manifest_name: str) -> None:
    main = _worker(manifest_name)["podTemplate"]["spec"]["containers"][0]
    script = _kv_transfer_config_python(main["args"][0])
    env = {
        **os.environ,
        "KVCR_CACHE_SLOT": "1",
        "KVCR_CACHE_SLOT_COUNT": "2",
        "KVCR_ENGINE_STARTUP_TIMEOUT_SECONDS": "30",
        "POD_IP": "192.0.2.1",
    }

    result = subprocess.run(
        ["python3", "-c", script],
        check=True,
        capture_output=True,
        env=env,
        text=True,
        timeout=30,
    )
    config = json.loads(result.stdout)
    owner = config["kv_connector_extra_config"]["dynamo_state_agent"][
        "cache_owner_ids"
    ]["0"]
    assert owner.endswith("/00000000000000000000000000000001")

    env["KVCR_CACHE_SLOT"] = "2"
    result = subprocess.run(
        ["python3", "-c", script],
        check=False,
        capture_output=True,
        env=env,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "outside [0, 2)" in result.stderr


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
    assert "env -u DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS" in command
    assert "DYN_DISCOVERY_BACKEND=etcd" in command
    assert "DYN_SYSTEM_PORT=9091" in command
    assert "DYN_HEALTH_CHECK_ENABLED=false" in command
    assert '--max-slots "$KVCR_CACHE_SLOT_COUNT"' in command
    assert 'wait -n "$state_agent_pid" "$vllm_pid"' in command
    assert "kvcr.kvcr_service" not in command
    assert 'case "$POD_INDEX" in' not in command
    assert '"cache_owner_ids": {"0": cache_owner_id()}' in command
    assert all(item["name"] != "POD_UID" for item in main["env"])

    assert "startupProbe" not in main
    assert "livenessProbe" not in main
    assert "readinessProbe" not in main


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
    assert sidecar["args"] == ["/opt/kvcr-startup/services.sh"]
    sidecar_command = (_MANIFEST_DIR / "services.sh").read_text()

    assert sidecar["name"] == "kvcr-services"
    assert "python3 -m kvcr.kvcr_service" in sidecar_command
    assert "python3 -m dynamo.kv_state_agent" in sidecar_command
    assert "--guard-count 1" in sidecar_command
    assert '--pool-sizes-gb "${KVCR_POOL_SIZE_GIB:-2}"' in sidecar_command
    assert 'if [ "$POD_INDEX" = "0" ]' in sidecar_command
    assert "DYN_HEALTH_CHECK_ENABLED=false" in sidecar_command
    assert "DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS" not in sidecar_command
    assert all(
        item["name"] != "DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS"
        for item in sidecar["env"]
    )
    assert '--max-slots "$KVCR_CACHE_SLOT_COUNT"' in sidecar_command
    assert "wait_any_exit" in sidecar_command
    assert (
        'wait_for_ready http://127.0.0.1:9092/ready "${KVCR_STARTUP_TIMEOUT_SECONDS}"'
        in main_command
    )
    assert '"kvcr_service_socket_path": "/run/kvcr/memory.sock"' in main_command
    assert "/run/kvcr/hold-engine-start" not in main_command
    assert 'case "$POD_INDEX" in' not in main_command
    assert '"cache_owner_ids": {"0": cache_owner_id()}' in main_command
    assert all(item["name"] != "POD_UID" for item in main["env"])
    pod_uid = next(item for item in sidecar["env"] if item["name"] == "POD_UID")
    assert pod_uid["valueFrom"]["fieldRef"]["fieldPath"] == "metadata.uid"

    for resource_class in ("requests", "limits"):
        assert sidecar["resources"][resource_class]["${DYNAMO_RDMA_RESOURCE}"] == "1"
    assert "IPC_LOCK" in sidecar["securityContext"]["capabilities"]["add"]

    for container in (main, sidecar):
        mounts = {mount["name"] for mount in container["volumeMounts"]}
        assert mounts == {"kvcr-memory", "kvcr-socket", "kvcr-startup"}
    memory = next(
        volume for volume in pod_spec["volumes"] if volume["name"] == "kvcr-memory"
    )
    assert memory["emptyDir"]["medium"] == "Memory"

    assert sidecar["readinessProbe"]["httpGet"] == {"path": "/ready", "port": 9092}
    assert "startupProbe" not in sidecar
    assert "livenessProbe" not in sidecar
    assert "--health-dependency-url" in sidecar_command
    assert "--health-port" in sidecar_command


@pytest.mark.skipif(shutil.which("envsubst") is None, reason="envsubst is unavailable")
@pytest.mark.parametrize(
    "memory_service,pool_gib,memory_request_gib,native_timeouts",
    [
        ("false", 2, 3, None),
        ("true", 2, 3, None),
        ("true", 1024, 1035, (600, 700)),
    ],
)
def test_deploy_script_renders_selected_variant(
    memory_service: str,
    pool_gib: int,
    memory_request_gib: int,
    native_timeouts: tuple[int, int] | None,
) -> None:
    env = os.environ.copy()
    env.update(
        {
            "DYNAMO_RDMA_RESOURCE": "rdma/test",
            "DYNAMO_UCX_NET_DEVICES": "mlx5_test:1",
            "DYNAMO_KVCR_COMPATIBILITY_DIGEST": "qwen3-0.6b-example-v1",
            "DYNAMO_VLLM_IMAGE": "runtime:1.4.0@sha256:test",
            "KVCR_MEMORY_SERVICE_ENABLED": memory_service,
            "KVCR_POOL_SIZE_GIB": str(pool_gib),
            "KVCR_STARTUP_TIMEOUT_SECONDS": "900",
            "KVCR_TERMINATION_GRACE_PERIOD_SECONDS": "180"
            if pool_gib == 1024
            else "30",
        }
    )

    if native_timeouts is not None:
        guard_timeout, engine_timeout = native_timeouts
        env["KVCR_GUARD_STARTUP_TIMEOUT_SECONDS"] = str(guard_timeout)
        env["KVCR_ENGINE_STARTUP_TIMEOUT_SECONDS"] = str(engine_timeout)
    else:
        guard_timeout = engine_timeout = 30

    result = subprocess.run(
        [str(_MANIFEST_DIR / "deploy.sh"), "--render-only"],
        check=True,
        capture_output=True,
        env=env,
        text=True,
        timeout=30,
    )

    rendered = yaml.safe_load(result.stdout)
    assert rendered["kind"] == "DynamoGraphDeployment"
    if memory_service == "true":
        worker = next(
            c for c in rendered["spec"]["components"] if c["name"] == "worker"
        )
        pod_spec = worker["podTemplate"]["spec"]
        main, sidecar = pod_spec["containers"]
        assert 'wait_for_ready http://127.0.0.1:9092/ready "900"' in main["args"][0]
        assert pod_spec["terminationGracePeriodSeconds"] == (
            180 if pool_gib == 1024 else 30
        )
        env_values = {e["name"]: e.get("value") for e in sidecar["env"]}
        assert env_values["KVCR_POOL_SIZE_GIB"] == str(pool_gib)
        assert env_values["KVCR_GUARD_STARTUP_TIMEOUT_SECONDS"] == str(guard_timeout)
        main_env = {e["name"]: e["value"] for e in main["env"] if "value" in e}
        assert main_env["KVCR_ENGINE_STARTUP_TIMEOUT_SECONDS"] == str(engine_timeout)
        config_result = subprocess.run(
            [sys.executable, "-c", _kv_transfer_config_python(main["args"][0])],
            check=True,
            capture_output=True,
            env={
                **env,
                **main_env,
                "KVCR_CACHE_SLOT": "0",
                "POD_IP": "192.0.2.1",
            },
            text=True,
            timeout=30,
        )
        tier = json.loads(config_result.stdout)["kv_connector_extra_config"][
            "secondary_tiers"
        ][0]
        assert tier["startup_timeout_seconds"] == engine_timeout
        assert sidecar["resources"]["requests"]["memory"] == f"{memory_request_gib}Gi"
        assert sidecar["resources"]["limits"]["memory"] == f"{memory_request_gib + 1}Gi"
        memory = next(v for v in pod_spec["volumes"] if v["name"] == "kvcr-memory")
        assert memory["emptyDir"]["sizeLimit"] == f"{pool_gib + 1}Gi"
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
        bundle = subprocess.run(
            [str(_MANIFEST_DIR / "deploy.sh"), "--render-configmap"],
            check=True,
            capture_output=True,
            env=env,
            text=True,
            timeout=30,
        )
        configmap = json.loads(bundle.stdout)
        assert configmap["immutable"] is True
        assert "binaryData" not in configmap
        assert (
            configmap["data"]["services.sh"]
            == (_MANIFEST_DIR / "services.sh").read_text()
        )
        assert (
            configmap["data"]["launch_utils.sh"]
            == (_REPO_ROOT / "examples/common/launch_utils.sh").read_text()
        )
        volume = next(
            v
            for v in worker["podTemplate"]["spec"]["volumes"]
            if v["name"] == "kvcr-startup"
        )
        assert volume["configMap"]["name"] == configmap["metadata"]["name"]
    expected_name = (
        "vllm-agg-kvcr-memory-service" if memory_service == "true" else "vllm-agg-kvcr"
    )
    assert f"name: {expected_name}" in result.stdout


@pytest.mark.parametrize(
    "option,value",
    [
        ("KVCR_POOL_SIZE_GIB", "0"),
        ("KVCR_STARTUP_TIMEOUT_SECONDS", "0"),
        ("KVCR_GUARD_STARTUP_TIMEOUT_SECONDS", "0"),
        ("KVCR_ENGINE_STARTUP_TIMEOUT_SECONDS", "0"),
        ("KVCR_TERMINATION_GRACE_PERIOD_SECONDS", "0"),
        ("KVCR_POOL_SIZE_GIB", "01"),
        ("KVCR_POOL_SIZE_GIB", "-1"),
        ("KVCR_POOL_SIZE_GIB", "1.5"),
        ("KVCR_POOL_SIZE_GIB", "abc"),
    ],
)
def test_deploy_script_rejects_invalid_pool_and_timeouts(option, value) -> None:
    result = subprocess.run(
        [str(_MANIFEST_DIR / "deploy.sh"), "--render-only"],
        check=False,
        capture_output=True,
        env={
            **os.environ,
            "DYNAMO_UCX_NET_DEVICES": "mlx5_test:1",
            "DYNAMO_VLLM_IMAGE": "runtime:1.4.0@sha256:test",
            "DYNAMO_KVCR_COMPATIBILITY_DIGEST": "test",
            "KVCR_MEMORY_SERVICE_ENABLED": "true",
            option: value,
        },
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert result.stdout == ""
    assert "positive decimal integers without leading zeros" in result.stderr


def test_deploy_script_rejects_unknown_argument() -> None:
    env = os.environ.copy()
    env.update(
        {
            "DYNAMO_UCX_NET_DEVICES": "mlx5_test:1",
            "DYNAMO_VLLM_IMAGE": "runtime:1.4.0@sha256:test",
            "KVCR_MEMORY_SERVICE_ENABLED": "false",
        }
    )

    result = subprocess.run(
        [str(_MANIFEST_DIR / "deploy.sh"), "--render-onyl"],
        check=False,
        capture_output=True,
        env=env,
        text=True,
        timeout=30,
    )

    assert result.returncode == 2
    assert "usage:" in result.stderr


@pytest.mark.skipif(shutil.which("envsubst") is None, reason="envsubst is unavailable")
@pytest.mark.parametrize("size", [256 * 1024, 1024 * 1024])
def test_startup_bundle_wheel_overlay_and_size_limit(tmp_path, size):
    wheel = tmp_path / "fixture.whl"
    wheel.write_bytes(b"x" * size)
    env = {
        **os.environ,
        "DYNAMO_UCX_NET_DEVICES": "test:1",
        "DYNAMO_VLLM_IMAGE": "runtime:test",
        "KVCR_MEMORY_SERVICE_ENABLED": "true",
        "DYNAMO_KVCR_COMPATIBILITY_DIGEST": "test",
        "KVCR_WHEEL": str(wheel),
        "NAMESPACE": "kvcr-test",
    }
    result = subprocess.run(
        [str(_MANIFEST_DIR / "deploy.sh"), "--render-configmap"],
        check=False,
        capture_output=True,
        env=env,
        text=True,
        timeout=30,
    )
    if size == 1024 * 1024:
        assert result.returncode != 0
        assert "exceeds 1 MiB" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        bundle = json.loads(result.stdout)
        assert base64.b64decode(bundle["binaryData"]["kvcr.whl"]) == wheel.read_bytes()
        without_wheel = subprocess.run(
            [str(_MANIFEST_DIR / "deploy.sh"), "--render-configmap"],
            check=True,
            capture_output=True,
            env={**env, "KVCR_WHEEL": ""},
            text=True,
            timeout=30,
        )
        assert (
            bundle["metadata"]["name"]
            != json.loads(without_wheel.stdout)["metadata"]["name"]
        )
        # This bundle fits the startup limit but not a client-side apply annotation.
        assert 256 * 1024 < len(result.stdout.encode()) < 1024 * 1024
        kubectl = tmp_path / "kubectl"
        kubectl.write_text(
            f"#!{sys.executable}\n"
            "import json, sys, yaml\n"
            "args = sys.argv[1:]\n"
            "if args[0] == 'api-resources':\n"
            "    print('podcliquesets.grove.io')\n"
            "elif args[0] == 'apply':\n"
            "    print(json.dumps({'args': args, 'object': yaml.safe_load(sys.stdin)}))\n"
            "elif args[0] == 'get':\n"
            "    print('grove')\n"
            "else:\n"
            "    raise AssertionError(args)\n"
        )
        kubectl.chmod(0o755)
        applied = subprocess.run(
            [str(_MANIFEST_DIR / "deploy.sh")],
            check=True,
            capture_output=True,
            env={**env, "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}"},
            text=True,
            timeout=30,
        )
        configmap_call, deployment_call = map(json.loads, applied.stdout.splitlines())
        assert configmap_call["object"] == bundle
        assert configmap_call["args"] == [
            "apply",
            "--server-side",
            "--field-manager=kvcr-startup",
            "-n",
            "kvcr-test",
            "-f",
            "-",
        ]
        assert deployment_call["object"]["kind"] == "DynamoGraphDeployment"
        assert deployment_call["args"] == ["apply", "-n", "kvcr-test", "-f", "-"]


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
        timeout=30,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert "unsupported character" in result.stderr
