# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the frontend readiness probe in a CPU-only Kubernetes namespace.

Install the DynamoWorkerMetadata CRD and load a frontend image containing the
change under test into Kind, then run:

    python3 -m pytest -n 0 tests/deploy/test_frontend_readiness_probe.py \
        --namespace default --frontend-image <image>

The test creates its own namespaced resources. The Pod status patch stands in
for the readiness controller so the test can run on the frontend stack alone.
"""

import json
import subprocess
import time
from pathlib import Path
from uuid import uuid4

import pytest
import requests

from tests.utils.managed_process import ManagedProcess

pytestmark = [
    pytest.mark.post_merge,
    pytest.mark.e2e,
    pytest.mark.gpu_0,
    pytest.mark.deploy,
    pytest.mark.k8s,
    pytest.mark.timeout(300),
]

GATE = "nvidia.com/dynamo-frontend-admitted"
TEST_LABEL = "nvidia.com/dynamo-readiness-test"
# These ports are per Pod. The host port-forward uses a fixture-allocated port.
FRONTEND_PORT = 8000
RPC_PORT = 9091
FIXTURES = "/readiness-test"
CACHE_DIR = "/home/dynamo/.cache"
COUNT_FILE = f"{CACHE_DIR}/readiness-request-count"
DISCOVERY_LABELS = {
    "nvidia.com/dynamo-discovery-backend": "kubernetes",
    "nvidia.com/dynamo-discovery-enabled": "true",
}


def _kubectl(namespace, *args, input_text=None, timeout=15, check=True):
    command = ["kubectl", f"--request-timeout={timeout}s", "-n", namespace, *args]
    result = subprocess.run(
        command,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout + 5,
        check=False,
    )
    if check:
        assert result.returncode == 0, f"{command}: {result.stderr}"
    return result.stdout


def _resource(kind, name, run, api_version="v1", **fields):
    return {
        "apiVersion": api_version,
        "kind": kind,
        "metadata": {"name": name, "labels": {TEST_LABEL: run}},
        **fields,
    }


def _pod(name, run, image, args, env, *, worker=False, held=False):
    field_env = {
        "POD_NAME": "metadata.name",
        "POD_NAMESPACE": "metadata.namespace",
        "POD_UID": "metadata.uid",
        "DYN_TCP_RPC_HOST": "status.podIP",
    }
    container = {
        "name": "main",
        "image": image,
        "imagePullPolicy": "IfNotPresent",
        "command": ["python3", *args],
        "env": [
            {"name": name, "valueFrom": {"fieldRef": {"fieldPath": path}}}
            for name, path in field_env.items()
        ]
        + [{"name": name, "value": value} for name, value in env.items()],
        "volumeMounts": [
            {"name": "fixtures", "mountPath": FIXTURES, "readOnly": True},
            {"name": "scratch", "mountPath": CACHE_DIR},
        ],
        "readinessProbe": {
            **(
                {"tcpSocket": {"port": RPC_PORT}}
                if worker
                else {"httpGet": {"path": "/health", "port": FRONTEND_PORT}}
            ),
            "periodSeconds": 1,
        },
    }
    spec = {
        "restartPolicy": "Never",
        "terminationGracePeriodSeconds": 5,
        "serviceAccountName": run,
        "containers": [container],
        "volumes": [
            {"name": "fixtures", "configMap": {"name": run}},
            {"name": "scratch", "emptyDir": {"medium": "Memory", "sizeLimit": "32Mi"}},
        ],
    }
    if held:
        spec["readinessGates"] = [{"conditionType": GATE}]
    pod = _resource("Pod", name, run, spec=spec)
    if worker:
        pod["metadata"]["labels"].update(DISCOVERY_LABELS)
    return pod


def _apply(namespace, *items):
    _kubectl(
        namespace,
        "apply",
        "-f",
        "-",
        input_text=json.dumps({"apiVersion": "v1", "kind": "List", "items": items}),
    )


def _wait_for(namespace, pod, condition):
    _kubectl(
        namespace,
        "wait",
        f"pod/{pod}",
        f"--for=condition={condition}",
        "--timeout=45s",
        timeout=50,
    )


def _conditions(namespace, pod):
    result = json.loads(_kubectl(namespace, "get", f"pod/{pod}", "-o", "json"))
    return {
        condition["type"]: condition["status"]
        for condition in result["status"]["conditions"]
    }


def _instance_id(namespace, pod):
    deadline = time.monotonic() + 30
    while True:
        result = _kubectl(
            namespace,
            "get",
            f"dynamoworkermetadata/{pod}",
            "--ignore-not-found",
            "-o",
            "json",
        )
        if result:
            endpoints = json.loads(result)["spec"]["data"].get("endpoints", {}).values()
            for endpoint in endpoints:
                if endpoint["endpoint"] == "generate":
                    return endpoint["instance_id"]
        assert time.monotonic() < deadline, f"No generate endpoint registered by {pod}"
        time.sleep(0.2)


def _count(namespace, pod):
    return int(
        _kubectl(
            namespace,
            "exec",
            pod,
            "--",
            "python3",
            "-c",
            "import pathlib, sys; p = pathlib.Path(sys.argv[1]); "
            "print(p.read_text() if p.exists() else 0)",
            COUNT_FILE,
        )
    )


def _expect_status(response, status):
    assert response.status_code == status, f"{response.status_code}: {response.text}"


def _eventually_status(request, status, *, transient=(503,)):
    deadline = time.monotonic() + 30
    while True:
        response = request()
        if response.status_code not in transient or time.monotonic() >= deadline:
            _expect_status(response, status)
            return
        time.sleep(0.2)


def test_held_candidate_completion_probe(
    request, namespace, dynamo_dynamic_ports, tmp_path
):
    image = request.config.getoption("--frontend-image")
    assert (
        image and namespace
    ), "Pass --frontend-image and --namespace for this Kubernetes test"
    run = f"readiness-probe-{uuid4().hex[:8]}"
    frontend, old, candidate = (f"{run}-{role}" for role in ("frontend", "old", "new"))
    root = Path(__file__).resolve().parents[2]
    model = root / "lib/llm/tests/data/sample-models/mock-llama-3.1-8b-instruct"
    files = {path.name: path.read_text() for path in sorted(model.glob("*.json"))}
    files["counter_worker.py"] = (root / "tests/router/counter_worker.py").read_text()

    rbac = "rbac.authorization.k8s.io/v1"
    service = _resource(
        "Service",
        run,
        run,
        spec={
            "clusterIP": "None",
            "selector": {TEST_LABEL: run, **DISCOVERY_LABELS},
            "ports": [{"name": "rpc", "port": RPC_PORT, "targetPort": RPC_PORT}],
        },
    )
    service["metadata"]["labels"].update(DISCOVERY_LABELS)
    env = {
        "DYN_NAMESPACE": run,
        "DYN_DISCOVERY_BACKEND": "kubernetes",
        "DYN_KUBE_DISCOVERY_MODE": "pod",
        "DYN_REQUEST_PLANE": "tcp",
        "DYN_EVENT_PLANE": "zmq",
        "DYN_HEALTH_CHECK_ENABLED": "false",
    }
    worker_env = {**env, "DYN_TCP_RPC_PORT": str(RPC_PORT)}
    worker_args = [
        f"{FIXTURES}/counter_worker.py",
        COUNT_FILE,
        "cpu",
        f"{run}.workers.generate",
        "--model-path",
        FIXTURES,
        "--completions",
    ]
    port = dynamo_dynamic_ports.frontend_port
    base = f"http://127.0.0.1:{port}"

    def completion(max_tokens, worker_id=None):
        body = {"model": "counter", "prompt": "hello", "max_tokens": max_tokens}
        if worker_id is not None:
            body["nvext"] = {"backend_instance_id": worker_id}
        return requests.post(f"{base}/v1/completions", json=body, timeout=5)

    try:
        _apply(
            namespace,
            _resource("ConfigMap", run, run, data=files),
            _resource("ServiceAccount", run, run),
            _resource(
                "Role",
                run,
                run,
                rbac,
                rules=[
                    {
                        "apiGroups": [""],
                        "resources": ["pods"],
                        "verbs": ["get", "list", "watch"],
                    },
                    {
                        "apiGroups": ["discovery.k8s.io"],
                        "resources": ["endpointslices"],
                        "verbs": ["get", "list", "watch"],
                    },
                    {
                        "apiGroups": ["nvidia.com"],
                        "resources": ["dynamoworkermetadatas"],
                        "verbs": [
                            "get",
                            "list",
                            "watch",
                            "create",
                            "patch",
                            "update",
                            "delete",
                        ],
                    },
                ],
            ),
            _resource(
                "RoleBinding",
                run,
                run,
                rbac,
                subjects=[
                    {"kind": "ServiceAccount", "name": run, "namespace": namespace}
                ],
                roleRef={
                    "apiGroup": "rbac.authorization.k8s.io",
                    "kind": "Role",
                    "name": run,
                },
            ),
            service,
            _pod(old, run, image, worker_args, worker_env, worker=True),
            _pod(
                frontend,
                run,
                image,
                ["-m", "dynamo.frontend", "--router-mode", "round-robin"],
                {
                    **env,
                    "DYN_KUBE_DISCOVERY_HELD_PODS": "true",
                    "DYN_HTTP_HOST": "0.0.0.0",
                    "DYN_HTTP_PORT": str(FRONTEND_PORT),
                },
            ),
        )
        _wait_for(namespace, frontend, "Ready")
        with ManagedProcess(
            command=[
                "kubectl",
                "-n",
                namespace,
                "port-forward",
                f"pod/{frontend}",
                f"{port}:{FRONTEND_PORT}",
            ],
            health_check_urls=[f"{base}/health"],
            timeout=30,
            log_dir=str(tmp_path),
            terminate_all_matching_process_names=False,
        ):
            _eventually_status(lambda: completion(1), 200, transient=(404, 503))
            assert _count(namespace, old) > 0

            _apply(
                namespace,
                _pod(
                    candidate,
                    run,
                    image,
                    worker_args,
                    worker_env,
                    worker=True,
                    held=True,
                ),
            )
            _wait_for(namespace, candidate, "ContainersReady")
            assert _conditions(namespace, candidate)["Ready"] == "False"
            worker_id = _instance_id(namespace, candidate)
            old_worker_id = _instance_id(namespace, old)
            old_count = _count(namespace, old)

            _eventually_status(lambda: completion(0, worker_id), 204)
            # The test namespace has exactly two worker IDs.
            unknown_id = next(
                i for i in (0, 1, 2) if i not in (worker_id, old_worker_id)
            )
            _expect_status(completion(0, unknown_id), 503)
            assert _count(namespace, old) == old_count
            assert _count(namespace, candidate) == 0
            assert _conditions(namespace, candidate)["Ready"] == "False"

            for _ in range(4):
                _expect_status(completion(1), 200)
            assert _count(namespace, old) == old_count + 4
            assert _count(namespace, candidate) == 0

            # Simulate the controller's status write only after the probe passes.
            patch = {"status": {"conditions": [{"type": GATE, "status": "True"}]}}
            _kubectl(
                namespace,
                "patch",
                f"pod/{candidate}",
                "--subresource=status",
                "--type=strategic",
                "-p",
                json.dumps(patch),
            )
            _wait_for(namespace, candidate, "Ready")
            # The old worker keeps serving while the frontend observes the change.
            deadline = time.monotonic() + 30
            while _count(namespace, candidate) == 0:
                assert (
                    time.monotonic() < deadline
                ), "Released worker received no requests"
                _expect_status(completion(1), 200)
                time.sleep(0.2)
    except Exception:
        print(
            _kubectl(
                namespace,
                "get",
                "pods",
                "-l",
                f"{TEST_LABEL}={run}",
                "-o",
                "wide",
                check=False,
            )
        )
        for pod in (frontend, old, candidate):
            print(
                f"{pod}:\n{_kubectl(namespace, 'logs', pod, '--tail=100', check=False)}"
            )
        raise
    finally:
        _kubectl(
            namespace,
            "delete",
            "pod,service,configmap,serviceaccount,role,rolebinding",
            "-l",
            f"{TEST_LABEL}={run}",
            "--ignore-not-found",
            "--wait=false",
        )
