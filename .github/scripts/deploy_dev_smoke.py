# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise deploy-runner access, one GPU, and the shared GAIE gateway."""

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

CONTEXT = os.environ["KUBE_CONTEXT"]
NAMESPACE = "dynamo-deploy-test"
GATEWAY_NAMESPACE = "agentgateway-system"
NAME = (
    f"ci-access-smoke-{os.environ['GITHUB_RUN_ID']}-{os.environ['GITHUB_RUN_ATTEMPT']}"
)
HOST = f"{NAME}.example.invalid"
LABEL = {"ci-access-smoke": NAME}
ARTIFACTS = Path(os.environ["RUNNER_TEMP"]) / NAME
ARTIFACTS.mkdir(parents=True, exist_ok=True)
KUBE = ["kubectl", "--context", CONTEXT, "--request-timeout=30s"]
MANIFEST = ARTIFACTS / "resources.json"


def kube(*args, check=True):
    result = subprocess.run([*KUBE, *args], capture_output=True, text=True, timeout=360)
    if check and result.returncode:
        raise RuntimeError(f"kubectl {' '.join(args)}: {result.stderr}")
    return result


def check_access(verb, resource, namespace=None, expected="yes"):
    args = ["auth", "can-i", verb, resource]
    if resource in {"pods/portforward", "pods/exec"}:
        parent, subresource = resource.split("/", 1)
        args = ["auth", "can-i", verb, parent, "--subresource", subresource]
    if namespace:
        args += ["-n", namespace]
    result = kube(*args, check=False)
    if result.stdout.strip() != expected:
        raise RuntimeError(
            f"Expected {expected}: {' '.join(args)}; "
            f"got {result.stdout} {result.stderr}"
        )
    print(f"PASS {verb} {resource} in {namespace or 'cluster'}: {expected}", flush=True)


def resource(kind, api_version, name, spec):
    return {
        "apiVersion": api_version,
        "kind": kind,
        "metadata": {"name": name, "namespace": NAMESPACE, "labels": LABEL},
        "spec": spec,
    }


def run():
    assert os.environ["KUBECONFIG"] == "/teleport/kubeconfig.yaml"
    contexts = kube("config", "get-contexts", "-o", "name").stdout.splitlines()
    if CONTEXT not in contexts:
        raise RuntimeError(f"Target {CONTEXT} missing from mounted tbot kubeconfig")
    identity = kube("auth", "whoami", "-o", "json").stdout
    (ARTIFACTS / "identity.json").write_text(identity)
    username = json.loads(identity)["status"]["userInfo"]["username"]
    if username != "bot-dynamo-deploy-test":
        raise RuntimeError(f"Unexpected Kubernetes identity: {username}")
    for verb in ["get", "create", "patch", "delete"]:
        for obj in [
            "deployments.apps",
            "services",
            "jobs.batch",
            "dynamographdeployments.nvidia.com",
            "httproutes.gateway.networking.k8s.io",
        ]:
            check_access(verb, obj, NAMESPACE)
    for verb in ["get", "create"]:
        check_access(verb, "pods/portforward", GATEWAY_NAMESPACE)
    check_access("get", "services/inference-gateway", GATEWAY_NAMESPACE)
    check_access("list", "pods", GATEWAY_NAMESPACE)
    for verb, obj, ns in [
        ("create", "namespaces", None),
        ("delete", "namespaces", None),
        ("create", "clusterrolebindings.rbac.authorization.k8s.io", None),
        ("create", "deployments.apps", "default"),
        ("delete", "deployments.apps", GATEWAY_NAMESPACE),
        ("create", "pods/exec", GATEWAY_NAMESPACE),
        ("get", "secrets", GATEWAY_NAMESPACE),
    ]:
        check_access(verb, obj, ns, expected="no")
    quota = json.loads(
        kube(
            "get",
            "resourcequota",
            "deploy-test-gpu-quota",
            "-n",
            NAMESPACE,
            "-o",
            "json",
        ).stdout
    )
    assert quota["spec"]["hard"]["requests.nvidia.com/gpu"] == "16"
    assert quota["spec"]["hard"]["limits.nvidia.com/gpu"] == "16"
    print("PASS 16-GPU quota", flush=True)
    kube("get", "service", "inference-gateway", "-n", GATEWAY_NAMESPACE)
    items = [
        resource(
            "Deployment",
            "apps/v1",
            NAME,
            {
                "replicas": 1,
                "selector": {"matchLabels": LABEL},
                "template": {
                    "metadata": {"labels": LABEL},
                    "spec": {
                        "automountServiceAccountToken": False,
                        "containers": [
                            {
                                "name": "http",
                                "image": "python:3.12-slim",
                                "command": [
                                    "sh",
                                    "-c",
                                    f"echo '{NAME}' > /tmp/index.html; "
                                    "exec python -m http.server 8080 --directory /tmp",
                                ],
                                "ports": [{"containerPort": 8080}],
                                "readinessProbe": {
                                    "httpGet": {"path": "/", "port": 8080}
                                },
                                "resources": {
                                    "requests": {"cpu": "100m", "memory": "64Mi"},
                                    "limits": {"cpu": "500m", "memory": "128Mi"},
                                },
                            }
                        ],
                    },
                },
            },
        ),
        resource(
            "Service",
            "v1",
            NAME,
            {
                "selector": LABEL,
                "ports": [{"port": 80, "targetPort": 8080}],
            },
        ),
        resource(
            "HTTPRoute",
            "gateway.networking.k8s.io/v1",
            NAME,
            {
                "parentRefs": [
                    {"name": "inference-gateway", "namespace": GATEWAY_NAMESPACE}
                ],
                "hostnames": [HOST],
                "rules": [
                    {
                        "matches": [{"path": {"type": "PathPrefix", "value": "/"}}],
                        "backendRefs": [{"name": NAME, "port": 80}],
                    }
                ],
            },
        ),
        resource(
            "Job",
            "batch/v1",
            NAME,
            {
                "backoffLimit": 0,
                "activeDeadlineSeconds": 300,
                "ttlSecondsAfterFinished": 600,
                "template": {
                    "metadata": {"labels": LABEL},
                    "spec": {
                        "restartPolicy": "Never",
                        "automountServiceAccountToken": False,
                        "containers": [
                            {
                                "name": "gpu",
                                "image": "nvidia/cuda:12.9.1-base-ubuntu22.04",
                                "command": [
                                    "nvidia-smi",
                                    "--query-gpu=name",
                                    "--format=csv,noheader",
                                ],
                                "resources": {
                                    "requests": {
                                        "cpu": "100m",
                                        "memory": "128Mi",
                                        "nvidia.com/gpu": 1,
                                    },
                                    "limits": {
                                        "cpu": "1",
                                        "memory": "256Mi",
                                        "nvidia.com/gpu": 1,
                                    },
                                },
                            }
                        ],
                    },
                },
            },
        ),
    ]
    MANIFEST.write_text(
        json.dumps({"apiVersion": "v1", "kind": "List", "items": items})
    )
    print(kube("apply", "-f", str(MANIFEST)).stdout, flush=True)
    print(
        kube(
            "rollout", "status", f"deployment/{NAME}", "-n", NAMESPACE, "--timeout=180s"
        ).stdout,
        flush=True,
    )
    print(
        kube(
            "wait",
            "--for=condition=complete",
            f"job/{NAME}",
            "-n",
            NAMESPACE,
            "--timeout=300s",
        ).stdout,
        flush=True,
    )
    gpu_log = kube("logs", f"job/{NAME}", "-n", NAMESPACE).stdout
    (ARTIFACTS / "gpu.log").write_text(gpu_log)
    if os.environ["EXPECTED_GPU"] not in gpu_log:
        raise RuntimeError(f"Unexpected GPU: {gpu_log}")
    print(f"PASS GPU execution: {gpu_log.strip()}", flush=True)
    with (ARTIFACTS / "portforward.log").open("w") as log:
        proc = subprocess.Popen(
            [
                *KUBE,
                "-n",
                GATEWAY_NAMESPACE,
                "port-forward",
                "service/inference-gateway",
                "18080:80",
                "--address=127.0.0.1",
            ],
            stdout=log,
            stderr=log,
        )
        try:
            for _ in range(60):
                if proc.poll() is not None:
                    raise RuntimeError(
                        "Gateway port-forward exited; see portforward.log"
                    )
                try:
                    req = urllib.request.Request(
                        "http://127.0.0.1:18080/", headers={"Host": HOST}
                    )
                    with urllib.request.urlopen(req, timeout=3) as response:
                        body = response.read().decode().strip()
                    if body == NAME:
                        print(
                            "PASS HTTPRoute traffic through gateway pod port-forward",
                            flush=True,
                        )
                        break
                except (urllib.error.URLError, TimeoutError):
                    pass
                time.sleep(2)
            else:
                raise RuntimeError("Gateway did not route the smoke HTTP request")
        finally:
            proc.terminate()
            proc.wait(timeout=10)


if __name__ == "__main__":
    try:
        run()
    finally:
        if MANIFEST.exists():
            for kind in ["pods", "jobs", "deployments", "httproutes"]:
                result = kube(
                    "get",
                    kind,
                    "-n",
                    NAMESPACE,
                    "-l",
                    f"ci-access-smoke={NAME}",
                    "-o",
                    "yaml",
                    check=False,
                )
                (ARTIFACTS / f"{kind}.yaml").write_text(result.stdout + result.stderr)
            result = kube(
                "get",
                "events",
                "-n",
                NAMESPACE,
                "--sort-by=.lastTimestamp",
                check=False,
            )
            (ARTIFACTS / "events.txt").write_text(result.stdout + result.stderr)
            # Delete only resources named by this run; never delete namespaces.
            result = kube(
                "delete", "-f", str(MANIFEST), "--ignore-not-found", "--timeout=90s"
            )
            print(result.stdout, flush=True)
