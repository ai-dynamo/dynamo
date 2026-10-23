# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Deploy the demo into an existing owned namespace; never changes kube context."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

OMNI_REVISION = "257301672237f04db12cd6aed3730c2824bba56b"
OMNI_IMAGE = "hongccc/sglang-omni@sha256:06ed306707bcf3d9e9beb5ba7ce66f09a666cac11c23455ece50c34199c0e985"
ROOT = Path(__file__).resolve().parent
LABELS = {
    "app.kubernetes.io/part-of": "dynamo-voicechat",
    "owner": "rmccormick",
    "purpose": "sglang-omni-duplex",
}
PYTHON_IMAGE = "python:3.12-slim-bookworm"
INSTALL_RUNTIME = "pip install --disable-pip-version-check ai-dynamo-runtime==1.6.0 aiohttp==3.14.3 && exec python /demo/worker.py"
INSTALL_FRONTEND = "pip install --disable-pip-version-check ai-dynamo==1.6.0 uvloop && exec python -m dynamo.frontend --http-port 8080 --discovery-backend etcd --request-plane tcp --event-plane zmq"
BOOTSTRAP = f"""#!/usr/bin/env bash
set -euo pipefail
git clone --filter=blob:none https://github.com/sgl-project/sglang-omni.git /workspace/omni
git -C /workspace/omni checkout {OMNI_REVISION}
git -C /workspace/omni apply /demo/playback.patch
git -C /workspace/omni apply /demo/tools.patch
python3 -m pip install --no-deps --no-build-isolation -e /workspace/omni
export PYTHONPATH=/demo:/workspace/omni
exec python3 /demo/backend.py
"""


def deployment(name, containers, *, volumes=(), spec=None):
    labels = {**LABELS, "app": name}
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": name, "labels": LABELS},
        "spec": {
            "replicas": 1,
            "strategy": {"type": "Recreate"},
            "selector": {"matchLabels": {"app": name}},
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "terminationGracePeriodSeconds": 60,
                    "containers": containers,
                    "volumes": list(volumes),
                    **(spec or {}),
                },
            },
        },
    }


def service(name, port):
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": name, "labels": LABELS},
        "spec": {
            "selector": {"app": name},
            "ports": [{"port": port, "targetPort": port}],
        },
    }


def manifests(args):
    code_volume = {
        "name": "code",
        "configMap": {"name": "voicechat-code", "defaultMode": 0o755},
    }
    code_mount = {"name": "code", "mountPath": "/demo", "readOnly": True}
    common_env = [
        {"name": "ETCD_ENDPOINTS", "value": "http://voicechat-etcd:2379"},
        {"name": "DYN_SYSTEM_PORT", "value": "-1"},
        {"name": "PYTHONUNBUFFERED", "value": "1"},
        {"name": "POD_IP", "valueFrom": {"fieldRef": {"fieldPath": "status.podIP"}}},
        {"name": "DYN_TCP_RPC_HOST", "value": "$(POD_IP)"},
        {"name": "DYN_TCP_RESPONSE_STREAM_HOST", "value": "$(POD_IP)"},
    ]
    runtime_resources = {
        "requests": {"cpu": "500m", "memory": "512Mi"},
        "limits": {"cpu": "2", "memory": "2Gi"},
    }
    config = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": "voicechat-code", "labels": LABELS},
        "data": {
            **{p.name: p.read_text() for p in ROOT.glob("*.py")},
            "playback.patch": (ROOT / "playback.patch").read_text(),
            "tools.patch": (ROOT / "tools.patch").read_text(),
            "tools.js": (ROOT / "tools.js").read_text(),
            "bootstrap.sh": BOOTSTRAP,
        },
    }
    etcd = deployment(
        "voicechat-etcd",
        [
            {
                "name": "etcd",
                "image": "registry.k8s.io/etcd:3.5.21-0",
                "command": [
                    "etcd",
                    "--listen-client-urls=http://0.0.0.0:2379",
                    "--advertise-client-urls=http://voicechat-etcd:2379",
                    "--data-dir=/data",
                ],
                "resources": {
                    "requests": {"cpu": "100m", "memory": "128Mi"},
                    "limits": {"cpu": "1", "memory": "512Mi"},
                },
                "volumeMounts": [{"name": "data", "mountPath": "/data"}],
                "readinessProbe": {"tcpSocket": {"port": 2379}},
            }
        ],
        volumes=[{"name": "data", "emptyDir": {}}],
    )
    frontend = deployment(
        "voicechat-frontend",
        [
            {
                "name": "frontend",
                "image": PYTHON_IMAGE,
                "command": ["bash", "-lc", INSTALL_FRONTEND],
                "env": common_env,
                "volumeMounts": [code_mount],
                "resources": runtime_resources,
                "readinessProbe": {
                    "httpGet": {"path": "/health", "port": 8080},
                    "periodSeconds": 10,
                    "timeoutSeconds": 8,
                },
            },
            {
                "name": "web",
                "image": PYTHON_IMAGE,
                "command": [
                    "bash",
                    "-lc",
                    "pip install --disable-pip-version-check aiohttp==3.14.3 && exec python /demo/web.py",
                ],
                "volumeMounts": [code_mount],
                "resources": runtime_resources,
                "readinessProbe": {
                    "httpGet": {"path": "/health", "port": 8082},
                    "timeoutSeconds": 8,
                },
            },
        ],
        volumes=[code_volume],
    )
    native = {
        "name": "omni",
        "image": OMNI_IMAGE,
        "command": ["bash", "/demo/bootstrap.sh"],
        "env": [
            {"name": "HF_HOME", "value": "/models/voicechat-demo/huggingface"},
            {"name": "PYTHONUNBUFFERED", "value": "1"},
        ],
        "resources": {
            "requests": {
                "cpu": "8",
                "memory": "80Gi",
                "nvidia.com/gpu": "1",
                "ephemeral-storage": "40Gi",
            },
            "limits": {
                "cpu": "24",
                "memory": "192Gi",
                "nvidia.com/gpu": "1",
                "ephemeral-storage": "100Gi",
            },
        },
        "volumeMounts": [
            code_mount,
            {"name": "models", "mountPath": "/models"},
            {"name": "shm", "mountPath": "/dev/shm"},
        ],
        "startupProbe": {
            "httpGet": {"path": "/health", "port": 8097},
            "periodSeconds": 10,
            "failureThreshold": 360,
        },
        "readinessProbe": {
            "httpGet": {"path": "/health", "port": 8097},
            "periodSeconds": 10,
        },
    }
    registrar = {
        "name": "dynamo-worker",
        "image": PYTHON_IMAGE,
        "command": ["bash", "-lc", INSTALL_RUNTIME],
        "env": [*common_env, {"name": "OMNI_URL", "value": "http://$(POD_IP):8097"}],
        "volumeMounts": [code_mount],
        "resources": runtime_resources,
    }
    backend_spec = {
        "tolerations": [
            {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}
        ]
    }
    if args.node:
        backend_spec["nodeSelector"] = {"kubernetes.io/hostname": args.node}
    backend = deployment(
        "voicechat-backend",
        [native, registrar],
        volumes=[
            code_volume,
            {"name": "models", "persistentVolumeClaim": {"claimName": args.cache_pvc}},
            {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "32Gi"}},
        ],
        spec=backend_spec,
    )
    for resource, files in (
        (frontend, ("web.py", "tools.js", "demo_tools.py")),
        (
            backend,
            (
                "worker.py",
                "backend.py",
                "demo_tools.py",
                "native_tools.py",
                "bootstrap.sh",
                "playback.patch",
                "tools.patch",
            ),
        ),
    ):
        checksum = hashlib.sha256(
            json.dumps(
                {name: config["data"][name] for name in files}, sort_keys=True
            ).encode()
        ).hexdigest()
        resource["spec"]["template"]["metadata"]["annotations"] = {
            "checksum/code": checksum
        }
    resources = [
        config,
        etcd,
        service("voicechat-etcd", 2379),
        frontend,
        service("voicechat-frontend", 8080),
    ]
    resources[-1]["spec"]["ports"].append(
        {"name": "web", "port": 8082, "targetPort": 8082}
    )
    resources[-1]["spec"]["ports"][0]["name"] = "frontend"
    if not args.control_only:
        resources += [backend, service("voicechat-backend", 8097)]
    for resource in resources:
        resource["metadata"]["namespace"] = args.namespace
    return {"apiVersion": "v1", "kind": "List", "items": resources}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--cache-pvc", default="shared-model-cache")
    parser.add_argument("--node", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--control-only",
        action="store_true",
        help="Start CPU services before allocating the GPU",
    )
    parser.add_argument("--delete", action="store_true")
    args = parser.parse_args()
    command = ["kubectl", "--context", args.context, "--namespace", args.namespace]
    if args.delete:
        subprocess.run(
            [
                *command,
                "delete",
                "deployment,service,configmap",
                "-l",
                "app.kubernetes.io/part-of=dynamo-voicechat",
            ],
            check=True,
        )
    elif args.dry_run:
        print(json.dumps(manifests(args), indent=2))
    else:
        subprocess.run(
            [*command, "apply", "-f", "-"],
            input=json.dumps(manifests(args)),
            text=True,
            check=True,
        )
