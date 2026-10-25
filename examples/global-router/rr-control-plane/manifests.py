# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
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
"""Render an isolated Global Ingress + MCS discovery deployment as Kubernetes JSON."""

import argparse
import json
import re
from pathlib import Path


def render(namespace, image, routing):
    if not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", namespace):
        raise ValueError("namespace must be a DNS label")
    if "@sha256:" not in image:
        raise ValueError("use a digest-pinned runtime image")
    labels = {"app.kubernetes.io/part-of": "global-router-round-robin"}

    def meta(name):
        return {"name": name, "namespace": namespace, "labels": labels}

    objects = [
        {"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": namespace}},
        {
            "apiVersion": "v1",
            "kind": "ServiceAccount",
            "metadata": meta("global-discovery"),
        },
        {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": meta("global-discovery-routing"),
            "data": {"routing.json": json.dumps(routing, sort_keys=True)},
        },
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "ClusterRole",
            "metadata": {"name": namespace + "-discovery", "labels": labels},
            "rules": [
                {
                    "apiGroups": ["discovery.k8s.io"],
                    "resources": ["endpointslices"],
                    "verbs": ["get", "list", "watch"],
                }
            ],
        },
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "ClusterRoleBinding",
            "metadata": {"name": namespace + "-discovery", "labels": labels},
            "roleRef": {
                "apiGroup": "rbac.authorization.k8s.io",
                "kind": "ClusterRole",
                "name": namespace + "-discovery",
            },
            "subjects": [
                {
                    "kind": "ServiceAccount",
                    "name": "global-discovery",
                    "namespace": namespace,
                }
            ],
        },
    ]
    for name, port, command, args, health in [
        (
            "global-discovery",
            8080,
            "global-discovery-service",
            [
                "--store",
                "karmada",
                "--router-namespace",
                namespace,
                "--router-service",
                "global-router",
                "--router-port-name",
                "http",
                "--routing-config",
                "/etc/global-discovery/routing.json",
            ],
            "/healthz",
        ),
        (
            "global-router",
            8000,
            "global-ingress",
            [
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
                "--discovery-url",
                "http://global-discovery:8080/",
                "--discovery-interval-secs",
                "2",
            ],
            "/health",
        ),
    ]:
        pod = {
            "serviceAccountName": "global-discovery"
            if name == "global-discovery"
            else "default",
            "automountServiceAccountToken": name == "global-discovery",
            "securityContext": {
                "runAsNonRoot": True,
                "runAsUser": 65532,
                "runAsGroup": 65532,
                "seccompProfile": {"type": "RuntimeDefault"},
            },
            "containers": [
                {
                    "name": name,
                    "image": image,
                    "command": [command],
                    "args": args,
                    "env": [
                        {
                            "name": "DYN_LOG",
                            "value": "info,dynamo_global_ingress=debug",
                        },
                        {"name": "DYN_LOGGING_CONSOLE_FORMAT", "value": "jsonl"},
                    ],
                    "ports": [{"name": "http", "containerPort": port}],
                    "readinessProbe": {
                        "httpGet": {"path": health, "port": "http"},
                        "periodSeconds": 3,
                    },
                    "livenessProbe": {
                        "tcpSocket": {"port": "http"},
                        "initialDelaySeconds": 20,
                    },
                    "resources": {
                        "requests": {"cpu": "100m", "memory": "128Mi"},
                        "limits": {"memory": "512Mi"},
                    },
                    "securityContext": {
                        "allowPrivilegeEscalation": False,
                        "readOnlyRootFilesystem": True,
                        "capabilities": {"drop": ["ALL"]},
                    },
                }
            ],
        }
        if name == "global-discovery":
            pod["volumes"] = [
                {"name": "routing", "configMap": {"name": "global-discovery-routing"}}
            ]
            pod["containers"][0]["volumeMounts"] = [
                {
                    "name": "routing",
                    "mountPath": "/etc/global-discovery",
                    "readOnly": True,
                }
            ]
        objects.extend(
            [
                {
                    "apiVersion": "v1",
                    "kind": "Service",
                    "metadata": meta(name),
                    "spec": {
                        "selector": {"app": name},
                        "ports": [{"name": "http", "port": port, "targetPort": "http"}],
                    },
                },
                {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "metadata": meta(name),
                    "spec": {
                        "replicas": 1,
                        "selector": {"matchLabels": {"app": name}},
                        "template": {
                            "metadata": {"labels": {"app": name, **labels}},
                            "spec": pod,
                        },
                    },
                },
            ]
        )
    return {"apiVersion": "v1", "kind": "List", "items": objects}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default="global-router-rr")
    parser.add_argument("--image", required=True)
    parser.add_argument("--routing-config", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            render(
                args.namespace, args.image, json.loads(args.routing_config.read_text())
            ),
            indent=2,
        )
    )
