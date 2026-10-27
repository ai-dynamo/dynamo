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


def render(
    namespace,
    image,
    routing,
    replicas=1,
    router_policy=None,
    router_cpu=None,
    node_selector=None,
    tolerations=(),
    ingress_args=(),
):
    """``ingress_args`` are appended to the Global Ingress arguments, for
    example ``["--region", "us-east-2"]``."""
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
                *(["--router-policy", router_policy] if router_policy else []),
                *ingress_args,
            ],
            "/health",
        ),
    ]:
        requests = {"cpu": "100m", "memory": "128Mi"}
        limits = {"memory": "512Mi"}
        if name == "global-router" and router_cpu:
            requests["cpu"] = limits["cpu"] = router_cpu
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
                    "resources": {"requests": requests, "limits": limits},
                    "securityContext": {
                        "allowPrivilegeEscalation": False,
                        "readOnlyRootFilesystem": True,
                        "capabilities": {"drop": ["ALL"]},
                    },
                }
            ],
        }
        if node_selector:
            pod["nodeSelector"] = node_selector
        if tolerations:
            pod["tolerations"] = list(tolerations)
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
                        "replicas": replicas,
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
    parser.add_argument(
        "--replicas", type=int, default=1, help="Global Ingress and GDS replicas each"
    )
    parser.add_argument(
        "--router-policy",
        choices=["round-robin", "least-in-flight"],
        help="Global Ingress pool selection; omitted uses the binary's default",
    )
    parser.add_argument("--router-cpu", help="Global Ingress CPU request and limit")
    parser.add_argument(
        "--node-selector", action="append", default=[], metavar="KEY=VALUE"
    )
    parser.add_argument(
        "--toleration", action="append", default=[], metavar="KEY=VALUE:EFFECT"
    )
    args = parser.parse_args()
    tolerations = []
    for value in args.toleration:
        key, rest = value.split("=", 1)
        toleration_value, effect = rest.rsplit(":", 1)
        tolerations.append(
            {
                "key": key,
                "operator": "Equal",
                "value": toleration_value,
                "effect": effect,
            }
        )
    print(
        json.dumps(
            render(
                args.namespace,
                args.image,
                json.loads(args.routing_config.read_text()),
                replicas=args.replicas,
                router_policy=args.router_policy,
                router_cpu=args.router_cpu,
                node_selector=dict(value.split("=", 1) for value in args.node_selector),
                tolerations=tolerations,
            ),
            indent=2,
        )
    )
