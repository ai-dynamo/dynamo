# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Render an isolated CPU Mocker pool using ordinary Kubernetes resources."""

import argparse
import json
import re

ETCD_IMAGE = (
    "registry.k8s.io/etcd@sha256:"
    "397189418d1a00e500c0605ad18d1baf3b541a1004d768448c367e48071622e5"
)


def render(
    namespace,
    stem,
    site_id,
    planner_image,
    relay_image,
    model,
    node_selector,
    worker_replicas=2,
    mocker_args=(),
    tolerations=(),
    frontend_cpu=None,
):
    for value in (namespace, stem):
        if len(value) > 40 or not re.fullmatch(r"[a-z0-9]([a-z0-9-]*[a-z0-9])?", value):
            raise ValueError(
                "namespace and stem must be DNS labels of at most 40 characters"
            )
    for image in (planner_image, relay_image):
        if not re.search(r"@sha256:[a-f0-9]{64}$", image):
            raise ValueError(
                "planner and Relay images must use immutable sha256 digests"
            )
    shared = {
        "app.kubernetes.io/part-of": "global-router-v2-rr",
        "app.kubernetes.io/instance": stem,
    }

    def labels(component):
        return {**shared, "app.kubernetes.io/component": component}

    def meta(name):
        return {"name": name, "namespace": namespace, "labels": shared}

    def resources(cpu, memory, cpu_limit, memory_limit):
        return {
            "requests": {"cpu": cpu, "memory": memory},
            "limits": {"cpu": cpu_limit, "memory": memory_limit},
        }

    env = [
        {"name": name, "value": value}
        for name, value in {
            "DYN_DISCOVERY_BACKEND": "etcd",
            "DYN_EVENT_PLANE": "zmq",
            "DYN_REQUEST_PLANE": "tcp",
            "DYN_NAMESPACE": stem,
            "ETCD_ENDPOINTS": f"http://{stem}-etcd:2379",
            "RUST_LOG": "warn",
        }.items()
    ]

    def deploy(component, container, replicas=1, volumes=None):
        pod = {"automountServiceAccountToken": False, "containers": [container]}
        if node_selector:
            pod["nodeSelector"] = node_selector
        if tolerations:
            pod["tolerations"] = list(tolerations)
        if volumes:
            pod["volumes"] = volumes
        if component == "worker":
            pod["affinity"] = {
                "podAntiAffinity": {
                    "preferredDuringSchedulingIgnoredDuringExecution": [
                        {
                            "weight": 100,
                            "podAffinityTerm": {
                                "topologyKey": "kubernetes.io/hostname",
                                "labelSelector": {"matchLabels": labels(component)},
                            },
                        }
                    ]
                }
            }
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": meta(f"{stem}-{component}"),
            "spec": {
                "replicas": replicas,
                "selector": {"matchLabels": labels(component)},
                "template": {"metadata": {"labels": labels(component)}, "spec": pod},
            },
        }

    def service(component, ports):
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": meta(f"{stem}-{component}"),
            "spec": {
                "type": "ClusterIP",
                "selector": labels(component),
                "ports": ports,
            },
        }

    def tcp_probe(port):
        return {
            "tcpSocket": {"port": port},
            "periodSeconds": 5,
            "timeoutSeconds": 2,
            "failureThreshold": 6,
        }

    etcd = {
        "name": "etcd",
        "image": ETCD_IMAGE,
        "command": ["/usr/local/bin/etcd"],
        "args": [
            "--data-dir=/data",
            "--listen-client-urls=http://0.0.0.0:2379",
            f"--advertise-client-urls=http://{stem}-etcd:2379",
            "--listen-peer-urls=http://0.0.0.0:2380",
            f"--initial-advertise-peer-urls=http://{stem}-etcd:2380",
            f"--initial-cluster=default=http://{stem}-etcd:2380",
        ],
        "ports": [{"name": "client", "containerPort": 2379}],
        "readinessProbe": tcp_probe(2379),
        "resources": resources("100m", "128Mi", "1", "512Mi"),
        "volumeMounts": [{"name": "data", "mountPath": "/data"}],
    }
    worker = {
        "name": "mocker",
        "image": planner_image,
        "command": ["python3", "-m", "dynamo.mocker"],
        "args": [
            "--model-path",
            model,
            "--model-name",
            model,
            "--num-workers",
            "1",
            "--speedup-ratio",
            "1",
            # Later flags win, so these can override the defaults above.
            *mocker_args,
        ],
        "env": [
            *env,
            {"name": "DYN_SYSTEM_ENABLED", "value": "true"},
            {"name": "DYN_SYSTEM_PORT", "value": "9090"},
            {"name": "DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS", "value": '["generate"]'},
        ],
        "ports": [{"name": "system", "containerPort": 9090}],
        "startupProbe": {
            "httpGet": {"path": "/live", "port": 9090},
            "periodSeconds": 5,
            "failureThreshold": 120,
        },
        "readinessProbe": {
            "httpGet": {"path": "/health", "port": 9090},
            "periodSeconds": 5,
            "timeoutSeconds": 3,
        },
        "resources": resources("500m", "1Gi", "2", "4Gi"),
    }
    model_ready = (
        "import json,os,urllib.request; "
        "v=json.load(urllib.request.urlopen('http://127.0.0.1:8000/v1/models',timeout=2)); "
        "assert any(m['id']==os.environ['MODEL_NAME'] for m in v['data'])"
    )
    frontend = {
        "name": "frontend",
        "image": planner_image,
        "command": ["python3", "-m", "dynamo.frontend"],
        "args": ["--http-port", "8000", "--router-mode", "round-robin"],
        "env": [
            *env,
            {"name": "DYN_NAMESPACE_PREFIX", "value": stem},
            {"name": "MODEL_NAME", "value": model},
        ],
        "ports": [{"name": "http", "containerPort": 8000}],
        "startupProbe": {
            "httpGet": {"path": "/health", "port": 8000},
            "periodSeconds": 5,
            "failureThreshold": 120,
        },
        "readinessProbe": {
            "exec": {"command": ["python3", "-c", model_ready]},
            "periodSeconds": 5,
            "timeoutSeconds": 4,
        },
        "resources": resources(
            frontend_cpu or "250m", "512Mi", frontend_cpu or "1", "2Gi"
        ),
    }
    relay_config = {
        "dc_id": f"{site_id}-{stem}",
        "runtime_namespace": stem,
        "watch_namespaces": [stem],
        "wan_listen": "0.0.0.0:50051",
        "stats_listen": "0.0.0.0:50052",
        "stats_allow_non_loopback": True,
    }
    relay = {
        "name": "pool-relay",
        "image": relay_image,
        "args": ["/etc/relay/config.json"],
        "env": env,
        "ports": [
            {"name": "wan-grpc", "containerPort": 50051},
            {"name": "stats-grpc", "containerPort": 50052},
        ],
        "startupProbe": {
            "tcpSocket": {"port": 50051},
            "periodSeconds": 5,
            "failureThreshold": 120,
        },
        "readinessProbe": tcp_probe(50052),
        "resources": resources("100m", "128Mi", "1", "1Gi"),
        "securityContext": {
            "allowPrivilegeEscalation": False,
            "readOnlyRootFilesystem": True,
            "capabilities": {"drop": ["ALL"]},
        },
        "volumeMounts": [
            {"name": "config", "mountPath": "/etc/relay", "readOnly": True}
        ],
    }
    items = [
        {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {"name": namespace, "labels": shared},
        },
        {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": meta(f"{stem}-relay-config"),
            "data": {"config.json": json.dumps(relay_config)},
        },
        deploy(
            "etcd", etcd, volumes=[{"name": "data", "emptyDir": {"sizeLimit": "1Gi"}}]
        ),
        service("etcd", [{"name": "client", "port": 2379, "targetPort": 2379}]),
        deploy("worker", worker, replicas=worker_replicas),
        deploy("frontend", frontend),
        service("frontend", [{"name": "http", "port": 8000, "targetPort": 8000}]),
        deploy(
            "pool-relay",
            relay,
            volumes=[{"name": "config", "configMap": {"name": f"{stem}-relay-config"}}],
        ),
        service(
            "pool-relay",
            [
                {"name": "wan-grpc", "port": 50051, "targetPort": 50051},
                {"name": "stats-grpc", "port": 50052, "targetPort": 50052},
            ],
        ),
    ]
    return {"apiVersion": "v1", "kind": "List", "items": items}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default="global-router-v2-rr")
    parser.add_argument("--stem", default="rr-mocker")
    parser.add_argument("--site-id", required=True)
    parser.add_argument("--planner-image", required=True)
    parser.add_argument("--relay-image", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument(
        "--node-selector", action="append", default=[], metavar="KEY=VALUE"
    )
    parser.add_argument(
        "--toleration",
        action="append",
        default=[],
        metavar="KEY=VALUE:EFFECT",
        type=toleration,
    )
    parser.add_argument("--worker-replicas", type=int, default=2)
    parser.add_argument(
        "--mocker-arg",
        action="append",
        default=[],
        help="Extra Mocker argument; repeat for each token, e.g. --mocker-arg=--max-num-seqs --mocker-arg=16",
    )
    parser.add_argument(
        "--frontend-cpu", help="Frontend CPU request and limit, e.g. 2"
    )
    args = parser.parse_args()
    selectors = dict(value.split("=", 1) for value in args.node_selector)
    print(
        json.dumps(
            render(
                args.namespace,
                args.stem,
                args.site_id,
                args.planner_image,
                args.relay_image,
                args.model,
                selectors,
                worker_replicas=args.worker_replicas,
                mocker_args=args.mocker_arg,
                tolerations=args.toleration,
                frontend_cpu=args.frontend_cpu,
            ),
            indent=2,
        )
    )


def toleration(value):
    """A ``KEY=VALUE:EFFECT`` toleration, e.g. ``bench=true:NoSchedule``."""
    key, rest = value.split("=", 1)
    toleration_value, effect = rest.rsplit(":", 1)
    return {
        "key": key,
        "operator": "Equal",
        "value": toleration_value,
        "effect": effect,
    }


if __name__ == "__main__":
    main()
