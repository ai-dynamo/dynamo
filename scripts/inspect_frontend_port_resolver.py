# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify ``KubernetesConnector.resolve_frontend_http_port`` against a live
DGD's frontend pod.  Reads each frontend pod and prints both the
named-port-resolved value and the legacy fallback so the diff between them
is unambiguous.

Run inside the dev pod:
    NS=<your-namespace>
    kubectl cp scripts/inspect_frontend_port_resolver.py $NS/planner-dev:/tmp/
    kubectl exec -n $NS planner-dev -- python3 /tmp/inspect_frontend_port_resolver.py
"""
import os

from kubernetes import client, config

from dynamo.planner.connectors.kubernetes import KubernetesConnector

config.load_incluster_config()
v1 = client.CoreV1Api()

ns = os.environ["DYN_PARENT_DGD_K8S_NAMESPACE"]
dgd = os.environ["DYN_PARENT_DGD_K8S_NAME"]
selector = (
    f"nvidia.com/dynamo-graph-deployment-name={dgd},"
    "nvidia.com/dynamo-component-type=frontend"
)
pods = v1.list_namespaced_pod(ns, label_selector=selector).items
print(f"Frontend pods for dgd={dgd!r} in ns={ns!r}: {len(pods)}")
for pod in pods:
    declared_ports = [
        f"{p.name}={p.container_port}"
        for c in (pod.spec.containers or [])
        for p in (c.ports or [])
    ]
    resolved = KubernetesConnector.resolve_frontend_http_port(pod, fallback=9999)
    print(f"  {pod.metadata.name}")
    print(f"    declared ports: {declared_ports}")
    print(f"    resolved http port: {resolved}  (fallback would have been 9999)")
