# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quick sanity check: can the pod's SA access DGDs and Prometheus?

Run inside the dev pod:
    NS=<your-namespace>
    kubectl cp scripts/dev/test_k8s_access.py $NS/planner-dev:/tmp/
    kubectl exec -n $NS planner-dev -- env DYN_PARENT_DGD_K8S_NAMESPACE=$NS \\
        python3 /tmp/test_k8s_access.py
"""
import json
import os
import urllib.request

# 1. Test K8s API access via in-cluster config
from kubernetes import client, config

NS = os.environ.get("DYN_PARENT_DGD_K8S_NAMESPACE") or os.environ.get("POD_NAMESPACE")
if not NS:
    raise SystemExit(
        "Set DYN_PARENT_DGD_K8S_NAMESPACE (or POD_NAMESPACE) to your K8s namespace "
        "before running this probe."
    )

config.load_incluster_config()
api = client.CustomObjectsApi()
dgds = api.list_namespaced_custom_object(
    "nvidia.com", "v1alpha1", NS, "dynamographdeployments"
)
for item in dgds["items"]:
    name = item["metadata"]["name"]
    state = item.get("status", {}).get("state", "unknown")
    services = list(item.get("spec", {}).get("services", {}).keys())
    print(f"DGD: {name}  state={state}  services={services}")
print(f"K8s API access (ns={NS}): OK\n")

# 2. Test Prometheus connectivity
prom_url = "http://kube-prometheus-stack-prometheus.monitoring.svc.cluster.local:9090/api/v1/query?query=up"
try:
    resp = urllib.request.urlopen(prom_url, timeout=5)
    data = json.loads(resp.read())
    print(f"Prometheus: {data['status']} ({len(data['data']['result'])} targets)")
    print("Prometheus access: OK\n")
except Exception as e:
    print(f"Prometheus access: FAILED ({e})\n")

# 3. Test NATS connectivity
nats_url = "http://dynamo-platform-nats.dynamo-system.svc.cluster.local:8222/varz"
try:
    resp = urllib.request.urlopen(nats_url, timeout=5)
    data = json.loads(resp.read())
    print(f"NATS: version={data['version']} connections={data['connections']}")
    print("NATS access: OK")
except Exception as e:
    print(f"NATS access: FAILED ({e})")
