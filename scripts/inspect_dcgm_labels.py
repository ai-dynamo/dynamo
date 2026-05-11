"""Inspect DCGM label structure on the live cluster.

Run inside the dev pod:
    NS=<your-namespace>
    kubectl cp scripts/inspect_dcgm_labels.py $NS/planner-dev:/tmp/
    kubectl exec -n $NS planner-dev -- env DYN_PARENT_DGD_K8S_NAMESPACE=$NS \\
        python3 /tmp/inspect_dcgm_labels.py
"""
import json
import os
import urllib.parse
import urllib.request

PROM = "http://kube-prometheus-stack-prometheus.monitoring.svc.cluster.local:9090"
NS = os.environ.get("DYN_PARENT_DGD_K8S_NAMESPACE") or os.environ.get("POD_NAMESPACE")
if not NS:
    raise SystemExit(
        "Set DYN_PARENT_DGD_K8S_NAMESPACE (or POD_NAMESPACE) to your K8s namespace "
        "before running this probe."
    )


def q(query: str):
    url = f"{PROM}/api/v1/query?query={urllib.parse.quote(query)}"
    return json.loads(urllib.request.urlopen(url, timeout=10).read())["data"]["result"]


def show(label: str, query: str, n: int = 3):
    print(f"--- {label} ---")
    print(f"query: {query}")
    r = q(query)
    print(f"samples: {len(r)}")
    for s in r[:n]:
        print(json.dumps(s["metric"], indent=2))
        print()


show("DCGM_FI_DEV_POWER_USAGE (any)", "DCGM_FI_DEV_POWER_USAGE", 2)
show(
    "DCGM with exported_namespace label",
    'DCGM_FI_DEV_POWER_USAGE{exported_namespace!=""}',
    2,
)
show(
    f"DCGM with namespace={NS}",
    f'DCGM_FI_DEV_POWER_USAGE{{namespace="{NS}"}}',
    2,
)
show(
    f"DCGM with exported_namespace={NS}",
    f'DCGM_FI_DEV_POWER_USAGE{{exported_namespace="{NS}"}}',
    2,
)
