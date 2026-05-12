"""Probe Prometheus for DCGM workload-attribution coverage.

Run inside the dev pod:
    NS=<your-namespace>
    kubectl cp scripts/inspect_dcgm_attribution.py $NS/planner-dev:/tmp/
    kubectl exec -n $NS planner-dev -- python3 /tmp/inspect_dcgm_attribution.py
"""
import json
import urllib.request

PROM = "http://kube-prometheus-stack-prometheus.monitoring.svc.cluster.local:9090/api/v1/query"


def q(expr: str) -> list:
    url = f"{PROM}?query={urllib.parse.quote(expr)}"
    with urllib.request.urlopen(url) as r:
        return json.load(r).get("data", {}).get("result", [])


import urllib.parse  # noqa: E402

print("=== All DCGM_FI_DEV_POWER_USAGE samples ===")
all_samples = q("DCGM_FI_DEV_POWER_USAGE")
print(f"  total: {len(all_samples)}")
hosts = sorted({s["metric"].get("Hostname", "?") for s in all_samples})
print(f"  hosts: {hosts}")

print("\n=== Samples with non-empty exported_pod ===")
attributed = q('DCGM_FI_DEV_POWER_USAGE{exported_pod!=""}')
print(f"  total: {len(attributed)}")
for s in attributed:
    m = s["metric"]
    ns = m.get("exported_namespace", "")
    pod = m.get("exported_pod", "")
    host = m.get("Hostname", "")
    watts = s["value"][1]
    print(f"  ns={ns!r} pod={pod!r} host={host} watts={watts}")

print("\n=== qwen3-quickstart-specific match ===")
qwen = q('DCGM_FI_DEV_POWER_USAGE{exported_pod=~"^qwen3-quickstart-[0-9]+-.*"}')
print(f"  total: {len(qwen)}")
for s in qwen:
    m = s["metric"]
    print(
        f"  exported_namespace={m.get('exported_namespace')!r} exported_pod={m.get('exported_pod')!r} watts={s['value'][1]}"
    )
