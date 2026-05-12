# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check DCGM pod-attribution for the qwen3 worker."""
import json
import urllib.parse
import urllib.request

PROM = "http://kube-prometheus-stack-prometheus.monitoring.svc.cluster.local:9090"


def q(query: str):
    url = f"{PROM}/api/v1/query?query={urllib.parse.quote(query)}"
    return json.loads(urllib.request.urlopen(url, timeout=10).read())["data"]["result"]


r = q('DCGM_FI_DEV_POWER_USAGE{exported_pod=~".*qwen3.*"}')
print(f"qwen3 via exported_pod samples: {len(r)}")
for s in r[:1]:
    print(json.dumps(s["metric"], indent=2))

r = q("count by (exported_namespace) (DCGM_FI_DEV_POWER_USAGE)")
print()
print("DCGM samples grouped by exported_namespace:")
for s in r:
    ns = s["metric"].get("exported_namespace", "<no exported_namespace>")
    print(f"  {ns}: {s['value'][1]} samples")

r = q('DCGM_FI_DEV_POWER_USAGE{exported_pod=~"qwen3-quickstart.*"}')
print(f"\nexported_pod=~qwen3-quickstart.* samples: {len(r)}")
