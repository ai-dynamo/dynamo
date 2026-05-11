"""Live cluster metric label inspector — used to ground integration test assumptions."""
import json
import urllib.parse
import urllib.request

PROM = "http://kube-prometheus-stack-prometheus.monitoring.svc.cluster.local:9090"


def q(query: str):
    url = f"{PROM}/api/v1/query?query={urllib.parse.quote(query)}"
    return json.loads(urllib.request.urlopen(url, timeout=10).read())["data"]["result"]


def show(label: str, query: str, n: int = 2):
    print(f"--- {label} ---")
    print(f"query: {query}")
    r = q(query)
    print(f"samples: {len(r)}")
    for s in r[:n]:
        print(json.dumps(s["metric"], indent=2))
    print()


show(
    "DCGM power on qwen3 pods",
    'DCGM_FI_DEV_POWER_USAGE{pod=~".*qwen3.*"}',
)
show(
    "Planner-style DCGM regex (worker)",
    'DCGM_FI_DEV_POWER_USAGE{pod=~"qwen3-quickstart-decode-.*"}',
)
show(
    "router_requests_total",
    "dynamo_component_router_requests_total",
)
show(
    "frontend_requests_total",
    "dynamo_frontend_requests_total",
)
show(
    "router TTFT count",
    "dynamo_component_router_time_to_first_token_seconds_count",
)
show(
    "frontend TTFT count",
    "dynamo_frontend_time_to_first_token_seconds_count",
)
