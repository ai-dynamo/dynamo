"""Probe the worker /metrics endpoint and the frontend /metrics endpoint.

Goal: figure out which (if any) endpoint actually exposes
``dynamo_frontend_worker_*`` per-worker gauges that the planner's
``DirectRouterMetricsClient`` is designed to scrape.

Run inside the dev pod:
    NS=<your-namespace>
    kubectl cp scripts/inspect_worker_metrics.py $NS/planner-dev:/tmp/
    kubectl exec -n $NS planner-dev -- python3 /tmp/inspect_worker_metrics.py
"""
from __future__ import annotations

import os
import urllib.error
import urllib.request

DGD = os.environ.get("DYN_PARENT_DGD_K8S_NAME", "qwen3-quickstart")

TARGET_PREFIXES = (
    "dynamo_frontend_worker_",
    "dynamo_frontend_",
    "dynamo_component_",
    "dynamo_router_",
    "dynamo_",
)
PLANNER_TARGET_NAMES = {
    "dynamo_frontend_worker_active_decode_blocks",
    "dynamo_frontend_worker_active_prefill_tokens",
    "dynamo_frontend_worker_last_input_sequence_tokens",
    "dynamo_frontend_worker_last_inter_token_latency_seconds",
    "dynamo_frontend_worker_last_time_to_first_token_seconds",
}

ENDPOINTS = {
    f"worker (vllmworker:9090)": f"http://{DGD}-vllmworker:9090/metrics",
    # The frontend metrics service can be reached via the frontend Service.
    # Operator names the frontend Service `<dgd>-frontend` on port 8000 by
    # default — let's scrape it too.
    f"frontend (frontend:8000)": f"http://{DGD}-frontend:8000/metrics",
    f"frontend (frontend:9090)": f"http://{DGD}-frontend:9090/metrics",
}


def fetch(url: str, timeout: float = 5.0) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        print(f"  ! Unreachable: {exc}")
        return None


def summarize(text: str) -> None:
    """Print a count-by-prefix summary plus the unique metric names that match
    each interesting prefix.
    """
    metric_names: list[str] = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        # First whitespace-delimited token of a sample line is the metric
        # name (possibly with `{labels}` appended).
        first = line.split()[0]
        name = first.split("{", 1)[0]
        metric_names.append(name)
    unique = sorted(set(metric_names))

    print(f"  total samples: {len(metric_names)}  unique metric names: {len(unique)}")
    for prefix in TARGET_PREFIXES:
        matching = [n for n in unique if n.startswith(prefix)]
        if matching:
            print(f"  prefix {prefix!r}: {len(matching)} metric(s)")
            for m in matching:
                marker = "  ***" if m in PLANNER_TARGET_NAMES else ""
                print(f"    {m}{marker}")


for label, url in ENDPOINTS.items():
    print(f"\n=== {label} -> {url} ===")
    text = fetch(url)
    if text is None:
        continue
    summarize(text)

print("\n=== Planner expects these (asterisks above mark which were found) ===")
for n in sorted(PLANNER_TARGET_NAMES):
    print(f"  {n}")
