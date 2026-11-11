#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Check that one frontend serves every model of a DGD and that each request reaches
# the worker of the model it names.
#
# It lists the models on the frontend, then sends a different number of chat requests
# to each model, all in parallel, and compares that with the counter
# vllm:request_success_total read from each worker's metrics port before and after.
# A request for a model that does not exist must return 404.
#
# The check runs inside the frontend pod, which can reach both its own HTTP port and
# the worker pods, so it needs nothing but kubectl on your side.
#
# Usage: ./verify-routing.sh [dgd-name]          (default: qwen3-4b-shared)
#
# Environment:
#   NAMESPACE   namespace of the DGD                          (default: default)
#   KUBECTL     kubectl command, e.g. "microk8s kubectl"      (default: kubectl)
#   REQUESTS    requests sent to the first model; each next
#               model gets one more                           (default: 3)
set -euo pipefail

DGD_NAME="${1:-qwen3-4b-shared}"
NAMESPACE="${NAMESPACE:-default}"
REQUESTS="${REQUESTS:-3}"
read -r -a KUBECTL_CMD <<< "${KUBECTL:-kubectl}"

FRONTEND_POD="$("${KUBECTL_CMD[@]}" -n "${NAMESPACE}" get pods \
  -l "experiment.nvidia.com/dgd=${DGD_NAME},experiment.nvidia.com/role=frontend" \
  --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')"
# "<model>=<pod IP>" for every running worker of the DGD.
WORKERS="$("${KUBECTL_CMD[@]}" -n "${NAMESPACE}" get pods \
  -l "experiment.nvidia.com/dgd=${DGD_NAME},experiment.nvidia.com/role=worker" \
  --field-selector=status.phase=Running \
  -o jsonpath='{range .items[*]}{.metadata.labels.experiment\.nvidia\.com/model}={.status.podIP}{" "}{end}')"

if [ -z "${FRONTEND_POD}" ] || [ -z "${WORKERS}" ]; then
  echo "no running frontend or workers found for DGD ${DGD_NAME} in namespace ${NAMESPACE}" >&2
  exit 1
fi

# shellcheck disable=SC2086  # WORKERS is a space-separated argument list on purpose
"${KUBECTL_CMD[@]}" -n "${NAMESPACE}" exec -i "${FRONTEND_POD}" -- \
  python3 - "${REQUESTS}" ${WORKERS} <<'PY'
import json
import sys
import threading
import urllib.error
import urllib.request

FRONTEND = "http://localhost:8000"
first = int(sys.argv[1])
workers = dict(arg.split("=", 1) for arg in sys.argv[2:])  # model -> pod IP


def finished(ip: str) -> float:
    """Requests this engine has finished, summed over finish reasons."""
    with urllib.request.urlopen(f"http://{ip}:9090/metrics", timeout=10) as r:
        return sum(
            float(line.rsplit(" ", 1)[1])
            for line in r.read().decode().splitlines()
            if line.startswith("vllm:request_success_total")
        )


def chat(model: str) -> int:
    body = json.dumps({"model": model, "max_tokens": 8,
                       "messages": [{"role": "user", "content": "Reply with one word."}]}).encode()
    req = urllib.request.Request(f"{FRONTEND}/v1/chat/completions", body, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


with urllib.request.urlopen(f"{FRONTEND}/v1/models", timeout=10) as r:
    listed = sorted(m["id"] for m in json.load(r)["data"])
missing = sorted(set(workers) - set(listed))
print(f"frontend lists {len(listed)} models; {len(workers)} workers deployed; missing: {missing or 'none'}")

plan = {model: first + i for i, model in enumerate(sorted(workers))}
before = {model: finished(ip) for model, ip in workers.items()}
statuses: list[int] = []
lock = threading.Lock()


def send(model: str) -> None:
    status = chat(model)
    with lock:
        statuses.append(status)


threads = [threading.Thread(target=send, args=(model,)) for model, n in plan.items() for _ in range(n)]
for t in threads:
    t.start()
for t in threads:
    t.join()
unknown = chat("no-such-model")

ok = not missing and all(s == 200 for s in statuses) and unknown == 404
print(f"sent {len(statuses)} requests, {sum(s == 200 for s in statuses)} returned 200; unknown model returned {unknown}")
for model, ip in sorted(workers.items()):
    got = int(finished(ip) - before[model])
    verdict = "PASS" if got == plan[model] else "FAIL"
    ok = ok and got == plan[model]
    print(f"  {model}: worker finished +{got}, sent {plan[model]}  {verdict}")
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
