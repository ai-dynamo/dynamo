#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Send continuous chat-completion requests to the mocker frontend.
# Keeps enough traffic in flight for the planner to observe non-zero metrics.
#
# Usage:
#   ./send_load.sh <namespace>
#
# In a second terminal, watch the planner observe router metrics:
#   kubectl logs -n <namespace> -l nvidia.com/dynamo-component-type=planner -f \
#     | grep -E "Observed|Predicted|scaling|replicas"

set -euo pipefail

NAMESPACE="${1:?Usage: $0 <namespace>}"
DGD_NAME="mocker-router-metrics-test"
MODEL="nvidia/Llama-3.1-8B-Instruct-FP8"
CONCURRENCY=8          # parallel requests in flight
LOCAL_PORT=18000       # local port for port-forward

PAYLOAD=$(cat <<EOF
{
  "model": "${MODEL}",
  "messages": [{"role": "user", "content": "Count from 1 to 50."}],
  "max_tokens": 128,
  "stream": false
}
EOF
)

# Find the frontend service (operator names it <dgd-name>-<service-name-lowercase>)
SVC="${DGD_NAME}-frontend"
echo "Waiting for service ${SVC} to exist in namespace ${NAMESPACE}..."
kubectl wait --for=jsonpath='{.metadata.name}'="${SVC}" \
    service/"${SVC}" -n "${NAMESPACE}" --timeout=120s 2>/dev/null || true

echo "Port-forwarding ${SVC}:8000 -> localhost:${LOCAL_PORT} ..."
kubectl port-forward -n "${NAMESPACE}" "svc/${SVC}" "${LOCAL_PORT}:8000" &
PF_PID=$!
trap "kill ${PF_PID} 2>/dev/null || true" EXIT
sleep 2

echo "Sending ${CONCURRENCY} parallel requests in a loop. Ctrl-C to stop."
echo ""
echo "To watch planner metrics in real time:"
echo "  kubectl logs -n ${NAMESPACE} -l nvidia.com/dynamo-component-type=planner -f \\"
echo "    | grep -E 'Observed|Predicted|scaling|replicas'"
echo ""
echo "To query Prometheus directly (requires port-forward to Prometheus):"
echo "  curl -sG http://localhost:9090/api/v1/query \\"
echo "    --data-urlencode 'query=dynamo_component_router_requests_total' | python3 -m json.tool"
echo ""

REQ=0
while true; do
    for _ in $(seq 1 "${CONCURRENCY}"); do
        curl -sf -o /dev/null \
            -H "Content-Type: application/json" \
            -d "${PAYLOAD}" \
            "http://127.0.0.1:${LOCAL_PORT}/v1/chat/completions" &
    done
    wait
    REQ=$(( REQ + CONCURRENCY ))
    echo "Sent ${REQ} total requests"
done
