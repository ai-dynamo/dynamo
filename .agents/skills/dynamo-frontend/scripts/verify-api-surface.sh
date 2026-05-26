#!/usr/bin/env bash
# verify-api-surface.sh — post-apply OpenAI API surface probe.
#
# Confirms /v1/models is populated and at least one sample inference
# request returns a non-empty response. Uses the check() pattern from
# SKILL_AUTHORING.md §8.4 (A10).
#
# Usage:
#   bash scripts/verify-api-surface.sh -d <dgd-name> [-n <ns>] [-p <port>]

set -uo pipefail

usage() {
    cat <<USAGE
Usage: $0 -d <dgd-name> [-n <ns>] [-p <local-port>]
  -d  DGD name. Required.
  -n  Namespace. Default: from \$KUBECTL_NAMESPACE or 'default'.
  -p  Local port for port-forward. Default: 8000.
  -h  Show this help.
USAGE
}

DGD_NAME=""
NAMESPACE="${KUBECTL_NAMESPACE:-default}"
PORT=8000

while getopts ":d:n:p:h" opt; do
    case "$opt" in
        d) DGD_NAME="$OPTARG" ;;
        n) NAMESPACE="$OPTARG" ;;
        p) PORT="$OPTARG" ;;
        h) usage; exit 0 ;;
        \?) echo "Unknown flag -$OPTARG" >&2; usage >&2; exit 2 ;;
        :)  echo "-$OPTARG requires an argument" >&2; exit 2 ;;
    esac
done

if [ -z "$DGD_NAME" ]; then
    echo "Error: -d <dgd-name> is required." >&2
    usage >&2
    exit 2
fi

PASS=0
FAIL=0

check() {
    local desc="$1" cmd="$2" pattern="$3"
    local out
    if out=$(bash -c "$cmd" 2>&1) && echo "$out" | grep -q "$pattern"; then
        ((PASS++))
        echo "PASS: $desc"
    else
        ((FAIL++))
        echo "FAIL: $desc"
    fi
}

# Locate the Frontend service.
frontend_svc=$(kubectl get svc -n "$NAMESPACE" -l "nvidia.com/dgd-name=$DGD_NAME,app.kubernetes.io/component=frontend" -o name 2>/dev/null | head -1 | sed 's|service/||')
if [ -z "$frontend_svc" ]; then
    echo "FAIL: no Frontend service for DGD $DGD_NAME in $NAMESPACE"
    exit 1
fi

# Port-forward.
kubectl port-forward -n "$NAMESPACE" "svc/$frontend_svc" "$PORT:8000" &
pf_pid=$!
# shellcheck disable=SC2329  # invoked via trap below
cleanup_pf() { kill "${pf_pid:-}" 2>/dev/null || true; }
trap cleanup_pf EXIT
sleep 3

# 1. Frontend reachable.
check "Frontend reachable" \
      "curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://localhost:$PORT/v1/models" \
      "200"

# 2. /v1/models populated (allow 60s).
deadline=$(( $(date +%s) + 60 ))
populated=0
while [ "$(date +%s)" -lt "$deadline" ]; do
    if curl -s --max-time 5 "http://localhost:$PORT/v1/models" | python3 -c 'import json,sys; sys.exit(0 if json.load(sys.stdin).get("data") else 1)' 2>/dev/null; then
        populated=1
        break
    fi
    sleep 5
done
if [ "$populated" = "1" ]; then
    model_count=$(curl -s "http://localhost:$PORT/v1/models" | python3 -c 'import json,sys; print(len(json.load(sys.stdin).get("data",[])))')
    ((PASS++))
    echo "PASS: /v1/models populated (${model_count} model(s))"
else
    ((FAIL++))
    echo "FAIL: /v1/models still empty after 60s (see D2 / D3 in known-issues.md)"
fi

# 3. Sample inference (only if we have at least one model).
if [ "$populated" = "1" ]; then
    first_model=$(curl -s "http://localhost:$PORT/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
    check "sample chat completion" \
          "curl -s -m 30 -X POST http://localhost:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"$first_model\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello.\"}],\"max_tokens\":16}'" \
          'choices'
fi

# 4. Metrics endpoint.
check "Frontend /metrics reachable" \
      "curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://localhost:$((PORT + 1))/metrics" \
      "200"

# 5. DynamoModel CRs in this namespace (informational).
dm_count=$(kubectl get dynamomodel -n "$NAMESPACE" --no-headers 2>/dev/null | wc -l | tr -d ' ')
if [ "$dm_count" -gt 0 ]; then
    ((PASS++))
    echo "PASS: DynamoModel CRs ($dm_count in $NAMESPACE)"
else
    echo "INFO: no DynamoModel CRs in $NAMESPACE (Frontend can still serve via DGD's worker --model flags)"
fi

echo
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -gt 0 ] && exit 1
exit 0
