#!/usr/bin/env bash
# precheck-target.sh — pre-benchmark target readiness check.
#
# Confirms a Dynamo deployment is ready for benchmarking:
#   1. DGD exists and reports Ready
#   2. Frontend service is reachable
#   3. /v1/models returns at least one model
#   4. Pods are Running (no CrashLoopBackOff)
#   5. AIPerf is installed
#
# Implements PASS/FAIL/WARN per SKILL_AUTHORING.md §8.3 (A9).
#
# Usage:
#   bash scripts/precheck-target.sh -d <dgd-name> [-n <ns>] [--port-forward]

set -uo pipefail

usage() {
    cat <<USAGE
Usage: $0 -d <dgd-name> [-n <ns>] [--port-forward]
  -d  DGD name. Required.
  -n  Namespace. Default: from \$KUBECTL_NAMESPACE or 'default'.
  --port-forward  Open a kubectl port-forward and probe /v1/models (default: skip)
  -h  Show this help.
USAGE
}

DGD_NAME=""
NAMESPACE="${KUBECTL_NAMESPACE:-default}"
PORT_FORWARD=0

while [ $# -gt 0 ]; do
    case "$1" in
        -d) DGD_NAME="$2"; shift 2 ;;
        -n) NAMESPACE="$2"; shift 2 ;;
        --port-forward) PORT_FORWARD=1; shift ;;
        -h) usage; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [ -z "$DGD_NAME" ]; then
    echo "Error: -d <dgd-name> is required." >&2
    usage >&2
    exit 2
fi

PASS=0
FAIL=0
WARN=0
RESULTS=()

pass() { ((PASS++)); RESULTS+=("PASS|$1|$2"); }
fail() { ((FAIL++)); RESULTS+=("FAIL|$1|$2"); }
warn() { ((WARN++)); RESULTS+=("WARN|$1|$2"); }

# 1. DGD exists and is Ready.
if kubectl get dgd "$DGD_NAME" -n "$NAMESPACE" &>/dev/null; then
    state=$(kubectl get dgd "$DGD_NAME" -n "$NAMESPACE" -o jsonpath='{.status.state}')
    if [ "$state" = "successful" ] || [ "$state" = "Ready" ] || [ "$state" = "Deployed" ]; then
        pass "DGD ready" "state=$state"
    else
        fail "DGD ready" "state=$state (expected successful/Ready/Deployed)"
    fi
else
    fail "DGD exists" "$DGD_NAME not found in $NAMESPACE"
    echo "===== Pre-Check Summary ====="
    for row in "${RESULTS[@]}"; do echo "$row"; done
    exit 1
fi

# 2. Frontend service exists.
frontend_svc=$(kubectl get svc -n "$NAMESPACE" -l "nvidia.com/dgd-name=$DGD_NAME,app.kubernetes.io/component=frontend" -o name 2>/dev/null | head -1 | sed 's|service/||')
if [ -n "$frontend_svc" ]; then
    pass "Frontend service" "$frontend_svc"
else
    # Fall back to any frontend service in the namespace.
    frontend_svc=$(kubectl get svc -n "$NAMESPACE" -l app.kubernetes.io/component=frontend -o name 2>/dev/null | head -1 | sed 's|service/||')
    if [ -n "$frontend_svc" ]; then
        warn "Frontend service" "$frontend_svc (no DGD label; could be a sibling deployment)"
    else
        fail "Frontend service" "no Frontend service in $NAMESPACE"
    fi
fi

# 3. Pods Running (no CrashLoopBackOff).
running=$(kubectl get pods -n "$NAMESPACE" -l "nvidia.com/dgd-name=$DGD_NAME" --no-headers 2>/dev/null | awk '$3=="Running"' | wc -l | tr -d ' ')
total=$(kubectl get pods -n "$NAMESPACE" -l "nvidia.com/dgd-name=$DGD_NAME" --no-headers 2>/dev/null | wc -l | tr -d ' ')
crashloop=$(kubectl get pods -n "$NAMESPACE" -l "nvidia.com/dgd-name=$DGD_NAME" --no-headers 2>/dev/null | awk '$3=="CrashLoopBackOff"' | wc -l | tr -d ' ')

if [ "$crashloop" -gt 0 ]; then
    fail "no crashloops" "$crashloop pod(s) in CrashLoopBackOff"
elif [ "$running" -eq "$total" ] && [ "$total" -gt 0 ]; then
    pass "pods running" "$running/$total"
else
    warn "pods running" "$running/$total Running (some pods may still be starting)"
fi

# 4. /v1/models populated (optional port-forward).
if [ "$PORT_FORWARD" = "1" ] && [ -n "$frontend_svc" ]; then
    kubectl port-forward -n "$NAMESPACE" "svc/$frontend_svc" 8000:8000 &
    pf_pid=$!
    # shellcheck disable=SC2329  # invoked via trap below
    cleanup_pf() { kill "${pf_pid:-}" 2>/dev/null || true; }
    trap cleanup_pf EXIT
    sleep 3

    if models=$(curl -s --max-time 5 http://localhost:8000/v1/models 2>/dev/null); then
        if echo "$models" | python3 -c 'import json,sys; sys.exit(0 if json.load(sys.stdin).get("data") else 1)' 2>/dev/null; then
            pass "/v1/models populated" "$(echo "$models" | python3 -c 'import json,sys; print(len(json.load(sys.stdin).get("data",[])))') model(s)"
        else
            fail "/v1/models populated" "empty data array (worker registration window — see dynamo-deploy/references/known-issues.md D3)"
        fi
    else
        warn "/v1/models populated" "curl failed; check port-forward"
    fi

    kill $pf_pid 2>/dev/null || true
elif [ "$PORT_FORWARD" = "1" ]; then
    warn "/v1/models populated" "no Frontend service to port-forward against"
else
    warn "/v1/models populated" "skipped (run with --port-forward to probe)"
fi

# 5. AIPerf installed.
if command -v aiperf >/dev/null 2>&1; then
    ver=$(aiperf --version 2>/dev/null | head -1 || echo "unknown")
    pass "aiperf installed" "$ver"
else
    fail "aiperf installed" "aiperf not on PATH; install: pip install ai-perf"
fi

# Summary
echo
echo "===== Pre-Check Summary ====="
for row in "${RESULTS[@]}"; do echo "$row"; done
echo
echo "Passed: $PASS   Failed: $FAIL   Warned: $WARN"
[ "$FAIL" -gt 0 ] && exit 1
exit 0
