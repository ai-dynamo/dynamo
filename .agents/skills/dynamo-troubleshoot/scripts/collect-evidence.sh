#!/usr/bin/env bash
# collect-evidence.sh — passive day-2 evidence collector for a Dynamo resource.
#
# Captures the canonical artifact set for a DGD, DGDR, DGDSA, or DCD into
# a single output directory. Does NOT mutate the cluster.
#
# Implements PASS/FAIL/WARN per SKILL_AUTHORING.md §8.3 (A9).
#
# Usage:
#   bash scripts/collect-evidence.sh -k <kind> -r <name> [-n <ns>] -o <out-dir>

set -uo pipefail

usage() {
    cat <<USAGE
Usage: $0 -k <kind> -r <name> [-n <ns>] -o <out-dir>
  -k  Resource kind: dgd | dgdr | dgdsa | dcd
  -r  Resource name. Required.
  -n  Namespace. Default: from \$KUBECTL_NAMESPACE or 'default'.
  -o  Output directory. Required.
  -h  Show this help.
USAGE
}

KIND=""
NAME=""
NAMESPACE="${KUBECTL_NAMESPACE:-default}"
OUT=""

while getopts ":k:r:n:o:h" opt; do
    case "$opt" in
        k) KIND="$OPTARG" ;;
        r) NAME="$OPTARG" ;;
        n) NAMESPACE="$OPTARG" ;;
        o) OUT="$OPTARG" ;;
        h) usage; exit 0 ;;
        \?) echo "Unknown flag -$OPTARG" >&2; usage >&2; exit 2 ;;
        :)  echo "-$OPTARG requires an argument" >&2; exit 2 ;;
    esac
done

if [ -z "$KIND" ] || [ -z "$NAME" ] || [ -z "$OUT" ]; then
    echo "Error: -k <kind>, -r <name>, and -o <out-dir> are required." >&2
    usage >&2
    exit 2
fi

case "$KIND" in
    dgd|dgdr|dgdsa|dcd) ;;
    *) echo "Error: -k must be one of dgd|dgdr|dgdsa|dcd" >&2; exit 2 ;;
esac

mkdir -p "$OUT"

PASS=0
FAIL=0
WARN=0
RESULTS=()

pass() { ((PASS++)); RESULTS+=("PASS|$1|$2"); }
fail() { ((FAIL++)); RESULTS+=("FAIL|$1|$2"); }
warn() { ((WARN++)); RESULTS+=("WARN|$1|$2"); }

# Resource exists
if kubectl get "$KIND" "$NAME" -n "$NAMESPACE" &>/dev/null; then
    pass "resource exists" "$KIND/$NAME in $NAMESPACE"
else
    fail "resource exists" "$KIND/$NAME in $NAMESPACE not found"
    echo "===== Collection Summary ====="
    for row in "${RESULTS[@]}"; do echo "$row"; done
    exit 1
fi

# 1. Resource YAML
if kubectl get "$KIND" "$NAME" -n "$NAMESPACE" -o yaml > "$OUT/$NAME.yaml" 2>/dev/null; then
    pass "resource yaml" "$OUT/$NAME.yaml"
else
    fail "resource yaml" "kubectl get -o yaml failed"
fi

# 2. Status conditions
if kubectl get "$KIND" "$NAME" -n "$NAMESPACE" -o jsonpath='{.status.conditions}' 2>/dev/null \
    | python3 -m json.tool > "$OUT/conditions.txt" 2>/dev/null; then
    pass "status conditions" "$OUT/conditions.txt"
else
    warn "status conditions" "no .status.conditions or parse failed"
fi

# 3. Pod listing — use the appropriate selector per kind.
case "$KIND" in
    dgd)  selector="nvidia.com/dgd-name=$NAME" ;;
    dgdr) selector="nvidia.com/dgdr-name=$NAME" ;;
    dgdsa) selector="nvidia.com/dgdsa-name=$NAME" ;;
    dcd)  selector="nvidia.com/dcd-name=$NAME" ;;
esac

if kubectl get pods -n "$NAMESPACE" -l "$selector" -o wide > "$OUT/pods.txt" 2>&1; then
    pass "pod listing" "$OUT/pods.txt"
else
    warn "pod listing" "no pods matched selector $selector"
fi

# 4. Per-pod describe + logs
pods=$(kubectl get pods -n "$NAMESPACE" -l "$selector" -o name 2>/dev/null | sed 's|pod/||')
if [ -n "$pods" ]; then
    while read -r pod; do
        [ -z "$pod" ] && continue
        if kubectl describe pod "$pod" -n "$NAMESPACE" > "$OUT/describe-$pod.txt" 2>&1; then
            pass "describe-$pod" "$OUT/describe-$pod.txt"
        else
            warn "describe-$pod" "describe failed"
        fi

        if kubectl logs "$pod" -n "$NAMESPACE" --all-containers --tail=2000 > "$OUT/logs-$pod.txt" 2>&1; then
            pass "logs-$pod" "$OUT/logs-$pod.txt"
        else
            warn "logs-$pod" "logs failed"
        fi

        if kubectl logs "$pod" -n "$NAMESPACE" --all-containers --previous --tail=2000 \
            > "$OUT/logs-$pod-previous.txt" 2>&1; then
            pass "logs-$pod-previous" "$OUT/logs-$pod-previous.txt"
        else
            warn "logs-$pod-previous" "no previous logs"
        fi
    done <<< "$pods"
fi

# 5. Recent events
if kubectl get events -n "$NAMESPACE" \
    --field-selector "involvedObject.name=$NAME" \
    --sort-by='.lastTimestamp' > "$OUT/events.txt" 2>&1; then
    pass "events" "$OUT/events.txt"
else
    warn "events" "no events for $NAME"
fi

# 6. Operator logs
operator_pod=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=dynamo-operator -o name 2>/dev/null | head -1 | sed 's|pod/||')
if [ -n "$operator_pod" ]; then
    if kubectl logs "$operator_pod" -n "$NAMESPACE" --tail=1000 > "$OUT/operator.log" 2>&1; then
        pass "operator log" "$OUT/operator.log"
    else
        warn "operator log" "could not fetch operator logs"
    fi
else
    warn "operator log" "no dynamo-operator pod in $NAMESPACE; try the platform install namespace"
fi

# 7. Helm status (if dynamo-platform is in this namespace)
if helm status dynamo-platform -n "$NAMESPACE" > "$OUT/helm-status.txt" 2>&1; then
    pass "helm status" "$OUT/helm-status.txt"
else
    warn "helm status" "dynamo-platform release not in $NAMESPACE (may be in a different ns)"
fi

# Summary
echo
echo "===== Collection Summary ====="
for row in "${RESULTS[@]}"; do echo "$row"; done
echo
echo "Output: $OUT"
echo "Passed: $PASS   Failed: $FAIL   Warned: $WARN"
echo
echo "Next step (Phase 3 Diagnose): walk references/symptom-signatures.md"
echo "against the artifacts in $OUT/."
[ "$FAIL" -gt 0 ] && exit 1
exit 0
