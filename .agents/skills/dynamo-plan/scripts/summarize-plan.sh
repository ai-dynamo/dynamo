#!/usr/bin/env bash
# summarize-plan.sh — produce a one-screen summary of a planning result.
#
# Reads either an AIConfigurator planning.json or a DGDR status, and
# prints a uniform summary: parallelism, replicas, KV budget, expected
# TTFT/ITL, and the hand-off command for dynamo-deploy.
#
# Implements PASS/FAIL/WARN per SKILL_AUTHORING.md §8.3 (A9).
#
# Usage:
#   bash scripts/summarize-plan.sh -i <planning.json>
#   bash scripts/summarize-plan.sh -d <dgdr-name> [-n <ns>]

set -uo pipefail

usage() {
    cat <<USAGE
Usage:
  $0 -i <planning.json>
  $0 -d <dgdr-name> [-n <namespace>]
  $0 -h
USAGE
}

PLANNING_JSON=""
DGDR_NAME=""
NAMESPACE="${KUBECTL_NAMESPACE:-default}"

while getopts ":i:d:n:h" opt; do
    case "$opt" in
        i) PLANNING_JSON="$OPTARG" ;;
        d) DGDR_NAME="$OPTARG" ;;
        n) NAMESPACE="$OPTARG" ;;
        h) usage; exit 0 ;;
        \?) echo "Unknown flag -$OPTARG" >&2; usage >&2; exit 2 ;;
        :)  echo "-$OPTARG requires an argument" >&2; exit 2 ;;
    esac
done

if [ -z "$PLANNING_JSON" ] && [ -z "$DGDR_NAME" ]; then
    echo "Error: provide -i <planning.json> or -d <dgdr-name>" >&2
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

if [ -n "$PLANNING_JSON" ]; then
    if [ ! -f "$PLANNING_JSON" ]; then
        fail "planning.json present" "$PLANNING_JSON not found"
    else
        pass "planning.json present" "$PLANNING_JSON"
        python3 - "$PLANNING_JSON" <<'PYBLOCK'
import json, sys
data = json.load(open(sys.argv[1]))
print()
print("=== AIConfigurator Planning Result ===")
print(f"model:         {data.get('model','<unknown>')}")
print(f"backend:       {data.get('backend','<unknown>')}")
print(f"parallelism:   TP={data.get('tp','?')} PP={data.get('pp','?')} EP={data.get('ep','?')}")
print(f"replicas:      {data.get('replicas','?')}")
print(f"kv_cache_mb:   {data.get('kv_cache_mb','?')}")
print(f"expected_ttft: {data.get('expected_ttft_ms','?')} ms")
print(f"expected_itl:  {data.get('expected_itl_ms','?')} ms")
print()
print("Hand-off to dynamo-deploy:")
print(f"  Convert this planning.json to a DGDR with autoApply: true,")
print(f"  or to a DGD using `dynamo-deploy` Phase 2 patterns.")
PYBLOCK
    fi
elif [ -n "$DGDR_NAME" ]; then
    if ! kubectl get dgdr "$DGDR_NAME" -n "$NAMESPACE" &>/dev/null; then
        fail "DGDR exists" "$DGDR_NAME in $NAMESPACE not found"
    else
        pass "DGDR exists" "$DGDR_NAME in $NAMESPACE"
        phase=$(kubectl get dgdr "$DGDR_NAME" -n "$NAMESPACE" -o jsonpath='{.status.phase}')
        echo
        echo "=== DGDR Planning Result ==="
        echo "name:    $DGDR_NAME"
        echo "phase:   $phase"
        if [ "$phase" = "Ready" ] || [ "$phase" = "Deploying" ] || [ "$phase" = "Deployed" ]; then
            pass "DGDR phase" "$phase"
            kubectl get dgdr "$DGDR_NAME" -n "$NAMESPACE" \
                -o jsonpath='{.status.profilingResults.selectedConfig}' \
                | python3 -m json.tool 2>/dev/null || warn "selectedConfig parse" "not valid JSON"
            echo
            echo "Hand-off to dynamo-deploy:"
            echo "  kubectl patch dgdr $DGDR_NAME -n $NAMESPACE \\"
            echo "    --type=merge -p '{\"spec\":{\"autoApply\":true}}'"
        else
            warn "DGDR phase" "still $phase — planning not complete"
        fi
    fi
fi

echo
echo "===== Summary ====="
for row in "${RESULTS[@]}"; do echo "$row"; done
echo "Passed: $PASS   Failed: $FAIL   Warned: $WARN"
[ "$FAIL" -gt 0 ] && exit 1
exit 0
