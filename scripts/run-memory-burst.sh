#!/usr/bin/env bash
# scripts/run-memory-burst.sh — wrapper around dynamo-ft kv_router_memory_stability_a4
# that satisfies observe's strict-data-capture rule with per-launch FE-knob injection.
#
# Usage:
#   run-memory-burst.sh <label> <image-tag> <namespace> [fe_knob=val[;fe_knob2=val2…]]

set -uo pipefail

LABEL="${1:-}"
IMAGE="${2:-}"
NS="${3:-}"
FE_KNOBS="${4:-}"

if [[ -z "$LABEL" || -z "$IMAGE" || -z "$NS" ]]; then
    echo "usage: $0 <label> <image-tag> <namespace> [fe_knob=val[;…]]" >&2
    exit 2
fi

OUTBASE_PARENT="${DYN_TEST_OUTPUT_PATH_BASE:-$HOME/.dynamo-ft/test_outputs}"
OUTBASE="${OUTBASE_PARENT}_${LABEL}"
TESTNAME='test_kv_router_memory_stability[a4-recommended-2-2.0-3.0-18.0-3.0-2.0]'
RAW_DIR="$OUTBASE/$TESTNAME"
FINAL_DIR="$OUTBASE/${TESTNAME}.${LABEL}"
LOG="/tmp/memory-burst-${LABEL}-$(date +%H%M%S).log"
mkdir -p "$OUTBASE"

echo "=== run-memory-burst.sh ==="
echo "  label:     $LABEL"
echo "  image:     $IMAGE"
echo "  namespace: $NS"
echo "  fe knobs:  ${FE_KNOBS:-<none>}"
echo "  log:       $LOG"
echo "  outbase:   $OUTBASE"
echo "  final:     $FINAL_DIR"

EXISTING=$(kubectl -n "$NS" get pods --no-headers 2>/dev/null | grep -E 'vllm-q3|audit-' | wc -l)
if (( EXISTING > 0 )); then
    echo "ERROR: $NS has $EXISTING leftover vllm/audit pods. Clean first." >&2
    exit 3
fi

cd "$(dirname "$0")/../tests/fault_tolerance/deploy"
LAUNCH_OUT="/tmp/memory-burst-launch-${LABEL}-$(date +%H%M%S).out"
DYN_TEST_FE_KNOBS="$FE_KNOBS" \
DYN_TEST_OUTPUT_PATH="$OUTBASE" \
./dynamo-ft kv_router_memory_stability_a4 \
    --image "$IMAGE" \
    --namespace "$NS" \
    --storage-class dgxc-enterprise-file \
    --bg \
    --log "$LOG" 2>&1 | tee "$LAUNCH_OUT" | tee /dev/stderr | head -10

# Same launch-PID-detection fix as run-cascade-burst.sh: grep
# dynamo-ft's stdout, not the pytest log file. pgrep with -n (newest)
# avoids matching stale residuals from prior killed runs.
TEST_PID=$(grep -m1 '^PID:' "$LAUNCH_OUT" 2>/dev/null | awk '{print $2}')
for _ in 1 2 3 4 5; do
    [[ -n "$TEST_PID" ]] || { sleep 2; TEST_PID=$(pgrep -nf "pytest.*memory_stability.*$NS" 2>/dev/null | head -1); }
done
if [[ -z "$TEST_PID" ]] || ! kill -0 "$TEST_PID" 2>/dev/null; then
    echo "ERROR: could not determine live test PID. LAUNCH_OUT=$LAUNCH_OUT" >&2
    exit 4
fi
echo "  test PID:  $TEST_PID"

# Wait for pods Ready, snapshot
deadline=$(( $(date +%s) + 720 ))
ready=0
while (( $(date +%s) < deadline )); do
    READY=$(kubectl -n "$NS" get pods --no-headers 2>/dev/null \
        | awk '/vllm-q3/ {split($2,a,"/"); if(a[1]==a[2]) c++} END{print c+0}')
    if (( READY >= 8 )); then ready=1; break; fi
    sleep 15
done

if (( ready )); then
    echo "  pods Ready — snapshot capture"
    SNAP="$RAW_DIR/.burst-snapshot-${LABEL}"
    mkdir -p "$SNAP"
    kubectl -n "$NS" get pods -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.containers[*].image}{"\n"}{end}' > "$SNAP/image-tags.txt" 2>&1
    {
        for pt in frontend vllmdecodeworker vllmprefillworker; do
            POD=$(kubectl -n "$NS" get pods --no-headers 2>/dev/null | grep "$pt" | head -1 | awk '{print $1}')
            if [[ -n "$POD" ]]; then
                echo "## $pt ($POD)"
                kubectl -n "$NS" exec "$POD" -c main -- env 2>/dev/null | sort
                echo ""
            fi
        done
    } > "$SNAP/env-vars.txt" 2>&1
    kubectl -n "$NS" get dynamographdeployment vllm-q3-prod-units -o yaml > "$SNAP/dgd.yaml" 2>&1
    cat > "$SNAP/aiperf-config.json" <<EOF2
{
  "test": "$TESTNAME",
  "image_label": "$LABEL",
  "image_tag": "$IMAGE",
  "namespace": "$NS",
  "fe_knobs": "${FE_KNOBS:-}",
  "captured_at": "$(date -u +%FT%TZ)"
}
EOF2
fi

# Wait for test exit (memory tests take ~45 min)
while kill -0 "$TEST_PID" 2>/dev/null; do sleep 60; done
echo "  test exited at $(date -u +%FT%TZ)"
tail -25 "$LOG" 2>/dev/null | grep -E 'PASSED|FAILED|in [0-9]+\.' | head -3

[[ -d "$RAW_DIR" ]] && {
    [[ -d "$FINAL_DIR" ]] && mv "$FINAL_DIR" "${FINAL_DIR}.prev-$(date +%s)"
    mv "$RAW_DIR" "$FINAL_DIR"
    echo "  renamed: $FINAL_DIR"
}
echo "=== DONE: $LABEL → $FINAL_DIR ==="
