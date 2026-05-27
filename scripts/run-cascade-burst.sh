#!/usr/bin/env bash
# scripts/run-cascade-burst.sh — wrapper around dynamo-ft cascade_sanity_natural
# that satisfies the strict-data-capture rule from observe's 5hr-burst handoff.
#
# Per observe's `the 5hr-burst test handoff`, EVERY cascade run must produce:
#   - profile_export.jsonl / profile_export_aiperf.json / server_metrics_export.jsonl
#   - pod_memory_growth.tsv
#   - frontend/<pod>.log, vllmprefillworker/<pod>.log, vllmdecodeworker/<pod>.log
#   - dgd.yaml (the actual applied DGD)
#   - env-vars.txt (kubectl exec -- env on one FE pod + one worker pod)
#   - image-tags.txt (per-pod image label)
#   - aiperf-config.json (request timeout + concurrency + duration + dataset)
#
# Plus: per-image output dirs must be renamed immediately so the next run
# doesn't overwrite. Last cycle's bug was the baseline cliff data getting
# clobbered by the PR9858 run.
#
# Usage:
#   run-cascade-burst.sh <image-label> <image-tag> <namespace> [knob=val[;knob2=val2…]]
#
# Example:
#   run-cascade-burst.sh dis-2105 \
#       nvcr.io/nvidian/dynamo-dev/vllm-runtime:dis-2105-v1.1.1-53f1dd83a7-neelays \
#       neelays-test-b \
#       DYN_TCP_WORK_QUEUE_SIZE=256

set -uo pipefail

LABEL="${1:-}"
IMAGE="${2:-}"
NS="${3:-}"
KNOBS="${4:-}"

if [[ -z "$LABEL" || -z "$IMAGE" || -z "$NS" ]]; then
    echo "usage: $0 <image-label> <image-tag> <namespace> [knob=val[;…]]" >&2
    exit 2
fi

# Per-launch unique output dir. Without this, two parallel runs (different
# images, different namespaces) collide on the same local path because the
# test name is identical — and conftest.py's auto-clean-on-rerun then wipes
# whichever sibling started last. We learned this the hard way on 2026-05-22:
# 3 parallel cascade tests stomped on each other's snapshot dirs and rung
# data.
OUTBASE_PARENT="${DYN_TEST_OUTPUT_PATH_BASE:-$HOME/.dynamo-ft/test_outputs}"
OUTBASE="${OUTBASE_PARENT}_${LABEL}"
TESTNAME='test_overload_cascade_sanity_natural[2-64-160-8]'
RAW_DIR="$OUTBASE/$TESTNAME"
FINAL_DIR="$OUTBASE/${TESTNAME}.${LABEL}"
LOG="/tmp/cascade-burst-${LABEL}-$(date +%H%M%S).log"
mkdir -p "$OUTBASE"

echo "=== run-cascade-burst.sh ==="
echo "  label:     $LABEL"
echo "  image:     $IMAGE"
echo "  namespace: $NS"
echo "  knobs:     ${KNOBS:-<none>}"
echo "  log:       $LOG"
echo "  outbase:   $OUTBASE (per-launch unique)"
echo "  raw out:   $RAW_DIR"
echo "  final:     $FINAL_DIR"
echo ""

# Pre-flight: namespace must be empty (no leftover DGDs)
EXISTING=$(kubectl -n "$NS" get pods --no-headers 2>/dev/null | grep -E 'vllm-q3|audit-' | wc -l)
if (( EXISTING > 0 )); then
    echo "ERROR: $NS has $EXISTING leftover vllm/audit pods. Clean first." >&2
    kubectl -n "$NS" get pods --no-headers 2>&1 | head
    exit 3
fi

# Launch test in background, with knobs injected via DYN_TEST_WORKER_KNOBS
# and the unique per-launch output dir set explicitly. The dynamo-ft wrapper
# falls back to a default OUTPUT_PATH only if we don't pass one; we do.
cd "$(dirname "$0")/../tests/fault_tolerance/deploy"
# dynamo-ft prints "PID: <n>" to stdout in --bg mode. Capture that
# stdout to a dedicated file so we can grep it for the launch PID —
# previous version grepped $LOG (which is the pytest log file passed to
# --log) and found nothing, then pgrep'd a too-broad pattern that
# matched stale residual processes from killed prior runs.
LAUNCH_OUT="/tmp/cascade-burst-launch-${LABEL}-$(date +%H%M%S).out"
DYN_TEST_WORKER_KNOBS="$KNOBS" \
DYN_TEST_OUTPUT_PATH="$OUTBASE" \
./dynamo-ft cascade_sanity_natural \
    --image "$IMAGE" \
    --namespace "$NS" \
    --storage-class dgxc-enterprise-file \
    --bg \
    --log "$LOG" 2>&1 | tee "$LAUNCH_OUT" | tee /dev/stderr | head -10

# Grab the PID dynamo-ft printed
TEST_PID=$(grep -m1 '^PID:' "$LAUNCH_OUT" 2>/dev/null | awk '{print $2}')
# Fallback: filter pgrep by --newest so stale residuals don't win
for _ in 1 2 3 4 5; do
    if [[ -z "$TEST_PID" ]]; then
        sleep 2
        TEST_PID=$(pgrep -nf "pytest.*cascade.*$NS" 2>/dev/null | head -1)
    fi
done
if [[ -z "$TEST_PID" ]] || ! kill -0 "$TEST_PID" 2>/dev/null; then
    echo "ERROR: could not determine live test PID. LAUNCH_OUT=$LAUNCH_OUT  LOG=$LOG" >&2
    exit 4
fi
echo "  test PID:  $TEST_PID"

# Wait for pods to reach Ready (8 pods for units=2 cascade DGD).
# Capture env-vars.txt + image-tags.txt + dgd.yaml WHILE pods are alive.
# Timeout this at 12 min in case pods stick in PodInitializing.
echo "  waiting for pods to reach Ready..."
deadline=$(( $(date +%s) + 720 ))
ready=0
while (( $(date +%s) < deadline )); do
    READY=$(kubectl -n "$NS" get pods --no-headers 2>/dev/null \
        | awk '/vllm-q3/ {split($2,a,"/"); if(a[1]==a[2]) c++} END{print c+0}')
    if (( READY >= 8 )); then
        ready=1
        break
    fi
    sleep 15
done

if (( ready )); then
    echo "  all 8 pods Ready — capturing snapshot artifacts"
    SNAP="$RAW_DIR/.burst-snapshot-${LABEL}"
    mkdir -p "$SNAP"
    # image-tags.txt
    kubectl -n "$NS" get pods -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.containers[*].image}{"\n"}{end}' \
        > "$SNAP/image-tags.txt" 2>&1
    # env-vars.txt — exec into one FE + one decode + one prefill, dump env
    {
        for podtype in frontend vllmdecodeworker vllmprefillworker; do
            POD=$(kubectl -n "$NS" get pods --no-headers -l 'nvidia.com/dynamo-graph-deployment-name=vllm-q3-prod-units' 2>/dev/null \
                | grep "$podtype" | head -1 | awk '{print $1}')
            if [[ -n "$POD" ]]; then
                echo "## $podtype ($POD)"
                kubectl -n "$NS" exec "$POD" -c main -- env 2>/dev/null | sort
                echo ""
            fi
        done
    } > "$SNAP/env-vars.txt" 2>&1
    # dgd.yaml
    kubectl -n "$NS" get dynamographdeployment vllm-q3-prod-units -o yaml \
        > "$SNAP/dgd.yaml" 2>&1
    # aiperf-config sentinel — derive from the launch params
    cat > "$SNAP/aiperf-config.json" <<EOF
{
  "test": "$TESTNAME",
  "image_label": "$LABEL",
  "image_tag": "$IMAGE",
  "namespace": "$NS",
  "knobs": "${KNOBS:-}",
  "request_timeout_seconds": 40,
  "warmup_concurrency": 64,
  "cliff_concurrency": 160,
  "cliff_duration_min": 8,
  "captured_at": "$(date -u +%FT%TZ)"
}
EOF
    echo "  snapshot: $SNAP"
else
    echo "  WARN: pods did not all reach Ready in 12 min; capturing what's there"
    SNAP="$RAW_DIR/.burst-snapshot-${LABEL}"
    mkdir -p "$SNAP"
    kubectl -n "$NS" get pods -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.containers[*].image}{"\t"}{.status.phase}{"\n"}{end}' \
        > "$SNAP/image-tags.txt" 2>&1
    echo "{\"warning\":\"pods not all Ready at capture time\",\"image_label\":\"$LABEL\"}" \
        > "$SNAP/missing-data.txt"
fi

# Wait for test completion
echo "  waiting for test PID $TEST_PID to exit..."
while kill -0 "$TEST_PID" 2>/dev/null; do sleep 30; done
echo "  test exited at $(date -u +%FT%TZ)"

# Brief AIPerf result peek for the live log
tail -25 "$LOG" 2>/dev/null | grep -E 'Requests:|Errors:|Throughput:|PASSED|FAILED|short test' | head -15

# Rename output dir to the per-image label
if [[ -d "$RAW_DIR" ]]; then
    if [[ -d "$FINAL_DIR" ]]; then
        BACKUP="${FINAL_DIR}.prev-$(date +%s)"
        echo "  $FINAL_DIR exists; moving aside to $BACKUP"
        mv "$FINAL_DIR" "$BACKUP"
    fi
    mv "$RAW_DIR" "$FINAL_DIR"
    echo "  renamed: $FINAL_DIR"
else
    echo "  WARN: $RAW_DIR not found — nothing to rename"
fi

# Final integrity check against the strict-data-capture list
echo ""
echo "=== strict-data-capture integrity check (final: $FINAL_DIR) ==="
required=(
    "pod_memory_growth.tsv"
    "test.log.txt"
    "frontend"
    "vllmprefillworker"
    "vllmdecodeworker"
    ".burst-snapshot-${LABEL}/image-tags.txt"
    ".burst-snapshot-${LABEL}/env-vars.txt"
    ".burst-snapshot-${LABEL}/dgd.yaml"
    ".burst-snapshot-${LABEL}/aiperf-config.json"
)
missing=()
for f in "${required[@]}"; do
    if [[ -e "$FINAL_DIR/$f" ]]; then
        echo "  OK    $f"
    else
        echo "  MISS  $f"
        missing+=("$f")
    fi
done

# Per-rung data
for rung in warmup cliff; do
    dirs=$(find "$FINAL_DIR/load" -maxdepth 1 -type d -name "load-${rung}-*" 2>/dev/null | head -1)
    if [[ -n "$dirs" ]]; then
        for need in profile_export.jsonl profile_export_aiperf.json server_metrics_export.jsonl; do
            if [[ -e "$dirs/$need" ]]; then
                echo "  OK    $(basename "$dirs")/$need"
            else
                echo "  MISS  $(basename "$dirs")/$need"
                missing+=("$(basename "$dirs")/$need")
            fi
        done
    else
        echo "  MISS  load-${rung}-*/"
        missing+=("load-${rung}-*/")
    fi
done

if (( ${#missing[@]} > 0 )); then
    echo ""
    echo "  MISSING ITEMS noted to: $FINAL_DIR/missing-data.txt"
    printf '%s\n' "${missing[@]}" > "$FINAL_DIR/missing-data.txt"
fi

echo ""
echo "=== DONE: $LABEL → $FINAL_DIR ==="
