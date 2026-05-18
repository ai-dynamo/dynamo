#!/usr/bin/env bash
# Live-watch a running cascade-console driver: pulls the load PVC's current
# server_metrics_export.jsonl, runs cascade_timeline, repeats every 30s.
#
# Requires:
#   - The driver pytest is running with `--log-pvc qwen3-30b-logs`
#   - host has kubectl + cluster context selected
#   - a peek pod gets created in the test namespace and reused
#
# Usage:
#   scripts/watch_live.sh                                        # default ns + pvc
#   scripts/watch_live.sh -n <ns> -p <pvc> --metrics queued,...   # full custom

set -euo pipefail

CTX_DEFAULT=nv-prd-dgxc.teleport.sh-dynamo-aws-dev-02
NS_DEFAULT=neelays-test
PVC_DEFAULT=qwen3-30b-logs

CTX="$CTX_DEFAULT"
NS="$NS_DEFAULT"
PVC="$PVC_DEFAULT"
METRICS="queued,inflight,ttft_p99,running,waiting,kv_pct,nixl_p99"
INTERVAL=30

while [[ $# -gt 0 ]]; do
  case "$1" in
    --context|-c) CTX="$2"; shift 2 ;;
    --namespace|-n) NS="$2"; shift 2 ;;
    --pvc|-p) PVC="$2"; shift 2 ;;
    --metrics|-m) METRICS="$2"; shift 2 ;;
    --interval|-i) INTERVAL="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

OUTDIR=$(mktemp -d)
trap 'rm -rf "$OUTDIR"; kubectl --context "$CTX" -n "$NS" delete pod pvc-peek-watch --wait=false 2>/dev/null || true' EXIT

# Spin up peek pod (idempotent)
cat <<EOF | kubectl --context "$CTX" -n "$NS" apply -f - >/dev/null
apiVersion: v1
kind: Pod
metadata:
  name: pvc-peek-watch
spec:
  restartPolicy: Never
  containers:
  - name: peek
    image: alpine:3.20
    command: ["sleep", "86400"]
    volumeMounts:
    - {name: logs, mountPath: /pvc}
  volumes:
  - name: logs
    persistentVolumeClaim:
      claimName: $PVC
EOF

# Wait for peek pod
echo "waiting for peek pod ..." >&2
until kubectl --context "$CTX" -n "$NS" get pod pvc-peek-watch \
        -o jsonpath='{.status.phase}' 2>/dev/null | grep -q Running; do
  sleep 1
done
echo "peek pod ready" >&2

mkdir -p "$OUTDIR/load"

while true; do
  # Pull the current snapshot of the live JSONL
  if kubectl --context "$CTX" -n "$NS" cp \
        "pvc-peek-watch:/pvc/aiperf/server_metrics_export.jsonl" \
        "$OUTDIR/load/server_metrics_export.jsonl" 2>/dev/null; then
    sz=$(wc -l < "$OUTDIR/load/server_metrics_export.jsonl" 2>/dev/null || echo 0)
    if [[ "$sz" -gt 0 ]]; then
      clear
      echo "=== live cascade view — $(date) — $sz JSONL records ==="
      echo
      python3 "$(dirname "$0")/cascade_timeline.py" "$OUTDIR" \
        --metrics "$METRICS" --width 80 2>&1 | tail -200
    else
      echo "[$(date +%H:%M:%S)] no records yet (load Job may be initializing)"
    fi
  else
    echo "[$(date +%H:%M:%S)] couldn't pull jsonl (PVC busy or path missing); retrying"
  fi
  sleep "$INTERVAL"
done
