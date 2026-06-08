#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Deploy one Qwen3.5-27B config (agg or disagg) + launch its GSM8K accuracy job.
# Usage: ./run-config.sh <agg|disagg> <DGD_NAME> <ASYNC_FLAG> <RUN_LABEL> [TP]
#   ./run-config.sh agg    q27-agg-async     --async-scheduling    agg-async
#   ./run-config.sh disagg q27-disagg-async  --async-scheduling    disagg-async
#   ./run-config.sh disagg q27-disagg-noasync --no-async-scheduling disagg-noasync
#   ./run-config.sh disagg q27-disagg-async-tp2 --async-scheduling  disagg-async-tp2 2
# TP defaults to 1; GPUs/worker = TP (disagg = 2 workers => 2*TP GPUs total).
set -euo pipefail

MODE="${1:?mode: agg|disagg}"; DGD_NAME="${2:?DGD_NAME}"
ASYNC_FLAG="${3:?ASYNC_FLAG}"; RUN_LABEL="${4:?RUN_LABEL}"; TP="${5:-1}"; GPU_COUNT="$TP"
MODEL="${MODEL:-Qwen/Qwen3.5-27B}"   # override via env, e.g. MODEL=Qwen/Qwen3.5-35B-A3B-FP8

NS=qiwa
CTX=nv-prd-dgxc.teleport.sh-dynamo-aws-dev-02
KC="kubectl --context=$CTX -n $NS"
HERE="$(cd "$(dirname "$0")" && pwd)"

case "$MODE" in
  agg)    YAML="$HERE/deploy/agg.yaml";    NEED="frontend vllmworker" ;;
  disagg) YAML="$HERE/deploy/disagg.yaml"; NEED="frontend vllmprefillworker vllmdecodeworker" ;;
  *) echo "mode must be agg|disagg"; exit 2 ;;
esac

set -a; . "$HERE/hw/h100.env"; set +a   # VLLM_IMAGE, selectors, IMAGE_PULL_SECRET
export DGD_NAME ASYNC_FLAG TP GPU_COUNT MODEL

echo "[deploy] $MODE  $DGD_NAME  MODEL=$MODEL  TP=$TP  async='$ASYNC_FLAG'  image=$VLLM_IMAGE"
envsubst '$VLLM_IMAGE $HW_NODE_SELECTOR $HW_TOLERATIONS $DGD_NAME $ASYNC_FLAG $TP $GPU_COUNT $MODEL' \
  < "$YAML" | $KC apply -f -

# --- patch the operator-stamped SA with the private-registry pull secret ---
SA="${DGD_NAME}-k8s-service-discovery"
echo "[sa] waiting for $SA ..."
for _ in $(seq 1 60); do
  if $KC get sa "$SA" >/dev/null 2>&1; then
    $KC patch sa "$SA" -p "{\"imagePullSecrets\":[{\"name\":\"$IMAGE_PULL_SECRET\"}]}" >/dev/null
    echo "[sa] patched $SA -> $IMAGE_PULL_SECRET"; break
  fi
  sleep 2
done

# --- wait for every required component Ready (model load); fix ImagePull once ---
SEL='nvidia.com/dynamo-graph-deployment-name='"$DGD_NAME"
echo "[wait] components Ready: $NEED (model load ~3-6 min)..."
DEADLINE=$(( $(date +%s) + 1500 ))
while true; do
  line=$($KC get pods -l "$SEL" --no-headers 2>/dev/null || true)
  printf '%s\n' "$line"
  printf '%s\n' "$line" | awk '$3 ~ /ImagePull|ErrImage/ {print $1}' | while read -r p; do
    [ -n "$p" ] && $KC delete pod "$p" --wait=false >/dev/null 2>&1 || true
  done
  all_ok=1
  for comp in $NEED; do
    ok=$(printf '%s\n' "$line" | grep -i "$comp" | awk '$3=="Running"{split($2,a,"/"); if(a[1]==a[2]&&a[2]>0) print 1}' | head -1)
    [ "${ok:-0}" = "1" ] || all_ok=0
  done
  if [ "$all_ok" = "1" ]; then echo "[wait] all Ready"; break; fi
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then echo "[wait] TIMEOUT"; $KC get pods -l "$SEL"; exit 1; fi
  sleep 15
done

# --- frontend service: operator names it <dgd>-frontend (no DGD-name label) ---
FE="${DGD_NAME}-frontend"
$KC get svc "$FE" >/dev/null 2>&1 || { echo "[frontend] svc $FE not found"; $KC get svc | grep "$DGD_NAME" || true; exit 1; }
echo "[frontend] service = $FE"

# --- launch the GSM8K accuracy job ---
export BENCH_ACC_POD="${RUN_LABEL}-acc" BENCH_FRONTEND="$FE" BENCH_RUN_LABEL="$RUN_LABEL"
envsubst '$HW_NODE_SELECTOR $HW_TOLERATIONS $BENCH_ACC_POD $BENCH_FRONTEND $BENCH_RUN_LABEL $MODEL' \
  < "$HERE/accuracy-job.yaml" | $KC apply -f -
echo "[acc] launched ${RUN_LABEL}-acc -> $FE   (poll: $KC logs ${RUN_LABEL}-acc -f)"
echo "DONE $MODE / $DGD_NAME / $RUN_LABEL / frontend=$FE"
