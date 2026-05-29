#!/bin/bash
# Tests the connector through Dynamo's own runtime (python -m dynamo.vllm),
# which is what the customer's DGD invokes. Single-GPU layout: one frontend,
# one prefill worker, one decode worker — all sharing GPU 0.
#
# Confirms two things:
#   1. The customer's exact launch path (python -m dynamo.vllm with our
#      --kv-transfer-config) successfully starts the worker with our custom
#      connector loaded.
#   2. After decode is killed mid-transfer, the prefill worker's /metrics
#      shows vllm:nixl_num_pending_sends > 0 while all standard scheduler-
#      state metrics are at zero.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPONENTS_SRC="$(cd "$SCRIPT_DIR/../../../../" && pwd)"
GPU_ID=0
FRONTEND_PORT=8000
PREFILL_SYSTEM_PORT=8082
DECODE_SYSTEM_PORT=8081
PREFILL_SIDE_CHANNEL=20097
MODEL="Qwen/Qwen2.5-0.5B-Instruct"

KV_CONFIG='{
  "kv_connector":"NixlConnectorWithPendingMetrics",
  "kv_role":"kv_both",
  "kv_connector_module_path":"dynamo.vllm.custom_connectors.nixl_with_pending_metrics"
}'

# pkill -f matches the full command line; use python invocation pattern to
# avoid matching this script's own path (which contains "dynamo/vllm").
pkill -f "python.* -m dynamo\.vllm" 2>/dev/null || true
pkill -f "python.* -m dynamo\.frontend" 2>/dev/null || true
sleep 3

mkdir -p /tmp/dynamo_runtime_test

# Self-contained single-host setup — no NATS/etcd dependency.
export DYN_REQUEST_PLANE=tcp
export DYN_DISCOVERY_BACKEND=file

echo "[launch] dynamo.frontend on port $FRONTEND_PORT"
PYTHONPATH="$COMPONENTS_SRC" \
  DYN_HTTP_PORT=$FRONTEND_PORT \
  DYN_REQUEST_PLANE=tcp \
  DYN_DISCOVERY_BACKEND=file \
  DYN_EVENT_PLANE=zmq \
  ${PYTHON:-python} -m dynamo.frontend > /tmp/dynamo_runtime_test/frontend.log 2>&1 &
FRONTEND_PID=$!

echo "[launch] decode worker (python -m dynamo.vllm --disaggregation-mode decode) — system_port $DECODE_SYSTEM_PORT"
CUDA_VISIBLE_DEVICES=$GPU_ID \
  PYTHONPATH="$COMPONENTS_SRC" \
  DYN_SYSTEM_PORT=$DECODE_SYSTEM_PORT \
  VLLM_NIXL_ABORT_REQUEST_TIMEOUT=600 \
  DYN_REQUEST_PLANE=tcp \
  DYN_DISCOVERY_BACKEND=file \
  DYN_EVENT_PLANE=zmq \
  ${PYTHON:-python} -m dynamo.vllm \
    --model "$MODEL" \
    --enforce-eager \
    --max-model-len 512 --max-num-seqs 4 \
    --gpu-memory-utilization 0.30 \
    --disaggregation-mode decode \
    --kv-transfer-config "$KV_CONFIG" \
    > /tmp/dynamo_runtime_test/decode.log 2>&1 &
DECODE_PID=$!

echo "[launch] prefill worker (python -m dynamo.vllm --disaggregation-mode prefill) — system_port $PREFILL_SYSTEM_PORT"
CUDA_VISIBLE_DEVICES=$GPU_ID \
  PYTHONPATH="$COMPONENTS_SRC" \
  DYN_SYSTEM_PORT=$PREFILL_SYSTEM_PORT \
  VLLM_NIXL_SIDE_CHANNEL_PORT=$PREFILL_SIDE_CHANNEL \
  VLLM_NIXL_ABORT_REQUEST_TIMEOUT=600 \
  DYN_REQUEST_PLANE=tcp \
  DYN_DISCOVERY_BACKEND=file \
  DYN_EVENT_PLANE=zmq \
  ${PYTHON:-python} -m dynamo.vllm \
    --model "$MODEL" \
    --enforce-eager \
    --max-model-len 512 --max-num-seqs 4 \
    --gpu-memory-utilization 0.30 \
    --disaggregation-mode prefill \
    --kv-transfer-config "$KV_CONFIG" \
    > /tmp/dynamo_runtime_test/prefill.log 2>&1 &
PREFILL_PID=$!

# Wait for frontend to be responsive
echo "[wait] frontend"
for i in $(seq 1 60); do
  if curl -fs localhost:$FRONTEND_PORT/v1/models > /dev/null 2>&1; then
    echo "[wait] frontend OK"
    break
  fi
  sleep 1
done

# Wait for worker system ports (where /metrics lives)
for PORT in $DECODE_SYSTEM_PORT $PREFILL_SYSTEM_PORT; do
  echo "[wait] worker system port $PORT"
  for i in $(seq 1 480); do
    if curl -fs localhost:$PORT/metrics > /dev/null 2>&1; then
      echo "[wait] $PORT OK"
      break
    fi
    sleep 1
  done
  if ! curl -fs localhost:$PORT/metrics > /dev/null 2>&1; then
    echo "[FAIL] system port $PORT never came up"
    echo "--- decode.log last 20 ---"; tail -20 /tmp/dynamo_runtime_test/decode.log
    echo "--- prefill.log last 20 ---"; tail -20 /tmp/dynamo_runtime_test/prefill.log
    kill $FRONTEND_PID $DECODE_PID $PREFILL_PID 2>/dev/null || true
    exit 1
  fi
done

# /metrics responding != vLLM loaded. Wait until at least one vllm: line shows.
echo "[wait] vLLM engine loaded (vllm: metrics present)"
for i in $(seq 1 480); do
  if [ "$(curl -s localhost:$PREFILL_SYSTEM_PORT/metrics 2>/dev/null | grep -c '^vllm:')" -gt 0 ] && \
     [ "$(curl -s localhost:$DECODE_SYSTEM_PORT/metrics 2>/dev/null | grep -c '^vllm:')" -gt 0 ]; then
    echo "[wait] vllm: metrics OK on both workers"
    break
  fi
  sleep 2
done

# Confirm the connector loaded by greping the prefill log
echo
echo "[verify-1] confirm our connector loaded in prefill worker:"
grep -E "NixlConnectorWithPendingMetrics|kv_connector_module_path|Creating v1 connector" /tmp/dynamo_runtime_test/prefill.log | head -3

dump_metrics() {
  local label="$1"
  echo
  echo "===================================================================="
  echo "  $label"
  echo "===================================================================="
  curl -s localhost:$PREFILL_SYSTEM_PORT/metrics 2>/dev/null | python3 -c "
import sys, re
keep = [
    'vllm:num_requests_running',
    'vllm:num_requests_waiting',
    'vllm:kv_cache_usage_perc',
    'vllm:nixl_num_pending_sends',
    'vllm:nixl_num_in_process_reqs',
    'vllm:nixl_num_kv_expired_reqs_total',
    'vllm:num_preemptions_total',
]
text = sys.stdin.read()
print(f'  {\"metric\":<48}  {\"value\":>12}')
print('  ' + '-' * 70)
for name in keep:
    pat = re.compile(r'^' + re.escape(name) + r'(\{[^}]*\})?\s+(\S+)', re.M)
    matches = pat.findall(text)
    if matches:
        for labels, value in matches:
            print(f'  {name:<48}  {value:>12}')
    else:
        print(f'  {name:<48}  {\"<missing>\":>12}')
"
}

dump_metrics "T0: cold start — through Dynamo runtime"

echo
echo "[T1] sending request via Dynamo frontend on port $FRONTEND_PORT..."
RESP=$(curl -s -X POST localhost:$FRONTEND_PORT/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"'$MODEL'","messages":[{"role":"user","content":"What is 7+8? One sentence."}],"max_tokens":32,"temperature":0.0}')
echo "[T1] response (150 chars): ${RESP:0:150}"

dump_metrics "T1: after successful round-trip via Dynamo frontend"

echo
echo "[T2] firing another request, killing decode mid-transfer..."
(curl -s -X POST localhost:$FRONTEND_PORT/v1/chat/completions \
   -H 'Content-Type: application/json' \
   -d '{"model":"'$MODEL'","messages":[{"role":"user","content":"Write a 6-line poem about a robot."}],"max_tokens":128,"temperature":0.0}' \
   > /tmp/dynamo_runtime_test/req2.log 2>&1) &
REQ2_PID=$!
sleep 3
echo "[T2] killing decode (pid=$DECODE_PID)"
kill -9 $DECODE_PID 2>/dev/null || true
sleep 3

dump_metrics "T2: 3s after decode killed — invisible occupancy, prefill side"

sleep 10
dump_metrics "T3: 13s after decode killed — strand holds, no traffic"

echo
echo "[cleanup]"
kill $FRONTEND_PID $PREFILL_PID $REQ2_PID 2>/dev/null || true
wait 2>/dev/null
echo "[done] logs in /tmp/dynamo_runtime_test/"
