#!/bin/bash
# Single-GPU prefill + decode disagg test with our custom connector.
# Run two vLLM processes sharing one GPU; use the upstream toy_proxy_server
# to orchestrate P->D handoff; query /metrics to verify pending_sends moves.
set -e

GPU_ID=0
PREFILL_PORT=8100
DECODE_PORT=8200
PROXY_PORT=8000
PREFILL_SCP=5559
DECODE_SCP=5659
MODEL="Qwen/Qwen2.5-0.5B-Instruct"

KV_CONFIG='{
  "kv_connector":"NixlConnectorWithPendingMetrics",
  "kv_role":"kv_both",
  "kv_connector_module_path":"dynamo.vllm.custom_connectors.nixl_with_pending_metrics"
}'

# Cleanup any prior runs
pkill -f "vllm serve" 2>/dev/null || true
pkill -f "toy_proxy_server" 2>/dev/null || true
sleep 1

mkdir -p /tmp/pd_logs

echo "[launch] starting PREFILL on GPU $GPU_ID port $PREFILL_PORT"
CUDA_VISIBLE_DEVICES=$GPU_ID \
  PYTHONPATH=/home/krish/repos/amz-ads/dynamo/components/src \
  VLLM_NIXL_SIDE_CHANNEL_PORT=$PREFILL_SCP \
  vllm serve "$MODEL" \
    --port $PREFILL_PORT \
    --enforce-eager \
    --max-model-len 512 \
    --max-num-seqs 4 \
    --gpu-memory-utilization 0.30 \
    --kv-transfer-config "$KV_CONFIG" \
    > /tmp/pd_logs/prefill.log 2>&1 &
PREFILL_PID=$!
echo "[launch] PREFILL pid=$PREFILL_PID"

echo "[launch] starting DECODE on GPU $GPU_ID port $DECODE_PORT"
CUDA_VISIBLE_DEVICES=$GPU_ID \
  PYTHONPATH=/home/krish/repos/amz-ads/dynamo/components/src \
  VLLM_NIXL_SIDE_CHANNEL_PORT=$DECODE_SCP \
  vllm serve "$MODEL" \
    --port $DECODE_PORT \
    --enforce-eager \
    --max-model-len 512 \
    --max-num-seqs 4 \
    --gpu-memory-utilization 0.30 \
    --kv-transfer-config "$KV_CONFIG" \
    > /tmp/pd_logs/decode.log 2>&1 &
DECODE_PID=$!
echo "[launch] DECODE pid=$DECODE_PID"

# Wait for both to become healthy
echo "[wait] for /v1/models to respond on both ports..."
for PORT in $PREFILL_PORT $DECODE_PORT; do
  for i in $(seq 1 240); do
    if curl -fs localhost:$PORT/v1/models > /dev/null 2>&1; then
      echo "[wait] port $PORT OK"
      break
    fi
    sleep 1
  done
  if ! curl -fs localhost:$PORT/v1/models > /dev/null 2>&1; then
    echo "[wait] port $PORT FAILED to come up"
    echo "--- prefill.log last 30 ---"
    tail -30 /tmp/pd_logs/prefill.log
    echo "--- decode.log last 30 ---"
    tail -30 /tmp/pd_logs/decode.log
    kill $PREFILL_PID $DECODE_PID 2>/dev/null || true
    exit 1
  fi
done

echo "[proxy] starting toy_proxy_server on port $PROXY_PORT"
python /home/krish/repos/amz-ads/verification/toy_proxy_server.py \
  --prefiller-host localhost --prefiller-port $PREFILL_PORT \
  --decoder-host localhost --decoder-port $DECODE_PORT \
  --port $PROXY_PORT > /tmp/pd_logs/proxy.log 2>&1 &
PROXY_PID=$!
sleep 3
echo "[proxy] pid=$PROXY_PID"

echo
echo "=================================================================="
echo "[T0] Before any traffic - both gauges should be 0"
echo "=================================================================="
curl -s localhost:$PREFILL_PORT/metrics | grep -E "nixl_num_pending_sends|nixl_num_in_process_reqs" | grep -v "^#" || true
echo
echo "[T1] Sending one chat completion through the proxy..."
RESP=$(curl -s -X POST localhost:$PROXY_PORT/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"'$MODEL'","messages":[{"role":"user","content":"What is 2+2? Answer in one short sentence."}],"max_tokens":32,"temperature":0.0}')
echo "[T1] response (first 200 chars): ${RESP:0:200}"
echo
echo "[T2] After successful round-trip - pending should be back at 0 (transfer completed)"
echo "=================================================================="
curl -s localhost:$PREFILL_PORT/metrics | grep -E "nixl_num_pending_sends|nixl_num_in_process_reqs" | grep -v "^#" || true
echo
echo "[T3] Examining ALL nixl metrics on PREFILL (post-traffic):"
curl -s localhost:$PREFILL_PORT/metrics | grep -E "^vllm:nixl" | grep -v "^#" | head -20
echo
echo
echo "=================================================================="
echo "[T4] Now testing the stranded-block case: kill DECODE mid-transfer"
echo "=================================================================="
echo "[T4] sending request in background then killing decode..."
(
  curl -s -X POST localhost:$PROXY_PORT/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"'$MODEL'","messages":[{"role":"user","content":"Tell me a 100-word story about a cat."}],"max_tokens":128,"temperature":0.0}' \
    > /tmp/pd_logs/req2.log 2>&1
) &
REQ_PID=$!
# Sleep briefly so prefill has time to enter pending state
sleep 4
echo "[T4] killing DECODE (pid=$DECODE_PID)"
kill -9 $DECODE_PID 2>/dev/null || true
sleep 2
echo
echo "[T5] After decode kill - pending_sends on PREFILL should be > 0:"
echo "=================================================================="
curl -s localhost:$PREFILL_PORT/metrics | grep -E "nixl_num_pending_sends|nixl_num_in_process_reqs|nixl_num_kv_expired" | grep -v "^#" || true
echo
echo "[T5] (waiting 5s and re-reading to confirm value sticks while timeout is pending)"
sleep 5
curl -s localhost:$PREFILL_PORT/metrics | grep -E "nixl_num_pending_sends|nixl_num_in_process_reqs|nixl_num_kv_expired" | grep -v "^#" || true
echo
echo "[cleanup]"
kill $PREFILL_PID $PROXY_PID $REQ_PID 2>/dev/null || true
wait 2>/dev/null
echo "[done] logs in /tmp/pd_logs/"
