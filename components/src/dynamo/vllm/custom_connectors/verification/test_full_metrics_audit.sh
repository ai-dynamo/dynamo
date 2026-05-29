#!/bin/bash
# Thorough audit: dump the FULL prefill /metrics output at three phases and
# compute a diff between them, then we can map each non-zero metric back to
# its v0.19.0 source to confirm whether it tracks the stranded request.
set -e

GPU_ID=0
PREFILL_PORT=8100
DECODE_PORT=8200
PROXY_PORT=8000
MODEL="Qwen/Qwen2.5-0.5B-Instruct"

KV_CONFIG='{
  "kv_connector":"NixlConnectorWithPendingMetrics",
  "kv_role":"kv_both",
  "kv_connector_module_path":"dynamo.vllm.custom_connectors.nixl_with_pending_metrics"
}'

pkill -f "vllm serve" 2>/dev/null || true
pkill -f "toy_proxy_server" 2>/dev/null || true
sleep 3

mkdir -p /tmp/pd_audit

CUDA_VISIBLE_DEVICES=$GPU_ID \
  PYTHONPATH=/home/krish/repos/amz-ads/dynamo/components/src \
  VLLM_NIXL_SIDE_CHANNEL_PORT=5559 \
  VLLM_NIXL_ABORT_REQUEST_TIMEOUT=600 \
  vllm serve "$MODEL" --port $PREFILL_PORT \
    --enforce-eager --max-model-len 512 --max-num-seqs 4 \
    --gpu-memory-utilization 0.30 \
    --kv-transfer-config "$KV_CONFIG" \
    > /tmp/pd_audit/prefill.log 2>&1 &
PREFILL_PID=$!

CUDA_VISIBLE_DEVICES=$GPU_ID \
  PYTHONPATH=/home/krish/repos/amz-ads/dynamo/components/src \
  VLLM_NIXL_SIDE_CHANNEL_PORT=5659 \
  vllm serve "$MODEL" --port $DECODE_PORT \
    --enforce-eager --max-model-len 512 --max-num-seqs 4 \
    --gpu-memory-utilization 0.30 \
    --kv-transfer-config "$KV_CONFIG" \
    > /tmp/pd_audit/decode.log 2>&1 &
DECODE_PID=$!

for PORT in $PREFILL_PORT $DECODE_PORT; do
  for i in $(seq 1 240); do
    curl -fs localhost:$PORT/v1/models > /dev/null 2>&1 && break || sleep 1
  done
done

python /home/krish/repos/amz-ads/verification/toy_proxy_server.py \
  --prefiller-host localhost --prefiller-port $PREFILL_PORT \
  --decoder-host localhost --decoder-port $DECODE_PORT \
  --port $PROXY_PORT > /tmp/pd_audit/proxy.log 2>&1 &
PROXY_PID=$!
sleep 3

# Strip helper: keep only metric value lines, drop _created (these are
# Prometheus-internal "this counter was created at time" floats that don't
# represent state).
snap() {
  curl -s localhost:$PREFILL_PORT/metrics 2>/dev/null \
    | grep -E '^vllm:' \
    | grep -v '_created' \
    | sort
}

echo "[T0] capturing baseline (no traffic yet)..."
snap > /tmp/pd_audit/T0_baseline.txt

# Send a normal request that completes successfully — note: this WILL change
# many counter values (request_total, prompt_tokens, etc.). The point of this
# phase is to show those are unrelated to stranded state.
echo "[T1] sending request that completes successfully..."
curl -s -X POST localhost:$PROXY_PORT/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"'$MODEL'","messages":[{"role":"user","content":"What is 7 times 8? One sentence."}],"max_tokens":32,"temperature":0.0}' \
  > /tmp/pd_audit/req1.log 2>&1
sleep 2
snap > /tmp/pd_audit/T1_after_success.txt

# Send a second request, then kill decode mid-transfer to strand it.
echo "[T2] sending second request + killing decode mid-transfer..."
(curl -s -X POST localhost:$PROXY_PORT/v1/chat/completions \
   -H 'Content-Type: application/json' \
   -d '{"model":"'$MODEL'","messages":[{"role":"user","content":"Write a 6-line poem about robots."}],"max_tokens":128,"temperature":0.0}' \
   > /tmp/pd_audit/req2.log 2>&1) &
REQ2_PID=$!
sleep 3
kill -9 $DECODE_PID 2>/dev/null || true
sleep 3
snap > /tmp/pd_audit/T2_stranded.txt

# Wait 10 more seconds (no traffic) to confirm nothing else moves
sleep 10
snap > /tmp/pd_audit/T3_still_stranded.txt

echo
echo "===================================================================="
echo "DIFF: T2 (stranded) vs T1 (last successful state)"
echo "===================================================================="
echo "Lines that CHANGED between successful state and stranded state."
echo "Any non-zero change here is a metric that the strand event TRIGGERED."
echo ""
diff /tmp/pd_audit/T1_after_success.txt /tmp/pd_audit/T2_stranded.txt | head -40

echo
echo "===================================================================="
echo "DIFF: T3 (12s later, still stranded) vs T2 (just-stranded)"
echo "===================================================================="
echo "Lines that changed between two snapshots of the STRANDED state with NO"
echo "intervening traffic. Anything in here is moving WITHOUT new requests."
echo ""
diff /tmp/pd_audit/T2_stranded.txt /tmp/pd_audit/T3_still_stranded.txt | head -40

echo
echo "===================================================================="
echo "ALL non-zero metrics at the stranded state (T2):"
echo "===================================================================="
grep -v ' 0\.0$\|^$\|^# ' /tmp/pd_audit/T2_stranded.txt | head -50

echo
echo "===================================================================="
echo "Metrics that explicitly track in-flight scheduler state (at T2):"
echo "===================================================================="
grep -E '^vllm:(num_requests_running|num_requests_waiting|kv_cache_usage_perc|num_preemptions|nixl_num_pending_sends|nixl_num_in_process|nixl_num_kv_expired)' /tmp/pd_audit/T2_stranded.txt \
  | grep -v '_created'

echo
kill $PREFILL_PID $PROXY_PID $REQ2_PID 2>/dev/null || true
wait 2>/dev/null
echo "[done] snapshots saved in /tmp/pd_audit/"
