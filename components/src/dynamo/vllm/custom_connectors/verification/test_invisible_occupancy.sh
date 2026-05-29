#!/bin/bash
# Demonstrates the diagnostic gap our new metric closes:
#
# When decode dies mid-transfer, prefill holds the KV blocks but the request
# is invisible to standard scheduler-state metrics. We dump the full set of
# relevant gauges side-by-side at each phase so the contrast is obvious.
#
# Expected at "T2 after decode dies":
#    vllm:num_requests_running   = 0     ← invisible
#    vllm:num_requests_waiting   = 0     ← invisible
#    vllm:kv_cache_usage         > 0     ← KV is pinned (but no clue why)
#    vllm:nixl_num_pending_sends > 0     ← NEW: explains where it went
#    vllm:nixl_num_kv_expired_reqs_total = 0  ← timeout hasn't fired yet
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

mkdir -p /tmp/pd_logs_invis

echo "[launch] prefill on port $PREFILL_PORT"
CUDA_VISIBLE_DEVICES=$GPU_ID \
  PYTHONPATH=/home/krish/repos/amz-ads/dynamo/components/src \
  VLLM_NIXL_SIDE_CHANNEL_PORT=5559 \
  VLLM_NIXL_ABORT_REQUEST_TIMEOUT=60 \
  vllm serve "$MODEL" \
    --port $PREFILL_PORT \
    --enforce-eager --max-model-len 512 --max-num-seqs 4 \
    --gpu-memory-utilization 0.30 \
    --kv-transfer-config "$KV_CONFIG" \
    > /tmp/pd_logs_invis/prefill.log 2>&1 &
PREFILL_PID=$!

echo "[launch] decode on port $DECODE_PORT"
CUDA_VISIBLE_DEVICES=$GPU_ID \
  PYTHONPATH=/home/krish/repos/amz-ads/dynamo/components/src \
  VLLM_NIXL_SIDE_CHANNEL_PORT=5659 \
  vllm serve "$MODEL" \
    --port $DECODE_PORT \
    --enforce-eager --max-model-len 512 --max-num-seqs 4 \
    --gpu-memory-utilization 0.30 \
    --kv-transfer-config "$KV_CONFIG" \
    > /tmp/pd_logs_invis/decode.log 2>&1 &
DECODE_PID=$!

for PORT in $PREFILL_PORT $DECODE_PORT; do
  for i in $(seq 1 240); do
    curl -fs localhost:$PORT/v1/models > /dev/null 2>&1 && break || sleep 1
  done
done
echo "[wait] both up"

python /home/krish/repos/amz-ads/verification/toy_proxy_server.py \
  --prefiller-host localhost --prefiller-port $PREFILL_PORT \
  --decoder-host localhost --decoder-port $DECODE_PORT \
  --port $PROXY_PORT > /tmp/pd_logs_invis/proxy.log 2>&1 &
PROXY_PID=$!
sleep 3

# Pretty-print only the metrics we care about, from prefill's /metrics
dump_metrics() {
  local label="$1"
  echo
  echo "===================================================================="
  echo "  $label"
  echo "===================================================================="
  curl -s localhost:$PREFILL_PORT/metrics 2>/dev/null | python3 -c "
import sys, re
keep = [
    ('vllm:num_requests_running',          'standard'),
    ('vllm:num_requests_waiting',          'standard'),
    ('vllm:kv_cache_usage_perc',           'standard'),
    ('vllm:nixl_num_pending_sends',        'NEW'),
    ('vllm:nixl_num_in_process_reqs',      'NEW'),
    ('vllm:nixl_num_kv_expired_reqs_total','standard'),
    ('vllm:num_preemptions_total',         'standard'),
]
text = sys.stdin.read()
print(f'  {\"metric\":<48}  {\"value\":>10}  source')
print('  ' + '-' * 78)
for name, source in keep:
    pat = re.compile(r'^' + re.escape(name) + r'(\{[^}]*\})?\s+(\S+)', re.M)
    matches = pat.findall(text)
    if matches:
        for labels, value in matches:
            tag = '[NEW]' if source == 'NEW' else ''
            print(f'  {name:<48}  {value:>10}  {tag}')
    else:
        print(f'  {name:<48}  {\"<missing>\":>10}')
"
}

dump_metrics "T0: cold start, no traffic"

echo
echo "[T1] sending one request..."
RESP=$(curl -s -X POST localhost:$PROXY_PORT/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"'$MODEL'","messages":[{"role":"user","content":"What is 7 times 8? Answer in one short sentence."}],"max_tokens":32,"temperature":0.0}')
echo "[T1] response (first 150 chars): ${RESP:0:150}..."

dump_metrics "T1: after successful round-trip"

# Now strand a request by killing decode mid-transfer
echo
echo "[T2] firing another request and killing decode mid-transfer..."
(curl -s -X POST localhost:$PROXY_PORT/v1/chat/completions \
   -H 'Content-Type: application/json' \
   -d '{"model":"'$MODEL'","messages":[{"role":"user","content":"Write a short poem about a robot, 6 lines."}],"max_tokens":128,"temperature":0.0}' \
   > /tmp/pd_logs_invis/req2.log 2>&1) &
REQ2_PID=$!
sleep 3
kill -9 $DECODE_PID 2>/dev/null || true
echo "[T2] decode killed"
sleep 2

dump_metrics "T2: 2s after decode killed - HERE IS THE INVISIBLE OCCUPANCY"

echo
echo "[T3] waiting 10s more without traffic - state should hold..."
sleep 10
dump_metrics "T3: 12s after decode killed - still pinned, sweep hasn't fired"

echo
echo "[cleanup]"
kill $PREFILL_PID $PROXY_PID $REQ2_PID 2>/dev/null || true
wait 2>/dev/null
echo "[done] full logs in /tmp/pd_logs_invis/"
