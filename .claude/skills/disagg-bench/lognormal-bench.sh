#!/bin/bash
# Lognormal ISL AIPerf benchmark: 1P+1D KVBM disaggregated serving.
#
# Runs prefill on GPU 0 and decode on GPU 1, waits for both to be
# healthy, then runs AIPerf with synthetic ISL=2048 (representative of
# lognormal(mu=7, sigma=1.5) median ~1100 tokens, SOLBench-shape).
#
# Run inside a Pyxis/srun container (same as disagg-smoke scripts).
# The KVBM hub binary must already be built (cargo build -p kvbm-hub).
# Use the build wrapper in disagg-bringup/env.sh + maturin develop first.
#
# kv_role is always "kv_both" for both prefill and decode roles — the
# prefill/decode distinction is in disagg.role, NOT kv_role.
#
# Env knobs:
#   KVBM_BENCH_MODEL      HF model ID (for aiperf --model-names)
#   KVBM_BENCH_MODEL_DIR  Local model dir (default: $HF_HOME/qwen3-8b)
#   KVBM_BENCH_N          Number of AIPerf requests (default: 200)
#   KVBM_BENCH_CONCURRENCY                           (default: 8)
#   KVBM_BENCH_ISL        Synthetic ISL tokens       (default: 2048)
#   KVBM_BENCH_OSL        Fixed output length        (default: 256)
#   KVBM_BENCH_GPU_PREFILL GPU for prefill           (default: 0)
#   KVBM_BENCH_GPU_DECODE  GPU for decode            (default: 1)
#   KVBM_BENCH_MEM_UTIL   GPU memory utilization     (default: 0.45)
#   KVBM_BENCH_MAX_LEN    max-model-len for vLLM     (default: 8192)
#
# Honored from environment (set by disagg-bringup/env.sh):
#   KVBM_REPO, KVBM_HUB_BIN, KVBM_EXPERIMENTS_DIR, HF_HOME
#
# Usage (inside srun Pyxis container, after sourcing env.sh):
#   export KVBM_BENCH_MODEL_DIR=/tmp/hf_cache/qwen3-8b
#   bash .claude/skills/disagg-bench/lognormal-bench.sh

set -uo pipefail

DYNAMO=${KVBM_REPO:-/home/ryan/repos/dynamo}
HUB_BIN=${KVBM_HUB_BIN:-$DYNAMO/target/debug/kvbm_hub}
EXP_BASE=${KVBM_EXPERIMENTS_DIR:-/tmp/kvbm-experiments}
HF_HOME=${HF_HOME:-/tmp/hf_cache}

MODEL=${KVBM_BENCH_MODEL:-Qwen/Qwen3-8B}
MODEL_DIR=${KVBM_BENCH_MODEL_DIR:-$HF_HOME/qwen3-8b}
N=${KVBM_BENCH_N:-200}
CONCURRENCY=${KVBM_BENCH_CONCURRENCY:-8}
ISL=${KVBM_BENCH_ISL:-2048}
OSL=${KVBM_BENCH_OSL:-256}
GPU_PREFILL=${KVBM_BENCH_GPU_PREFILL:-0}
GPU_DECODE=${KVBM_BENCH_GPU_DECODE:-1}
MEM_UTIL=${KVBM_BENCH_MEM_UTIL:-0.45}
MAX_LEN=${KVBM_BENCH_MAX_LEN:-8192}

HUB_URL="http://127.0.0.1:1337"
PREFILL_PORT=8000
DECODE_PORT=8001

mkdir -p "$EXP_BASE"
EXP_DIR=$(bash "$DYNAMO/.claude/skills/disagg-bringup/new-experiment.sh" lognormal-bench)
echo "EXP=$EXP_DIR"

# Teardown stale processes and wait for GPU memory to clear
pkill -f vllm-entrypoints 2>/dev/null || true
pkill -x kvbm_hub 2>/dev/null || true
sleep 5
STALE=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -v '^$' || true)
[ -n "$STALE" ] && echo "$STALE" | xargs -r kill -9 2>/dev/null || true
sleep 2
echo "[GPU state after cleanup]"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || echo "(empty)"

# Hub
"$HUB_BIN" --port 1337 > "$EXP_DIR/hub.log" 2>&1 &
sleep 2

# Build KV transfer configs — kv_role=kv_both for BOTH roles
# disagg.role ("prefill"/"decode") is the actual role discriminator
KV_PREFILL='{"kv_buffer_size":1e9,"kv_connector":"DynamoConnector","kv_connector_module_path":"kvbm.v2.vllm.schedulers.connector","kv_role":"kv_both","kv_connector_extra_config":{"leader":{"disagg":{"hub_url":"'"$HUB_URL"'","role":"prefill"},"cache":{"host":{"cache_size_gb":1.0}},"tokio":{"worker_threads":2}},"worker":{"nixl":{"backends":{"UCX":{},"POSIX":{}}},"tokio":{"worker_threads":2}}}}'
KV_DECODE='{"kv_buffer_size":1e9,"kv_connector":"DynamoConnector","kv_connector_module_path":"kvbm.v2.vllm.schedulers.connector","kv_role":"kv_both","kv_connector_extra_config":{"leader":{"disagg":{"hub_url":"'"$HUB_URL"'","role":"decode"},"cache":{"host":{"cache_size_gb":1.0}},"tokio":{"worker_threads":2}},"worker":{"nixl":{"backends":{"UCX":{},"POSIX":{}}},"tokio":{"worker_threads":2}}}}'

CUDA_VISIBLE_DEVICES=$GPU_PREFILL python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" --port $PREFILL_PORT --host 0.0.0.0 \
  --max-model-len $MAX_LEN --gpu-memory-utilization $MEM_UTIL \
  --max-num-seqs 32 --enable-chunked-prefill \
  --kv-transfer-config "$KV_PREFILL" \
  > "$EXP_DIR/prefill.log" 2>&1 & PFILL=$!

CUDA_VISIBLE_DEVICES=$GPU_DECODE python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" --port $DECODE_PORT --host 0.0.0.0 \
  --max-model-len $MAX_LEN --gpu-memory-utilization $MEM_UTIL \
  --max-num-seqs 32 --enable-chunked-prefill \
  --kv-transfer-config "$KV_DECODE" \
  > "$EXP_DIR/decode.log" 2>&1 & DECODE=$!

# Wait for both — with live process check to catch early crashes
echo "Waiting for vLLMs (up to 240s)..."
for i in $(seq 1 120); do
  P=$(curl -sf "http://127.0.0.1:$PREFILL_PORT/health" 2>/dev/null && echo ok || echo no)
  D=$(curl -sf "http://127.0.0.1:$DECODE_PORT/health"  2>/dev/null && echo ok || echo no)
  [ "$P" = "ok" ] && [ "$D" = "ok" ] && echo "BOTH UP $(date -u +%FT%TZ)" && break
  kill -0 $PFILL  2>/dev/null || { echo "Prefill died!"; tail -5 "$EXP_DIR/prefill.log"; exit 1; }
  kill -0 $DECODE 2>/dev/null || { echo "Decode died!";  tail -5 "$EXP_DIR/decode.log";  exit 1; }
  [ $((i % 15)) -eq 0 ] && echo "  still waiting ${i}x2s P=$P D=$D"
  sleep 2
done

echo "=== AIPerf START (n=$N c=$CONCURRENCY isl=$ISL osl=$OSL) ==="
# Note: use --request-count (not --num-prompts) to set the number of benchmark
# requests. --num-prompts controls dataset size; the profiling phase defaults to
# concurrency*2 if --request-count is not given.
# Note: Qwen3-8B thinking mode produces empty streaming chunks for ~14% of requests
# at c=1 (InvalidInferenceResultError). Use a non-thinking model or add:
#   --extra-inputs enable_thinking:false --extra-inputs ignore_eos:true
# ignore_eos:true forces the model to generate exactly --osl tokens, eliminating
# InvalidInferenceResultError from short outputs (~11% of requests otherwise).
aiperf profile \
  --url "http://127.0.0.1:$DECODE_PORT" \
  --model-names "$MODEL" \
  --endpoint-type chat \
  --request-count "$N" \
  --concurrency "$CONCURRENCY" \
  --isl "$ISL" \
  --osl "$OSL" \
  --extra-inputs enable_thinking:false \
  --extra-inputs ignore_eos:true \
  --streaming \
  2>&1 | tee "$EXP_DIR/aiperf.log"
AIPERF_RC=${PIPESTATUS[0]}

echo "AIPERF_RC=$AIPERF_RC"
echo "Results at: $EXP_DIR/aiperf.log"

pkill -f vllm-entrypoints 2>/dev/null || true
pkill -x kvbm_hub 2>/dev/null || true

exit $AIPERF_RC
