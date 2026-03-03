#!/usr/bin/env bash
# Disagg planner-driven experiments: separate prefill/decode workers with
# KV-aware routing. Planner scales decode workers based on TTFT/ITL breaches.
#
# Disagg planner-specific env vars:
#   TTFT_TARGET          TTFT SLA target in ms (default: 2000)
#   ITL_TARGET           ITL SLA target in ms (default: 50)
#   MIN_REPLICAS         Minimum replicas for prefill and decode (default: 1)
#   MAX_GPU_BUDGET       Total GPU budget across prefill+decode (default: 32)
#   PLANNER_INTERVAL     Load-based adjustment interval in seconds (default: 5)
#   TARGET_NODE_COUNT    Number of target nodes to reserve (default: 2)

# Pass KUBE_CONTEXT through to run scripts for parallel cluster support
export KUBE_CONTEXT="${KUBE_CONTEXT:-}"

RUN_COUNT=0

run() {
  local desc="$1"
  RUN_COUNT=$((RUN_COUNT + 1))
  echo ""
  echo "##############################################################"
  echo "# Experiment $RUN_COUNT: $desc"
  echo "# Model: $MODEL_NAME | TP=$TP_SIZE DP=$DP_SIZE | Tag: $RUN_TAG"
  echo "# Planner (disagg): TTFT=${TTFT_TARGET}ms ITL=${ITL_TARGET}ms budget=${MAX_GPU_BUDGET} interval=${PLANNER_INTERVAL}s"
  echo "# $(date)"
  echo "##############################################################"
  echo ""
  ./mx_benchmarks/run_disagg_planner.sh
  local rc=$?
  echo ""
  echo "##############################################################"
  echo "# Experiment $RUN_COUNT FINISHED (exit=$rc): $desc"
  echo "# $(date)"
  echo "##############################################################"
  echo ""
}

# ------------- Run deepseek-ai/DeepSeek-V3 -------------

# Common settings for all DSv3 runs
export DYNAMO_IMAGE_TAG=vllm-runtime-0c6e86e33-mx_ref-64f9249
export MX_SERVER_IMAGE_TAG=64f9249
export RUN_ID=dsv3-0-disagg
export MODEL_NAME=deepseek-ai/DeepSeek-V3
export VLLM_USE_DEEP_GEMM=1
export VLLM_USE_FLASHINFER_MOE_FP8=0
export ENABLE_EP=1
export INPUT_TRACE=conversation_trace.jsonl
export MAX_ISL=1024
export MAX_OSL=1024
export STEP_DURATIONS=120,900,900,200,120
export STEP_RATES=1,3,5,3,1
export DATASET=staircase_trace_r-${STEP_RATES}_d-${STEP_DURATIONS}
export SLICE_DURATION=30
export STAT=avg

# Disagg planner settings
export TTFT_TARGET=900
export ITL_TARGET=50
export MIN_REPLICAS=1
export MAX_GPU_BUDGET=32
export PLANNER_INTERVAL=5
export TARGET_NODE_COUNT=2

# TP=8 default (clear cache for new TP config)
export TP_SIZE=8
export DP_SIZE=0
export VLLM_EXTRA_ARGS=""
export CLEAR_COMPILE_CACHE=1
export RUN_TAG=tp8-ep-default-no_flashinfer_moe_fp8-disagg-planner
run "DSv3 TP=8 EP default (disagg planner)"

echo ""
echo "##############################################################"
echo "# All $RUN_COUNT experiments completed at $(date)"
echo "##############################################################"
