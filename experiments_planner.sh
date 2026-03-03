#!/usr/bin/env bash
# Planner-driven experiments: same models/configs as experiments.sh but using
# run_planner.sh (automatic scale-up via planner instead of manual sleep).
#
# Usage:
#   ./experiments_planner.sh
#   KUBE_CONTEXT=teleport-cluster-1 ./experiments_planner.sh
#
# Planner-specific env vars:
#   TTFT_TARGET          TTFT SLA target in ms (default: 2000)
#   MIN_REPLICAS         Minimum mx-target replicas (default: 0)
#   MAX_REPLICAS         Maximum mx-target replicas (default: 2)
#   PLANNER_INTERVAL     Load-based adjustment interval in seconds (default: 5)
#   COLD_START_THRESHOLD TTFT threshold (ms) for pre-regression scale-up (default: 0)

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
  echo "# Planner: TTFT=${TTFT_TARGET}ms interval=${PLANNER_INTERVAL}s max=${MAX_REPLICAS}"
  echo "# $(date)"
  echo "##############################################################"
  echo ""
  ./mx_benchmarks/run_planner.sh
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
export RUN_ID=dsv3-0
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

# Planner settings
export TTFT_TARGET=900
export MIN_REPLICAS=0
export MAX_REPLICAS=2
export PLANNER_INTERVAL=5

# TP=8 default (clear cache for new TP config)
export TP_SIZE=8
export DP_SIZE=0
export VLLM_EXTRA_ARGS=""
export CLEAR_COMPILE_CACHE=1
export RUN_TAG=tp8-ep-default-no_flashinfer_moe_fp8-planner
run "DSv3 TP=8 EP default (planner)"

## TP=8 + extra args (reuse cache from above)
#export VLLM_EXTRA_ARGS="\
#--gpu-memory-utilization 0.90 \
#--quantization fp8 \
#--max-model-len 8192"
#export CLEAR_COMPILE_CACHE=0
#export RUN_TAG=tp8-ep-fp8-cxl-8192-no_flashinfer_moe_fp8-planner
#run "DSv3 TP=8 EP fp8 max-model-len=8192 (planner)"

## DP=8 default (clear cache for new DP config)
#export TP_SIZE=0
#export DP_SIZE=8
#export VLLM_EXTRA_ARGS=""
#export CLEAR_COMPILE_CACHE=1
#export RUN_TAG=dp8-ep-default-no_flashinfer_moe_fp8-planner
#run "DSv3 DP=8 EP default (planner)"
#
## DP=8 + extra args (reuse cache from above)
#export VLLM_EXTRA_ARGS="\
#--gpu-memory-utilization 0.90 \
#--quantization fp8 \
#--max-model-len 8192"
#export CLEAR_COMPILE_CACHE=0
#export RUN_TAG=dp8-ep-fp8-cxl-8192-no_flashinfer_moe_fp8-planner
#run "DSv3 DP=8 EP fp8 max-model-len=8192 (planner)"


# ------------- Run deepseek-ai/DeepSeek-V3.2 -------------

## Common settings for all DSv3.2 runs
#export RUN_ID=dsv3-2-planner
#export MODEL_NAME=deepseek-ai/DeepSeek-V3.2
#
## TP=8 default (clear cache for new model + TP config)
#export TP_SIZE=8
#export DP_SIZE=0
#export VLLM_EXTRA_ARGS=""
#export CLEAR_COMPILE_CACHE=1
#export RUN_TAG=tp8-ep-default-no_deepgemm-planner
#run "DSv3.2 TP=8 EP default (planner)"
#
## TP=8 + extra args (reuse cache)
#export VLLM_EXTRA_ARGS="\
#--quantization fp8 \
#--max-model-len 8192"
#export CLEAR_COMPILE_CACHE=0
#export RUN_TAG=tp8-ep-fp8-cxl-8192-no_deepgemm-planner
#run "DSv3.2 TP=8 EP fp8 max-model-len=8192 (planner)"
#
## DP=8 default (clear cache for new DP config)
#export TP_SIZE=0
#export DP_SIZE=8
#export VLLM_EXTRA_ARGS=""
#export CLEAR_COMPILE_CACHE=1
#export RUN_TAG=dp8-ep-default-no_deepgemm-planner
#run "DSv3.2 DP=8 EP default (planner)"
#
## DP=8 + extra args (reuse cache)
#export VLLM_EXTRA_ARGS="\
#--quantization fp8 \
#--max-model-len 8192"
#export CLEAR_COMPILE_CACHE=0
#export RUN_TAG=dp8-ep-fp8-cxl-8192-no_deepgemm-planner
#run "DSv3.2 DP=8 EP fp8 max-model-len=8192 (planner)"

echo ""
echo "##############################################################"
echo "# All $RUN_COUNT experiments completed at $(date)"
echo "##############################################################"
