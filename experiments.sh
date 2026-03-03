#!/usr/bin/env bash
# Usage:
#   ./experiments.sh
#   KUBE_CONTEXT=teleport-cluster-1 ./experiments.sh   # run on a specific cluster
#
RUN_COUNT=0

run() {
  local desc="$1"
  RUN_COUNT=$((RUN_COUNT + 1))
  echo ""
  echo "##############################################################"
  echo "# Experiment $RUN_COUNT: $desc"
  echo "# Model: $MODEL_NAME | TP=$TP_SIZE DP=$DP_SIZE | Tag: $RUN_TAG"
  echo "# $(date)"
  echo "##############################################################"
  echo ""
  ./mx_benchmarks/run.sh
  local rc=$?
  echo ""
  echo "##############################################################"
  echo "# Experiment $RUN_COUNT FINISHED (exit=$rc): $desc"
  echo "# $(date)"
  echo "##############################################################"
  echo ""
}

# Common settings for all runs
export KUBE_CONTEXT="${KUBE_CONTEXT:-}"
export RDMA_RESOURCE=rdma/shared_ib
export RDMA_COUNT=8
export DYNAMO_IMAGE_TAG=vllm-runtime-ea86df298-mx_ref-64f9249-debug
export MX_SERVER_IMAGE_TAG=64f9249
export VLLM_USE_DEEP_GEMM=1
export VLLM_USE_FLASHINFER_MOE_FP8=0
export ENABLE_TORCH_COMPILE_CACHE=1
export DEEPGEMM_SOURCE_WARMUP_MODE=full
export DEEPGEMM_TARGET_WARMUP_MODE=skip
export ENABLE_EP=1
export INPUT_TRACE=conversation_trace.jsonl
export MAX_ISL=1024
export MAX_OSL=1024
export STEP_DURATIONS=180,900
export STEP_RATES=1,8
export CONSTANT_RATE=0
CONSTANT_SUFFIX=$( [ "$CONSTANT_RATE" = "1" ] && echo "-constant" || echo "" )
export DATASET=staircase_trace_r-${STEP_RATES}_d-${STEP_DURATIONS}${CONSTANT_SUFFIX}
export SLICE_DURATION=30
export SCALEUP_DELAY=300
export STAT=avg
export ENABLE_REQUEST_LOGGING=0

# ------------- Run moonshotai/Kimi-K2.5 -------------
export RUN_ID=kimi-k25-0
export MODEL_NAME=moonshotai/Kimi-K2.5

# TP=8 default (clear cache for new TP config)
export TP_SIZE=8
export DP_SIZE=0
export VLLM_EXTRA_ARGS=""
export RUN_TAG=tp8-ep-trial1
run "Kimi-K2.5 TP=8 EP default"

# ------------- Run deepseek-ai/DeepSeek-V3 -------------
# export RUN_ID=dsv3-0
# export MODEL_NAME=deepseek-ai/DeepSeek-V3

# # TP=8 default (clear cache for new TP config)
# export TP_SIZE=8
# export DP_SIZE=0
# export VLLM_EXTRA_ARGS=""
# # export RUN_TAG=tp8-ep-trial10
# export RUN_TAG=tp8-ep-default-no_flashinfer_moe_fp8-deepgemm_full-trial7
# run "DSv3 TP=8 EP default"

## TP=8 + extra args (reuse cache from above)
#export VLLM_EXTRA_ARGS="\
#--gpu-memory-utilization 0.90 \
#--quantization fp8 \
#--max-model-len 8192"
#export RUN_TAG=tp8-ep-fp8-cxl-8192-no_flashinfer_moe_fp8
#run "DSv3 TP=8 EP fp8 max-model-len=8192"

## DP=8 default (clear cache for new DP config)
#export TP_SIZE=0
#export DP_SIZE=8
#export VLLM_EXTRA_ARGS=""
#export RUN_TAG=dp8-ep-default-no_flashinfer_moe_fp8
#run "DSv3 DP=8 EP default"
#
## DP=8 + extra args (reuse cache from above)
#export VLLM_EXTRA_ARGS="\
#--gpu-memory-utilization 0.90 \
#--quantization fp8 \
#--max-model-len 8192"
#export RUN_TAG=dp8-ep-fp8-cxl-8192-no_flashinfer_moe_fp8
#run "DSv3 DP=8 EP fp8 max-model-len=8192"


# ------------- Run deepseek-ai/DeepSeek-V3.2 -------------
#export RUN_ID=dsv3-2
#export MODEL_NAME=deepseek-ai/DeepSeek-V3.2

## TP=8 default (clear cache for new model + TP config)
#export TP_SIZE=8
#export DP_SIZE=0
#export VLLM_EXTRA_ARGS=""
#export RUN_TAG=tp8-ep-default-no_flashinfer_moe_fp8-trial4
#run "DSv3.2 TP=8 EP default"

## TP=8 + extra args (reuse cache)
#export VLLM_EXTRA_ARGS="\
#--quantization fp8 \
#--max-model-len 8192"
#export RUN_TAG=tp8-ep-fp8-cxl-8192-no_deepgemm
#run "DSv3.2 TP=8 EP fp8 max-model-len=8192"
#
## DP=8 default (clear cache for new DP config)
#export TP_SIZE=0
#export DP_SIZE=8
#export VLLM_EXTRA_ARGS=""
#export RUN_TAG=dp8-ep-default-no_deepgemm
#run "DSv3.2 DP=8 EP default"
#
## DP=8 + extra args (reuse cache)
#export VLLM_EXTRA_ARGS="\
#--quantization fp8 \
#--max-model-len 8192"
#export RUN_TAG=dp8-ep-fp8-cxl-8192-no_deepgemm
#run "DSv3.2 DP=8 EP fp8 max-model-len=8192"

echo ""
echo "##############################################################"
echo "# All $RUN_COUNT experiments completed at $(date)"
echo "##############################################################"
