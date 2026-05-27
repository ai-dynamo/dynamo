# Source-only profile defaults for the Dynamo vLLM KVBM harness.
#
# Workflow scripts in this directory are topology-oriented. They should source
# this file for model, sizing, and placement defaults instead of embedding
# machine assumptions directly. Callers can override any value with env vars.

kvbm_xdc_require() {
  local name=$1
  if [ -z "${!name:-}" ]; then
    echo "$name is required for KVBM_HARDWARE_PROFILE=custom" >&2
    return 1
  fi
}

kvbm_xdc_apply_hardware_profile() {
  local scope=${1:-all}
  KVBM_HARDWARE_PROFILE=${KVBM_HARDWARE_PROFILE:-h100-a100}
  case "$KVBM_HARDWARE_PROFILE" in
    h100-a100)
      # Recorded experiment profile. Use one worker per H100/A100 GPU.
      : "${MODEL:=deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
      : "${GPU_MEMORY_UTILIZATION:=0.70}"
      : "${MAX_MODEL_LEN:=2048}"
      : "${MAX_NUM_SEQS:=8}"
      : "${CPU_CACHE_GB:=16}"
      if [ "$scope" != "worker" ]; then
        : "${DECODE_CUDA_VISIBLE_DEVICES:=0}"
        : "${PREFILL_CUDA_VISIBLE_DEVICES:=1}"
      fi
      : "${GPU_CLASS:=H100-or-A100}"
      ;;
    spark-gb10)
      # Local smoke profile. Keeps the same workflow runnable on constrained
      # Spark/GB10 systems; results from this profile are not experiment data.
      : "${MODEL:=Qwen/Qwen3-0.6B}"
      : "${GPU_MEMORY_UTILIZATION:=0.15}"
      : "${MAX_MODEL_LEN:=1024}"
      : "${MAX_NUM_SEQS:=8}"
      : "${CPU_CACHE_GB:=2}"
      : "${DECODE_CUDA_VISIBLE_DEVICES:=0}"
      : "${PREFILL_CUDA_VISIBLE_DEVICES:=0}"
      : "${GPU_CLASS:=GB10}"
      ;;
    custom)
      : "${GPU_CLASS:=custom}"
      if [ "$scope" = "model-only" ]; then
        kvbm_xdc_require MODEL
      elif [ "$scope" = "worker" ]; then
        for var in \
          MODEL \
          GPU_MEMORY_UTILIZATION \
          MAX_MODEL_LEN \
          MAX_NUM_SEQS \
          CPU_CACHE_GB; do
          kvbm_xdc_require "$var"
        done
      else
        for var in \
          MODEL \
          GPU_MEMORY_UTILIZATION \
          MAX_MODEL_LEN \
          MAX_NUM_SEQS \
          CPU_CACHE_GB \
          DECODE_CUDA_VISIBLE_DEVICES \
          PREFILL_CUDA_VISIBLE_DEVICES; do
          kvbm_xdc_require "$var"
        done
      fi
      ;;
    *)
      echo "KVBM_HARDWARE_PROFILE must be h100-a100, spark-gb10, or custom; got $KVBM_HARDWARE_PROFILE" >&2
      return 2
      ;;
  esac
  if [ "$scope" != "model-only" ]; then
    : "${PREFILL_GPU_MEMORY_UTILIZATION:=$GPU_MEMORY_UTILIZATION}"
    : "${DECODE_GPU_MEMORY_UTILIZATION:=$GPU_MEMORY_UTILIZATION}"
    export GPU_MEMORY_UTILIZATION PREFILL_GPU_MEMORY_UTILIZATION DECODE_GPU_MEMORY_UTILIZATION
    export MAX_MODEL_LEN MAX_NUM_SEQS CPU_CACHE_GB
    if [ "$scope" != "worker" ]; then
      export DECODE_CUDA_VISIBLE_DEVICES PREFILL_CUDA_VISIBLE_DEVICES
    fi
  fi
  export KVBM_HARDWARE_PROFILE MODEL GPU_CLASS
}
