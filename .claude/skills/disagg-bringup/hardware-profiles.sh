# Source-only hardware profile defaults for KVBM bring-up workflows.

kvbm_profile_require() {
  local name=$1
  if [ -z "${!name:-}" ]; then
    echo "$name is required for the selected hardware profile" >&2
    return 1
  fi
}

kvbm_apply_disagg_bringup_profile() {
  KVBM_HARDWARE_PROFILE=${KVBM_HARDWARE_PROFILE:-spark-gb10}
  case "$KVBM_HARDWARE_PROFILE" in
    h100-a100)
      : "${KVBM_MODEL:=deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
      : "${KVBM_MAX_MODEL_LEN:=2048}"
      : "${KVBM_MAX_NUM_SEQS:=8}"
      : "${KVBM_GPU_MEMORY_UTILIZATION:=0.70}"
      : "${KVBM_PREFILL_GPU_MEMORY_UTILIZATION:=$KVBM_GPU_MEMORY_UTILIZATION}"
      : "${KVBM_DECODE_GPU_MEMORY_UTILIZATION:=$KVBM_GPU_MEMORY_UTILIZATION}"
      : "${KVBM_SINGLE_GPU_MEMORY_UTILIZATION:=$KVBM_GPU_MEMORY_UTILIZATION}"
      : "${KVBM_CPU_CACHE_GB:=16}"
      : "${KVBM_DECODE_CUDA_VISIBLE_DEVICES:=0}"
      : "${KVBM_PREFILL_CUDA_VISIBLE_DEVICES:=1}"
      : "${KVBM_SINGLE_CUDA_VISIBLE_DEVICES:=0}"
      : "${KVBM_GPU_CLASS:=H100-or-A100}"
      ;;
    spark-gb10)
      : "${KVBM_MODEL:=Qwen/Qwen3-0.6B}"
      : "${KVBM_MAX_MODEL_LEN:=1024}"
      : "${KVBM_MAX_NUM_SEQS:=8}"
      : "${KVBM_GPU_MEMORY_UTILIZATION:=0.15}"
      : "${KVBM_PREFILL_GPU_MEMORY_UTILIZATION:=$KVBM_GPU_MEMORY_UTILIZATION}"
      : "${KVBM_DECODE_GPU_MEMORY_UTILIZATION:=$KVBM_GPU_MEMORY_UTILIZATION}"
      : "${KVBM_SINGLE_GPU_MEMORY_UTILIZATION:=0.30}"
      : "${KVBM_CPU_CACHE_GB:=2}"
      : "${KVBM_DECODE_CUDA_VISIBLE_DEVICES:=0}"
      : "${KVBM_PREFILL_CUDA_VISIBLE_DEVICES:=0}"
      : "${KVBM_SINGLE_CUDA_VISIBLE_DEVICES:=0}"
      : "${KVBM_GPU_CLASS:=GB10}"
      ;;
    custom)
      : "${KVBM_PREFILL_GPU_MEMORY_UTILIZATION:=${KVBM_GPU_MEMORY_UTILIZATION:-}}"
      : "${KVBM_DECODE_GPU_MEMORY_UTILIZATION:=${KVBM_GPU_MEMORY_UTILIZATION:-}}"
      : "${KVBM_SINGLE_GPU_MEMORY_UTILIZATION:=${KVBM_GPU_MEMORY_UTILIZATION:-}}"
      : "${KVBM_GPU_CLASS:=custom}"
      for var in \
        KVBM_MODEL \
        KVBM_MAX_MODEL_LEN \
        KVBM_MAX_NUM_SEQS \
        KVBM_CPU_CACHE_GB \
        KVBM_DECODE_CUDA_VISIBLE_DEVICES \
        KVBM_PREFILL_CUDA_VISIBLE_DEVICES \
        KVBM_SINGLE_CUDA_VISIBLE_DEVICES \
        KVBM_DECODE_GPU_MEMORY_UTILIZATION \
        KVBM_PREFILL_GPU_MEMORY_UTILIZATION \
        KVBM_SINGLE_GPU_MEMORY_UTILIZATION; do
        kvbm_profile_require "$var"
      done
      ;;
    *)
      echo "KVBM_HARDWARE_PROFILE must be h100-a100, spark-gb10, or custom; got $KVBM_HARDWARE_PROFILE" >&2
      return 2
      ;;
  esac
  export KVBM_HARDWARE_PROFILE KVBM_MODEL KVBM_MAX_MODEL_LEN KVBM_MAX_NUM_SEQS
  export KVBM_CPU_CACHE_GB KVBM_GPU_CLASS
  export KVBM_DECODE_CUDA_VISIBLE_DEVICES KVBM_PREFILL_CUDA_VISIBLE_DEVICES KVBM_SINGLE_CUDA_VISIBLE_DEVICES
  export KVBM_DECODE_GPU_MEMORY_UTILIZATION KVBM_PREFILL_GPU_MEMORY_UTILIZATION KVBM_SINGLE_GPU_MEMORY_UTILIZATION
}

kvbm_apply_p2p_profile() {
  P2P_HARDWARE_PROFILE=${P2P_HARDWARE_PROFILE:-${KVBM_HARDWARE_PROFILE:-h100-a100}}
  case "$P2P_HARDWARE_PROFILE" in
    h100-a100)
      : "${P2P_MODEL:=deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
      : "${P2P_GMU:=0.70}"
      : "${P2P_MAX_MODEL_LEN:=2048}"
      : "${P2P_CACHE_GB:=16}"
      : "${P2P_A_CUDA_VISIBLE_DEVICES:=0}"
      : "${P2P_B_CUDA_VISIBLE_DEVICES:=1}"
      : "${P2P_GPU_CLASS:=H100-or-A100}"
      ;;
    spark-gb10)
      : "${P2P_MODEL:=Qwen/Qwen3-0.6B}"
      : "${P2P_GMU:=0.15}"
      : "${P2P_MAX_MODEL_LEN:=1024}"
      : "${P2P_CACHE_GB:=2}"
      : "${P2P_A_CUDA_VISIBLE_DEVICES:=0}"
      : "${P2P_B_CUDA_VISIBLE_DEVICES:=0}"
      : "${P2P_GPU_CLASS:=GB10}"
      ;;
    custom)
      : "${P2P_GPU_CLASS:=custom}"
      for var in \
        P2P_MODEL \
        P2P_GMU \
        P2P_MAX_MODEL_LEN \
        P2P_CACHE_GB \
        P2P_A_CUDA_VISIBLE_DEVICES \
        P2P_B_CUDA_VISIBLE_DEVICES; do
        kvbm_profile_require "$var"
      done
      ;;
    *)
      echo "P2P_HARDWARE_PROFILE must be h100-a100, spark-gb10, or custom; got $P2P_HARDWARE_PROFILE" >&2
      return 2
      ;;
  esac
  export P2P_HARDWARE_PROFILE P2P_MODEL P2P_GMU P2P_MAX_MODEL_LEN P2P_CACHE_GB
  export P2P_A_CUDA_VISIBLE_DEVICES P2P_B_CUDA_VISIBLE_DEVICES P2P_GPU_CLASS
}
