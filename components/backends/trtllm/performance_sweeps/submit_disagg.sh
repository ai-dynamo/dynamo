#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if [[ -z ${MODEL_PATH} ]]; then
    echo "ERROR: MODEL_PATH was not set."
    echo "ERROR: MODEL_PATH must be set to either the HuggingFace ID or locally " \
         "downloaded path to the model weights. Since Deepseek R1 is large, it is " \
         "recommended to pre-download them to a shared location and provide the path."
    exit 1
fi

if [[ -z ${SERVED_MODEL_NAME} ]]; then
    echo "ERROR: SERVED_MODEL_NAME was not set."
    exit 1
fi

IMAGE="${IMAGE:-""}"

# For GB200, we use 4 tasks per node.
NTASKS_PER_NODE="${NTASKS_PER_NODE:-4}"

ISL="${ISL:-8150}"
OSL="${OSL:-1024}"

# Build slurm_args step-by-step with validation and defaults
slurm_args="--time=04:00:00"

# Add partition if set
if [[ -n "${SLURM_PARTITION:-}" ]]; then
    slurm_args="${slurm_args} --partition=${SLURM_PARTITION}"
fi

# Add account if set
if [[ -n "${SLURM_ACCOUNT:-}" ]]; then
    slurm_args="${slurm_args} --account=${SLURM_ACCOUNT}"
fi

# Add job name with sensible default
if [[ -n "${SLURM_JOB_NAME:-}" ]]; then
    slurm_args="${slurm_args} --job-name=${SLURM_JOB_NAME}"
fi

# Usage Instructions
usage() {
    echo "Usage: $0 <mtp_mode> <mode> [ctx_num] [gen_num] [gen_tp_size] [gen_batch_size] [gen_max_num_tokens] [gen_gpu_memory_fraction] [gen_eplb_num_slots] [gen_mtp_size] [gen_concurrency_list]"
    echo ""
    echo "MTP Modes:"
    echo "  mtp=off - Run without Multi-Token Prediction (gen_mtp_size=0)"
    echo "  mtp=on  - Run with Multi-Token Prediction (gen_mtp_size=1,2,3)"
    echo ""
    echo "Execution Modes:"
    echo "  all - Run all predefined GPU configurations (4, 8, 16, 32 GPUs)"
    echo "  tep - Run Tensor-Expert Parallel mode (attention_dp=false)"
    echo "  dep - Run Data-Expert Parallel mode (attention_dp=true)"
    echo "  4GPU, 8GPU, 16GPU, 32GPU - Run specific GPU configurations"
    echo ""
    echo "Parameters for tep/dep modes:"
    echo "  ctx_num: Number of context nodes"
    echo "  gen_num: Number of generation nodes"
    echo "  gen_tp_size: Generation tensor parallel size"
    echo "  gen_batch_size: Generation batch size"
    echo "  gen_max_num_tokens: Generation max number of tokens"
    echo "  gen_gpu_memory_fraction: GPU memory fraction (0.7-0.95)"
    echo "  gen_mtp_size: Multi-Token Prediction size (0 for mtp=off, 1-3 for mtp=on)"
    echo "  gen_eplb_num_slots: Expert load balancing slots (0, 256, 288)"
    echo "  gen_concurrency_list: Concurrency values (space-separated, quoted)"
    echo ""
    echo "Examples:"
    echo "  $0 mtp=off all                                    # Run all MTP0 predefined combinations"
    echo "  $0 mtp=on all                                     # Run all MTP predefined combinations"
    echo "  $0 mtp=off tep 1 3 4 128 128 0.9 0 0 \"1 2 4 8\" # Run MTP0 TEP with specific config"
    echo "  $0 mtp=on dep 2 3 8 256 256 0.8 0 256 \"256 512 1024\" # Run MTP DEP with specific config"
    exit 1
}

# Run single task
run_single() {
    local ctx_num=$1
    local ctx_tp_size=$2
    local ctx_ep_size=$3
    local ctx_enable_attention_dp=$4
    local gen_num=$5
    local gen_tp_size=$6
    local gen_ep_size=$7
    local gen_batch_size=$8
    local gen_max_num_tokens=$9
    local gen_enable_attention_dp=${10}
    local gen_gpu_memory_fraction=${11}
    local gen_eplb_num_slots=${12}
    local gen_mtp_size=${13}
    local gen_concurrency_list=${14}

    # TODO: expose kind to the command line
    local kind="dynamo_disagg"

    gen_nodes=$(((gen_tp_size + 3)/4 * gen_num))
    total_nodes=$((ctx_num + gen_nodes))
    total_tasks=$((total_nodes * 4))
    set -x
    sbatch --nodes=${total_nodes} --ntasks=${total_tasks} --ntasks-per-node=${NTASKS_PER_NODE} --segment=${total_nodes} ${slurm_args} benchmark_disagg.slurm ${ctx_num} ${ctx_tp_size} ${ctx_ep_size} ${ctx_enable_attention_dp} 30 20000 ${gen_num} ${gen_tp_size} ${gen_ep_size} ${gen_batch_size} ${gen_max_num_tokens} ${gen_enable_attention_dp} ${gen_gpu_memory_fraction} ${gen_eplb_num_slots} ${gen_mtp_size} "${gen_concurrency_list}" ${gen_nodes} ${kind} ${MODEL_PATH} ${SERVED_MODEL_NAME} ${IMAGE} ${ISL} ${OSL}
    set +x
}

# Main function
main() {
    local mtp_mode=$1
    local mode=$2

    # Validate MTP mode
    if [[ "$mtp_mode" != "mtp=off" && "$mtp_mode" != "mtp=on" ]]; then
        echo "Error: Invalid MTP mode '$mtp_mode'. Must be 'mtp=off' or 'mtp=on'"
        usage
    fi

    case $mode in
        "tep")
            if [ $# -ne 14 ]; then
                echo "Error: TEP mode requires 14 additional parameters (including mtp_mode)"
                usage
            fi

            local ctx_num=$3
            local ctx_tp_size=$4
            local ctx_ep_size=$5
            local ctx_enable_attention_dp=$6
            local gen_num=$7
            local gen_tp_size=$8
            local gen_batch_size=$9
            local gen_max_num_tokens=${10}
            local gen_gpu_memory_fraction=${11}
            local gen_mtp_size=${12}
            local gen_eplb_num_slots=${13}
            local gen_concurrency_list=${14}

            echo "Running TEP mode ($mtp_mode) with ctx_num=$ctx_num, gen_num=$gen_num, gen_tp_size=$gen_tp_size, gen_ep_size=$gen_tp_size, gen_batch_size=$gen_batch_size, gen_max_num_tokens=$gen_max_num_tokens, gen_gpu_memory_fraction=$gen_gpu_memory_fraction, gen_mtp_size=$gen_mtp_size, gen_eplb_num_slots=$gen_eplb_num_slots, gen_concurrency_list=\"$gen_concurrency_list\""

            # TEP mode: Use false to disable attention dp
            run_single $ctx_num $ctx_tp_size $ctx_ep_size $ctx_enable_attention_dp $gen_num $gen_tp_size $gen_tp_size $gen_batch_size $gen_max_num_tokens false $gen_gpu_memory_fraction $gen_mtp_size $gen_eplb_num_slots "$gen_concurrency_list"
            ;;
        "dep")
            if [ $# -ne 14 ]; then
                echo "Error: DEP mode requires 14 additional parameters (including mtp_mode)"
                usage
            fi

            local ctx_num=$3
            local ctx_tp_size=$4
            local ctx_ep_size=$5
            local ctx_enable_attention_dp=$6
            local gen_num=$7
            local gen_tp_size=$8
            local gen_batch_size=$9
            local gen_max_num_tokens=${10}
            local gen_gpu_memory_fraction=${11}
            local gen_mtp_size=${12}
            local gen_eplb_num_slots=${13}
            local gen_concurrency_list=${14}

            echo "Running DEP mode ($mtp_mode) with ctx_num=$ctx_num, ctx_tp_size=$ctx_tp_size, ctx_enable_attention_dp=$ctx_enable_attention_dp, gen_num=$gen_num, gen_tp_size=$gen_tp_size, gen_ep_size=$gen_tp_size, gen_batch_size=$gen_batch_size, gen_max_num_tokens=$gen_max_num_tokens, gen_gpu_memory_fraction=$gen_gpu_memory_fraction, gen_mtp_size=$gen_mtp_size, gen_eplb_num_slots=$gen_eplb_num_slots, gen_concurrency_list=\"$gen_concurrency_list\""

            run_single $ctx_num $ctx_tp_size $ctx_ep_size $ctx_enable_attention_dp $gen_num $gen_tp_size $gen_tp_size $gen_batch_size $gen_max_num_tokens true $gen_gpu_memory_fraction $gen_mtp_size $gen_eplb_num_slots "$gen_concurrency_list"
            ;;
        "tp")
            if [ $# -ne 14 ]; then
                echo "Error: TP mode requires 14 additional parameters (including mtp_mode)"
                usage
            fi

            local ctx_num=$3
            local ctx_tp_size=$4
            local ctx_ep_size=$5
            local ctx_enable_attention_dp=$6
            local gen_num=$7
            local gen_tp_size=$8
            local gen_batch_size=$9
            local gen_max_num_tokens=${10}
            local gen_gpu_memory_fraction=${11}
            local gen_mtp_size=${12}
            local gen_eplb_num_slots=${13}
            local gen_concurrency_list=${14}

            echo "Running TP mode ($mtp_mode) with ctx_num=$ctx_num, gen_num=$gen_num, gen_tp_size=$gen_tp_size, gen_ep_size=1, gen_batch_size=$gen_batch_size, gen_max_num_tokens=$gen_max_num_tokens, gen_gpu_memory_fraction=$gen_gpu_memory_fraction, gen_mtp_size=$gen_mtp_size, gen_eplb_num_slots=$gen_eplb_num_slots, gen_concurrency_list=\"$gen_concurrency_list\""

            run_single $ctx_num $ctx_tp_size $ctx_ep_size $ctx_enable_attention_dp $gen_num $gen_tp_size 1 $gen_batch_size $gen_max_num_tokens false $gen_gpu_memory_fraction $gen_mtp_size $gen_eplb_num_slots "$gen_concurrency_list"
            ;;
        *)
            echo "Error: Unknown mode '$mode'"
            usage
            ;;
    esac
}

# Check parameters
if [ $# -eq 0 ]; then
    usage
fi

# Run main function
main "$@"
