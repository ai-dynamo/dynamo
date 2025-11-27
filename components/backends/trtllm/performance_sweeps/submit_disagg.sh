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

# MTP0 Configuration (gen_mtp_size=0)
run_4_gpus_mtp0() {
    echo "Running 4 GPUs MTP0 combinations..."
    if (( ISL == OSL )); then
        run_single 1 5 4 128 128 false "0.9" 0 0 "1 2 4 8 16 32 64 128 192"
        run_single 1 5 4 64 64 true "0.85" 0 0 "256 384"
        run_single 1 4 4 128 128 true "0.85" 0 0 "512 768"
        run_single 2 5 4 256 256 true "0.85" 0 0 "1024 1536"
        run_single 1 2 4 512 512 true "0.85" 0 0 "2048 3072"
        run_single 2 3 4 768 768 true "0.85" 0 0 "3072 4096"
    else
        run_single 1 5 4 16 16 false "0.9" 0 0 "1 2 4 8 16 24"
        run_single 1 4 4 32 32 false "0.9" 0 0 "32 48"
        run_single 2 5 4 64 64 false "0.9" 0 0 "64 96"
        run_single 1 2 4 128 128 false "0.9" 0 0 "128 192"
        run_single 1 1 4 64 64 true "0.8" 0 0 "256 384"
        run_single 3 2 4 128 128 true "0.8" 0 0 "512 768"
    fi
}

run_8_gpus_mtp0() {
    echo "Running 8 GPUs MTP0 combinations..."
    if (( ISL == OSL )); then
        run_single 1 4 8 128 128 false "0.9" 0 0 "1 2 4 8 16 32 64 128 192 256"
        run_single 1 4 8 32 32 true "0.8" 0 0 "256 384"
        run_single 1 3 8 64 64 true "0.8" 0 0 "512 768"
        run_single 1 2 8 128 128 true "0.8" 0 0 "1024 1536"
        run_single 1 1 8 256 256 true "0.8" 0 0 "2048 3072"
        run_single 1 1 8 512 512 true "0.8" 0 0 "4096 6144"
        run_single 3 2 8 768 768 true "0.8" 0 0 "6144 8192"
        run_single 3 2 8 1024 1024 true "0.8" 0 0 "8192 12288"
    else
        run_single 1 4 8 16 16 false "0.9" 0 0 "1 2 4 8 16 24"
        run_single 1 3 8 32 32 false "0.9" 0 0 "32 48"
        run_single 1 2 8 64 64 false "0.9" 0 0 "64 96"
        run_single 1 1 8 128 128 false "0.9" 0 0 "128 192"
        run_single 3 2 8 32 32 true "0.8" 0 0 "256 384"
        run_single 5 2 8 64 64 true "0.8" 0 0 "512 768"
        run_single 4 1 8 128 128 true "0.8" 0 0 "1024 1536"
        run_single 5 1 8 256 256 true "0.8" 0 0 "2048 3072"
    fi
}

run_16_gpus_mtp0() {
    echo "Running 16 GPUs MTP0 combinations..."
    if (( ISL == OSL )); then
        run_single 1 1 16 64 64 true "0.75" 0 0 "16 32 64 128 256 512 1024 1536"
        run_single 2 1 16 128 128 true "0.75" 0 256 "2048 3072"
        run_single 2 1 16 256 256 true "0.75" 0 256 "4096 6144"
        run_single 3 1 16 512 512 true "0.75" 0 256 "8192 12288"
        run_single 3 1 16 768 768 true "0.75" 0 256 "12288 16384"
        run_single 3 1 16 1024 1024 true "0.75" 0 288 "16384 20480"
    else
        run_single 1 1 16 8 8 true "0.8" 0 0 "16 32 64 128 192" # 5
        run_single 2 1 16 16 16 true "0.8" 0 0 "256 384"        # 6
        run_single 3 1 16 32 32 true "0.8" 0 0 "512 768"       # 7
        run_single 6 1 16 64 64 true "0.8" 0 0 "1024 1536"     # 10
        run_single 8 1 16 128 128 true "0.8" 0 256 "2048 3072"   # 12
        run_single 10 1 16 256 256 true "0.8" 0 256 "4096 6144" # 14
    fi
}

run_32_gpus_mtp0() {
    echo "Running 32 GPUs MTP0 combinations..."
    if (( ISL == OSL )); then
        run_single 1 1 32 32 32 true "0.7" 0 0 "32 64 128 256 512 1024 1536"
        run_single 2 1 32 64 64 true "0.7" 0 256 "2048 3072"
        run_single 3 1 32 128 128 true "0.7" 0 288 "4096 6144"
        run_single 4 1 32 256 256 true "0.7" 0 288 "8192 12288"
        run_single 5 1 32 512 512 true "0.7" 0 288 "16384 20480"
    else
        run_single 1 1 32 4 4 true "0.7" 0 0 "32 64 128 192"  # 9
        run_single 2 1 32 8 8 true "0.7" 0 0 "256 384"          # 10
        run_single 4 1 32 16 16 true "0.7" 0 0 "512 768"       # 12
        run_single 7 1 32 32 32 true "0.7" 0 0 "1024 1536"     # 15
    fi
}

# MTP Configuration (gen_mtp_size=1,2,3)
run_4_gpus_mtp() {
    echo "Running 4 GPUs MTP combinations..."
    if (( ISL == OSL )); then
        run_single 1 5 4 32 128 false "0.9" 3 0 "1 2 4 8 16 32 48"
        run_single 1 5 4 32 128 true "0.9" 3 0 "64 128 192"
        run_single 1 4 4 64 256 true "0.9" 3 0 "256 384"
        run_single 1 3 4 128 512 true "0.9" 3 0 "512 768"
        run_single 1 2 4 256 768 true "0.9" 2 0 "1024 1536"
        run_single 2 3 4 512 1024 true "0.9" 1 0 "2048 3072"
        run_single 1 1 4 768 1536 true "0.9" 1 0 "3072 4096"
    else
        run_single 1 5 4 8 32 false "0.9" 3 0 "1 2 4 8 12"
        run_single 1 4 4 16 64 false "0.9" 3 0 "16 24"
        run_single 1 3 4 32 128 false "0.9" 3 0 "32 48"
        run_single 2 3 4 16 64 true "0.8" 3 0 "64 96"
        run_single 1 1 4 32 128 true "0.8" 3 0 "128 192"
        run_single 2 1 4 64 256 true "0.8" 2 0 "256 384"
        run_single 5 2 4 128 512 true "0.8" 1 0 "512 768"
    fi
}

run_8_gpus_mtp() {
    echo "Running 8 GPUs MTP combinations..."
    if (( ISL == OSL )); then
        run_single 1 4 8 32 128 false "0.9" 3 0 "1 2 4 8 16 32 48"
        run_single 1 4 8 16 64 true "0.8" 3 0 "64 128 192"
        run_single 1 3 8 32 128 true "0.8" 3 0 "256 384"
        run_single 1 2 8 64 256 true "0.8" 3 0 "512 768"
        run_single 1 1 8 128 512 true "0.8" 3 0 "1024 1536"
        run_single 1 1 8 256 512 true "0.8" 1 0 "2048 3072"
        run_single 3 2 8 512 1024 true "0.8" 1 0 "4096 6144"
        run_single 3 2 8 768 1536 true "0.8" 1 0 "6144 8192"
        run_single 3 2 8 1024 2048 true "0.8" 1 0 "8192 12288"
    else
        run_single 1 4 8 8 32 false "0.9" 3 0 "1 2 4 8 12"
        run_single 1 3 8 16 64 false "0.9" 3 0 "16 24"
        run_single 1 2 8 32 128 false "0.9" 3 0 "32 48"
        run_single 1 1 8 8 32 true "0.8" 3 0 "64 96"
        run_single 3 2 8 16 64 true "0.8" 3 0 "128 192"
        run_single 5 2 8 32 128 true "0.8" 3 0 "256 384"
        run_single 7 2 8 64 256 true "0.8" 2 0 "512 768"
        run_single 5 1 8 128 256 true "0.8" 1 0 "1024 1536"
        run_single 6 1 8 256 512 true "0.8" 1 0 "2048 3072"
    fi
}

run_16_gpus_mtp() {
    echo "Running 16 GPUs MTP combinations..."
    if (( ISL == OSL )); then
        run_single 1 1 16 32 128 true "0.7" 3 0 "16 32 64 128 256 512 768"
        run_single 1 1 16 64 256 true "0.7" 3 256 "1024 1536"
        run_single 2 1 16 128 256 true "0.7" 1 288 "2048 3072"
        run_single 2 1 16 256 512 true "0.7" 1 288 "4096 6144"
        run_single 3 1 16 512 1024 true "0.7" 1 288 "8192 12288"
        run_single 3 1 16 768 1536 true "0.7" 1 288 "12288 16384"
        run_single 3 1 16 1024 1024 true "0.75" 0 288 "16384 20480"
    else
        run_single 1 1 16 4 16 true "0.8" 3 0 "16 32 64 96" # 5
        run_single 2 1 16 8 32 true "0.8" 3 0 "128 192"       # 6
        run_single 4 1 16 16 64 true "0.8" 3 0 "256 384"      # 8
        run_single 6 1 16 32 128 true "0.8" 3 0 "512 768"    # 10
        run_single 8 1 16 64 256 true "0.8" 2 256 "1024 1536" # 13
        run_single 10 1 16 128 256 true "0.8" 1 256 "2048 3072" # 15
        run_single 12 1 16 256 512 true "0.8" 1 256 "4096 6144" # 16
    fi

}

run_32_gpus_mtp() {
    echo "Running 32 GPUs MTP combinations..."
    if (( ISL == OSL )); then
        run_single 1 1 32 16 64 true "0.6" 3 0 "32 64 128 256 512 768"
        run_single 2 1 32 32 128 true "0.6" 3 288 "1024 1536"
        run_single 3 1 32 64 256 true "0.6" 3 288 "2048 3072"
        run_single 3 1 32 128 256 true "0.6" 1 288 "4096 6144"
        run_single 4 1 32 256 512 true "0.6" 1 288 "8192 12288"
        run_single 5 1 32 512 1024 true "0.6" 1 288 "16384 20480"
    else
        run_single 1 1 32 1 4 true "0.7" 3 0 "32 48" # 9
        run_single 2 1 32 2 8 true "0.7" 3 0 "64 96" # 10
        run_single 3 1 32 4 16 true "0.7" 3 0 "128 192" # 11
        run_single 5 1 32 8 32 true "0.7" 3 0 "256 384" # 13
        run_single 8 1 32 16 64 true "0.7" 3 256 "512 768" # 16
    fi
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
        "all")
            echo "Running all GPU configurations for $mtp_mode mode..."
            if [[ "$mtp_mode" == "mtp=off" ]]; then
                run_4_gpus_mtp0
                run_8_gpus_mtp0
                run_16_gpus_mtp0
                run_32_gpus_mtp0
            else
                run_4_gpus_mtp
                run_8_gpus_mtp
                run_16_gpus_mtp
                run_32_gpus_mtp
            fi
            ;;
        "pareto")
            # 1k/1k
            export ISL=1024
            export OSL=1024
            export CACHE_TRANSCEIVER_MAX_NUM_TOKENS=4608

            if [[ "$mtp_mode" == "mtp=off" ]]; then
                # 1k/1k mtp=off
                run_single 1 4 8 128 128 false "0.9" 0 0 "1 2 4 8 16 32 64 141"
                run_single 1 1 32 32 32 true "0.7" 0 0 "1075"
                run_single 1 1 16 64 64 true "0.75" 0 0 "1075"
                run_single 2 1 16 256 256 true "0.75" 0 0 "2048 4300"
                run_single 1 1 8 512 512 true "0.8" 0 0 "4300"

            else
                # 1k/1k mtp=on
                run_single 1 4 8 32 128 false "0.9" 3 0 "1 2 4 8 16 36"
                run_single 1 1 16 64 256 true "0.7" 3 0 "512 1075"
                run_single 2 1 16 128 256 true "0.7" 1 0 "2150"
                run_single 1 1 32 16 64 true "0.6" 3 0 "512"
                run_single 1 1 8 256 512 true "0.8" 1 0 "2252"
            fi

            # 8k/1k
            export ISL=8192
            export OSL=1024
            export CACHE_TRANSCEIVER_MAX_NUM_TOKENS=8448

            if [[ "$mtp_mode" == "mtp=off" ]]; then
                # 8k/1k mtp=off
                run_single 1 3 8 32 32 false "0.9" 0 0 "1 2 4 8 16 34"
                run_single 4 1 32 16 16 true "0.7" 0 0 "256 538"
                run_single 7 1 32 32 32 true "0.7" 0 0 "1075" # remove if need 5 cofigs
                run_single 6 1 16 64 64 true "0.75" 0 0 "1075"
                run_single 8 1 16 128 128 true "0.75" 0 0 "2150"
                run_single 5 1 8 256 256 true "0.8" 0 0 "2150"
            else
                # 8k/1k mtp=on
                run_single 1 3 8 16 64 false "0.9" 3 0 "1 2 4 8 18"
                run_single 5 1 32 8 32 true "0.7" 3 0 "128 269"
                run_single 8 1 32 16 64 true "0.7" 3 0 "538"
                run_single 6 1 16 32 128 true "0.75" 3 0 "538" # remove if need 5 configs
                run_single 8 1 16 64 256 true "0.75" 2 0 "1075"
                run_single 5 1 8 128 256 true "0.8" 1 0 "1075" # remove if need 5 configs
                run_single 6 1 8 256 512 true "0.8" 1 0 "2150"
            fi
            ;;
        "4GPU")
            echo "Running 4 GPUs combinations for $mtp_mode mode..."
            if [[ "$mtp_mode" == "mtp=off" ]]; then
                run_4_gpus_mtp0
            else
                run_4_gpus_mtp
            fi
            ;;
        "8GPU")
            echo "Running 8 GPUs combinations for $mtp_mode mode..."
            if [[ "$mtp_mode" == "mtp=off" ]]; then
                run_8_gpus_mtp0
            else
                run_8_gpus_mtp
            fi
            ;;
        "16GPU")
            echo "Running 16 GPUs combinations for $mtp_mode mode..."
            if [[ "$mtp_mode" == "mtp=off" ]]; then
                run_16_gpus_mtp0
            else
                run_16_gpus_mtp
            fi
            ;;
        "32GPU")
            echo "Running 32 GPUs combinations for $mtp_mode mode..."
            if [[ "$mtp_mode" == "mtp=off" ]]; then
                run_32_gpus_mtp0
            else
                run_32_gpus_mtp
            fi
            ;;
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

            run_single $ctx_num $ctx_tp_size $ctx_ep_size $ctx_enable_attention_dp $gen_num $gen_tp_size $gen_tp_size $gen_batch_size $gen_max_num_tokens $gen_gpu_memory_fraction $gen_mtp_size $gen_eplb_num_slots "$gen_concurrency_list"
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
