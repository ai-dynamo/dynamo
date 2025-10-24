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

ISL="${ISL:-8150}"
OSL="${OSL:-1024}"

# For GB200, we use 4 tasks per node.
NTASKS_PER_NODE="${NTASKS_PER_NODE:-4}"

kind='dynamo_agg'

common_args="${kind} ${ISL} ${OSL} ${MODEL_PATH} ${SERVED_MODEL_NAME} ${IMAGE}"

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

# Add job name if set
if [[ -n "${SLURM_JOB_NAME:-}" ]]; then
    slurm_args="${slurm_args} --job-name=${SLURM_JOB_NAME}"
fi


# tep4
max_batch=1024
tp_size=4
ep_size=${tp_size}
enable_attention_dp=false
mtp=0
nodes_count=$(( (tp_size + NTASKS_PER_NODE - 1) / NTASKS_PER_NODE ))

concurrency_list="1 2 4 8 16 32 64 128 256 512 1024 2048"

max_num_tokens=$(( ((mtp+1)*max_batch+ISL+128+63)/64*64 ))
sbatch --nodes=${nodes_count} --ntasks=${tp_size} --ntasks-per-node=${NTASKS_PER_NODE} ${slurm_args} benchmark_agg.slurm ${tp_size} ${ep_size} ${max_batch} ${max_num_tokens} ${enable_attention_dp} "${concurrency_list}" ${mtp} ${common_args}

# dep4
max_batch=1024
tp_size=4
ep_size=${tp_size}
enable_attention_dp=true
mtp=0
nodes_count=$((tp_size/NTASKS_PER_NODE))

concurrency_list="32 64 128 256 512 1024"
max_num_tokens=$(( ((mtp+1)*max_batch+ISL+128+63)/64*64 ))
sbatch --nodes=${nodes_count} --ntasks=${tp_size} --ntasks-per-node=${NTASKS_PER_NODE} ${slurm_args} benchmark_agg.slurm ${tp_size} ${ep_size} ${max_batch} ${max_num_tokens} ${enable_attention_dp} "${concurrency_list}" ${mtp} ${common_args}

concurrency_list="2048 4096"
max_num_tokens=$(( ((mtp+1)*max_batch+ISL+128+63)/64*64 ))
sbatch --nodes=${nodes_count} --ntasks=${tp_size} --ntasks-per-node=${NTASKS_PER_NODE} ${slurm_args} benchmark_agg.slurm ${tp_size} ${ep_size} ${max_batch} ${max_num_tokens} ${enable_attention_dp} "${concurrency_list}" ${mtp} ${common_args}


# tep8
max_batch=1024
tp_size=8
ep_size=${tp_size}
enable_attention_dp=false
mtp=0
nodes_count=$((tp_size/NTASKS_PER_NODE))

concurrency_list="1 2 4 8 16 32 64 128 256 512 1024 2048"
max_num_tokens=$(( ((mtp+1)*max_batch+ISL+128+63)/64*64 ))
sbatch --nodes=${nodes_count} --ntasks=${tp_size} --ntasks-per-node=${NTASKS_PER_NODE} ${slurm_args} benchmark_agg.slurm ${tp_size} ${ep_size} ${max_batch} ${max_num_tokens} ${enable_attention_dp} "${concurrency_list}" ${mtp} ${common_args}

# dep8
max_batch=1024
tp_size=8
ep_size=${tp_size}
enable_attention_dp=true
mtp=0
nodes_count=$((tp_size/NTASKS_PER_NODE))

concurrency_list="32 64 128 256 512 1024"
max_num_tokens=$(( ((mtp+1)*max_batch+ISL+128+63)/64*64 ))
sbatch --nodes=${nodes_count} --ntasks=${tp_size} --ntasks-per-node=${NTASKS_PER_NODE} ${slurm_args} benchmark_agg.slurm ${tp_size} ${ep_size} ${max_batch} ${max_num_tokens} ${enable_attention_dp} "${concurrency_list}" ${mtp} ${common_args}

concurrency_list="2048 4096"
max_num_tokens=$(( ((mtp+1)*max_batch+ISL+128+63)/64*64 ))
sbatch --nodes=${nodes_count} --ntasks=${tp_size} --ntasks-per-node=${NTASKS_PER_NODE} ${slurm_args} benchmark_agg.slurm ${tp_size} ${ep_size} ${max_batch} ${max_num_tokens} ${enable_attention_dp} "${concurrency_list}" ${mtp} ${common_args}

# New: dep8 concurrency greater than 4096 as a separate group
concurrency_list="6144 8192"
max_num_tokens=$(( ((mtp+1)*max_batch+ISL+128+63)/64*64 ))
sbatch --nodes=${nodes_count} --ntasks=${tp_size} --ntasks-per-node=${NTASKS_PER_NODE} ${slurm_args} benchmark_agg.slurm  ${tp_size} ${ep_size} ${max_batch} ${max_num_tokens} ${enable_attention_dp} "${concurrency_list}" ${mtp} ${common_args}