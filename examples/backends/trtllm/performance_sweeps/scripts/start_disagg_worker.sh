#! /bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

config_file=$1
enable_pdl=$2
ctx_gpus=$3
model_name=$4
model_path=$5
disaggregation_mode=$6
unset UCX_TLS
echo "config_file: ${config_file}, enable_pdl: ${enable_pdl}, ctx_gpus: ${ctx_gpus}, disaggregation_mode: ${disaggregation_mode}"

# Read configuration values from the YAML config file
if [ ! -f "${config_file}" ]; then
    echo "Error: Config file ${config_file} not found"
    exit 1
fi

# Note: TensorRT-LLM config file is a YAML file may not respect the max_num_tokens,
# max_batch_size, max_seq_len when provided as yaml. Providing these values via
# command line to make sure they are respected.
max_num_tokens=$(grep "^max_num_tokens:" "${config_file}" | sed 's/.*: *//')
max_batch_size=$(grep "^max_batch_size:" "${config_file}" | sed 's/.*: *//')
max_seq_len=$(grep "^max_seq_len:" "${config_file}" | sed 's/.*: *//')


# Validate that we got the values
if [ -z "${max_num_tokens}" ] || [ -z "${max_batch_size}" ] || [ -z "${max_seq_len}" ]; then
    echo "Error: Failed to read required configuration values from ${config_file}"
    echo "max_num_tokens: ${max_num_tokens}"
    echo "max_batch_size: ${max_batch_size}"
    echo "max_seq_len: ${max_seq_len}"
    exit 1
fi

echo "Configuration loaded from ${config_file}:"
echo "  max_num_tokens: ${max_num_tokens}"
echo "  max_batch_size: ${max_batch_size}"
echo "  max_seq_len: ${max_seq_len}"

export TLLM_LOG_LEVEL=INFO
export TRTLLM_MOE_ENABLE_ALLTOALL_WITHOUT_ALLGATHER=1

if [ "${enable_pdl}" = "true" ]; then
    export TRTLLM_ENABLE_PDL=1
fi

trtllm-llmapi-launch python3 -m dynamo.trtllm \
    --model-path ${model_path} \
    --served-model-name ${model_name} \
    --max-num-tokens ${max_num_tokens} \
    --max-batch-size ${max_batch_size} \
    --max-seq-len ${max_seq_len} \
    --disaggregation-mode ${disaggregation_mode} \
    --extra-engine-args ${config_file}
