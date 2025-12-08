#! /bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

config_file=$1
ctx_gpus=$2
model_name=$3
model_path=$4
disaggregation_mode=$5
is_dep=$6

unset UCX_TLS
echo "config_file: ${config_file}, ctx_gpus: ${ctx_gpus}, disaggregation_mode: ${disaggregation_mode}, is_dep: ${is_dep}"

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
export TRTLLM_ENABLE_PDL=1

if [ "$is_dep" = "true" ]; then
    echo "Using DEP. Setting env vars."
    export TRTLLM_MOE_ALLTOALL_BACKEND="mnnvlthroughput"
    export TRTLLM_FORCE_ALLTOALL_METHOD="MNNVL"
    export TRTLLM_MOE_A2A_WORKSPACE_MB="2048"
fi 

if [[ "${model_path,,}" != *r1* ]]; then
    echo "Inferred gpt-oss style model. Setting OVERRIDE_QUANT_ALGO to W4A8_MXFP4_MXFP8"
    export OVERRIDE_QUANT_ALGO=W4A8_MXFP4_MXFP8
fi

trtllm-llmapi-launch python3 -m dynamo.trtllm \
    --model-path ${model_path} \
    --served-model-name ${model_name} \
    --max-num-tokens ${max_num_tokens} \
    --max-batch-size ${max_batch_size} \
    --max-seq-len ${max_seq_len} \
    --disaggregation-mode ${disaggregation_mode} \
    --extra-engine-args ${config_file}
