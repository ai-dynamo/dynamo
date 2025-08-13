#! /bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

config_file=$1
enable_pdl=$2
ctx_gpus=$3
model_name=$4
model_path=$5
disaggregation_mode=$6
unset UCX_TLS
echo "config_file: ${config_file}, enable_pdl: ${enable_pdl}, ctx_gpus: ${ctx_gpus}, disaggregation_mode: ${disaggregation_mode}"

export TLLM_LOG_LEVEL=INFO
export TRTLLM_MOE_ENABLE_ALLTOALL_WITHOUT_ALLGATHER=1

if [ "${enable_pdl}" = "true" ]; then
    export TRTLLM_ENABLE_PDL=1
fi

trtllm-llmapi-launch python3 -m dynamo.trtllm --model-path $model_path --served-model-name $model_name --disaggregation-mode $disaggregation_mode  --extra-engine-args $config_file
