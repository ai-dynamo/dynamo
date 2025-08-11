#! /bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

config_file=$1
enable_pdl=$2
ctx_gpus=$3
#work_dir=$4
model_name=$4
model_path=$5
disaggregation_mode=$6
unset UCX_TLS
echo "config_file: ${config_file}, enable_pdl: ${enable_pdl}, ctx_gpus: ${ctx_gpus}, work_dir: ${work_dir}, disaggregation_mode: ${disaggregation_mode}"

export TLLM_LOG_LEVEL=INFO
export TRTLLM_MOE_ENABLE_ALLTOALL_WITHOUT_ALLGATHER=1

if [ "${enable_pdl}" = "true" ]; then
    export TRTLLM_ENABLE_PDL=1
fi

#check if work_dir is provided
if [ -z "${work_dir}" ]; then
    echo "nsys is not enabled, start normal flow"
    trtllm-llmapi-launch python3 -m dynamo.trtllm --model-path $model_path --served-model-name $model_name --disaggregation-mode $disaggregation_mode  --extra-engine-args $config_file
else
    nsys_prefix=""
    nsys_file=${work_dir}/nsys_worker_proc_${SLURM_PROCID}
    export TLLM_PROFILE_RECORD_GC=1
    export TLLM_NVTX_DEBUG=1
    export TRTLLM_ENABLE_DUMMY_ALLREDUCE=1
    if [ ${SLURM_PROCID} -ge ${ctx_gpus} ]; then
        export TLLM_PROFILE_START_STOP=200-250
        nsys_prefix="nsys profile -e \"NSYS_MPI_STORE_TEAMS_PER_RANK=1\" -o ${nsys_file} -f true -t cuda,nvtx,python-gil -c cudaProfilerApi --cuda-graph-trace node --capture-range-end=stop --gpu-metrics-devices=none"
        echo "nsys_prefix: ${nsys_prefix}"
    else
        # export TLLM_PROFILE_START_STOP=10-30
        echo "nsys is not enabled on ctx_gpus"
    fi
    # nsys_prefix="nsys profile -e \"NSYS_MPI_STORE_TEAMS_PER_RANK=1\" -o ${nsys_file} -f true -t cuda,nvtx,python-gil -c cudaProfilerApi --cuda-graph-trace node --capture-range-end=stop --gpu-metrics-devices=all"
    # echo "nsys_prefix: ${nsys_prefix}"
    ${nsys_prefix} trtllm-llmapi-launch python3 -m dynamo.trtllm --model-path $model_path --served-model-name $model_name --disaggregation-mode $disaggregation_mode --extra-engine-args $config_file
fi


