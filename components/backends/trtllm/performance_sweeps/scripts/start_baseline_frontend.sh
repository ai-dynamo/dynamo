#! /bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -u
set -e
set -x

num_ctx_servers=$1
num_gen_servers=$2
work_dir=$3
script_dir=$4

python3 ${script_dir}/scripts/gen_baseline_server_config.py \
    --num_ctx_servers ${num_ctx_servers} \
    --num_gen_servers ${num_gen_servers} \
    --work_dir ${work_dir}
echo "server config generated to ${work_dir}/server_config.yaml"

trtllm-serve disaggregated -c ${work_dir}/server_config.yaml -t 7200 -r 7200