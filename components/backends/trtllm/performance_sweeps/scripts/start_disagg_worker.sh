#! /bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

config_file=$1
model_path=$2
served_model_name=$3
disagg_type=$4

# Read configuration values from the YAML config file
if [ ! -f "${config_file}" ]; then
    echo "Error: Config file ${config_file} not found"
    exit 1
fi

config_tmp_file=$(mktemp)

env_vars=$(python3 -c "
import yaml
config = yaml.safe_load(open('${config_file}'))
    
config_dict = config['${disagg_type}']
for k,v in config_dict['env'].items():
    print(f'export {k}={v}')
for k in config_dict.get('unset_env', []):
    print(f'unset {k}')	
yaml.dump(config_dict['config'], open('${config_tmp_file}', 'w'))
")

echo "Using TRTLLM config:"
cat "${config_tmp_file}"
echo "Setting environment variables from config:"
echo "${env_vars}"

eval "${env_vars}"

trtllm-llmapi-launch python3 -m dynamo.trtllm \
    --model-path ${model_path} \
    --served-model-name ${served_model_name} \
    --disaggregation-mode ${disagg_type} \
    --extra-engine-args ${config_tmp_file}
