<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# TensorRT-LLM Benchmark Scripts for DeepSeek R1 model

This directory contains scripts for benchmarking TensorRT-LLM performance with Dynamo using SLURM job scheduler.

## ⚠️ DISCLAIMER
**These scripts are currently not QA'ed and are provided for demonstration purposes only.**

Please note that:

- These scripts have not undergone formal quality assurance testing
- These scripts were tested on GB200 systems. To run all configurations, you will need at least 16 nodes, with each node equipped with 4 GPUs.
- They are intended for demonstration and educational purposes
- Use at your own risk in production environments
- Always review and test scripts thoroughly before running in your specific environment
- In disaggregated mode, using `--exclusive` flag to launch worker processes can impact runtime performance. Hence, these scripts specify nodelist explicitly in srun call.
- We are actively working on refining the configuration sweeps.

## Scripts Overview

### Core Scripts

1. `submit_disagg.sh` - Main entry point for submitting benchmark jobs for disaggregated configurations. This includes WideEP optimization for DEP>=16.
2. `submit_agg.sh` - Main entry point for submitting benchmark jobs for aggregated configurations.
3. `post_process.py` - Scan the aiperf results to produce a json with entries to each config point.
4. `plot_performance_comparison.py` - Takes the json result file for disaggregated and/or aggregated configuration sweeps and plots a pareto line for better visualization.

For more finer grained details on how to launch TRTLLM backend workers with DeepSeek R1 on GB200 slurm, please refer [multinode-examples.md](../../../../docs/backends/trtllm/multinode/multinode-examples.md). This guide shares similar assumption to the multinode examples guide.

## Usage

### Prerequisites

Before running the scripts, ensure you have:
1. Access to a SLURM cluster
2. Container image of Dynamo with TensorRT-LLM built using instructions from [here](https://github.com/ai-dynamo/dynamo/tree/main/docs/backends/trtllm/README.md#build-container).
3. Model files accessible on the cluster
4. Required environment variables set

### Setup

Within the login node of the cluster, set the following variables

```bash
# Set partition manually based on your slurm cluster's partition names
export SLURM_PARTITION=""

# Set account manually if this command doesn't work on your cluster
export SLURM_ACCOUNT="$(sacctmgr -nP show assoc where user=$(whoami) format=account)"

# Set a job name for your benchmarking runs
export SLURM_JOB_NAME=""

# NOTE: IMAGE must be set manually for now
# To build an iamge, see the steps here:
# https://github.com/ai-dynamo/dynamo/tree/main/docs/backends/trtllm/README.md#build-container
export IMAGE="<dynamo_trtllm_image>"

# NOTE: In general, Deepseek R1 is very large, so it is recommended to
# pre-download the model weights and save them in some shared location,
# NFS storage, HF_HOME, etc. and modify the `--model-path` below
# to reuse the pre-downloaded weights instead.
#
# On Blackwell systems (ex: GB200), it is recommended to use the FP4 weights:
# https://huggingface.co/nvidia/DeepSeek-R1-FP4
#
# On Hopper systems, FP4 isn't supported so you'll need to use the default weights:
# https://huggingface.co/deepseek-ai/DeepSeek-R1
export MODEL_PATH="<path_to_model_weights>"

# The name the model will be served/queried under, matching what's
# returned by the /v1/models endpoint.
#
# By default this is inferred from MODEL_PATH, but when using locally downloaded
# model weights, it can be nice to have explicit control over the name.
export SERVED_MODEL_NAME="nvidia/DeepSeek-R1-FP4"
```

## Launching benchmarking sweeps for different configurations

### Aggregated

```bash
# Queues the SLURM jobs for aggregated configurations for DeepSeek R1.
./submit_agg.sh
```

### Disaggregated (Includes WideEP) - MTP off

```bash
# Queues the SLURM jobs for disaggregated configurations for DeepSeek R1 without MTP
./submit_disagg.sh mtp=off all
```

### Disaggregated (Includes WideEP) - MTP on

```bash
# Queues the SLURM jobs for disaggregated configurations for DeepSeek R1 with MTP
./submit.sh mtp=on all
```

## Post-Processing Results

The above jobs use aiperf tool to benchmark each configuration point across different concurrency values. These get stored in `dynamo_disagg-bm-8150-1024/<config-setup>/aiperf_artifacts` and `dynamo_agg-bm-8150-1024/<config-setup>/aiperf_artifacts` for disaggregated and aggregated respectively.

After your benchmarking jobs have completed, you can use the `post_process.py` script to aggregate and summarize the results from the generated aiperf_artifacts.

To run the post-processing script, use:

### Aggregated

```bash
python3 post_process.py dynamo_agg-bm-8150-1024 --output-file agg_result.json
```

### Disaggregated

```bash
python3 post_process.py dynamo_disagg-bm-8150-1024 --output-file disagg_result.json
```

## Ploting Performance

You can now use the `plot_performance_comparison.py` like below to observe the performance.

```bash
python3 plot_performance_comparison.py dynamo_agg-bm-8150-1024/agg_result.json dynamo_disagg-bm-8150-1024/disagg_result.json -o performance_plot.png
```

This script will produce a scatter plot of all the configuration points with each concurrency on a Output Throughput per GPU vs Output Throughput per User. It will also include the roofline pareto line for both aggregated and disaggregated setups.

Refer to [Beyond the Buzz: A Pragmatic Take on Inference Disaggregation](https://arxiv.org/html/2506.05508v1) to learn how to interpret these plots.

## Known Issues

- Some jobs may time out if aiperf requires more time to complete all concurrency levels.
- Workers may encounter out-of-memory (OOM) errors during inference, especially with larger configurations.
- Configurations affected by these issues will result in missing data points on the performance plot.
