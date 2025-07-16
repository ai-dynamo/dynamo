#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This is one of the only variables that must be set currently. The image must contain
# genai-perf installation.
IMAGE_BENCH="${IMAGE_BENCH:-""}"

# Set to mount current host directory to /mnt inside the container as an example,
# but you may freely customize the mounts based on your cluster. A common practice
# is to mount paths to NFS storage for common scripts, model weights, etc.
# NOTE: This can be a comma separated list of multiple mounts as well.
DEFAULT_MOUNT="${PWD}/../:/mnt,${PWD}/../../../benchmarks:/benchmarks"
MOUNTS="${MOUNTS:-${DEFAULT_MOUNT}}"

# Set the default values for the benchmark parameters
export MODE="${MODE:-disaggregated}"
export KIND="${KIND:-dynamo_trtllm_wideep}"
export TP="${TP:-1}"
export DP="${DP:-1}"
export PREFILL_TP="${PREFILL_TP:-4}"
export PREFILL_DP="${PREFILL_DP:-7}"
export DECODE_TP="${DECODE_TP:-32}"
export DECODE_DP="${DECODE_DP:-1}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-nvidia/DeepSeek-R1-FP4}"
export INPUT_SEQ_LEN="${INPUT_SEQ_LEN:-8192}"
export OUTPUT_SEQ_LEN="${OUTPUT_SEQ_LEN:-1024}"
export CONCURRENCY="${CONCURRENCY:-1024}"
export ARTIFACTS_ROOT_DIR="${ARTIFACTS_ROOT_DIR:-/mnt/artifacts_root}"

# Automate settings of certain variables for convenience, but you are free
# to manually set these for more control as well.
ACCOUNT="$(sacctmgr -nP show assoc where user=$(whoami) format=account)"
export HEAD_NODE="${SLURMD_NODENAME}"
export HEAD_NODE_IP="$(hostname -i)"

if [[ -z ${IMAGE} ]]; then
  echo "ERROR: You need to set the IMAGE environment variable to the " \
       "Dynamo+TRTLLM docker image or .sqsh file from 'enroot import' " \
       "See how to build one from source here: " \
       "https://github.com/ai-dynamo/dynamo/tree/main/examples/tensorrt_llm#build-docker"
  exit 1
fi

# NOTE: Output streamed to stdout for ease of understanding the example, but
# in practice you would probably set `srun --output ... --error ...` to pipe
# the stdout/stderr to files.
echo "Launching Benchmarking Service"
srun \
  --overlap \
  --container-image "${IMAGE_BENCH}" \
  --container-mounts "${MOUNTS}" \
  --verbose \
  --label \
  -A "${ACCOUNT}" \
  -J "${ACCOUNT}-dynamo.trtllm" \
  --nodelist "${HEAD_NODE}" \
  --nodes 1 \
  --jobid "${SLURM_JOB_ID}" \
  --container-env HEAD_NODE_IP,MODE,KIND,TP,DP,PREFILL_TP,PREFILL_DP,DECODE_TP,DECODE_DP,SERVED_MODEL_NAME,INPUT_SEQ_LEN,OUTPUT_SEQ_LEN,CONCURRENCY,ARTIFACTS_ROOT_DIR \
  /mnt/multinode/start_benchmark.sh
