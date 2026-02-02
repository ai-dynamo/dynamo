---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---

# Example: Multi-node TRTLLM Workers with Dynamo on Slurm

> **Note:** The scripts referenced in this example (such as `srun_aggregated.sh` and `srun_disaggregated.sh`) can be found in [`examples/basics/multinode/trtllm/`](https://github.com/ai-dynamo/dynamo/tree/main/examples/basics/multinode/trtllm/).

To run a single Dynamo+TRTLLM Worker that spans multiple nodes (ex: TP16),
the set of nodes need to be launched together in the same MPI world, such as
via `mpirun` or `srun`. This is true regardless of whether the worker is
aggregated, prefill-only, or decode-only.

In this document we will demonstrate two examples launching multinode workers
on a slurm cluster with `srun`:
1. Deploying an aggregated nvidia/DeepSeek-R1 model as a multi-node TP16/EP16
   worker across 4 GB200 nodes
2. Deploying a disaggregated nvidia/DeepSeek-R1 model with a multi-node
   TP16/EP16 prefill worker (4 nodes) and a multi-node TP16/EP16 decode
   worker (4 nodes) across a total of 8 GB200 nodes.

NOTE: Some of the scripts used in this example like `start_frontend_services.sh` and
`start_trtllm_worker.sh` should be translatable to other environments like Kubernetes, or
using `mpirun` directly, with relative ease.

## Setup

For simplicity of the example, we will make some assumptions about your slurm cluster:
1. First, we assume you have access to a slurm cluster with multiple GPU nodes
   available. For functional testing, most setups should be fine. For performance
   testing, you should aim to allocate groups of nodes that are performantly
   inter-connected, such as those in an NVL72 setup.
2. Second, we assume this slurm cluster has the [Pyxis](https://github.com/NVIDIA/pyxis)
   SPANK plugin setup. In particular, the `srun_aggregated.sh` script in this
   example will use `srun` arguments like `--container-image`,
   `--container-mounts`, and `--container-env` that are added to `srun` by Pyxis.
   If your cluster supports similar container based plugins, you may be able to
   modify the script to use that instead.
3. Third, we assume you have already built a recent Dynamo+TRTLLM container image as
   described [here](https://github.com/ai-dynamo/dynamo/tree/main/docs/backends/trtllm/README.md#build-container).
   This is the image that can be set to the `IMAGE` environment variable in later steps.
4. Fourth, we assume you pre-allocate a group of nodes using `salloc`. We
   will allocate 8 nodes below as a reference command to have enough capacity
   to run both examples. If you plan to only run the aggregated example, you
   will only need 4 nodes. If you customize the configurations to require a
   different number of nodes, you can adjust the number of allocated nodes
   accordingly. Pre-allocating nodes is technically not a requirement,
   but it makes iterations of testing/experimenting easier.

   Make sure to set your `PARTITION` and `ACCOUNT` according to your slurm cluster setup:
    ```bash
    # Set partition manually based on your slurm cluster's partition names
    PARTITION=""
    # Set account manually if this command doesn't work on your cluster
    ACCOUNT="$(sacctmgr -nP show assoc where user=$(whoami) format=account)"
    salloc \
      --partition="${PARTITION}" \
      --account="${ACCOUNT}" \
      --job-name="${ACCOUNT}-dynamo.trtllm" \
      -t 05:00:00 \
      --nodes 8
    ```
5. Lastly, we will assume you are inside an interactive shell on one of your allocated
   nodes, which may be the default behavior after executing the `salloc` command above
   depending on the cluster setup. If not, then you should SSH into one of the allocated nodes.

### Environment Variable Setup

This example aims to automate as much of the environment setup as possible,
but all slurm clusters and environments are different, and you may need to
dive into the scripts to make modifications based on your specific environment.

Assuming you have already allocated your nodes via `salloc`, and are
inside an interactive shell on one of the allocated nodes, set the
following environment variables based:
```bash
# NOTE: IMAGE must be set manually for now
# To build an iamge, see the steps here:
# https://github.com/ai-dynamo/dynamo/tree/main/docs/backends/trtllm/README.md#build-container
export IMAGE="<dynamo_trtllm_image>"

# MOUNTS are the host:container path pairs that are mounted into the containers
# launched by each `srun` command.
#
# If you want to reference files, such as $MODEL_PATH below, in a
# different location, you can customize MOUNTS or specify additional
# comma-separated mount pairs here.
#
# NOTE: Currently, this example assumes that the local bash scripts and configs
# referenced are mounted into into /mnt inside the container. If you want to
# customize the location of the scripts, make sure to modify `srun_aggregated.sh`
# accordingly for the new locations of `start_frontend_services.sh` and
# `start_trtllm_worker.sh`.
#
# For example, assuming your cluster had a `/lustre` directory on the host, you
# could add that as a mount like so:
#
# export MOUNTS="${PWD}/../../../../:/mnt,/lustre:/lustre"
export MOUNTS="${PWD}/../../../../:/mnt"

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
export MODEL_PATH="nvidia/DeepSeek-R1-FP4"

# The name the model will be served/queried under, matching what's
# returned by the /v1/models endpoint.
#
# By default this is inferred from MODEL_PATH, but when using locally downloaded
# model weights, it can be nice to have explicit control over the name.
export SERVED_MODEL_NAME="nvidia/DeepSeek-R1-FP4"
```

## Aggregated WideEP

Assuming you have at least 4 nodes allocated following the setup steps above,
follow these steps below to launch an **aggregated** deployment across 4 nodes:

```bash
# Default set in srun_aggregated.sh, but can customize here.
# export ENGINE_CONFIG="/mnt/examples/backends/trtllm/engine_configs/deepseek-r1/agg/wide_ep/wide_ep_agg.yaml"

# Customize NUM_NODES to match the desired parallelism in ENGINE_CONFIG
# The product of NUM_NODES*NUM_GPUS_PER_NODE should match the number of
# total GPUs necessary to satisfy the requested parallelism. For example,
# 4 nodes x 4 gpus/node = 16 gpus total for TP16/EP16.
# export NUM_NODES=4

# GB200 nodes have 4 gpus per node, but for other types of nodes you can configure this.
# export NUM_GPUS_PER_NODE=4

# Launches:
# - frontend + etcd/nats on current (head) node
# - one large aggregated trtllm worker across multiple nodes via MPI tasks
./srun_aggregated.sh
```

## Disaggregated WideEP

Assuming you have at least 8 nodes allocated (4 for prefill, 4 for decode)
following the setup above, follow these steps below to launch a **disaggregated**
deployment across 8 nodes:

> [!TIP]
> Make sure you have a fresh environment and don't still have the aggregated
> example above still deployed on the same set of nodes.

```bash
# Defaults set in srun_disaggregated.sh, but can customize here.
# export PREFILL_ENGINE_CONFIG="/mnt/examples/backends/trtllm/engine_configs/deepseek-r1/disagg/wide_ep/wide_ep_prefill.yaml"
# export DECODE_ENGINE_CONFIG="/mnt/examples/backends/trtllm/engine_configs/deepseek-r1/disagg/wide_ep/wide_ep_decode.yaml"

# Customize NUM_PREFILL_NODES to match the desired parallelism in PREFILL_ENGINE_CONFIG
# Customize NUM_DECODE_NODES to match the desired parallelism in DECODE_ENGINE_CONFIG
# The products of NUM_PREFILL_NODES*NUM_GPUS_PER_NODE and
# NUM_DECODE_NODES*NUM_GPUS_PER_NODE should match the respective number of
# GPUs necessary to satisfy the requested parallelism in each config.
# export NUM_PREFILL_NODES=4
# export NUM_DECODE_NODES=4

# GB200 nodes have 4 gpus per node, but for other types of nodes you can configure this.
# export NUM_GPUS_PER_NODE=4

# Launches:
# - frontend + etcd/nats on current (head) node.
# - one large prefill trtllm worker across multiple nodes via MPI tasks
# - one large decode trtllm worker across multiple nodes via MPI tasks
./srun_disaggregated.sh
```

> [!TIP]
> To launch multiple replicas of the configured prefill/decode workers, you can set
> NUM_PREFILL_WORKERS and NUM_DECODE_WORKERS respectively (default: 1).

## Understanding the Output

1. The `srun_aggregated.sh` launches two `srun` jobs. The first launches
   etcd, NATS, and the OpenAI frontend on the head node only
   called "node1" in the example output below. The second launches
   a single TP16 Dynamo+TRTLLM worker spread across 4 nodes, each node
   using 4 GPUs each.
    ```
    # Frontend/etcd/nats services
    srun: launching StepId=453374.17 on host node1, 1 tasks: 0
    ...
    # TP16 TRTLLM worker split across 4 nodes with 4 gpus each
    srun: launching StepId=453374.18 on host node1, 4 tasks: [0-3]
    srun: launching StepId=453374.18 on host node2, 4 tasks: [4-7]
    srun: launching StepId=453374.18 on host node3, 4 tasks: [8-11]
    srun: launching StepId=453374.18 on host node4, 4 tasks: [12-15]
   ```
2. The OpenAI frontend will listen for and dynamically discover workers as
   they register themselves with Dynamo's distributed runtime:
   ```
   0: 2025-06-13T02:36:48.160Z  INFO dynamo_run::input::http: Watching for remote model at models
   0: 2025-06-13T02:36:48.161Z  INFO dynamo_llm::http::service::service_v2: Starting HTTP service on: 0.0.0.0:8000 address="0.0.0.0:8000"
   ```
3. The TRTLLM worker will consist of N (N=16 for TP16) MPI ranks, 1 rank on each
   GPU on each node, which will each output their progress while loading the model.
   You can see each rank's output prefixed with the rank at the start of each log line
   until the model succesfully finishes loading:
    ```
     8: rank8 run mgmn worker node with mpi_world_size: 16 ...
    10: rank10 run mgmn worker node with mpi_world_size: 16 ...
     9: rank9 run mgmn worker node with mpi_world_size: 16 ...
    11: rank11 run mgmn worker node with mpi_world_size: 16 ...
    ...
    15: Model init total -- 55.42s
    11: Model init total -- 55.91s
    12: Model init total -- 55.24s
    ```
4. After the model fully finishes loading on all ranks, the worker will register itself,
   and the OpenAI frontend will detect it, signaled by this output:
    ```
    0: 2025-06-13T02:46:35.040Z  INFO dynamo_llm::discovery::watcher: added model model_name="nvidia/DeepSeek-R1-FP4"
    ```
5. At this point, with the worker fully initialized and detected by the frontend,
   it is now ready for inference.
6. For `srun_disaggregated.sh`, it follows a very similar flow, but instead launches
   three srun jobs instead of two. One for frontend, one for prefill worker,
   and one for decode worker.

## Example Request

To verify the deployed model is working, send a `curl` request:
```bash
# NOTE: $HOST assumes running on head node, but can be changed to $HEAD_NODE_IP instead.
HOST=localhost
PORT=8000
# "model" here should match the model name returned by the /v1/models endpoint
curl -w "%{http_code}" ${HOST}:${PORT}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
  "model": "'${SERVED_MODEL_NAME}'",
  "messages": [
  {
    "role": "user",
    "content": "Tell me a story as if we were playing dungeons and dragons."
  }
  ],
  "stream": true,
  "max_tokens": 30
}'
```

## Cleanup

To cleanup background `srun` processes launched by `srun_aggregated.sh` or
`srun_disaggregated.sh`, you can run:
```bash
pkill srun
```

## Known Issues

- This example has only been tested on a 4xGB200 node setup with 16 GPUs using
  FP4 weights. In theory, the example should work on alternative setups such as
  H100 nodes with FP8 weights, but this hasn't been tested yet.
- WideEP configs in this directory are still being tested. A WideEP specific
  example with documentation will be added once ready.
- There are known issues where WideEP workers may not cleanly shut down:
    - This may lead to leftover shared memory files in `/dev/shm/moe_*`. For
      now, you must manually clean these up before deploying again on the
      same set of nodes.
    - Similarly, there may be GPU memory left in-use after killing the `srun`
      jobs. After cleaning up any leftover shared memory files as described
      above, the GPU memory may slowly come back. You can run `watch nvidia-smi`
      to check on this behavior. If you don't free the GPU memory before the
      next deployment, you may get a CUDA OOM error while loading the model.
    - There is mention of this issue in the relevant TRT-LLM blog
      [here](https://github.com/NVIDIA/TensorRT-LLM/blob/6021a439ab9c29f4c46f721eeb59f6b992c425ea/docs/source/blogs/tech_blog/blog4_Scaling_Expert_Parallelism_in_TensorRT-LLM.md#miscellaneous).
