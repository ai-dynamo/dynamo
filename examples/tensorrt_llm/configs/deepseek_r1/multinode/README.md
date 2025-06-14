<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Example: Multi-node TRTLLM Workers with Dynamo on Slurm

To run a single Dynamo+TRTLLM Worker that spans multiple nodes (ex: TP16),
the set of nodes need to be launched together in the same MPI world, such as
via `mpirun` or `srun`. This is true regardless of whether the worker is
aggregated, prefill-only, or decode-only.

In this document we will demonstrate an example of launching a multi-node TP16/EP16
aggregated worker on a slurm cluster with `srun`.

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
   SPANK plugin setup. In particular, the `srun_script.sh` script in this
   example will use `srun` arguments like `--container-image`,
   `--container-mounts`, and `--container-env` that are added to `srun` by Pyxis.
   If your cluster supports similar container based plugins, you may be able to
   modify the script to use that instead.
3. Third, we assume you have already built a recent Dynamo+TRTLLM container image as
   described [here](https://github.com/ai-dynamo/dynamo/tree/main/examples/tensorrt_llm#build-docker).
   This is the image that can be set to the `IMAGE` environment variable in later steps.
4. Fourth, we assume you pre-allocate a group of nodes using `salloc`. We
   will allocate 4 nodes below as a reference command. This is technically not
   a requirement, but makes iterations of testing/experimenting easier when
   you have a reserved set of nodes for a period of time. Make sure to set your
   `PARTITION` and `ACCOUNT` according to your slurm cluster setup:
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
      --nodes 4
    ```
5. Lastly, we will assume you are inside an interactive shell on one of your allocated
   nodes, which should be the default behavior after executing the `salloc` command above.
   If not, then you should SSH into one of the allocated nodes.

## Launching Slurm Jobs

This example aims to automate as much of the environment setup as possible,
but all slurm clusters and environments are different, and you may need to
dive into the scripts to make modifications based on your specific environment.

Assuming you have already allocated at least 4 nodes via `salloc`, and are
inside an interactive shell on one of the allocated nodes:
```bash
# NOTE: IMAGE must be set manually for now
# To build an iamge, see the steps here:
# https://github.com/ai-dynamo/dynamo/tree/main/examples/tensorrt_llm#build-docker
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
# customize the location of the scripts, make sure to modify `srun_script.sh`
# accordingly for the new locations of `start_frontend_services.sh` and
# `start_trtllm_worker.sh`.
#
# For example, assuming your cluster had a `/lustre` directory on the host, you
# could add that as a mount like so:
#
# export MOUNTS="${PWD}:/mnt,/lustre:/lustre"
export MOUNTS="${PWD}:/mnt"

# NOTE: In general, Deepseek R1 is very large, so it is recommended to
# pre-download the model weights and save them in some shared location,
# NFS storage, HF_CACHE, etc. and modify the `--model-path` below
# to reuse the pre-downloaded weights instead.
#
# On Blackwell systems (ex: GB200), it is recommended to use the FP4 weights:
# https://huggingface.co/nvidia/DeepSeek-R1-FP4
#
# On Hopper systems, FP4 isn't supported so you'll need to use the default weights:
# https://huggingface.co/deepseek-ai/DeepSeek-R1
export MODEL_PATH="nvidia/DeepSeek-R1-FP4"

# NOTE: This path assumes you have mounted the config file into /mnt inside
# the container. See the MOUNTS variable in srun_script.sh
export ENGINE_CONFIG="/mnt/agg_DEP16_dsr1.yaml"

# Launches frontend + etcd/nats on current (head) node.
# Launches one large trtllm worker across multiple nodes via MPI tasks.
./srun_script.sh
```

## Understanding the Output

1. The `srun_script.sh` launches two `srun` jobs. The first launches
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
    0: 2025-06-13T02:46:35.040Z  INFO dynamo_llm::discovery::watcher: added model model_name="Deepseek-R1-FP4"
    ```
5. At this point, with the worker fully initialized and detected by the frontend,
   it is now ready for inference.


## Example Request

To verify the deployed model is working, send a `curl` request:
```bash
# NOTE: $HOST assumes running on head node, but can be changed to $HEAD_NODE_IP instead.
HOST=localhost
PORT=8000
MODEL=Deepseek-R1-FP4
curl -w "%{http_code}" ${HOST}:${PORT}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
  "model": "'${MODEL}'",
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

To cleanup background `srun` processes launched by `srun_script.sh`, you can run:
```bash
pkill srun
```

## Known Issues

- This example has only been tested on a 4xGB200 node setup with 16 GPUs using
  FP4 weights. In theory, the example should work on alternative setups such as
  H100 nodes with FP8 weights, but this hasn't been tested yet.
- This example only tests an aggregated model setup for now. A disaggregated
  serving example will be added in the near future.
