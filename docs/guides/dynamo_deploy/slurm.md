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

# Deploying Inference Graphs to SLURM

This guide explains how to deploy Dynamo inference graphs on SLURM clusters, using sbatch. A fully working example of a vLLM serving is available [here](../../examples/deployments/slurm/run.sbatch). It should be easy to modify it to serve another type of workload.

## Prerequisites

1. **SLURM+pyxis Cluster**: Access to a SLURM-managed cluster with GPU nodes and the [pyxis plugin](https://github.com/nvidia/pyxis) installed
2. **Dynamo Container**: A containerized Dynamo environment (`.sqsh` file created with [enroot](https://github.com/nvidia/enroot))
3. **HuggingFace Token**: For downloading models (set `HF_TOKEN` environment variable)

## How to run


First, modify the `#SBATCH --output` line to redirect the output to your desired location (refer to sbatch documentation for more). And change the `DYNAMO_CONTAINER` to be the path of the downloaded squash container.

The example sbatch script spawns three different `srun` command, one for the `Frontend`, one for all the `VllmPrefillWorker` instances and one for all the `VllmDecodeWorker` instances. This is specific to this vLLM example, for other configurations simply adapt the `srun` commands so that each node run the right script.

For vLLM or similar framework, modify these variables to use different models or configurations:

```bash
# Example for vLLM v0.3.2
GRAPH_DIR="examples/vllm_v1"
CONFIG="configs/disagg.yaml"
FRONTEND_GRAPH="graphs.agg:Frontend"
PREFILL_COMPONENT="components.worker:VllmPrefillWorker"
DECODE_COMPONENT="components.worker:VllmDecodeWorker"
```


| Backend | Prefill Component | Decode Component |
|---------|------------------|------------------|
| vLLM | `VllmPrefillWorker` | `VllmDecodeWorker` |
| SGLang | `SGLangPrefillWorker` | `SGLangDecodeWorker` |
| TensorRT-LLM | `TrtllmPrefillWorker` | `TrtllmDecodeWorker` |


Modify these variables (or use the associated parameters) to modify the scale:
```bash
NUM_PREFILL_WORKERS=$(( SLURM_JOB_NUM_NODES / 2 ))
GPUS_PER_PREFILL_WORKER=8

NUM_DECODE_WORKERS=$(( SLURM_JOB_NUM_NODES / 2 ))
GPUS_PER_DECODE_WORKER=8
```

Additional parameters or variables can be added as desired.


Run with the `sbatch` utility:
```bash
# Submit the job with default settings
sbatch <sbatch_parameters> run.sbatch

# Or customize the deployment
sbatch run.sbatch --num-prefill-workers 2 --gpus-per-prefill 4 --container /path/to/dynamo.sqsh
```

Refer to [sbatch docs](https://slurm.schedmd.com/sbatch.html) for details.

## How it works

This section explains how this specific example work, so that it will be easier to customize.

### Node Assignment

The script automatically assigns nodes based on your SLURM allocation:

1. **Frontend Node**: The node where the script runs (handles HTTP + decode), it is selected to be the first node of the list
2. **Decode Workers**: First `(NUM_DECODE_WORKERS-1)` after the front-end one
3. **Prefill Workers**: Remaining nodes

### Communication

- **NATS**: Message broker for inter-node communication (`nats://frontend:4222`)
- **ETCD**: Service discovery and coordination (`http://frontend:2379`)

## Deployment Process

### 1. Environment Setup

First the script decides which nodes are allocated to Frontend, Decode and Prefill:
```bash
DYNAMO_FRONTEND=$(hostname)
SLURM_NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST))
not_frontend_nodes=() 
for node in "${SLURM_NODES[@]}"; do
    if [[ "$node" != "$DYNAMO_FRONTEND" ]]; then
        not_frontend_nodes+=("$node")
    fi
done
DECODE_NODES=()
PREFILL_NODES=()
```

```bash
# Create HuggingFace cache on all nodes
srun --job-name="prepare_huggingface" -N $SLURM_JOB_NUM_NODES --ntasks-per-node=1 \
     mkdir -p ${XDG_CACHE_HOME}/huggingface

# Load container on all nodes
srun --job-name="load_container" -N $SLURM_JOB_NUM_NODES --ntasks-per-node=1 \
     --container-image=${DYNAMO_CONTAINER} --container-name=dynamo hostname
```

### 2. Service Launch

The script launches three types of services:

#### Frontend Service
```bash
srun -N 1 --gpus-per-node=$SLURM_GPUS_PER_NODE \
     --container-image=${DYNAMO_CONTAINER} \
     -w ${DYNAMO_FRONTEND} \
     dynamo serve ${FRONTEND_GRAPH} -f ${CONFIG}
```

#### Prefill Workers
```bash
srun -N $NUM_PREFILL_WORKERS --gpus-per-node=$SLURM_GPUS_PER_NODE \
     --container-image=${DYNAMO_CONTAINER} \
     -w ${PREFILL_NODES_STR} \
     dynamo serve ${PREFILL_COMPONENT} -f ${CONFIG}
```

#### Decode Workers
```bash
srun -N $((NUM_DECODE_WORKERS-1)) --gpus-per-node=$SLURM_GPUS_PER_NODE \
     --container-image=${DYNAMO_CONTAINER} \
     -w ${DECODE_NODES_STR} \
     dynamo serve ${DECODE_COMPONENT} -f ${CONFIG}
```