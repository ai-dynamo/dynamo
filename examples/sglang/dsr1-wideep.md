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

# Running DeepSeek-R1 Disaggregated with WideEP

Dynamo supports SGLang's implementation of wide expert parallelism and large scale P/D for DeepSeek-R1! You can read their blog post [here](https://www.nvidia.com/en-us/technologies/ai/deepseek-r1-large-scale-p-d-with-wide-expert-parallelism/) for more details. We provide a Dockerfile for this in `container/Dockerfile.sglang-deepep` and configurations to deploy this at scale. In this example, we will run 1 prefill worker on 4 H100 nodes and 1 decode worker on 9 H100 nodes (104 total GPUs). 

## Instructions

1. Build the SGLang DeepEP container.

```bash
git clone -b v0.4.8.post1 https://github.com/sgl-project/sglang.git
cd sglang/docker
docker build -f Dockerfile -t deepep .
```

You will now have a `deepep:latest` image

2. Build the Dynamo container

```bash
cd $DYNAMO_ROOT
docker build -f container/Dockerfile.sglang-deepep . -t dynamo-deepep --no-cache
```

3. You can run this container on each 8xH100 node using the following command.

> [!IMPORTANT]
> We recommend downloading DeepSeek-R1 and then mounting it to the container. You can find the model [here](https://huggingface.co/deepseek-ai/DeepSeek-R1)

```bash
docker run \
    --gpus all \
    -it \
    --rm \
    --network host \
    --volume /PATH_TO_DSR1_MODEL/:/model/ \
    --shm-size=10G \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --ulimit nofile=65536:65536 \
    --cap-add CAP_SYS_PTRACE \
    --ipc host \
    dynamo-deepep:latest
```

In each container, you should be in the `/sgl-workspace/dynamo/examples/sglang` directory.

4. On the head prefill node, start `nats-server` and `etcd` using the following commands

```bash
nats-server -js &
etcd --listen-client-urls http://0.0.0.0:2379 \
     --advertise-client-urls http://0.0.0.0:2379 \
     --listen-peer-urls http://0.0.0.0:2380 \
     --initial-cluster default=http://HEAD_PREFILL_NODE_IP:2380 &
```

5. On every other node, go ahead and export the `NATS_SERVER` and `ETCD_ENDPOINTS` environment variables

> [!IMPORTANT]
> You will need the IP address of your head prefill node and head decode node for the configuration files

```bash
# run this on every other node
export NATS_SERVER=nats://HEAD_PREFILL_NODE_IP:4222
export ETCD_ENDPOINTS=http://HEAD_PREFILL_NODE_IP:2379
```

6. Configure each configuration file to use the correct `dist-init-addr`, and `node-rank`

Each container contains the configuration file in `configs/dsr1-wideep.yaml`. For our example, we will make the following changes:

On the prefill head node, `vim` into the configs and change the following section of the `SGLangWorker`:

```yaml
SGLangWorker:
    ...
    dist-init-addr: HEAD_PREFILL_NODE_IP
    nnodes: 2
    node-rank: 0
    ...
```

On the other prefill node (since this example has 2 prefill nodes), change the following section of the `SGLangWorker`:

```yaml
SGLangWorker:
    ...
    dist-init-addr: HEAD_PREFILL_NODE_IP
    nnodes: 2
    node-rank: 1
    ...
```

On the decode head node, `vim` into the configs and change the following section of the `SGLangDecodeWorker`:

```yaml
SGLangDecodeWorker:
    ...
    dist-init-addr: HEAD_DECODE_NODE_IP
    nnodes: 4
    node-rank: 0
    ...
```

On the other decode nodes (this example has 4 decode nodes), change the following section of the `SGLangDecodeWorker`:

```yaml
SGLangDecodeWorker:
    ...
    dist-init-addr: HEAD_DECODE_NODE_IP
    nnodes: 4
    # depending on which node this will be 1, 2, and 3
    node-rank: 1
```

7. Start up the workers using the following commands

On prefill head node

```bash
dynamo serve graphs.agg:Frontend -f configs/dsr1-wideep.yaml
```

On prefill child node

```bash
dynamo serve graphs.agg:Frontend -f configs/dsr1-wideep.yaml --service-name SGLangWorker
```

On all decode nodes

```bash
dynamo serve graphs.disagg:Frontend -f configs/dsr1-wideep.yaml --service-name SGLangDecodeWorker
```

8. Run the warmup script to warm up the model

DeepGEMM kernels can sometimes take a while to warm up. Here we provide a small helper script that should help. You can run this as many times as you want before starting inference/benchmarking. You can exec into the head node and run this script standalone - it does not need a container.

```bash
./warmup.sh HEAD_PREFILL_NODE_IP
```

## Benchmarking 

In the official [blog post repro instructions](https://github.com/sgl-project/sglang/issues/6017), SGL uses batch inference to benchmark their prefill and decode workers. They do this by pretokenizing the ShareGPT dataset and then creating a batch of 8192 requests with ISL 4096 and OSL 5 (for prefill stress test) and a batch of 40000 with ISL 2000 and OSL 100 (for decode stress test). If you want to repro these benchmarks, you will need to uncomment the labeled flags in the `configs/dsr1.yaml` file inside of the container.

We currently provide 2 different ways to perform an end to end benchmark which includes using our OpenAI frontend and tokenization. We will continue to add better support for these sorts of large single batch workloads in the future.

1. **GenAI Perf to benchmark end to end performance with 8k ISL 256 OSL**
We've found that 8k ISL 256 OSL provides a good baseline for measuring end to end disaggregated serving performance for DSR1. As WideEP allows for a higher throughput, we provide a script that runs this workload at high concurrencies. DeepGEMM kernels can sometimes take a while to warm up. We provide a short ramping warmup script that can be used.

Example usage:
```bash
# warmup
./utils/bench.sh HEAD_PREFILL_NODE_IP --type warmup
# run benchmark
./utils/bench.sh HEAD_PREFILL_NODE_IP --type e2e
```

2. **GenAI Perf to benchmark completions with custom dataset**
We provide a script that generates a JSONL file of the ShareGPT dataset and then use GenAI Perf to benchmark the prefill and decode workers. We use ShareGPT in order to leverage the pre-existing EPLB distributions provided by the SGLang team. If you don't want to use ShareGPT - you can also use GenAIPerf's synthetic dataset setup But note you will have to use dynamic EPLB configurations or record your own as the `init-expert-location` provided by SGLang is tuned specifically for the ShareGPT dataset at a 4096 ISL and 5 OSL.

Example usage:
```bash
# generate data
python3 utils/generate_bench_data.py --output data.jsonl --num-prompts 8192 --input-len 4096 --output-len 5 --model deepseek-ai/DeepSeek-R1
# run benchmark
./utils/bench.sh HEAD_PREFILL_NODE_IP --type custom_completions
```

3. **Modified sglang `bench_one_batch_server.py` for single batch inference**
We also provide an **experimental** modified version of the `bench_one_batch_server.py` script that can be used to benchmark the prefill and decode workers. This script is modified to handle the maximum NATS message size limit and to chunk the batch into smaller requests. This script is a work in progress and will be updated over time to better support batch inference as we build out this capability in dynamo.

Example usage:
```bash
python3 utils/sgl_bench_one_batch_server.py --model deepseek-ai/DeepSeek-R1 --base-url http://HEAD_PREFILL_NODE_IP:8000 --batch-size 8192 --input-len 4096 --output-len 5 --skip-warmup
```