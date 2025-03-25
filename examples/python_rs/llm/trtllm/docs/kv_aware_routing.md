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

# KV Aware Routing

This document describes how to use the KV aware routing feature in Dynamo with TensorRT LLM serving.

The KV Router is a component that aggregates KV Events from all the workers and maintains a prefix tree of the cached tokens. It makes decisions on which worker to route requests to based on the length of the prefix match and the load on the workers.

## Prerequisites

Follow the instructions in the [README](../README.md#Deployment_Prerequisites) to setup the environment. Note that the preprocessor must be launched with `--min-workers` argument to wait for workers start. It also must be launched with the `--routing-strategy prefix` argument to enable the kv routing.

In addition, launch a KV router endpoint:
```bash
cd /workspace/examples/python_rs/llm/tensorrt_llm/
python3 -m kv_router --engine_args llm_api_config.yaml --routing-strategy prefix --kv-block-size 32 --min-workers 2 1>kv_router.log 2>&1 &
```

## KV Aware Routing with Aggregated Serving

### Workers

For KV aware routing to work, we need to launch multiple workers. To do this, you can use the following command for each worker:

```bash
cd /workspace/examples/python_rs/llm/tensorrt_llm/
# For 2 workers
CUDA_VISIBLE_DEVICES=0 mpirun --allow-run-as-root -n 1 --oversubscribe python3 -m agg_worker --publish-stats --publish-kv-cache-events --engine_args llm_api_config.yaml 1>worker1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 mpirun --allow-run-as-root -n 1 --oversubscribe python3 -m agg_worker --publish-stats --publish-kv-cache-events --engine_args llm_api_config.yaml 1>worker2.log 2>&1 &
```

Note the extra arguments `--publish-stats` and `--publish-kv-cache-events` to publish the stats and kv cache events from the workers for effective routing.
The config file [llm_api_config.yaml](../llm_api_config.yaml) specifies extra configuration for the LLM execution engine to support stats and kv cache events collection. These configurations are:
1. `enable_iter_perf_stats` in `pytorch_backend_config` to enable the iteration performance stats collection.
2. `event_buffer_max_size` in `kv_cache_config` to specify the maximum number of events that can be stored in the buffer.
3. `enable_block_reuse` in `kv_cache_config` to enable the block reuse feature for improved performance.


## KV Aware Routing with Disaggregated Serving

Follow the instructions in the [README](../README.md) to setup the environment for [disaggregated serving](../README.md#disaggregated-deployment).
All of the steps remain the same except launching the [workers and the router](../README.md#workers).

### Workers

To launch the workers, run the following command:

```bash
cd /workspace/examples/python_rs/llm/tensorrt_llm/
mpirun --allow-run-as-root --oversubscribe -n 5 python3 -m disagg_worker --publish-stats --publish-kv-cache-events --engine_args llm_api_config.yaml -c llmapi_disaggregated_configs/single_node_kv_aware_config.yaml --remote-prefill 1>disagg_workers.log 2>&1 &
```

The config file [single_node_kv_aware_config.yaml](disaggregated/llmapi_disaggregated_configs/single_node_kv_aware_config.yaml) specifies extra configuration for the LLM execution engine to support stats and kv cache events collection. These configurations are:

Note: The configuration also specifies 4 context servers and 1 generation server.


### Send Requests

The requests must be long (greater than kv_block_size number of tokens) for KV routing to work.

```bash
curl localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "prompt": "Quantum computing is a revolutionary approach to computation that leverages the principles of quantum mechanics to process information. Unlike classical computers, which use bits as the smallest unit of data (represented as 0s and 1s), quantum computers use quantum bits, or qubits. Qubits can exist in multiple states simultaneously due to a property called superposition, allowing quantum computers to perform many calculations at once. Another key feature ",
        "max_tokens": 50,
        "temperature": 0,
        "stream": false
    }'
```