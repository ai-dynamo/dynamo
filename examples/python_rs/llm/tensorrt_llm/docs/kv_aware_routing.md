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

This document describes how to use the KV aware routing feature in Dynamo with TensorRT LLM disaggregated serving.
The KV Router is a component that aggregates KV Events from all the workers and maintains a prefix tree of the cached tokens. It makes decisions on which worker to route requests to based on the length of the prefix match and the load on the workers.

## KV Aware Routing with Disaggregated Serving

Follow the instructions in the [README](../README.md) to setup the environment for [disaggregated serving](../README.md#disaggregated-deployment).
All of the steps remain the same except launching the [workers and the router](../README.md#workers).

### 1. Workers

To launch the workers and the router, run the following command:

```bash
cd /workspace/examples/python_rs/llm/tensorrt_llm/
mpirun --allow-run-as-root --oversubscribe -n 5 python3 -m disaggregated.worker --publish-stats --publish-kv-cache-events --engine_args llm_api_config.yaml -c disaggregated/llmapi_disaggregated_configs/single_node_kv_aware_config.yaml 1>disagg_workers.log 2>&1 &
```

Note the extra arguments `--publish-stats` and `--publish-kv-cache-events` to publish the stats and kv cache events from the workers for effective routing.
The config file [single_node_kv_aware_config.yaml](disaggregated/llmapi_disaggregated_configs/single_node_kv_aware_config.yaml) specifies extra configuration for the LLM execution engine to support stats and kv cache events collection. These configurations are:
1. `enable_iter_perf_stats` in `pytorch_backend_config` to enable the iteration performance stats collection.
2. `event_buffer_max_size` in `kv_cache_config` to specify the maximum number of events that can be stored in the buffer.
3. `enable_block_reuse` in `kv_cache_config` to enable the block reuse feature for improved performance.

Note: The configuration also specifies 4 context servers and 1 generation server.

### 2. Router

To launch the router, run the following command:

```bash
cd /workspace/examples/python_rs/llm/tensorrt_llm/
python3 -m disaggregated.kv_router --engine_args llm_api_config.yaml 1>kv_router.log 2>&1 &
```

The router will route the incoming requests to the appropriate context server based on the stats and kv cache events.

### 3. Send Requests

Follow the instructions in the [README](../README.md#send-requests) to send requests to the [HTTP server](../README.md#http-server).

