<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Planner

The planner monitors the state of the system and adjusts workers to ensure that the system runs efficiently.
Currently, the planner can scale the number of vllm workers up and down based on the kv cache load and prefill queue size:

|                     |   | Feature                                                             |
| :------------------ | - | :------------------------------------------------------------------ |
| **Backend**         | ✅ | Local                                                               |
|                     | ✅ | Kubernetes                                                          |
| **LLM Framework**   | ✅ | vLLM                                                                |
|                     | ❌ | TensorRT-LLM                                                        |
|                     | ❌ | SGLang                                                              |
|                     | ❌ | llama.cpp                                                           |
| **Serving Type**    | ✅ | Aggregated                                                          |
|                     | ✅ | Disaggregated                                                       |
| **Planner Actions** | ✅ | Load-based scaling up/down prefill/decode workers                   |
|                     | ✅ | SLA-based scaling up/down prefill/decode workers **<sup>[1]</sup>** |
|                     | ✅ | Adjusting engine knobs                                              |

**<sup>[1]</sup>** Supported with some limitations.

We currently provide two reference planner designs:
1. Load-based planner: [Load-based planner docs](load_planner.md)
2. SLA-based planner: [SLA-based planner docs](sla_planner.md)


## Backends

The planner supports local and kubernetes backends for worker management.

### Local Backend

The local backend uses Circus to control worker processes. A Watcher tracks each `serve_dynamo.py` process.
The planner adds or removes watchers to scale workers.

Note: Circus's `increment` feature doesn't support GPU scheduling variables, so we create separate watchers per process.

#### State Management

The planner maintains state in a JSON file at `~/.dynamo/state/{namespace}.json`. This file:

- Tracks worker names as `{namespace}_{component_name}`.

- Records GPU allocations from the allocator.

- Updates after each planner action.

- Cleans up automatically when the arbiter exits.

Example state file evolution:

```none
# Initial decode worker
{
  "dynamo_VllmWorker": {..., resources={...}}
}

# After adding worker
{
  "dynamo_VllmWorker": {..., resources={...}},
  "dynamo_VllmWorker_1": {..., resources={...}}
}

# After removing worker
{
  "dynamo_VllmWorker": {..., resources={...}}
}

# After removing last worker
{
  "dynamo_VllmWorker": {...}
}
```

> [!Note]
> Start with one replica per worker.
> Multiple initial replicas currently share a single watcher.

### Kubernetes Backend

The Kubernetes backend scales workers by updating DynamoGraphDeployment replica counts.
When scaling needs change, the planner:

1. Updates the deployment's replica count
2. Lets the Kubernetes operator create/remove pods
3. Maintains seamless scaling without manual intervention
