<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

The planner is a component that monitors the state of the system and makes adjustments to workers to ensure that the system is running efficiently. Currently, planner can scale up and down the number of vllm workers based on the kv cache load and prefill queue size:
* Backend:
  * local ✅
  * kubernetes ✅
* LLM framework:
  * vllm ✅
  * tensorrt-llm ❌
  * SGLang ❌
  * llama.cpp ❌
* Serving type:
  * Aggregated ✅
  * Disaggregated ✅
* Planner actions:
  * Load-based scaling up/down prefill/decode workers ✅
  * SLA-based scaling up/down prefill/decode workers ❌
  * Adjusting engine knobs ❌

## Load-based Scaling Up/Down Prefill/Decode Workers

To adjust the number of prefill/decode workers, planner monitors the following metrics:
* Prefill worker: planner monitors the number of requests pending in the prefill queue to estimate the prefill workload.
* Decode/aggregated worker: planner monitors the average KV cache utilization rate to estimate the decode/aggregated workload.

Every `metric-pulling-interval`, planner will gather the aforementioned metrics. Every `adjustment-interval`, planner compares the aggregated metrics in this interval with pre-set thresholds and decide to scale up/down prefill/decode workers. To avoid over-compensation, planner only changes the number of workers by 1 in one adjustment interval. In addition, when the number of workers is being adjusted, the planner will block the metric pulling and adjustment.

To scale up a prefill/decode worker, planner just need to launch the worker in the correct namespace. The auto-discovery mechanism will pick up the workers and add them to the routers. To scale down a prefill worker, planner send a SIGTERM signal to the prefill worker. The prefill worker store the signal and exit when it finishes the current request pulled from the prefill queue. This ensures that no remote prefill request is dropped. To scale down a decode worker, currently, planner revoke the etcd lease of the decode worker. When the etcd lease is revoked, the corresponding decode worker will be immediately removed from the router and will not get any new requests. The decode worker will then finish all the current requests in their original stream and exit gracefully.

There are two additional rules set by planner to prevent over-compensation:
1. After a new decode worker is added, since it needs time to populate the kv cache, planner will not scale down the number of decode workers in the next `NEW_DECODE_WORKER_GRACE_PERIOD=3` adjustment intervals.
1. We do not scale up prefill worker if the prefill queue size is estimated to reduce below the `--prefill-queue-scale-up-threshold` within the next `NEW_PREFILL_WORKER_QUEUE_BUFFER_PERIOD=3` adjustment intervals following the trend observed in the current adjustment interval.

## Usage
The planner is started automatically as part of Dynamo pipelines when running `dynamo serve`. You can configure the planner just as you would any other component in your pipeline either via YAML configuration or through CLI arguments.

Usage:
```bash
# Configure the planner through YAML configuration
dynamo serve graphs.disagg:Frontend -f disagg.yaml

# disagg.yaml
# ...
# Planner:
#   environment: local
#   no-operation: false
#   log-dir: log/planner

# Configure the planner through CLI arguments
dynamo serve graphs.disagg:Frontend -f disagg.yaml --Planner.environment=local --Planner.no-operation=false --Planner.log-dir=log/planner
```

The planner accepts the following configuration options:
* `namespace` (str, default: "dynamo"): Namespace planner will look at
* `served-model-name` (str, default: "vllm"): Model name that is being served`
* `no-operation` (bool, default: false): Do not make any adjustments, just observe the metrics and log to tensorboard.
* `log-dir` (str, default: None): Tensorboard logging directory
* `adjustment-interval` (int, default: 30): Interval in seconds between scaling adjustments
* `metric-pulling-interval` (int, default: 1): Interval in seconds between metric pulls
* `max-gpu-budget` (int, default: 8): Maximum number of GPUs to use, planner will not scale up more than this number of GPUs for prefill plus decode workers
* `min-gpu-budget` (int, default: 1): Minimum number of GPUs to use, planner will not scale down below this number of GPUs for prefill or decode workers
* `decode-kv-scale-up-threshold` (float, default: 0.9): KV cache utilization threshold to scale up decode workers
* `decode-kv-scale-down-threshold` (float, default: 0.5): KV cache utilization threshold to scale down decode workers
* `prefill-queue-scale-up-threshold` (float, default: 0.5): Queue utilization threshold to scale up prefill workers
* `prefill-queue-scale-down-threshold` (float, default: 0.2): Queue utilization threshold to scale down prefill workers
* `decode-engine-num-gpu` (int, default: 1): Number of GPUs per decode engine
* `prefill-engine-num-gpu` (int, default: 1): Number of GPUs per prefill engine

Alternatively, you can run the planner as a standalone python process. The configuration options above can be directly passed in as CLI arguments.
```bash
PYTHONPATH=/workspace/examples/llm python components/planner.py <arguments>

# Example
# PYTHONPATH=/workspace/examples/llm python components/planner.py --namespace=dynamo --served-model-name=vllm --no-operation --log-dir=log/planner
```


### Tensorboard

Planner logs to tensorboard to visualize the metrics and the scaling actions. You can start tensorboard with the following command:
```bash
tensorboard --logdir=<path-to-tensorboard-log-dir>
```

## Backends
We currently support two backends:
1. `local` - uses circus to start/stop worker subprocesses
2. `kubernetes` - uses kubernetes to scale up/down the number of worker pods by updating the replicas count of the DynamoGraphDeployment resource

### Local Backend

Circus is a Python program which can be used to monitor and control processes and sockets. Dynamo serve uses circus to start each node in a graph and monitors each subprocesses. We leverage a core feature to do this called `Watcher`. A `Watcher` is the target program that you would like to run (which in our case is `serve_dynamo.py`). When planner decides to scale up or down, it will either add or remove a watcher from the existing `circus`.

> [!NOTE]
> Although circus allows you to `increment` an existing watcher, it was not designed to allow variables to be passed in which does not allow us to schedule on a GPU. So instead we start a new watcher per process. When planner decides to add or remove a worker, we have logic to handle this adding/removing and incrementing/decrementing the workers.

#### Statefile

The statefile is a json file created when initially running `dynamo serve` and is filled in with custom leases in `serve_dynamo`. Each worker is named `{namespace}_{component_name}` when it is initially created. The `resources` come from the allocator and allows us to keep track of which GPUs are available. This statefile is read in by the LocalConnector and after each planner update we make the relevant change to the statefile. Currently, this statefile is locally saved in `~/.dynamo/state/{namespace}.json` (or in `DYN_LOCAL_STATE_DIR `) and is automatically cleaned up when the arbiter dies.

When one Decode worker is spun up, the statefile looks like:

```json
{
  "dynamo_VllmWorker": {..., resources={...}},
}
```

Now another decode worker is added:

```json
{
  "dynamo_VllmWorker": {..., resources={...}},
  "dynamo_VllmWorker_1": {..., resources={...}},
}
```

Then one decode worker is removed:

```json
{
  "dynamo_VllmWorker": {..., resources={...}},
}
```

If the last decode worker is removed, the statefile looks like:

```json
{
  "dynamo_VllmWorker": {...},
}
```

Note that we keep the initial non-suffix entry in order to know what cmd we will need to spin up another worker. This is the same for prefill workers as well.

> [!NOTE]
> At the moment - planner work best if your initial replicas per worker are 1. This is because if you specify replicas > 1 when you initially start `dynamo serve`, the current implementation in `serving.py` starts each process in the same watcher.

### Kubernetes Backend

The Kubernetes backend works by updating the replicas count of the DynamoGraphDeployment custom resource. When the planner detects the need to scale up or down a specific worker type, it uses the Kubernetes API to patch the DynamoGraphDeployment resource, modifying the replicas count for the appropriate component. The Kubernetes operator then reconciles this change by creating or removing the necessary pods. This provides a seamless scaling experience in Kubernetes environments without requiring manual intervention.