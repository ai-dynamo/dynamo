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

The planner monitors the state of the system and adjusts workers to ensure that the system runs efficiently. Currently, the planner can scale the number of vllm workers up and down based on the kv cache load and prefill queue size:
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
  * SLA-based scaling up/down prefill/decode workers ✅ (with some limitations)
  * Adjusting engine knobs ❌

## Load-based Scaling Up/Down Prefill/Decode Workers

To adjust the number of prefill/decode workers, planner monitors the following metrics:
* Prefill worker: planner monitors the number of requests pending in the prefill queue to estimate the prefill workload.
* Decode/aggregated worker: planner monitors the average KV cache utilization rate to estimate the decode/aggregated workload.

Every `metric-pulling-interval`, planner gathers the aforementioned metrics. Every `adjustment-interval`, planner compares the aggregated metrics in this interval with pre-set thresholds and decide to scale up/down prefill/decode workers. To avoid over-compensation, planner only changes the number of workers by 1 in one adjustment interval. In addition, when the number of workers is being adjusted, the planner blocks the metric pulling and adjustment.

To scale up a prefill/decode worker, planner just need to launch the worker in the correct namespace. The auto-discovery mechanism picks up the workers and add them to the routers. To scale down a prefill worker, planner send a SIGTERM signal to the prefill worker. The prefill worker store the signal and exit when it finishes the current request pulled from the prefill queue. This ensures that no remote prefill request is dropped. To scale down a decode worker, planner revokes the etcd lease of the decode worker. When the etcd lease is revoked, the corresponding decode worker is immediately removed from the router and won't get any new requests. The decode worker then finishes all the current requests in their original stream and exits gracefully.

There are two additional rules set by planner to prevent over-compensation:
1. After a new decode worker is added, since it needs time to populate the kv cache, planner doesn't scale down the number of decode workers in the next `NEW_DECODE_WORKER_GRACE_PERIOD=3` adjustment intervals.
1. We do not scale up prefill worker if the prefill queue size is estimated to reduce below the `--prefill-queue-scale-up-threshold` within the next `NEW_PREFILL_WORKER_QUEUE_BUFFER_PERIOD=3` adjustment intervals following the trend observed in the current adjustment interval.

## Comply with SLA

To ensure dynamo serve complies with the SLA, we provide a pre-deployment script to profile the model performance with different parallelization mappings and recommend the parallelization mapping for prefill and decode workers and planner configurations. To use this script, the user needs to provide the target ISL, OSL, TTFT SLA, and ITL SLA.

> [!NOTE]
> Currently, the script considers a fixed ISL/OSL without KV cache reuse. If the real ISL/OSL has a large variance or a significant amount of KV cache can be reused, the result might be inaccurate.
> Currently, we assume there is no piggy-backed prefill requests in the decode engine. Even if there are some short piggy-backed prefill requests in the decode engine, it should not affect the ITL too much in most conditions. However, if the piggy-backed prefill requests are too much, the ITL might be inaccurate.

```bash
python -m utils.profile_sla \
  --config <path-to-dynamo-config-file> \
  --output-dir <path-to-profile-results-dir> \
  --isl <target-isl> \
  --osl <target-osl> \
  --ttft <target-ttft-(ms)> \
  --itl <target-itl-(ms)>
```

The script will first detect the number of available GPUs on the current nodes (multi-node engine not supported yet). Then, it will profile the prefill and decode performance with different TP sizes. For prefill, since there is no in-flight batching (assume isl is long enough to saturate the GPU), the script directly measures the TTFT for a request with given isl without kv-reusing. For decode, since the ITL (or iteration time) is relevant with how many requests are in-flight, the script will measure the ITL under different number of in-flight requests. The range of the number of in-flight requests is from 1 to the maximum number of requests that the kv cache of the engine can hold. To measure the ITL without being affected by piggy-backed prefill requests, the script will enable kv-reuse and warm up the engine by issuing the same prompts before measuring the ITL. Since the kv cache is sufficient for all the requests, it can hold the kv cache of the pre-computed prompts and skip the prefill phase when measuring the ITL.

After the profiling finishes, two plots will be generated in the `output-dir`. For example, here are the profiling results for `examples/llm/configs/disagg.yaml`:

![Prefill Performance](../images/h100_prefill_performance.png)
![Decode Performance](../images/h100_decode_performance.png)

For the prefill performance, the script will plot the TTFT for different TP sizes and select the best TP size that meet the target TTFT SLA and delivers the best throughput per GPU. Based on how close the TTFT of the selected TP size is to the SLA, the script will also recommend the upper and lower bounds of the prefill queue size to be used in planner.

For the decode performance, the script will plot the ITL for different TP sizes and different in-flight requests. Similarly, it will select the best point that satisfies the ITL SLA and delivers the best throughput per GPU and recommend the upper and lower bounds of the kv cache utilization rate to be used in planner.

The following information will be printed out in the terminal:
```
2025-05-16 15:20:24 - __main__ - INFO - Analyzing results and generate recommendations...
2025-05-16 15:20:24 - __main__ - INFO - Suggested prefill TP:4 (TTFT 48.37 ms, throughput 15505.23 tokens/s/GPU)
2025-05-16 15:20:24 - __main__ - INFO - Suggested planner upper/lower bound for prefill queue size: 0.24/0.10
2025-05-16 15:20:24 - __main__ - INFO - Suggested decode TP:4 (ITL 4.83 ms, throughput 51.22 tokens/s/GPU)
2025-05-16 15:20:24 - __main__ - INFO - Suggested planner upper/lower bound for decode kv cache utilization: 0.20/0.10
```

After finding the best TP size for prefill and decode, the script will then interpolate the TTFT with ISL and ITL with active KV cache and decode context length. This is to provide a more accurate estimation of the performance when ISL and OSL changes. The results will be saved to `<output_dir>/<decode/prefill>_tp<best_tp>_interploation`.

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
* `environment` (str, default: "local"): Environment to run the planner in (local, kubernetes)
* `served-model-name` (str, default: "vllm"): Model name that is being served
* `no-operation` (bool, default: false): Do not make any adjustments, just observe the metrics and log to tensorboard
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

Circus is a Python program that can be used to monitor and control processes and sockets. Dynamo serve uses circus to start each node in a graph and monitors each subprocesses. We leverage a core feature to do this called `Watcher`. A `Watcher` is the target program that you would like to run (which in our case is `serve_dynamo.py`). When planner decides to scale up or down, it either adds or removes a watcher from the existing `circus`.

``` {note}
Although circus allows you to `increment` an existing watcher, it was not designed to allow variables to be passed in which does not allow us to schedule on a GPU. So instead we start a new watcher per process. When planner decides to add or remove a worker, we have logic to handle this adding/removing and incrementing/decrementing the workers.
```

#### Statefile

The statefile is a json file created when initially running `dynamo serve` and is filled in with custom leases in `serve_dynamo`. Each worker is named `{namespace}_{component_name}` when it is initially created. The `resources` come from the allocator and allows us to keep track of which GPUs are available. This statefile is read in by the LocalConnector and after each planner update we make the relevant change to the statefile. Currently, this statefile is locally saved in `~/.dynamo/state/{namespace}.json` (or in `DYN_LOCAL_STATE_DIR `) and is automatically cleaned up when the arbiter dies.

When one Decode worker is spun up, the statefile looks like:

```none
{
  "dynamo_VllmWorker": {..., resources={...}},
}
```

Now another decode worker is added:

```none
{
  "dynamo_VllmWorker": {..., resources={...}},
  "dynamo_VllmWorker_1": {..., resources={...}},
}
```

Then one decode worker is removed:

```none
{
  "dynamo_VllmWorker": {..., resources={...}},
}
```

If the last decode worker is removed, the statefile looks like:

```none
{
  "dynamo_VllmWorker": {...},
}
```

We keep the initial non-suffix entry in order to know what cmd we'll need to spin up another worker. This is the same for prefill workers as well.

``` {note}
At the moment - planner work best if your initial replicas per worker are 1. This is because if you specify replicas > 1 when you initially start `dynamo serve`, the current implementation in `serving.py` starts each process in the same watcher.
```

### Kubernetes Backend

The Kubernetes backend works by updating the replicas count of the DynamoGraphDeployment custom resource. When the planner detects the need to scale up or down a specific worker type, it uses the Kubernetes API to patch the DynamoGraphDeployment resource, modifying the replicas count for the appropriate component. The Kubernetes operator then reconciles this change by creating or removing the necessary pods. This provides a seamless scaling experience in Kubernetes environments without requiring manual intervention.
