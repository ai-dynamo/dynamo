---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Load-based Planner"
---

This document covers load-based planner in `examples/llm/components/planner.py`.

<Warning>
Load-based planner is inoperable as vllm, sglang, and trtllm examples all do not use prefill queues. Please use SLA planner for now.
</Warning>

<Warning>
Bare metal deployment with local connector is deprecated. The only option to deploy load-based planner is via k8s. We will update the examples in this document soon.
</Warning>

## Load-based Scaling Up/Down Prefill/Decode Workers

To adjust the number of prefill/decode workers, planner monitors the following metrics:
* Prefill worker: planner monitors the number of requests pending in the prefill queue to estimate the prefill workload.
* Decode/aggregated worker: planner monitors the average KV cache utilization rate to estimate the decode/aggregated workload.

Every `metric-pulling-interval`, planner gathers the aforementioned metrics. Every `adjustment-interval`, planner compares the aggregated metrics in this interval with pre-set thresholds and decide to scale up/down prefill/decode workers. To avoid over-compensation, planner only changes the number of workers by 1 in one adjustment interval. In addition, when the number of workers is being adjusted, the planner blocks the metric pulling and adjustment.

To scale up a prefill/decode worker, planner just need to launch the worker in the correct namespace. The auto-discovery mechanism picks up the workers and add them to the routers. To scale down a prefill worker, planner send a SIGTERM signal to the prefill worker. The prefill worker store the signal and exit when it finishes the current request pulled from the prefill queue. This ensures that no remote prefill request is dropped. To scale down a decode worker, planner revokes the etcd lease of the decode worker. When the etcd lease is revoked, the corresponding decode worker is immediately removed from the router and won't get any new requests. The decode worker then finishes all the current requests in their original stream and exits gracefully.

There are two additional rules set by planner to prevent over-compensation:
1. After a new decode worker is added, since it needs time to populate the kv cache, planner doesn't scale down the number of decode workers in the next `NEW_DECODE_WORKER_GRACE_PERIOD=3` adjustment intervals.
1. We do not scale up prefill worker if the prefill queue size is estimated to reduce below the `--prefill-queue-scale-up-threshold` within the next `NEW_PREFILL_WORKER_QUEUE_BUFFER_PERIOD=3` adjustment intervals following the trend observed in the current adjustment interval.

## SLA-based Scaling Up/Down Prefill/Decode Workers

See [SLA-Driven Profiling](../benchmarks/sla-driven-profiling.md) for more details.

## Usage

The planner integration with the new frontend + worker architecture is currently a work in progress. This documentation will be updated with the new deployment patterns and code examples once the planner component has been fully adapted to the new workflow.

Configuration options:
* `namespace` (str, default: "dynamo"): Target namespace for planner operations
* `environment` (str, default: "local"): Target environment (local, kubernetes)
* `no-operation` (bool, default: false): Run in observation mode only
* `log-dir` (str, default: None): Tensorboard log directory
* `adjustment-interval` (int, default: 30): Seconds between adjustments
* `metric-pulling-interval` (int, default: 1): Seconds between metric pulls
* `max-gpu-budget` (int, default: 8): Maximum GPUs for all workers
* `min-gpu-budget` (int, default: 1): Minimum GPUs per worker type
* `decode-kv-scale-up-threshold` (float, default: 0.9): KV cache threshold for scale-up
* `decode-kv-scale-down-threshold` (float, default: 0.5): KV cache threshold for scale-down
* `prefill-queue-scale-up-threshold` (float, default: 0.5): Queue threshold for scale-up
* `prefill-queue-scale-down-threshold` (float, default: 0.2): Queue threshold for scale-down
* `decode-engine-num-gpu` (int, default: 1): GPUs per decode engine
* `prefill-engine-num-gpu` (int, default: 1): GPUs per prefill engine

Run as standalone process:
```bash
PYTHONPATH=/workspace/examples/llm python components/planner.py --namespace=dynamo --served-model-name=vllm --no-operation --log-dir=log/planner
```

Monitor metrics with Tensorboard:
```bash
tensorboard --logdir=<path-to-tensorboard-log-dir>
```
