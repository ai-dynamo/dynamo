---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Run a DynoSim Simulation
subtitle: Replay a synthetic workload or saved trace against one simulated configuration
---

A DynoSim run evaluates one workload against one simulated Dynamo configuration. This is an
**offline replay**: the harness drives simulated engine cores directly without starting a frontend,
registering workers, or sending HTTP requests. It runs locally, does not require GPUs, and produces
an AIPerf-style summary plus a JSON report.

Use [Simulate a Local Deployment with Mocker](mocker-local.md) instead when you need to exercise the
live Dynamo frontend, discovery, routing, event publication, and worker lifecycle.

Use this tutorial to establish a working baseline before changing topology, routing, or timing
settings. For all flags, trace fields, constraints, and report fields, see the
[DynoSim Replay CLI Reference](../components/mocker/replay-cli-reference.mdx). For the internal
execution model, see [DynoSim Architecture](../design-docs/dynosim-architecture.md).

## Prerequisites

Run the commands from the repository root. Use the project virtual environment and build the runtime
bindings if they are not already available:

```bash
.venv/bin/maturin develop --release -m lib/bindings/python/Cargo.toml
uv pip install -e .
```

The release build is recommended because the simulation is CPU-bound.

## Run a synthetic workload

Start with a small aggregated simulation:

```bash
.venv/bin/python -m dynamo.replay \
  --input-tokens 2048 \
  --output-tokens 128 \
  --request-count 100 \
  --replay-mode offline \
  --replay-concurrency 16 \
  --num-workers 2 \
  --extra-engine-args '{"block_size":64}' \
  --report-json /tmp/dynosim-synthetic.json
```

The command prints a latency and throughput table. Confirm that all requests completed and that
`/tmp/dynosim-synthetic.json` was created.

## Add prefix reuse and multiple turns

Run a second synthetic workload with shared prefixes and three turns per session:

```bash
.venv/bin/python -m dynamo.replay \
  --input-tokens 5000 \
  --output-tokens 500 \
  --request-count 200 \
  --turns-per-session 3 \
  --shared-prefix-ratio 0.5 \
  --num-prefix-groups 8 \
  --inter-turn-delay-ms 250 \
  --replay-mode offline \
  --replay-concurrency 32 \
  --num-workers 2 \
  --extra-engine-args '{"block_size":64}' \
  --report-json /tmp/dynosim-prefix.json
```

Compare the prefix-cache reuse and latency metrics with the first run. Keep the worker count and
engine arguments fixed so the workload change is the only variable.

## Replay a saved trace

Download the public FAST'25 tool-agent trace:

```bash
curl -sL \
  https://raw.githubusercontent.com/kvcache-ai/Mooncake/refs/heads/main/FAST25-release/traces/toolagent_trace.jsonl \
  -o /tmp/toolagent_trace.jsonl
```

Replay it with the trace block size used by the dataset:

```bash
.venv/bin/python -m dynamo.replay /tmp/toolagent_trace.jsonl \
  --trace-block-size 512 \
  --replay-mode offline \
  --router-mode round_robin \
  --num-workers 4 \
  --extra-engine-args '{"block_size":64}' \
  --report-json /tmp/dynosim-trace.json
```

`--trace-block-size` describes how the trace encodes `hash_ids`. The engine `block_size` describes
the simulated KV-cache block size, so the two values do not need to match.

## Compare routing modes

Run the same trace through the KV router. Change only the routing settings:

```bash
.venv/bin/python -m dynamo.replay /tmp/toolagent_trace.jsonl \
  --trace-block-size 512 \
  --replay-mode offline \
  --router-mode kv_router \
  --num-workers 4 \
  --extra-engine-args '{"block_size":64}' \
  --router-config '{"router_queue_policy":"fcfs"}' \
  --report-json /tmp/dynosim-kv-router.json
```

Compare `/tmp/dynosim-trace.json` and `/tmp/dynosim-kv-router.json`. Review throughput, Time to First
Token (TTFT), Inter-Token Latency (ITL), and prefix-cache reuse before deciding whether the routing
change is promising.

## Simulate disaggregated serving

Use separate prefill and decode worker pools and engine arguments:

```bash
.venv/bin/python -m dynamo.replay /tmp/toolagent_trace.jsonl \
  --trace-block-size 512 \
  --replay-mode offline \
  --router-mode kv_router \
  --num-prefill-workers 2 \
  --num-decode-workers 2 \
  --prefill-engine-args '{"block_size":64,"worker_type":"prefill"}' \
  --decode-engine-args '{"block_size":64,"worker_type":"decode"}' \
  --report-json /tmp/dynosim-disagg.json
```

Compare this report with the aggregated baseline. If the result is worth exploring, use
[Sweep DynoSim Configurations](sweeps.md) to search more worker and topology combinations.

## Validate the result

DynoSim models scheduler, KV-cache, routing, and timing behavior, but it does not replace a
real-hardware benchmark. Use [Benchmarking with AIPerf](../tools/aiperf.md) against a deployed
candidate to validate frontend, transport, engine, and GPU behavior.
