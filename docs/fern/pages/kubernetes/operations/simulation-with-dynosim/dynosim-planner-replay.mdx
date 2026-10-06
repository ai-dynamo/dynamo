---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Benchmark Planner Decisions with DynoSim
subtitle: Compare topology and scaling decisions against a saved trace
---

Run the Dynamo Planner inside an offline DynoSim prediction to compare aggregated and
disaggregated topologies, service-level objective (SLO) targets, and worker startup delays without a
live cluster.

<Note>
The production Planner scales Kubernetes or Global Planner deployments; it does not autoscale a
local deployment. `aisimulate predict --stack dynamo` runs locally but evaluates Planner decisions
for the deployment described by its YAML input.
</Note>

For the general workflow, see
[Run a DynoSim Simulation](../../../cli/operations/simulation-with-dynosim/dynosim-replay.mdx). For
Planner field types and defaults, see the
[Planner Configuration reference](../../../reference/components/planner-configuration.mdx). For
the simulation adapter, see
[DynoSim Architecture](../../../developer-guide/knowledge-base/concepts/simulation/dynosim-architecture.md#planner-simulation-adapter).

## Prerequisites

Build the Rust runtime bindings and install Dynamo from the repository root:

```bash
source .venv/bin/activate
maturin develop --release -m lib/bindings/python/Cargo.toml
uv pip install -e .
```

Use a release build because repeated simulation runs are CPU-bound.

<Steps toc={true}>
<Step title="Run the aggregated baseline" id="run-the-aggregated-baseline">

Download the FAST'25 tool-agent trace:

```bash
mkdir -p traces/mooncake-fast25
curl -sL \
  https://raw.githubusercontent.com/kvcache-ai/Mooncake/refs/heads/main/FAST25-release/traces/toolagent_trace.jsonl \
  -o traces/mooncake-fast25/toolagent_trace.jsonl
```

Save this configuration as `planner-aggregated.yaml`:

```yaml
traffic:
  source:
    type: trace
    format: mooncake
    paths: [traces/mooncake-fast25/toolagent_trace.jsonl]
    block_size: 512
  load: {type: trace_timestamps, speedup: 1.0}
  stop: {max_virtual_time_seconds: 3600}
engine:
  mode: aggregated
  model: nvidia/Llama-3.1-8B-Instruct-FP8
  hardware: h200_sxm
  backend: vllm
  context_length: max
  workers:
    aggregated:
      parallelism: {replicas: 2, tensor: 1, pipeline: 1, attention_data: 1, moe_tensor: 1, moe_expert: 1}
      scheduler: {max_batched_tokens: 8192, max_sequences: 256}
      kv_cache: {block_size: 64, prefix_caching: true, capacity: {type: default, memory_fraction: 0.9}}
      timing: {type: default}
      startup_seconds: 0
router:
  policy: round_robin
  prefill_load_model: {type: none}
planner:
  policy: enabled
  target: sla
  enable_throughput_scaling: true
  enable_load_scaling: true
  throughput_adjustment_interval_seconds: 300
  load_adjustment_interval_seconds: 10
  load_scaling_down_sensitivity: 80
  load_min_observations: 5
  max_num_gpus: 2
  min_workers: 1
evaluation:
  sla: {ttft_ms: 1500, itl_ms: 50}
```

Run the prediction:

```bash
aisimulate predict \
  --stack dynamo \
  --config planner-aggregated.yaml \
  --output-dir planner-reports/aggregated
```

The command prints the summary and writes Planner decisions and metrics to
`planner-reports/aggregated/prediction.json`.

</Step>
<Step title="Run the disaggregated comparison" id="run-the-disaggregated-comparison">

Copy the baseline, then replace its `engine` mapping with separate prefill and decode roles:

```yaml
engine:
  mode: disaggregated
  model: nvidia/Llama-3.1-8B-Instruct-FP8
  hardware: h200_sxm
  backend: vllm
  context_length: max
  kv_transfer: {bytes_per_token: auto, bandwidth_gb_per_second: 400, timing_mode: destination_missing}
  workers:
    prefill:
      parallelism: {replicas: 1, tensor: 1, pipeline: 1, attention_data: 1, moe_tensor: 1, moe_expert: 1}
      scheduler: {max_batched_tokens: 8192, max_sequences: 64}
      kv_cache: {block_size: 64, prefix_caching: true, capacity: {type: default, memory_fraction: 0.9}}
      timing: {type: default}
      startup_seconds: 0
    decode:
      parallelism: {replicas: 1, tensor: 1, pipeline: 1, attention_data: 1, moe_tensor: 1, moe_expert: 1}
      scheduler: {max_batched_tokens: 8192, max_sequences: 256}
      kv_cache: {block_size: 64, prefix_caching: true, capacity: {type: default, memory_fraction: 0.9}}
      timing: {type: default}
      startup_seconds: 0
```

Also replace the `planner` mapping with role-specific minimums:

```yaml
planner:
  policy: enabled
  target: sla
  enable_throughput_scaling: true
  enable_load_scaling: true
  throughput_adjustment_interval_seconds: 300
  load_adjustment_interval_seconds: 10
  load_scaling_down_sensitivity: 80
  load_min_observations: 5
  max_num_gpus: 2
  min_workers: 1
  prefill_min_workers: 1
  decode_min_workers: 1
```

Save the result as `planner-disaggregated.yaml`, then run it:

```bash
aisimulate predict \
  --stack dynamo \
  --config planner-disaggregated.yaml \
  --output-dir planner-reports/disaggregated
```

Compare request metrics, scaling events, and cumulative GPU time between the two
`prediction.json` files.

</Step>
<Step title="Sweep aggregated startup time" id="sweep-aggregated-startup-time">

The `startup_seconds` field is concrete rather than a recommendation search dimension. Run the same
prediction with a sequence of CLI overrides:

```bash
for startup_seconds in $(seq 0 10 300); do
  output_dir=$(printf "planner-reports/startup-%03d" "$startup_seconds")
  aisimulate predict \
    --stack dynamo \
    --config planner-aggregated.yaml \
    --set "engine.workers.aggregated.startup_seconds=$startup_seconds" \
    --output-dir "$output_dir" \
    --overwrite
done
```

Read each `prediction.json` to compare TTFT, ITL, scaling events, and cumulative GPU time as startup
delay increases.

</Step>
</Steps>
