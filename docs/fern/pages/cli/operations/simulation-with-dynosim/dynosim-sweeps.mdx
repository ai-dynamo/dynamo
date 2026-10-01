---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Sweep DynoSim Configurations
subtitle: Recommend simulated topology, worker, and router choices before using GPU time
---

`aisimulate recommend --stack dynamo` searches simulated deployment configurations and writes each
selected candidate as a concrete prediction YAML. The search runs offline on CPUs; the GPU count is
a simulated constraint rather than a host requirement.

Use recommendation after a single [DynoSim prediction](dynosim-replay.mdx) works. For field and
domain semantics, see the
[DynoSim Sweep Reference](../../../reference/components/dynosim-sweep-reference.mdx).

## Prerequisites

Run from the repository root. Build the runtime bindings and install Dynamo, which installs the
pinned AISimulate release:

```bash
.venv/bin/maturin develop --release -m lib/bindings/python/Cargo.toml
uv pip install -e .
```

Do not install the standalone `aiconfigurator` package. AISimulate includes the performance-model
compatibility code used by the Dynamo stack.

<Steps toc={true}>
<Step title="Create a recommendation configuration" id="create-a-recommendation-configuration">

Save this configuration as `/tmp/dynosim-recommend.yaml`:

```yaml
traffic:
  source: {type: synthetic, input_tokens: 1024, output_tokens: 128}
  load: {type: concurrency, concurrency: 8}
  stop: {requests: 50}
engine:
  mode: aggregated
  model: Qwen/Qwen3-0.6B
  hardware: h200_sxm
  backend: vllm
  context_length: 8192
  workers:
    aggregated:
      parallelism:
        preset:
          - {replicas: 1, tensor: 1, pipeline: 1, attention_data: 1, moe_tensor: 1, moe_expert: 1}
          - {replicas: 2, tensor: 1, pipeline: 1, attention_data: 1, moe_tensor: 1, moe_expert: 1}
      scheduler:
        max_batched_tokens: {choices: [4096, 8192]}
        max_sequences: 256
      kv_cache: {block_size: 64, prefix_caching: true, capacity: {type: fixed, blocks: 32768}}
      timing: {type: fixed, prefill_ms: 2, decode_ms: 0.5}
router:
  policy: {choices: [round_robin, kv_router]}
  prefill_load_model: {type: none}
  overlap_score_credit: {choices: [0.5, 1.0]}
  prefill_load_scale: {choices: [0.5, 1.0]}
  temperature: {choices: [0.0, 0.2]}
optimization:
  target: throughput
  constraints: {max_candidate_gpus: 2}
optimizer:
  algorithm: random
  max_trials: 4
  parallelism: 2
  candidate_timeout_seconds: 30
  seed: 42
```

Each parallelism preset is a complete mapping and becomes one categorical choice. Router and
scheduler domains add independent search dimensions.

</Step>
<Step title="Run the recommendation" id="run-the-recommendation">

```bash
aisimulate recommend \
  --stack dynamo \
  --config /tmp/dynosim-recommend.yaml \
  --output-dir /tmp/dynosim-recommendations
```

The command prints ranked candidates and writes concrete files under
`/tmp/dynosim-recommendations/recommendations/`.

</Step>
<Step title="Predict the best candidate" id="predict-the-best-candidate">

Pass the highest-ranked recommendation directly to `predict`:

```bash
aisimulate predict \
  --stack dynamo \
  --config /tmp/dynosim-recommendations/recommendations/0001.yaml \
  --output-dir /tmp/dynosim-best-prediction
```

Compare the prediction metrics with the baseline before deploying the candidate.

</Step>
<Step title="Search against a trace" id="search-against-a-trace">

Download the public FAST'25 tool-agent trace:

```bash
curl -sL \
  https://raw.githubusercontent.com/kvcache-ai/Mooncake/refs/heads/main/FAST25-release/traces/toolagent_trace.jsonl \
  -o /tmp/toolagent_trace.jsonl
```

Override the workload while retaining the engine and search domains:

```bash
aisimulate recommend \
  --stack dynamo \
  --config /tmp/dynosim-recommend.yaml \
  --set 'traffic.source={type: trace, format: mooncake, paths: [/tmp/toolagent_trace.jsonl], block_size: 512}' \
  --set 'traffic.load={type: trace_timestamps, speedup: 1.0}' \
  --set 'traffic.stop={max_virtual_time_seconds: 3600}' \
  --output-dir /tmp/dynosim-trace-recommendations
```

Use a shorter virtual-time cutoff or trial budget while iterating on large traces.

</Step>
<Step title="Customize the objective" id="customize-the-objective">

Set `optimization.target` to `throughput`, `throughput_per_gpu`, `throughput_per_user`, `goodput`,
`goodput_per_gpu`, `ttft`, `e2e_latency`, or `pareto`. Goodput targets require `evaluation.sla`.
Pareto output contains the complete nondominated front rather than a scalar ranking.

Change one domain at a time. Use `choices` for categorical values, `range` for numeric domains, and
complete preset mappings for correlated knobs such as parallelism.

</Step>
<Step title="Validate a candidate" id="validate-a-candidate">

A recommendation is a heuristic simulation result, not proof of optimality. Run the generated YAML
through `aisimulate predict`, then deploy the candidate on its target hardware and benchmark it with
AIPerf using either the
[Kubernetes workflow](../../../kubernetes/operations/benchmarking-with-aiperf.mdx) or the
[local workflow](../benchmarking-with-aiperf.mdx).

</Step>
</Steps>
