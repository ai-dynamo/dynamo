---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Sweep DynoSim Configurations
subtitle: Recommend simulated topology, worker, and router choices before using GPU time
---

`aisimulate recommend --stack dynamo` searches simulated deployment configurations and writes each
selected candidate as a concrete prediction YAML. With `--output dgd`, it also renders the selected
candidate as a `DynamoGraphDeployment` (DGD). The search runs offline on CPUs; the GPU count is a
simulated constraint rather than a host requirement.

Use recommendation after a single [DynoSim prediction](dynosim-replay.mdx) works. For field and
domain semantics, see the
[DynoSim Sweep Reference](../../../reference/components/dynosim-sweep-reference.mdx).

## Prerequisites

Run from the repository root. Create and activate a virtual environment, build the runtime bindings,
and install Dynamo. The editable Dynamo installation installs the pinned AISimulate release and
registers the Dynamo-owned `dgd` output adapter:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install pip 'maturin[patchelf]'
maturin develop --release -m lib/bindings/python/Cargo.toml
uv pip install -e .
uv pip install scikit-learn==1.7.2
```

Do not install the standalone `aiconfigurator` package. AISimulate includes the performance-model
compatibility code used by the Dynamo stack.

<Steps toc={true}>
<Step title="Create one input file" id="create-a-recommendation-configuration">

Save the complete configuration below as `/tmp/dynosim-recommend.yaml`. It is one input file with
two parts:

- `traffic` through `optimizer` configure AISimulate's simulation and search.
- `dgd` configures the Dynamo output adapter selected by `--output dgd`.

```yaml
# AISimulate recommendation input: workload, search space, and optimizer.
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

# Dynamo output configuration, consumed only by --output dgd.
dgd:
  name: qwen
  output_file: deployment.yaml
  generator: aic
  format: manifest
  runtime_image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.6.0
  num_gpus_per_node: 8
```

Each parallelism preset is a complete mapping and becomes one categorical choice. Router and
scheduler domains add independent search dimensions. `engine.model` identifies the model that
AISimulate evaluates and that the generated DGD serves. `dgd.name` is only the Kubernetes resource
name, while `dgd.output_file` is the output filename relative to `--output-dir`. Neither field
selects the model. `dgd.namespace` is optional; when omitted or empty, the output manifest does not
set `metadata.namespace`.

To search Planner settings too, add `planner: {}`. The default search keeps the scaling presets
compatible with the optimization target: throughput and latency objectives retain disabled and
load-only policies; goodput objectives can also use throughput and hybrid policies when both
TTFT and ITL thresholds are supplied. An explicit `scaling_policy.preset` list must contain only
compatible choices. Other explicit preset lists, including load-predictor candidates, stay within
the selected subset.

FPM sampling is searched only if a retained policy uses throughput scaling; load sensitivity is
searched only if one uses load scaling. With `preset: false`, independent domains follow the
same filtering, and invalid combinations are skipped as infeasible candidates. Concrete
recommendations resolve through the same `PlannerConfig` as the production Planner and preserve
its effective defaults, scaling flags, and GPU limits when passed to `predict`.

</Step>
<Step title="Run the recommendation" id="run-the-recommendation">

```bash
aisimulate recommend \
  --stack dynamo \
  --config /tmp/dynosim-recommend.yaml \
  --output dgd \
  --output-dir /tmp/dynosim-recommendations
```

AISimulate first applies any `--set` overrides. Because the command requests `--output dgd`, it then
removes the top-level `dgd` mapping before validating the recommendation input and passes that mapping
to the Dynamo adapter. Omitting the `dgd` mapping while requesting `--output dgd` is an error.

The command prints ranked candidates and produces:

```text
/tmp/dynosim-recommendations/
├── recommendations/   # AISimulate's concrete ranked candidate configurations
├── deployment.yaml     # DGD rendered from the selected scalar candidate
└── index.json          # Index of artifacts written by the DGD adapter
```

The `--stack` option selects the simulation implementation. The independent `--output dgd` option
selects DGD generation.

The `dgd.generator` field defaults to `aic`. Set it to `direct` to compare the direct Dynamo generator.
The output target depends on `dgd.format`:

- `manifest` requires `output_file` and rejects `output_dir`.
- `kustomize` requires `output_dir` and rejects `output_file`. The directory is relative to the CLI
  `--output-dir` and contains `deploy.yaml` and `kustomization.yaml`.

`output_file` must be a bare `.yaml` or `.yml` filename, and `output_dir` must be one directory name,
not a path. Neither accepts `.` or `..`; `{index}` is the only supported placeholder.

For example, this writes the Kustomize bundle to `/tmp/dynosim-recommendations/qwen/`:

```yaml
dgd:
  name: qwen
  format: kustomize
  output_dir: qwen
  runtime_image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.6.0
  num_gpus_per_node: 8
```

For a Pareto optimization, `{index}` is required in `name` and in the active output field
(`output_file` or `output_dir`); scalar targets reject the placeholder. For example,
`name: candidate-{index}` with `output_dir: candidate-{index}` creates one independently deployable
Kustomize bundle per selected candidate. The adapter substitutes a three-digit, zero-based index:
`candidate-000`, `candidate-001`, and so on. AISimulate's recommendation files use a separate
four-digit, one-based sequence, so `candidate-000` is the DGD for `recommendations/0001.yaml`.

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
  --output dgd \
  --output-dir /tmp/dynosim-trace-recommendations
```

Use a shorter virtual-time cutoff or trial budget while iterating on large traces.

</Step>
<Step title="Customize the objective" id="customize-the-objective">

Set `optimization.target` to `throughput`, `throughput_per_gpu`, `throughput_per_user`, `goodput`,
`goodput_per_gpu`, `ttft`, `e2e_latency`, or `pareto`. Goodput targets require `evaluation.sla`.
Pareto output contains the complete nondominated front rather than a scalar ranking.
Before changing the example to `optimization.target: pareto`, update `dgd.name` and the active output
field with the required `{index}` placeholder as described in the recommendation step.

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
