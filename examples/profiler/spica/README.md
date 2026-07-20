<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Spica Examples

> [!WARNING]
> **Experimental.** Spica is intended for evaluation and feedback, not production capacity
> planning. Its Python API, configuration schema, search behavior, and output may change without a
> standard deprecation period. Spica does not guarantee SLA compliance, prediction accuracy, or
> globally optimal configurations.

These examples run Spica's replay-backed configuration search from a Dynamo source checkout.

## Prerequisites

Install Dynamo with the Spica optional dependencies:

```bash
uv sync --extra spica
```

The Planner/Profiler image includes the bindings required for Replay. For a local build, compile
the Python bindings with the performance-model and KVBM offload features:

```bash
cd lib/bindings/python
maturin develop --uv --release --features aic-forward-pass,mocker-kvbm-offload
cd ../../..
```

## Run a Search

Validate and run the general search example:

```bash
python -m dynamo.profiler.spica \
  --config examples/profiler/spica/configs/smart_sweep.yaml
```

Run the GLM-5-FP8 Pareto-front search:

> [!IMPORTANT]
> This configuration uses `kv_load_ratio` and requires an AI Configurator release that provides
> `aiconfigurator.sdk.memory`. It fails closed in the default Planner/Profiler image, which currently
> retains AI Configurator 0.9. Trace workloads and fixed `concurrency` workloads remain usable in
> the default image.

```bash
python -m dynamo.profiler.spica \
  --config examples/profiler/spica/configs/glm5-disagg-pareto-frontier.yaml
```

Update `workload.trace_path` before running a trace-backed configuration.

## Generate a Synthetic Trace

Generate a Mooncake-format trace whose request rate follows a sine wave:

```bash
python examples/profiler/spica/tools/gen_sine_trace.py \
  --out /tmp/spica-sine-trace.jsonl
```

Compare Planner load predictors on that trace:

```bash
python examples/profiler/spica/tools/run_load_predictor_sweep.py \
  --trace /tmp/spica-sine-trace.jsonl \
  --policies throughput_180_5 throughput_600_5
```

## Documentation

Read the [Spica documentation](https://github.com/ai-dynamo/dynamo/tree/main/docs/components/profiler/spica)
for the search flow, workload schema, optimization goals, and search-space reference.
