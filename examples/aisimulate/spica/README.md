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

Spica requires the matching Dynamo runtime from the same source revision. Build the bindings and
install both distributions from the repository root:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install pip "maturin[patchelf]"
cd lib/bindings/python
maturin develop --uv --release --features aic-forward-pass,mocker-kvbm-offload
cd ../../..
uv pip install --no-deps -e .
uv pip install -e ./aisimulate
```

The `dynamo-planner` image already builds and installs both wheels from the same commit.

## Run a Search

Validate and run the general search example:

```bash
python -m aisimulate.spica \
  --config examples/aisimulate/spica/configs/smart_sweep.yaml
```

The GLM-5-FP8 Pareto-front configuration captures the setup from a previous experiment.

Update `workload.trace_path` before running a trace-backed configuration.

## Generate a Synthetic Trace

Generate a Mooncake-format trace whose request rate follows a sine wave:

```bash
python examples/aisimulate/spica/tools/gen_sine_trace.py \
  --out /tmp/spica-sine-trace.jsonl
```

Compare Planner load predictors on that trace:

```bash
python examples/aisimulate/spica/tools/run_load_predictor_sweep.py \
  --trace /tmp/spica-sine-trace.jsonl \
  --policies throughput_180_5 throughput_600_5
```

## Documentation

Read the [Spica documentation](../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/ai-simulate-experimental/spica-experimental/overview.md)
for the search flow, workload schema, optimization goals, and search-space reference.
