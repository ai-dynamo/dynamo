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

These examples run Spica's backend-neutral configuration search. The provided runner script
composes Spica with Dynamo Replay and discovers the selected Dynamo adapters.

## Prerequisites

Backend-only Spica does not require Dynamo. To use the Dynamo replay runner or the
`dynamo.planner` and `dynamo.router` adapters, build the matching Dynamo runtime and install the
simulation dependencies from the repository root:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install pip "maturin[patchelf]"
cd lib/bindings/python
maturin develop --uv --release --features aic-forward-pass
cd ../../..
uv pip install --no-deps -e .
uv pip install -e ./aisimulate
uv pip install -r container/deps/requirements.planner.txt
```

For published wheels, `uv pip install "ai-dynamo[simulation]"` installs the matching simulation
bundle. The `dynamo-planner` image already builds and installs both wheels from the same commit.

## Run a Search

Run the general search example with the explicit Dynamo runner:

```bash
python examples/aisimulate/spica/tools/run_sweep.py \
  --config examples/aisimulate/spica/configs/smart_sweep.yaml
```

To use another replay implementation, pass its `RunnerFactory` to
`aisimulate.spica.run_smart_search`.

The GLM-5-FP8 configuration demonstrates a disaggregated Pareto search over
`kv_load_ratio`.

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

Read the [Spica documentation](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/developer-guide/knowledge-base/modular-components/ai-simulate-experimental/spica-experimental/overview.md)
for the search flow, workload schema, optimization goals, and search-space reference.
