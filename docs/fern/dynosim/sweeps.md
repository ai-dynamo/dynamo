---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Sweep DynoSim Configurations
subtitle: Search simulated topology, worker, and router choices before using GPU time
---

A DynoSim sweep evaluates many simulated configurations and ranks candidates against latency and GPU
budget constraints. The sweep runs locally, but its candidates can be used for either Kubernetes or
local CLI deployment. Use a sweep after a single [DynoSim run](runs.md) works and you want to identify
configurations to validate on real hardware.

The current Python API is `dynamo.profiler.utils.replay_optimize`. For the input models, field names,
result object, and output columns, see the
[DynoSim Sweep Reference](../components/mocker/sweep-reference.mdx).

## Prerequisites

Run from the repository root. Build the runtime bindings and install the project:

```bash
.venv/bin/maturin develop --release -m lib/bindings/python/Cargo.toml
uv pip install -e .
```

The checked-in example uses AIConfigurator-backed timing. Install AIConfigurator into the project
environment:

```bash
uv pip install --python .venv/bin/python aiconfigurator
```

When running directly from the source checkout, expose the in-repository Python components and
runtime bindings:

```bash
export PYTHONPATH=components/src:lib/bindings/python/src
```

## Run the example sweep

Run the checked-in driver with four parallel evaluations:

```bash
.venv/bin/python components/src/dynamo/profiler/utils/replay_optimize/example.py \
  --max-parallel-evals 4
```

The default example searches a synthetic disaggregated workload with a KV router. It prints the best
feasible state and a table of highly ranked feasible configurations. The GPU budget is a simulated
constraint; the machine running the sweep does not need that number of physical GPUs.

Start with a low parallel-evaluation count. Increase it only after confirming that the host has
enough CPU and memory for concurrent trials.

## Inspect the result

Review the best feasible candidate and the evaluated table. At minimum, compare:

- prefill and decode tensor-parallel sizes
- prefill and decode worker counts
- total simulated GPUs
- output throughput
- TTFT, ITL, and end-to-end latency
- prefix-cache reuse
- router overlap and prefill-load settings

A candidate is feasible only when it meets every configured SLA and GPU-budget constraint. Keep the
best infeasible result as a diagnostic: it can show which bound prevented an otherwise useful
configuration from being selected.

## Run a sweep against a trace

Download the public FAST'25 tool-agent trace:

```bash
curl -sL \
  https://raw.githubusercontent.com/kvcache-ai/Mooncake/refs/heads/main/FAST25-release/traces/toolagent_trace.jsonl \
  -o /tmp/toolagent_trace.jsonl
```

Run the same search against the trace:

```bash
.venv/bin/python components/src/dynamo/profiler/utils/replay_optimize/example.py \
  --trace-file /tmp/toolagent_trace.jsonl \
  --arrival-speedup-ratio 1.0 \
  --max-parallel-evals 4
```

The trace replaces the synthetic input-length, output-length, request-count, concurrency, and prefix
settings in the example. Keep `--arrival-speedup-ratio 1.0` for the first run so the original arrival
shape remains unchanged.

## Customize the search

Copy the example driver and change one search dimension at a time. Common experiments include:

- changing `HardwareSpec.totalGpus`
- adding TTFT, ITL, end-to-end latency, or p95 bounds to `SLASpec`
- changing `RouterSpec.overlapCredits`
- changing `RouterSpec.prefillLoadScales`
- changing shared-prefix settings in `WorkloadSpec`
- switching `RouterSpec.mode` to compare routing strategies
- changing the base prefill or decode engine arguments

Run the modified script again and compare the feasible tables. Persist `evaluated_df` and
`feasible_df` to CSV or Parquet when you need repeatable analysis across multiple searches.

## Validate a candidate

A sweep is a heuristic search over simulated states, not an exhaustive proof that one configuration
is optimal. Take the highest-ranked feasible candidates through these checks:

1. Run the configuration again with [Run a DynoSim Simulation](runs.md).
2. Exercise the live frontend and router with [Simulate a Local Deployment](mocker-local.md) or
   [Simulate a Kubernetes Deployment](mocker.md).
3. Deploy the candidate on the target hardware and benchmark it with AIPerf using either the
   [Kubernetes workflow](../tools/aiperf.md) or the [local CLI workflow](../cli/aiperf.mdx).
