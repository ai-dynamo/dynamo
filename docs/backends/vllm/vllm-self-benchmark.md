---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Self-Benchmarking
subtitle: Generate a startup sweep or run an explicit set of pure prefill and decode points
---

Dynamo's vLLM backend can benchmark the engine during startup, before the
worker registers for serving. The scheduler injects synthetic requests,
records forward-pass metrics (FPMs), and writes the results to
`/tmp/benchmark_results.json` by default.

Use the `python -m dynamo.vllm` entry point shown below. The unified vLLM
entry point does not yet gate worker registration on benchmark completion.

Enable a generated sweep with `--benchmark-mode`:

```bash
python -m dynamo.vllm \
  --model Qwen/Qwen3-0.6B \
  --benchmark-mode agg
```

The supported modes are:

| Mode | Points executed | Required non-empty JSON arrays |
|---|---|---|
| `prefill` | Pure prefill points | `prefill` |
| `decode` | Pure decode points | `decode` |
| `agg` | All pure prefill points, then all pure decode points | `prefill` and `decode` |

`agg` describes the composition of the sweep. It does not create benchmark
batches containing both prefill and decode requests. Normal serving can form
mixed scheduler iterations, but self-benchmark points are deliberately
phase-isolated so each result has an unambiguous prefill or decode shape.

## Supply Explicit Points

Use `--benchmark-points-file` to replace generated grid formation and sampling
with an ordered JSON manifest:

```json
{
  "schema_version": 1,
  "prefill": [
    {
      "total_prefill_tokens": 512,
      "total_kv_read_tokens": 0,
      "batch_size": 1
    },
    {
      "total_prefill_tokens": 1024,
      "total_kv_read_tokens": 2048,
      "batch_size": 4
    }
  ],
  "decode": [
    {
      "total_kv_read_tokens": 4096,
      "batch_size": 8
    },
    {
      "total_kv_read_tokens": 16384,
      "batch_size": 32
    }
  ]
}
```

Then launch the benchmark:

```bash
python -m dynamo.vllm \
  --model Qwen/Qwen3-0.6B \
  --benchmark-mode agg \
  --benchmark-points-file /path/to/points.json
```

The equivalent environment variable is `DYN_BENCHMARK_POINTS_FILE`.

The manifest is strict:

- `schema_version`, `prefill`, and `decode` are required. Both arrays are
  validated even when the selected mode executes only one of them.
- Values must be JSON integers. Booleans, floating-point values, strings,
  missing fields, and unknown fields are rejected.
- Prefill points contain exactly `total_prefill_tokens`,
  `total_kv_read_tokens`, and `batch_size`. Decode points contain exactly
  `total_kv_read_tokens` and `batch_size`.
- Totals are iteration totals on each DP rank. Dynamo balances each total
  across the point's requests while preserving the exact requested total;
  requests can differ by one token (or one cache block for aligned prefill KV
  reads) when a total is not evenly divisible by `batch_size`.
- Points are phase-pure and uniform across DP ranks. Rank-specific fields and
  mixed prefill-plus-decode points are not supported in schema version 1.

The parent process reads and normalizes the file once, computes a digest, and
forwards those contents to the engine processes. Every attention-DP rank
receives the same manifest and executes the same rank-local points. Dynamo
checks the manifest, realized grid, and synchronized scheduler output across
ranks. The merged iteration time is the maximum rank time.

## Validation and Failures

Static schema errors identify the exact field, for example
`prefill[2].batch_size`. Engine-dependent constraints are checked after vLLM
has initialized, including scheduler token and request limits, model length,
KV-cache capacity, prefix-cache availability and block alignment, and CUDA
graph metadata.

An explicit point is never silently filtered or resampled. If it cannot be
realized, the benchmark fails with its source index, such as `prefill[2]` or
`decode[5]`. Runtime injection or measured-shape mismatches fail in the same
way. Generated sweeps retain their existing behavior of filtering infeasible
candidate points.

Because an explicit file completely replaces generated sampling, it cannot be
combined with explicitly supplied grid sampling or deprecated granularity
options. Operational controls remain available, including:

- `--benchmark-warmup-iterations`
- `--benchmark-timeout`
- `--benchmark-output-path`

The points file must not be the output path or its `.tmp` sidecar; benchmark
startup clears stale output artifacts before the sweep begins.

## Results

Explicit-point result artifacts keep the existing self-benchmark result
schema and add provenance:

```json
{
  "point_source": {
    "kind": "external_json",
    "schema_version": 1,
    "placement": "uniform_per_dp_rank",
    "sha256": "..."
  }
}
```

The output records the realized point, including its generated
`benchmark_id`, live CUDA graph expectations, and `sample_reasons`. Runs that
use the generated grid do not add `point_source`, preserving their existing
artifact shape.
