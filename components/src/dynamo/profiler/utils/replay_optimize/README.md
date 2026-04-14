<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Replay Optimize

This directory contains helpers for heuristic offline replay search over dense aggregated and
disaggregated serving configurations. This README documents one concrete experiment setup: the
synthetic disaggregated KV-router sweep that we have been using for non-trivial `replay_optimize`
runs.

For the search logic itself, start with [search.py](search.py). For unit-tested examples, see
[../../tests/test_replay_optimize.py](../../tests/test_replay_optimize.py).

For adjacent trace tooling, see:

- [Prefix Data Generator](../../../../../../benchmarks/prefix_data_generator/README.md)
  for Mooncake trace format details plus `datagen analyze` and `datagen synthesize`
- [KV Router A/B Benchmarking Guide](../../../../../../docs/benchmarks/kv-router-ab-testing.md)
  for the public toolagent trace URL and benchmark context

## Experiment Goal

This experiment searches over disaggregated replay states to answer a concrete question:

- given a fixed GPU budget
- for a workload with real prefix overlap
- and latency constraints that still permit meaningful throughput

which `(prefill_tp, decode_tp, prefill_workers, decode_workers, overlap_score_weight)` combination
produces the best offline replay result?

This is a heuristic search over replay states, not an exact optimizer over all feasible
configurations.

## Prerequisites

Run from the repository root.

Use the project virtual environment:

```bash
.venv/bin/python --version
```

If the Python bindings are not importable yet, build them first:

```bash
.venv/bin/maturin develop --uv -m lib/bindings/python/Cargo.toml
```

When running directly from a source checkout, expose the in-repo Python packages:

```bash
export PYTHONPATH=lib/bindings/python/src:components/src
```

If the replay search uses multiple worker processes, prefer a real script file over a heredoc. This
matters on macOS because `ProcessPoolExecutor` child workers need a stable module path.

For KV-router replay logs, this filter keeps the run readable without hiding useful `info` output:

```bash
export DYN_LOG='info,dynamo_kv_router::scheduling::selector=warn'
```

## Experiment Setup

This sweep uses:

- model: `Qwen/Qwen3-32B`
- backend: `vllm`
- system: `h200_sxm`
- router mode: `kv_router`
- workload type: `SyntheticReplayWorkload`
- GPU budget: `16`

The synthetic workload is intentionally large enough to make worker allocation and router settings
matter:

- `isl=32768`
- `osl=256`
- `request_count=5000`
- `replay_concurrency=200`
- `shared_prefix_ratio=0.5`
- `num_prefix_groups=50`

The base engine args stay conservative:

- `block_size=512`
- `num_gpu_blocks=20000`
- `enable_prefix_caching=True`
- explicit `worker_type` for prefill vs decode

This setup does not force scheduler-specific bottlenecks such as:

- `enable_chunked_prefill`
- a small `max_num_seqs`
- a pinned `max_num_batched_tokens`

Only add those when the experiment is specifically about scheduler limits.

## Driver Script

Write the driver to a stable temp file such as `/tmp/dynamo_replay_kv_router_disagg_sweep.py`:

Treat this script as a starting point, not a frozen harness. Modify it as needed for your search:

- change the workload shape
- swap `SyntheticReplayWorkload` for `TraceReplayWorkload`
- change constraints
- change `overlap_score_weights`
- print different columns from `result.evaluated_df` or `result.feasible_df`
- persist the tables to CSV or parquet if you want downstream analysis

If you need to understand which knobs are available, see [models.py](models.py), [search.py](search.py),
and [evaluate.py](evaluate.py).

```python
from dynamo.llm import KvRouterConfig, MockEngineArgs
from dynamo.profiler.utils.replay_optimize import (
    SyntheticReplayWorkload,
    optimize_dense_disagg_with_replay,
)

result = optimize_dense_disagg_with_replay(
    model="Qwen/Qwen3-32B",
    backend="vllm",
    system="h200_sxm",
    workload=SyntheticReplayWorkload(
        isl=32768,
        osl=256,
        request_count=5000,
        replay_concurrency=200,
        shared_prefix_ratio=0.5,
        num_prefix_groups=50,
    ),
    base_prefill_engine_args=MockEngineArgs(
        block_size=512,
        num_gpu_blocks=20000,
        enable_prefix_caching=True,
        worker_type="prefill",
    ),
    base_decode_engine_args=MockEngineArgs(
        block_size=512,
        num_gpu_blocks=20000,
        enable_prefix_caching=True,
        worker_type="decode",
    ),
    base_router_config=KvRouterConfig(),
    max_total_gpus=16,
    constraints={
        "mean_ttft_ms": 50000.0,
        "mean_tpot_ms": 100.0,
        "mean_e2e_latency_ms": 60000.0,
    },
    overlap_score_weights=[0.0, 0.5, 1.0, 2.0],
)

cols = [
    "prefill_tp",
    "decode_tp",
    "prefill_workers",
    "decode_workers",
    "overlap_score_weight",
    "total_gpus_used",
    "output_throughput_tok_s",
    "prefix_cache_reused_ratio",
    "mean_ttft_ms",
    "mean_tpot_ms",
    "mean_e2e_latency_ms",
]

print("Best feasible:")
print(result.best_feasible)
print()

print("Top feasible states:")
print(result.feasible_df[cols].head(10).to_string(index=False))
```

## Run Command

From the repo root:

```bash
cat >/tmp/dynamo_replay_kv_router_disagg_sweep.py <<'PYCODE'
from dynamo.llm import KvRouterConfig, MockEngineArgs
from dynamo.profiler.utils.replay_optimize import (
    SyntheticReplayWorkload,
    optimize_dense_disagg_with_replay,
)

result = optimize_dense_disagg_with_replay(
    model="Qwen/Qwen3-32B",
    backend="vllm",
    system="h200_sxm",
    workload=SyntheticReplayWorkload(
        isl=32768,
        osl=256,
        request_count=5000,
        replay_concurrency=200,
        shared_prefix_ratio=0.5,
        num_prefix_groups=50,
    ),
    base_prefill_engine_args=MockEngineArgs(
        block_size=512,
        num_gpu_blocks=20000,
        enable_prefix_caching=True,
        worker_type="prefill",
    ),
    base_decode_engine_args=MockEngineArgs(
        block_size=512,
        num_gpu_blocks=20000,
        enable_prefix_caching=True,
        worker_type="decode",
    ),
    base_router_config=KvRouterConfig(),
    max_total_gpus=16,
    constraints={
        "mean_ttft_ms": 50000.0,
        "mean_tpot_ms": 100.0,
        "mean_e2e_latency_ms": 60000.0,
    },
    overlap_score_weights=[0.0, 0.5, 1.0, 2.0],
)

cols = [
    "prefill_tp",
    "decode_tp",
    "prefill_workers",
    "decode_workers",
    "overlap_score_weight",
    "total_gpus_used",
    "output_throughput_tok_s",
    "prefix_cache_reused_ratio",
    "mean_ttft_ms",
    "mean_tpot_ms",
    "mean_e2e_latency_ms",
]

print("Best feasible:")
print(result.best_feasible)
print()

print("Top feasible states:")
print(result.feasible_df[cols].head(10).to_string(index=False))
PYCODE

DYN_LOG='info,dynamo_kv_router::scheduling::selector=warn' \
PYTHONPATH=lib/bindings/python/src:components/src \
  .venv/bin/python /tmp/dynamo_replay_kv_router_disagg_sweep.py
```

## Expected Outputs

The returned object is a `DenseReplayOptimizationResult` with:

- `best_feasible`: best visited state that satisfies all constraints
- `best_infeasible`: best visited state that misses at least one constraint
- `evaluated_df`: all visited states
- `feasible_df`: only the feasible visited states

Useful columns to inspect:

- topology: `prefill_tp`, `decode_tp`, `prefill_workers`, `decode_workers`
- routing: `router_mode`, `overlap_score_weight`
- budget: `total_gpus_used`
- throughput: `output_throughput_tok_s`
- cache behavior: `prefix_cache_reused_ratio`
- latency: `mean_ttft_ms`, `mean_tpot_ms`, `mean_e2e_latency_ms`

In local testing, this setup produced a non-trivial mean-E2E winner around:

- `prefill_tp=2`
- `decode_tp=1`
- `prefill_workers=2`
- `decode_workers=4`
- `overlap_score_weight=0.5`

Ballpark metrics for that point were:

- `prefix_cache_reused_ratio ~= 0.5`
- `output_throughput_tok_s ~= 4500`
- `mean_ttft_ms ~= 4500`
- `mean_tpot_ms ~= 26`
- `mean_e2e_latency_ms ~= 11150`

Treat those as sanity-check ranges, not fixed assertions.

## Tuning This Sweep

To broaden or shift the search, vary one axis at a time:

- `max_total_gpus`
- `overlap_score_weights`
- `shared_prefix_ratio`
- `num_prefix_groups`
- base prefill/decode engine args

If you want to compare routing strategies directly, use `router_mode="both"` instead of the default
KV-router-only search.

## Real Traffic Replay

`replay_optimize` is wired up for trace-driven replay. In
[evaluate.py](evaluate.py), `TraceReplayWorkload` goes through `run_trace_replay(...)`, while
`SyntheticReplayWorkload` goes through `run_synthetic_trace_replay(...)`.

Use a separate trace-driven experiment when you want to evaluate the same search structure against a
real Mooncake-style workload instead of the synthetic shared-prefix workload above.

### Download a Mooncake Trace

For a public starting point, use the FAST'25 toolagent trace:

```bash
curl -sL \
  https://raw.githubusercontent.com/kvcache-ai/Mooncake/refs/heads/main/FAST25-release/traces/toolagent_trace.jsonl \
  -o /tmp/toolagent_trace.jsonl
```

```bash
wget -O /tmp/toolagent_trace.jsonl \
  https://raw.githubusercontent.com/kvcache-ai/Mooncake/refs/heads/main/FAST25-release/traces/toolagent_trace.jsonl
```

If you want to slow the arrival process to `0.80x` of the original trace rate:

```bash
.venv/bin/python - <<'PY'
import json

with open("/tmp/toolagent_trace.jsonl", encoding="utf-8") as src, open(
    "/tmp/toolagent_trace_080x.jsonl", "w", encoding="utf-8"
) as dst:
    for line in src:
        rec = json.loads(line)
        rec["timestamp"] = int(rec["timestamp"] / 0.80)
        dst.write(json.dumps(rec) + "\n")
PY
```

### Replace the Synthetic Workload

In the main driver, replace:

```python
workload=SyntheticReplayWorkload(
    isl=32768,
    osl=256,
    request_count=5000,
    replay_concurrency=200,
    shared_prefix_ratio=0.5,
    num_prefix_groups=50,
),
```

with:

```python
from dynamo.profiler.utils.replay_optimize import TraceReplayWorkload

workload=TraceReplayWorkload(
    trace_file="/tmp/toolagent_trace.jsonl",
    arrival_speedup_ratio=1.0,
),
```

or, if you rewrote the timestamps:

```python
workload=TraceReplayWorkload(
    trace_file="/tmp/toolagent_trace_080x.jsonl",
    arrival_speedup_ratio=1.0,
),
```

The main behavioral change is that the workload stops generating requests in memory and instead
replays request arrivals from the JSONL trace. In this path:

- `trace_file` points at the Mooncake-style JSONL input
- `arrival_speedup_ratio` compresses or stretches the trace arrival process
- synthetic-only knobs such as `isl`, `osl`, `request_count`, `replay_concurrency`,
  `shared_prefix_ratio`, and `num_prefix_groups` no longer apply at the workload level

Important notes for the public toolagent trace:

- the dataset uses Mooncake-style `hash_ids` with `512` tokens per block
- the underlying `run_trace_replay(...)` API defaults `trace_block_size` to `512`
- the current `TraceReplayWorkload` wrapper does not expose a separate `trace_block_size` field
- the prefix-data-generator tools in
  [Prefix Data Generator](../../../../../../benchmarks/prefix_data_generator/README.md)
  are useful if you want to inspect the trace first or synthesize a larger derivative trace before
  running this search

So this path is a good fit for the standard public Mooncake/toolagent trace as-is. If you need a
different dataset block size, extend the replay-optimize workload/evaluation path rather than
assuming a non-`512` trace will be interpreted correctly.
