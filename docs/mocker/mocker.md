---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Mocker
---

The Mocker is a lightweight, high-fidelity simulation of an LLM inference engine, implemented entirely in Rust. It replicates the core scheduling, memory management, and timing behaviors of production engines without requiring a GPU, making it invaluable for testing Dynamo's routing, KV cache events, disaggregated serving, and planner components.

## Overview

The mocker simulates:

- **Block-based KV cache management** with LRU eviction
- **Continuous batching scheduler** with watermark-based admission control
- **Prefix caching** with hash-based block deduplication
- **Chunked prefill** for better batching efficiency
- **Realistic timing models** for prefill and decode phases
- **Disaggregated serving** (prefill/decode separation)
- **KV event publishing** for router integration
- **Data parallelism** (multiple DP ranks per engine)

> **Note:** While the mocker uses vLLM as its primary reference implementation, these core components—block-based KV cache management, continuous batching schedulers, LRU evictors, and prefix caching—are fundamental to all modern LLM inference engines, including SGLang and TensorRT-LLM. The architectural patterns simulated here are engine-agnostic and apply broadly across the inference ecosystem.

## Quick Start

### Basic Usage

```bash
# Launch a single mocker worker
python -m dynamo.mocker --model-path Qwen/Qwen3-0.6B

# Launch with custom KV cache configuration
python -m dynamo.mocker \
    --model-path Qwen/Qwen3-0.6B \
    --num-gpu-blocks-override 8192 \
    --block-size 64 \
    --max-num-seqs 256

# Launch with timing speedup for faster testing
python -m dynamo.mocker \
    --model-path Qwen/Qwen3-0.6B \
    --speedup-ratio 10.0
```

### Disaggregated Serving

```bash
# Launch prefill worker
python -m dynamo.mocker \
    --model-path Qwen/Qwen3-0.6B \
    --disaggregation-mode prefill \
    --bootstrap-ports 50100

# Launch decode worker (in another terminal)
python -m dynamo.mocker \
    --model-path Qwen/Qwen3-0.6B \
    --disaggregation-mode decode
```

### Multiple Workers in One Process

```bash
# Launch 4 mocker workers sharing the same tokio runtime
python -m dynamo.mocker \
    --model-path Qwen/Qwen3-0.6B \
    --num-workers 4
```

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-path` | Required | HuggingFace model ID or local path for tokenizer |
| `--endpoint` | Auto-derived | Dynamo endpoint string. Defaults are namespace-dependent, and prefill workers use a different default endpoint than aggregated/decode workers |
| `--model-name` | Derived from model-path | Model name for API responses |
| `--trace-file` | None | Run offline trace replay from a Mooncake-style JSONL trace file |
| `--output-file` | `<trace stem>.replay.json` | Write replay metrics JSON to this path |
| `--replay-concurrency` | None | Run offline replay in closed-loop concurrency mode with this many in-flight requests |
| `--num-gpu-blocks-override` | 16384 | Number of KV cache blocks |
| `--block-size` | 64 | Tokens per KV cache block |
| `--max-num-seqs` | 256 | Maximum concurrent sequences |
| `--max-num-batched-tokens` | 8192 | Maximum tokens per batch |
| `--enable-prefix-caching` | True | Enable prefix caching |
| `--no-enable-prefix-caching` | - | Disable prefix caching |
| `--enable-chunked-prefill` | True | Enable chunked prefill |
| `--no-enable-chunked-prefill` | - | Disable chunked prefill |
| `--preemption-mode` | `lifo` | Decode eviction policy under memory pressure: `lifo` (vLLM v1 style) or `fifo` |
| `--watermark` | 0.01 | KV cache watermark (fraction reserved) |
| `--speedup-ratio` | 1.0 | Timing speedup factor |
| `--decode-speedup-ratio` | 1.0 | Decode-only speedup multiplier (e.g. for Eagle speculation) |
| `--data-parallel-size` | 1 | Number of DP replicas |
| `--startup-time` | None | Simulated startup delay (seconds) |
| `--planner-profile-data` | None | Path to either a mocker-format `.npz` file or a profiler results directory |
| `--num-workers` | 1 | Workers per process |
| `--reasoning` | None | JSON config for emitting reasoning token spans, with `start_thinking_token_id`, `end_thinking_token_id`, and `thinking_ratio` |
| `--engine-type` | `vllm` | Engine simulation type: `vllm` or `sglang` |
| `--sglang-schedule-policy` | `fifo` / `fcfs` | SGLang scheduling policy override |
| `--sglang-page-size` | 1 | SGLang radix-cache page size in tokens |
| `--sglang-max-prefill-tokens` | 16384 | SGLang max prefill-token budget per batch |
| `--sglang-chunked-prefill-size` | 8192 | SGLang chunked-prefill chunk size |
| `--sglang-clip-max-new-tokens` | 4096 | SGLang admission-budget cap for max new tokens |
| `--sglang-schedule-conservativeness` | 1.0 | SGLang schedule conservativeness factor |
| `--aic-perf-model` | False | Use AIC SDK for latency prediction instead of interpolated/polynomial models. Requires `aiconfigurator` SDK installed (install with `pip install ai-dynamo[mocker]`) |
| `--aic-system` | `h200_sxm` | AIC system name (e.g., `h200_sxm`). Used with `--aic-perf-model` |
| `--aic-backend-version` | Auto | AIC backend engine version (e.g., `0.12.0` for vLLM). If not set, uses the default version for the backend |
| `--aic-tp-size` | 1 | Tensor parallel size for AIC latency prediction. Only affects AIC performance model lookups, not mocker scheduling |
| `--extra-engine-args` | None | Path to a JSON file with mocker configuration; overrides individual CLI arguments |
| `--stagger-delay` | -1 (auto) | Delay between worker launches (seconds). 0 disables, -1 enables auto mode |
| `--disaggregation-mode` | `agg` | Worker mode: `agg` (aggregated), `prefill`, or `decode` |
| `--durable-kv-events` | False | Deprecated JetStream KV-event mode; prefer the local indexer / event-plane subscriber path |
| `--zmq-kv-events-ports` | None | Comma-separated ZMQ PUB base ports for KV event publishing, one per worker |
| `--zmq-replay-ports` | None | Comma-separated ZMQ ROUTER base ports for gap recovery, one per worker |
| `--bootstrap-ports` | None | Comma-separated rendezvous base ports, one per worker in disaggregated mode |
| `--kv-transfer-bandwidth` | 64.0 | KV cache transfer bandwidth in GB/s. Set to 0 to disable |
| `--kv-cache-dtype` | auto | KV cache dtype for bytes-per-token computation |
| `--kv-bytes-per-token` | Auto-computed | KV cache bytes per token (override auto-computation) |
| `--discovery-backend` | Env-driven (`etcd`) | Discovery backend: `kubernetes`, `etcd`, `file`, or `mem` |
| `--request-plane` | Env-driven (`tcp`) | Request transport: `nats`, `http`, or `tcp` |
| `--event-plane` | Env-driven (`nats`) | Event transport: `nats` or `zmq` |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DYN_MOCKER_KV_CACHE_TRACE` | off | Set to `1` or `true` to log structured KV cache allocation and eviction traces |

> **Note:** For local scale tests and router benchmarks, prefer `--num-workers` over launching many separate mocker processes. All workers share one tokio runtime and thread pool, which is both lighter weight and closer to how the test harnesses exercise the mocker.

## Offline Trace Replay

The mocker also supports an offline replay mode for Mooncake-style traces:

```bash
python -m dynamo.mocker \
    --trace-file /path/to/mooncake_trace.jsonl \
    --model-path Qwen/Qwen3-0.6B
```

This mode writes a replay report JSON and prints a `Replay Summary` table without launching a runtime or router.

For full usage, constraints, and benchmarking guidance, see [Mocker Offline Trace Replay](../benchmarks/mocker-trace-replay.md).

## Performance Modeling Setup

By default, the mocker uses hardcoded polynomial formulas to estimate prefill and decode timing. For more realistic simulations, pass `--planner-profile-data` with either:

- a mocker-format `.npz` file, or
- a profiler output directory

The mocker automatically accepts profiler-style results directories and converts them internally.

It also accepts older raw-data directories containing:

- `prefill_raw_data.json`
- `decode_raw_data.json`

```bash
python -m dynamo.mocker \
    --model-path nvidia/Llama-3.1-8B-Instruct-FP8 \
    --planner-profile-data tests/planner/profiling_results/H200_TP1P_TP1D \
    --speedup-ratio 1.0
```

### AIC Performance Model

To use the AIC SDK for latency prediction:

```bash
pip install ai-dynamo[mocker]

python -m dynamo.mocker \
    --model-path nvidia/Llama-3.1-8B-Instruct-FP8 \
    --engine-type vllm \
    --aic-perf-model \
    --aic-system h200_sxm
```

The AIC model automatically uses `--model-path` and `--engine-type` to select the appropriate performance data. Available systems include `h200_sxm`, `h100_sxm`, etc. (see AIC SDK documentation for the full list).

Example `--reasoning` configuration:

```bash
python -m dynamo.mocker \
    --model-path Qwen/Qwen3-0.6B \
    --reasoning '{"start_thinking_token_id":123,"end_thinking_token_id":456,"thinking_ratio":0.6}'
```

The profile results directory should contain:

- `selected_prefill_interpolation/raw_data.npz`
- `selected_decode_interpolation/raw_data.npz`

To generate profile data for your own model and hardware, run the profiler and then point `--planner-profile-data` at the resulting output directory.

## Event Transport and Router Testing

The default event path uses the local indexer / event-plane subscriber flow. The older durable KV-events mode is still available through `--durable-kv-events`, but it is deprecated and should not be the preferred setup for new tests.

For router and indexer experiments that need native wire-format event forwarding, the mocker also supports a ZMQ path:

- `--event-plane zmq`
- `--zmq-kv-events-ports` for per-worker PUB base ports
- `--zmq-replay-ports` for optional replay/gap-recovery ROUTER base ports

When set, each worker binds on its base port plus `dp_rank`, so the number of comma-separated base ports must match `--num-workers`.

## Disaggregation Port Layout

`--bootstrap-ports` takes a comma-separated list of base ports, one per worker. In multi-worker mode, the number of listed ports must exactly match `--num-workers`.

Prefill workers listen on these ports and publish the bootstrap endpoint through discovery. Decode workers use the matching ports to rendezvous before decode begins.

## Kubernetes Deployment

The mocker can be deployed through example `DynamoGraphDeployment` manifests for both aggregated and disaggregated setups:

```bash
kubectl apply -f examples/backends/mocker/deploy/agg.yaml
kubectl apply -f examples/backends/mocker/deploy/disagg.yaml
```

## Architecture

The mocker is organized into several cooperating components that mirror the internal architecture of production LLM inference engines.

### Scheduler

The scheduler implements continuous batching, maintaining three logical queues:

1. **Waiting Queue** - Newly arrived requests awaiting scheduling
2. **Prefill Queue** - Requests scheduled for prefill
3. **Decode Queue** - Requests actively decoding (ordered by age for preemption)

Each iteration, the scheduler receives incoming requests, moves eligible requests from waiting to prefill based on available memory and compute budgets, simulates the prefill phase for queued requests, runs one decode step for all active sequences, and publishes metrics about current resource utilization.

When resources become constrained, the scheduler employs preemption: the oldest decoding request is evicted back to the waiting queue, its KV blocks are freed, and it will be rescheduled later. This mirrors how real engines handle memory pressure.

### KV Block Manager

The block manager tracks KV cache blocks using reference counting and an LRU eviction policy. Blocks exist in one of two pools:

- **Active Pool** - Blocks currently in use by one or more sequences, tracked with reference counts
- **Inactive Pool** - Blocks no longer actively referenced but kept for potential reuse (prefix caching)

When a sequence needs blocks, the manager first checks if they already exist (cache hit). If not, it allocates new blocks, potentially evicting the least-recently-used inactive blocks to make room. When a sequence completes or is preempted, its blocks are either moved to the inactive pool (for potential reuse) or freed entirely.

The following diagram illustrates the block lifecycle, based on vLLM's block manager design:

```mermaid
stateDiagram-v2
    [*] --> Active : alloc
    Active --> Inactive : deref
    Inactive --> Active : cache hit (reuse)
    Inactive --> Freed : evict
    Active --> Freed : destroy (preemption)
    Freed --> [*]

    state Active {
        [*] --> Tracked : ref_count tracked
    }
    state Inactive {
        [*] --> Ordered : LRU order
    }
```

### Evictor

The LRU evictor maintains blocks ordered by a monotonic counter, enabling O(log n) eviction of the lowest-priority block. Each `insert` assigns the next counter value, so blocks inserted later have higher counters and survive longer.

This produces a **depth-aware eviction policy**: when a sequence completes, `free_signal` releases its blocks in reverse order (tail first). Deeper suffix blocks therefore receive lower counters and are evicted before shallower prefix blocks. This keeps shared prefixes cached longer, improving cache hit rates across requests with common prefixes.

The evictor also supports front-insertion (negative counters) for marking blocks for immediate eviction, though this is not currently used in the scheduler.

### Sequence Tracking

Each active request is tracked as a sequence, managing its token blocks and generation state. As tokens are generated, the sequence tracks which blocks are partial (still being filled) versus full (complete and hashable for prefix caching). When a partial block fills up, it gets "promoted" to a full block with a content-based hash, enabling future cache hits from requests with matching prefixes.

### Performance Model

The mocker supports three timing prediction modes:

**Polynomial Model (Default):** Uses hardcoded polynomial formulas that approximate typical GPU behavior. Prefill time scales quadratically with token count, while decode time depends on the total active KV cache size.

**Interpolated Model:** Loads actual profiling data from an NPZ file containing measured prefill and decode latencies. The mocker interpolates between data points to predict timing for any input size. This enables high-fidelity simulation matching a specific hardware configuration.

**AIC Model (`--aic-perf-model`):** Uses the NVIDIA AI Configurator (AIC) SDK for latency prediction. AIC provides calibrated performance models for specific GPU/model/engine combinations, predicting prefill and decode latency as a function of batch size, sequence length, and prefix cache hits. The model path is automatically derived from `--model-path`, and the engine type from `--engine-type`. This mode requires the `aiconfigurator` SDK, installable via `pip install ai-dynamo[mocker]`.

### Bootstrap Rendezvous (Disaggregated Serving)

For disaggregated prefill/decode deployments, prefill and decode workers coordinate via a simple TCP-based rendezvous protocol. The decode worker connects to the prefill worker's bootstrap port and waits until the prefill phase completes and KV cache is ready. Either side can arrive first—the rendezvous completes when both are ready.

### KV Transfer Latency Simulation

The mocker simulates KV cache transfer time between prefill and decode workers. Before the prefill worker emits its first (and only) token, it sleeps for a duration based on:

- **kv_bytes_per_token** (auto-computed from model config): `num_layers * 2 * num_kv_heads * head_dim * dtype_bytes`. The `dtype_bytes` is determined by `--kv-cache-dtype`: when set to `auto` (default), it uses the model's `dtype` from config; when explicitly set (e.g., `fp8`), it uses the specified dtype instead. It can also be overridden directly with `--kv-bytes-per-token`.
- **kv_transfer_bandwidth** (default: 64.0 GB/s, inter-node InfiniBand)
- **Transfer time**: `num_input_tokens * kv_bytes_per_token / bandwidth`

This delay is injected after the scheduler's prefill compute simulation completes, modeling the sequential flow: prefill computation → KV transfer → decode begins. Set `--kv-transfer-bandwidth 0` to disable.

## Integration with Dynamo

### KV Event Publishing

When prefix caching is enabled, the mocker publishes KV cache events to the distributed runtime. These events notify the system when blocks are stored (new content cached) or removed (evicted). This enables the KV-aware router to make intelligent routing decisions based on which workers have which prefixes cached.

### Metrics Publishing

Each scheduler publishes metrics about its current state, including the number of active decode blocks per DP rank. The router uses these metrics for load-aware routing decisions.

## Testing Scenarios

The mocker is particularly useful for:

1. **Router Testing** - Validate KV-aware routing without GPUs
2. **Planner Testing** - Test SLA-based planners with realistic timing
3. **Fault Tolerance** - Test request migration, graceful shutdown
4. **Disaggregation** - Test P/D separation and KV transfer coordination
5. **Performance Modeling** - Prototype scheduling policies
6. **CI/CD** - Fast integration tests without hardware dependencies

## Comparison with Real Engines

| Feature | Real Engine | Mocker |
|---------|-------------|--------|
| GPU Required | Yes | No |
| Block Manager | Paged KV cache | Simulated blocks |
| Scheduler | Continuous batching | Continuous batching |
| Prefix Caching | Hash-based | Hash-based |
| Chunked Prefill | Supported | Supported |
| Preemption | Recompute/swap | Recompute (simulated) |
| Timing | Real execution | Model-based |
| KV Events | Native | Compatible |
| Data Parallelism | Multi-GPU | Simulated |

## Next Steps

| Document | Description |
|----------|-------------|
| [Benchmarking Dynamo Deployments](../benchmarks/benchmarking.md) | Run AIPerf against a mocker-backed deployment to measure latency, TTFT, throughput, and scaling behavior |
| [Aggregated Mocker Deployment Example](../../examples/backends/mocker/deploy/agg.yaml) | Deploy a mocker-backed aggregated DynamoGraphDeployment on Kubernetes |
| [Disaggregated Mocker Deployment Example](../../examples/backends/mocker/deploy/disagg.yaml) | Deploy separate prefill and decode mocker workers for disaggregated-serving benchmarks |
| [Global Planner Mocker Example](../../examples/global_planner/global-planner-mocker-test.yaml) | Advanced multi-pool mocker setup for planner and global-router experiments |

## Feature Gaps (WIP)

The following features are not yet supported by the mocker:

- **Multi-tier memory** - No support for offloading KV cache to CPU/disk or onboarding back to GPU; potential future integration with KVBM
- **Multimodal support** - Currently only simulates text token processing; no vision encoder or cross-attention simulation
- **Native Rust reference counting** - Work in progress to use native Rc/Arc for block reference counting, enabling natural RAII patterns for simpler tracking
