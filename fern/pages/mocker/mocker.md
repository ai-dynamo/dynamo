---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---

# Mocker: LLM Engine Simulation in Rust

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
    --is-prefill-worker \
    --bootstrap-ports 50100

# Launch decode worker (in another terminal)
python -m dynamo.mocker \
    --model-path Qwen/Qwen3-0.6B \
    --is-decode-worker
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
| `--endpoint` | `dyn://dynamo.backend.generate` | Dynamo endpoint string |
| `--model-name` | Derived from model-path | Model name for API responses |
| `--num-gpu-blocks-override` | 16384 | Number of KV cache blocks |
| `--block-size` | 64 | Tokens per KV cache block |
| `--max-num-seqs` | 256 | Maximum concurrent sequences |
| `--max-num-batched-tokens` | 8192 | Maximum tokens per batch |
| `--enable-prefix-caching` | True | Enable prefix caching |
| `--enable-chunked-prefill` | True | Enable chunked prefill |
| `--watermark` | 0.01 | KV cache watermark (fraction reserved) |
| `--speedup-ratio` | 1.0 | Timing speedup factor |
| `--data-parallel-size` | 1 | Number of DP replicas |
| `--startup-time` | None | Simulated startup delay (seconds) |
| `--planner-profile-data` | None | Path to NPZ file with timing data |
| `--num-workers` | 1 | Workers per process |
| `--stagger-delay` | -1 (auto) | Delay between worker launches (seconds). 0 disables, -1 enables auto mode |
| `--is-prefill-worker` | False | Prefill-only mode |
| `--is-decode-worker` | False | Decode-only mode |
| `--enable-local-indexer` | False | Enable local KV indexer |
| `--bootstrap-ports` | None | Ports for P/D rendezvous |

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

```
                        ┌───── Cache hit (Use) ────┐
                        │                          │
                        ▼                          │
┌───────────┐       ┌───────────┐       ┌──────────┴──────┐       ┌───────────┐
│ New Block │──────►│  Active   │──────►│    Inactive     │──────►│   Freed   │
└───────────┘ alloc │   Pool    │ deref │      Pool       │ evict └───────────┘
                    │(ref_count)│       │   (LRU order)   │
                    └─────┬─────┘       └─────────────────┘
                          │
                          │ destroy (preemption)
                          ▼
                    ┌───────────┐
                    │   Freed   │
                    └───────────┘
```

### Evictor

The LRU evictor maintains blocks ordered by their last access time, enabling O(1) eviction of the oldest unused block. It supports both normal insertion (for completed sequences) and front-insertion (for preempted sequences that should be evicted first if memory pressure continues).

### Sequence Tracking

Each active request is tracked as a sequence, managing its token blocks and generation state. As tokens are generated, the sequence tracks which blocks are partial (still being filled) versus full (complete and hashable for prefix caching). When a partial block fills up, it gets "promoted" to a full block with a content-based hash, enabling future cache hits from requests with matching prefixes.

### Performance Model

The mocker supports two timing prediction modes:

**Polynomial Model (Default):** Uses hardcoded polynomial formulas that approximate typical GPU behavior. Prefill time scales quadratically with token count, while decode time depends on the total active KV cache size.

**Interpolated Model:** Loads actual profiling data from an NPZ file containing measured prefill and decode latencies. The mocker interpolates between data points to predict timing for any input size. This enables high-fidelity simulation matching a specific hardware configuration.

### Bootstrap Rendezvous (Disaggregated Serving)

For disaggregated prefill/decode deployments, prefill and decode workers coordinate via a simple TCP-based rendezvous protocol. The decode worker connects to the prefill worker's bootstrap port and waits until the prefill phase completes and KV cache is ready. Either side can arrive first—the rendezvous completes when both are ready.

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

## Feature Gaps (WIP)

The following features are not yet supported by the mocker:

- **KV transfer latency simulation** - Disaggregated serving simulates the rendezvous handshake but does not model the actual KV cache transfer time between prefill and decode workers
- **Multi-tier memory** - No support for offloading KV cache to CPU/disk or onboarding back to GPU; potential future integration with KVBM
- **Multimodal support** - Currently only simulates text token processing; no vision encoder or cross-attention simulation
- **Native Rust reference counting** - Work in progress to use native Rc/Arc for block reference counting, enabling natural RAII patterns for simpler tracking
