---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: DynoSim Simulation Model
subtitle: Engine behavior, timing sources, KV movement, and fidelity boundaries
---

DynoSim builds a distributed serving simulation from Mocker engine cores. Each core owns
engine-specific scheduler and KV-cache state. Offline prediction and the retained internal worker
runtime drive the same core through virtual-time and wall-clock execution, respectively.

> [!WARNING]
> The public online Mocker CLI is temporarily unavailable. The internal online behavior described
> here remains an implementation detail and will return through the unified AISimulate CLI in a
> future release.

## Engine Behavior

Mocker exposes three engine modes through two scheduler cores. The TensorRT-LLM mode uses the
vLLM-shaped core with a different capacity policy.

| Behavior | vLLM | SGLang | TensorRT-LLM |
|---|---|---|---|
| Scheduler model | Waiting and running queues with a shared token budget | Cache-aware waiting and running queues | vLLM-shaped core with capacity-first admission |
| KV representation | Native block pool | Token pool and radix cache | Native block pool |
| Memory pressure | LIFO or FIFO recompute preemption | Decode retraction with cached-prefix preservation | `GUARANTEED_NO_EVICT`; reserves prompt plus maximum output at admission |
| Prefix reuse | Block-hash matching when prefix caching is enabled | Radix-prefix matching | Block-hash matching when prefix caching is enabled |
| Default block or page size | 64 tokens | 1 token, or `sglang.page_size` | 32 tokens |
| Aggregated simulation | Supported | Supported | Supported |
| Prefill/decode disaggregation | Supported | Supported | Not supported |

Data-parallel ranks own independent scheduler and KV-pool state. The internal worker runtime and
offline prediction compose those ranks into one logical worker with a shared pass barrier.

### KV Managers

The shared vLLM/TensorRT-LLM core uses Mocker's self-contained physical block pool to model GPU KV
capacity, prefix reuse, request ownership, least-recently-used eviction, and router-visible KV
events. The SGLang core uses its own token-pool and radix-cache implementation.

## Timing Sources

Scheduler state determines the batch and cache-hit inputs to the timing source. Choose one timing
source for each worker role with `engine.workers.<role>.timing.type`.

The removed public Mocker CLI options for profile-derived interpolation and direct `--aic-*`
configuration are not part of the unified CLI. Use only the timing types accepted by the
AISimulate YAML schema.

### AISimulate Performance Model

`timing.type: default` uses the AISimulate performance model. AISimulate selects model data from
`engine.model`, `engine.hardware`, `engine.backend`, the optional `engine.backend_version`, and the
role's parallelism mapping. Unsupported combinations fail instead of silently using an
uncalibrated timing model.

KV-capacity selection is independent of the timing source. With `kv_cache.capacity.type: default`,
AISimulate estimates capacity from the model, backend, hardware, parallelism, block size, and memory
fraction. With `type: fixed`, set the concrete block count in `blocks`.

### Fixed Timing

Set `timing.type: fixed` and provide both `prefill_ms` and `decode_ms` to apply constant durations:

```yaml
timing: {type: fixed, prefill_ms: 2, decode_ms: 0.5}
```

Use fixed timing for functional tests or controlled comparisons. These values replace the
AISimulate timing lookup but do not disable default KV-capacity estimation.

### Polynomial Baseline

Set `timing.type: polynomial` to use the uncalibrated synthetic baseline. Prefill latency follows a
polynomial over the uncached tokens scheduled in the pass. Decode latency follows a polynomial over
active KV-cache utilization.

## Prefill/Decode Handoff

For `engine.mode: disaggregated`, define `engine.workers.prefill` and
`engine.workers.decode`. Configure the modeled transfer under `engine.kv_transfer`:

```yaml
kv_transfer:
  bytes_per_token: auto
  bandwidth_gb_per_second: 400
  timing_mode: destination_missing
```

The offline simulation uses a per-request line-rate model:

```text
transfer time = transferred KV bytes / kv_transfer_bandwidth
transferred KV bytes = charged tokens * kv_bytes_per_token
```

`timing_mode: full_prompt` charges the full logical prompt. `destination_missing` charges only the
prompt footprint absent from the destination and can produce zero delay on a full destination hit.
Concurrent requests do not contend for a shared link.

Set `bytes_per_token: auto` to derive the value from model metadata and parallelism, or provide a
positive integer. Set a positive `bandwidth_gb_per_second` to apply the transfer delay. TensorRT-LLM
does not support disaggregated simulation.

## Distributed Signals

The internal worker runtime publishes the same categories of signals consumed by the distributed
runtime:

- stored and removed KV events for KV-aware routing;
- engine-shaped Prometheus scheduler and request metrics;
- per-rank cache, queue, running-request, and preemption counters;
- forward-pass metrics with scheduled and queued prefill/decode work.

Offline KV-router replay captures the corresponding scheduler events and applies them to an
in-process indexer at deterministic event boundaries.

## Fidelity Boundaries

Interpret simulation results within these boundaries:

- Timing accuracy depends on the selected timing source and its calibration range.
- KV capacity and state transitions are modeled; KV tensor payloads are not.
- The internal worker runtime includes Dynamo component and transport overhead but does not measure GPU kernel or
  inference-engine overhead.
- Offline replay replaces external services and wall-clock concurrency with an event queue and
  shared logical clock.
- TensorRT-LLM disaggregation and multi-tier KV offload are not modeled.
- Mocker simulates text-token processing; it does not model multimodal encoder or cross-attention
  compute.

Use offline prediction for broad algorithm and configuration exploration. Validate the distributed
path with focused GPU benchmarks until online simulation returns.
