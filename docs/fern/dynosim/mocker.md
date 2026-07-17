---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Live Simulation with Mocker
---

Mocker is the live simulated engine in DynoSim. It runs as a Dynamo backend, registers workers, publishes KV events, and exercises the real frontend/router/planner path without requiring GPUs.

The mocker core is implemented in Rust and models the scheduling, memory management, and timing behavior of production engines. It can use polynomial timing, profile-derived timing, or AIC-backed timing. AIC predicts prefill/decode duration; Mocker still owns the scheduler, KV cache lifecycle, prefix-cache behavior, and request execution model.

This page covers running the live worker. For the flag reference, see the [Mocker CLI Reference](../components/mocker/mocker-cli-reference.mdx); for the engine internals, see [Mocker Engine Architecture](../design-docs/mocker-architecture.md); for offline, GPU-free batch simulation, see [DynoSim Runs](runs.md) and [DynoSim Sweeps](sweeps.md).

## Overview

The mocker simulates:

- **Block-based KV cache management** with LRU eviction
- **Engine-specific continuous batching schedulers** for vLLM and SGLang
- **Prefix caching** with hash-based block deduplication
- **Chunked prefill** for better batching efficiency
- **Realistic timing models** for prefill and decode phases
- **Disaggregated serving** (prefill/decode separation)
- **KV event publishing** for router integration
- **Data parallelism** (multiple DP ranks per engine)

> **Note:** While the mocker uses vLLM as its primary reference implementation, these core components—block-based KV cache management, continuous batching schedulers, LRU evictors, and prefix caching—are fundamental to all modern LLM inference engines, including SGLang and TensorRT-LLM. The architectural patterns simulated here are engine-agnostic and apply broadly across the inference ecosystem.

## Quick Start

The mocker runs as a real Dynamo worker: it registers with the frontend, publishes KV events, and exercises the router and planner paths. Deploy it as a DGD to simulate a live Dynamo deployment without GPUs, or launch it locally for quick iteration.

### On Kubernetes (DGD)

Deploy a Frontend plus a `dynamo.mocker` worker. The mocker ships in the `dynamo-planner` image (no GPU resources requested). This DGD is adapted from [`examples/backends/mocker/deploy/v1beta1/agg.yaml`](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/mocker/deploy/v1beta1/agg.yaml):

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: mocker-agg
spec:
  components:
  - name: Frontend
    type: frontend
    replicas: 1
    podTemplate:
      spec:
        containers:
        - name: main
          image: ${PLANNER_IMAGE}       # nvcr.io/nvidia/ai-dynamo/dynamo-planner:<tag>
  - name: decode
    type: decode
    replicas: 1
    podTemplate:
      spec:
        containers:
        - name: main
          image: ${PLANNER_IMAGE}
          workingDir: /workspace
          envFrom:
          - secretRef:
              name: hf-token-secret
          command:
          - python3
          - -m
          - dynamo.mocker
          args:
          - --model-path
          - Qwen/Qwen3-0.6B
          - --speedup-ratio
          - "1.0"
```

Apply it, then port-forward the Frontend (`kubectl port-forward svc/mocker-agg-frontend 8000:8000 -n ${NAMESPACE}`) and send OpenAI-compatible requests as you would to a real backend. For a disaggregated simulation, add a second component with `--disaggregation-mode prefill` (see [`.../deploy/v1beta1/disagg.yaml`](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/mocker/deploy/v1beta1/disagg.yaml)).

### Locally

For no-cluster iteration, launch the mocker directly. The same flags apply.

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

### Disaggregated Serving (local)

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

### Multiple Workers in One Process (local)

```bash
# Launch 4 mocker workers sharing the same tokio runtime
python -m dynamo.mocker \
    --model-path Qwen/Qwen3-0.6B \
    --num-workers 4
```

> **Note:** For local scale tests and router benchmarks, prefer `--num-workers` over launching many separate mocker processes. All workers share one tokio runtime and thread pool, which is both lighter weight and closer to how the test harnesses exercise the mocker.

## CLI Reference

For the complete list of command-line flags — model and KV cache settings, scheduling, timing, disaggregation, AIC, and transport — see the [Mocker CLI Reference](../components/mocker/mocker-cli-reference.mdx). To load a full configuration from a file instead of individual flags, pass `--extra-engine-args` pointing at a JSON file.

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
    --planner-profile-data components/src/dynamo/planner/tests/data/profiling_results/H200_TP1P_TP1D \
    --speedup-ratio 1.0
```

The profile results directory should contain:

- `selected_prefill_interpolation/raw_data.npz`
- `selected_decode_interpolation/raw_data.npz`

To generate profile data for your own model and hardware, run the profiler and then point `--planner-profile-data` at the resulting output directory.

### AIC Performance Model

To use the AIC SDK for latency prediction:

```bash
uv pip install '.[mocker]'

python -m dynamo.mocker \
    --model-path nvidia/Llama-3.1-8B-Instruct-FP8 \
    --engine-type vllm \
    --aic-perf-model \
    --aic-system h200_sxm
```

The AIC model automatically uses `--model-path` and `--engine-type` to select the appropriate performance data. Available systems include `h200_sxm`, `h100_sxm`, etc. (see AIC SDK documentation for the full list).

Important notes:

- AIC is opt-in. If you do not pass `--aic-perf-model`, `python -m dynamo.mocker` does not use AIC.
- **Pure-Rust callback.** When AIC is enabled, the mocker builds an `aiconfigurator_core::AicEngine` once at startup and answers per-step prefill/decode latency predictions from Rust with no GIL on the hot path (this is what lets predictions scale across threads in the live/concurrent path). It requires a build with the `aic-forward-pass` feature (release wheels enable it). There is no Python fallback: if the Rust engine cannot be built for the requested model/system/backend, mocker fails fast with a clear error rather than silently degrading to the slower GIL-bound Python op-walk. (`aiconfigurator`'s `compile_engine` covers every supported config, so a build failure indicates a real problem — missing perf data or an unsupported config.)
- `aiconfigurator` must be able to load the requested performance database for the selected `system/backend/version`. If the SDK is installed but the backing systems data is missing or unreadable, mocker fails fast at startup with a clear error instead of failing later on first request.
- In development environments, this may require pointing Python at a source checkout of `aiconfigurator` with real Git LFS payloads materialized in its `systems/` directory.

This mocker AIC path configures the simulated worker's own timing model. It is separate from the router-side prefill-load estimator and from the engine-timing AIC surface used by DynoSim runs; for those, see [DynoSim Runs](runs.md).

Example `--reasoning` configuration for emitting reasoning token spans:

```bash
python -m dynamo.mocker \
    --model-path Qwen/Qwen3-0.6B \
    --reasoning '{"start_thinking_token_id":123,"end_thinking_token_id":456,"thinking_ratio":0.6}'
```

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

## DynoSim Runs and Sweeps

The mocker also powers offline, GPU-free batch simulation through the `python -m dynamo.replay` CLI. Use [DynoSim Runs](runs.md) to drive one trace or synthetic workload through a simulated configuration and get an AIPerf-style report, and [DynoSim Sweeps](sweeps.md) to search many candidate topologies and router settings against SLA and GPU-budget constraints. Both reuse the same mocker engine and accept these engine settings as JSON through `--extra-engine-args` rather than as individual CLI flags.

## Architecture

The mocker's engine internals — the vLLM- and SGLang-style schedulers, the kvbm-logical block manager and eviction backends, sequence tracking, the three timing models, and the disaggregated bootstrap and KV-transfer simulation — are documented in [Mocker Engine Architecture](../design-docs/mocker-architecture.md).

## Testing Scenarios

The mocker is particularly useful for:

1. **Router Testing** - Validate KV-aware routing without GPUs
2. **Planner Testing** - Test SLA-based planners with realistic timing
3. **Fault Tolerance** - Test request migration, graceful shutdown
4. **Disaggregation** - Test P/D separation and KV transfer coordination
5. **Performance Modeling** - Prototype scheduling policies
6. **CI/CD** - Fast integration tests without hardware dependencies

## Next Steps

| Document | Description |
|----------|-------------|
| [Benchmarking Dynamo Deployments](../benchmarks/benchmarking.md) | Run AIPerf against a mocker-backed deployment to measure latency, TTFT, throughput, and scaling behavior |
| [Aggregated Mocker Deployment Example](../../examples/backends/mocker/deploy/agg.yaml) | Deploy a mocker-backed aggregated DynamoGraphDeployment on Kubernetes |
| [Disaggregated Mocker Deployment Example](../../examples/backends/mocker/deploy/disagg.yaml) | Deploy separate prefill and decode mocker workers for disaggregated-serving benchmarks |
| [Global Planner Mocker Example](../../examples/global_planner/global-planner-mocker-test.yaml) | Advanced multi-pool mocker setup for planner and global-router experiments |
