<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

Dynamo + vLLM recipe for **DeepSeek-V4-Pro-0813** (native FP8) on **GB200**, tuned for an
agentic coding and tool-use workload with long shared prefixes.

This is a different checkpoint from the undated `deepseek-ai/DeepSeek-V4-Pro` recipes under
`recipes/deepseek-v4/`: 0813 ships native FP8 weights, uses DSpark speculative decoding
rather than MTP, and requires a different MoE backend on aarch64. The configurations are not
interchangeable, so they live in separate trees.

## Configurations

| Recipe | GPUs | Topology | Parallelism | Speculative decoding | Context |
| --- | --- | --- | --- | --- | --- |
| `vllm/agg-gb200-agentic` | 8 (2 nodes) | Aggregated | TP8 + expert parallel | DSpark, k=5, probabilistic | 1,048,576 |

GB200 is 4 GPUs per node, so TP8 spans two nodes (`multinode.nodeCount: 2`). One replica is
8 GPUs.

> [!NOTE]
> A disaggregated GB200 profile is not published yet. It will be added once it has a
> measured SLA knee on the same trace. Aggregated is the recommended and only supported
> profile in this tree today.

## Supported features

- Aggregated serving with Dynamo's Kubernetes operator
- Tensor parallelism (TP8) with expert parallelism
- DSpark speculative decoding with real rejection sampling
- 1M context (`--max-model-len 1048576`)
- FP8 KV cache
- Prefix caching and chunked prefill

## Prerequisites

- A Kubernetes cluster with **8 GB200 GPUs across 2 nodes** available to one worker
- The Dynamo Kubernetes operator installed — see
  [the Kubernetes quickstart](../../docs/fern/pages/kubernetes/getting-started/quickstart.mdx)
- A `model-cache` PVC with at least **1 TiB** free. The checkpoint is 892.7 GB.
- Hugging Face access to `deepseek-ai/DeepSeek-V4-Pro-0813`

## Quick start

```bash
kubectl apply -f model-cache/model-cache.yaml
kubectl apply -f model-cache/model-download.yaml   # 892.7 GB; allow time
kubectl apply -f vllm/agg-gb200-agentic/deploy.yaml
```

Fetch the benchmark trace (it is a symlink into the `kimi-k2.6` recipe, which owns the
LFS pointer — name that path explicitly, a glob will not match it):

```bash
git lfs pull --include "recipes/kimi-k2.6/perf/traces/*"
```

Then follow [perf/README.md](perf/README.md) to replay it.

## Optimization targets

The workload is a Mooncake trace replay: **64K median input, 400 median output, 90% KV
cache reuse**, i.e. agentic coding with long shared prefixes.

The SLA is **both** of:

- P50 time to first token **< 5 s**
- P50 output **>= 50 tokens/s/user**

The reported figure is system output tokens/s divided by **total** GPUs.

## Benchmark Results

Measured on the full 3,541-request trace, single replica, 8 GB200 GPUs.

| Recipe | GPUs | Topology | Spec. dec. | Concurrency | User output (tok/s, P50) | TTFT (ms, P50) | System output per GPU (tok/s) | SLA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `agg-gb200-agentic` | 8 | Aggregated | DSpark k=5 | **8** | **55.906** | **383.3** | **64.88** | ✅ pass |
| `agg-gb200-agentic` | 8 | Aggregated | DSpark k=5 | 10 | 49.968 | 482.5 | 69.29 | ❌ fail |

**Concurrency 8 is the SLA knee** — the highest concurrency where both SLA terms hold.
Concurrency 10 yields a higher per-GPU figure (69.29) but misses the per-user floor at
49.968, reproducibly: two independent runs scored 49.968 and 49.894 (spread 0.074).
Both rows are 3,526 valid / 15 errors, with 13 of the 15 being requests that legitimately
exceed the 1M context window.

## Known Issues

- **`--kv-cache-dtype fp8` is required, not an optimization.** Without it the engine cannot
  start: `fp8_ds_mla` asserts on this checkpoint. There is therefore no "stock vLLM"
  baseline for this model.
- **MTP speculative decoding does not work on this checkpoint.** The FP8 weights carry
  DSpark-shaped tensors, and loading an MTP config fails with
  `KeyError: 'model.layers.61.mtp_block.main_norm.weight'`. Use DSpark.
- **`--moe-backend` must be set explicitly.** Auto-selection picks a MARLIN kernel that
  segfaults on aarch64. `deep_gemm_mega_moe` is required on GB200.
- **`--max-num-batched-tokens 32768` does not initialise.** Four independent attempts on
  different nodes crash-looped during worker startup with
  `torch.distributed.DistNetworkError`. 16384 is the shipped and highest working value;
  2048 is 22.6% slower.
- **Pipeline parallelism is unavailable.** This model does not implement vLLM's `SupportsPP`
  interface, so `--pipeline-parallel-size > 1` fails at startup. TP, DP and EP are the only
  parallelism axes.
- **GB200 multinode needs an IMEX channel.** TP across nodes uses MNNVL, which requires a
  `ComputeDomain` plus a `resourceClaims` entry on the worker (both shipped in
  `deploy.yaml`). Without them the workers hang in a Gloo barrier during `torch.distributed`
  init and fail with `RuntimeError: Application timeout caused pair closure` *before* loading
  any weights — so a longer startup probe does not help. This needs the NVIDIA DRA driver for
  compute domains installed on the cluster.
- **`/dev/shm` must be Memory-backed and large at TP8.** The Kubernetes 64 MB default causes
  `shm_broadcast.acquire_read` timeouts that surface as `EngineDeadError`. This recipe mounts
  200Gi.
- **The startup probe must allow for a 892.7 GB load.** Over a shared filesystem this can take
  more than 90 minutes; a shorter budget SIGKILLs the pod mid-load and reports `exit 137` with
  `reason=Error`, which is easily mistaken for an out-of-memory kill.
- **Do not benchmark with `speculative-config-synthetic`.** Forced acceptance length
  overstates per-user throughput by roughly 36.5% because it skips the draft-model compute.
  See [perf/README.md](perf/README.md).
