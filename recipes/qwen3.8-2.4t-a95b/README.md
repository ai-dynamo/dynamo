<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3.8-2.4T-A95B Recipes

Recipes for [Qwen3.8-2.4T-A95B](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B) on Dynamo + vLLM and SGLang.

Qwen3.8-2.4T-A95B is a hybrid gated-delta-net + MoE model: gated delta-net (GDN, linear attention with a
short convolution state) interleaved with full grouped-query attention (GQA), a 512-expert MoE, and a
262,144-token context. Weights are FP8.

## Configurations

### vLLM — Chat

Dynamo + vLLM chat deployment profiles (8k total context / 1k output, 70% KV cache reuse):

|                             | [GB300 aggregated](vllm/agg-gb300-chat/deploy.yaml) | [GB200 aggregated](vllm/agg-gb200-chat/deploy.yaml) |
| --------------------------- | ---------------------------------------------------- | ---------------------------------------------------- |
| **GPU** (per worker)        | 16x GB300 (4 nodes × 4)                              | 16x GB200 (4 nodes × 4)                              |
| **Replicas**                | 1 — 16 GPUs / 4 nodes                                | 1 — 16 GPUs / 4 nodes                                |
| **Mode**                    | Aggregated                                           | Aggregated                                           |
| **Precision**               | FP8 weights, FP8 KV                                  | FP8 weights, FP8 KV                                  |
| **Parallelism**             | TP16 over MNNVL                                      | TP16 over MNNVL                                      |
| **MoE backend**             | `flashinfer_trtllm` (FP8)                            | `flashinfer_trtllm`                                  |
| **AllReduce backend**       | FlashInfer MNNVL                                     | FlashInfer MNNVL                                     |
| **CUDA graphs**             | `FULL_AND_PIECEWISE` (≤512)                          | `FULL_AND_PIECEWISE` (≤512)                          |
| **Scheduler**               | Synchronous (async off)                              | Synchronous (async off)                              |
| **Batching**                | 12288 tok / 512 seqs                                 | 12288 tok / 512 seqs                                 |
| **GPU memory util**         | 0.90                                                 | 0.90                                                 |
| **KV transfer**             | —                                                    | —                                                    |
| **Prefix caching**          | Enabled                                              | Enabled                                              |

### vLLM — Agentic

Dynamo + vLLM agentic deployment profiles (64k total context / 400 output, 90% KV cache reuse).

|                             | [GB300 aggregated](vllm/agg-gb300-agentic/deploy.yaml) | [GB200 aggregated](vllm/agg-gb200-agentic/deploy.yaml) |
| --------------------------- | ------------------------------------------------------- | ------------------------------------------------------- |
| **GPU** (per worker)        | 16x GB300 (4 nodes × 4)                                 | 16x GB200 (4 nodes × 4)                                 |
| **Replicas**                | 1 — 16 GPUs / 4 nodes                                   | 1 — 16 GPUs / 4 nodes                                   |
| **Mode**                    | Aggregated                                              | Aggregated                                              |
| **Precision**               | FP8 weights, FP8 KV                                     | FP8 weights, FP8 KV                                     |
| **Parallelism**             | TP16 over MNNVL                                         | TP16 over MNNVL                                         |
| **MoE backend**             | `flashinfer_trtllm` (FP8)                               | `flashinfer_trtllm`                                     |
| **AllReduce backend**       | FlashInfer MNNVL                                        | FlashInfer MNNVL                                        |
| **CUDA graphs**             | `FULL_AND_PIECEWISE` (≤512)                             | `FULL_AND_PIECEWISE` (≤512)                             |
| **Scheduler**               | Synchronous (async off)                                 | Synchronous (async off)                                 |
| **Batching**                | 12288 tok / 512 seqs                                    | 12288 tok / 512 seqs                                    |
| **GPU memory util**         | 0.90                                                    | 0.90                                                    |
| **KV transfer**             | —                                                       | —                                                       |
| **Prefix caching**          | Enabled                                                 | Enabled                                                 |

### SGLang — Chat

Dynamo + SGLang chat deployment profiles:

|                          | [GB300 agg chat](sglang/agg-gb300-chat/deploy.yaml) | [GB300 disagg chat](sglang/disagg-gb300-chat/deploy.yaml) | [GB200 agg chat](sglang/agg-gb200-chat/deploy.yaml) | [GB200 disagg chat](sglang/disagg-gb200-chat/deploy.yaml) |
| ------------------------ | ---------------------------------------------------- | ---------------------------------------------------------- | ---------------------------------------------------- | ---------------------------------------------------------- |
| **GPU** (per worker)     | 16x GB300 (4 nodes × 4)                              | 16x GB300 prefill + 16x GB300 decode (4 nodes × 4 each)   | 16x GB200 (4 nodes × 4)                              | 16x GB200 × 2 prefill + 16x GB200 decode (4 nodes × 4 each) |
| **Replicas**             | 1 — 16 GPUs / 4 nodes                                | 1P1D — 32 GPUs / 8 nodes                                  | 1 — 16 GPUs / 4 nodes                                | 2P1D — 48 GPUs / 12 nodes                                 |
| **Mode**                 | Aggregated                                           | Disaggregated                                             | Aggregated                                           | Disaggregated                                             |
| **Precision**            | FP8 weights, FP8 KV                                  | FP8 weights, FP8 KV                                       | FP8 weights, FP8 KV                                  | FP8 weights, FP8 KV                                       |
| **Parallelism**          | TP16 over MNNVL                                      | TP16 over MNNVL, both roles                               | TP16 over MNNVL                                      | TP16 over MNNVL, both roles                               |
| **MoE backend**          | `flashinfer_trtllm`                                  | `flashinfer_trtllm`                                       | `flashinfer_trtllm`                                  | `flashinfer_trtllm`                                       |
| **Attention backend**    | `trtllm_mha`                                         | `trtllm_mha`                                              | `trtllm_mha`                                         | `trtllm_mha`                                              |
| **CUDA graph (prefill)** | `breakable`                                          | `breakable`                                               | `breakable`                                          | `breakable`                                               |
| **CUDA graph (decode)**  | `full`                                               | `full`                                                    | `full`                                               | `full`                                                    |
| **AllReduce backend**    | FlashInfer MNNVL                                     | FlashInfer MNNVL                                          | FlashInfer MNNVL                                     | FlashInfer MNNVL                                          |
| **mem-fraction-static**  | 0.90                                                 | 0.90 both roles                                           | 0.92                                                 | 0.85 prefill / 0.92 decode                                |
| **Routing**              | KV-aware                                             | KV-aware                                                  | KV-aware                                             | KV-aware, load-balanced across prefill replicas           |
| **Prefix caching**       | Enabled (radix cache)                                | Prefill-side only                                         | Enabled (radix cache)                                | Prefill-side only                                         |
| **KV transfer**          | —                                                    | NIXL over MNNVL                                           | —                                                    | NIXL over cuda_ipc + MNNVL (~900 GB/s/GPU)               |
| **Context length**       | 278,528                                              | 278,528 both roles                                        | 262,144                                              | 262,144                                                   |
| **Watchdog timeout**     | 900s                                                 | 900s                                                      | 900s                                                 | 7200s (breakable graph capture takes 30–60 min)           |

## Supported features

- Modalities: **Text only** (no vision tower)
- Reasoning (`qwen3` reasoning parser, engine and frontend side)
- Tool calling (`qwen3_coder` tool-call parser)
- KV-aware routing and prefix caching
- Multi-replica scale-out
- Disaggregated serving (GB200, GB300)

## Prerequisites

1. **Dynamo Platform installed** — see [Kubernetes Deployment Guide](../../docs/fern/pages/kubernetes/getting-started/quickstart.mdx).
2. **DRA / ComputeDomain controller** for the cross-node NVLink channel:
   ```bash
   kubectl get crd | grep computedomain
   ```
   The manifest creates its own `ComputeDomain` (`qwen38max-compute-domain`) and the workers claim its channel.
3. **Hugging Face token** with access to `Qwen/Qwen3.8-2.4T-A95B`. The workers read the weights from the
   `model-cache` PVC — see [Download the model](#3-download-the-model).

## Quick Start

### 1. Create namespace and secret

```bash
export NAMESPACE=your-namespace
kubectl create namespace ${NAMESPACE}
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="your-token" \
  -n ${NAMESPACE}
```

### 2. Create storage

> [!NOTE]
> Edit `model-cache/model-cache.yaml` and set `storageClassName` to a ReadWriteMany storage class
> available on the target cluster — `kubectl get storageclass` lists the candidates.

```bash
kubectl apply -f model-cache/model-cache.yaml -n ${NAMESPACE}
```

### 3. Download the model

```bash
kubectl apply -f model-cache/model-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=14400s
```

The Job sets `HF_HOME=/model-cache`, so the checkpoint lands in the PVC's Hugging Face cache; each
worker passes the repo id (`Qwen/Qwen3.8-2.4T-A95B`) to `--model` and resolves the weights from there.

> [!NOTE]
> The containers run as `runAsUser: 0` because the cached weight files are root-owned while the image
> defaults to a non-root uid.

> [!WARNING]
> The checkpoint loads from the shared filesystem on every cold start — each worker pod pulls the full
> checkpoint, so first-load time is bounded by the storage backend, not the GPUs. The startup probe
> budgets 120 minutes per worker (`failureThreshold: 720`); raise it if your storage is slower.

### 4. Deploy the DGD

```bash
kubectl apply -f vllm/agg-gb300-chat/deploy.yaml -n ${NAMESPACE}

kubectl wait --for=condition=Ready pod \
  -l nvidia.com/dynamo-graph-deployment-name=qwen38max-agg \
  -n ${NAMESPACE} --timeout=7200s
```

First launch loads the checkpoint across the TP16 ranks and warms CUDA graphs; the multinode pod set
only reports ready once all four nodes have joined the TP group.

### 5. Smoke test

```bash
kubectl port-forward svc/qwen38max-agg-frontend 8000:8000 -n ${NAMESPACE} &
```

#### Text

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3.8-2.4T-A95B",
    "messages": [{"role": "user", "content": "Hello, who are you?"}],
    "max_tokens": 128
  }'
```

Chain-of-thought lands in `choices[0].message.reasoning_content` and the answer in
`choices[0].message.content`. If `reasoning_content` is `null` and raw `</think>` markers show up in
`content`, the parsers are not wired — check `--reasoning-parser qwen3` and `--dyn-reasoning-parser
qwen3` on the worker command.

#### Tool calling

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3.8-2.4T-A95B",
    "messages": [{"role": "user", "content": "What is the weather in San Francisco?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
          "type": "object",
          "properties": {"location": {"type": "string", "description": "City name"}},
          "required": ["location"]
        }
      }
    }],
    "max_tokens": 300
  }' | python3 -m json.tool
```

Expected: `choices[0].message.tool_calls[0].function.name` is `get_weather` and `finish_reason` is `tool_calls`.

## Configuration notes

Non-obvious knobs, all already set in the manifest:

- **Model resolution.** Each worker pod mounts the `model-cache` PVC at `/model-cache` with
  `HF_HOME=/model-cache` and passes the repo id (`Qwen/Qwen3.8-2.4T-A95B`) to `--model`, so vLLM loads
  the checkpoint out of the PVC's Hugging Face cache. `envFrom: hf-token-secret` covers the hub lookup
  at startup. The frontend does not mount the PVC.
- **Async scheduling off.** `--no-async-scheduling` is required for these fixed serving shapes.
- **Event-driven KV routing.** Both workers publish KV events (`--kv-events-config` over ZMQ), and the
  frontend uses `--router-mode kv --router-kv-events`. Missing either worker flag can silently route
  traffic as aggregated/random instead of a verifiable P-to-D path.
- **MNNVL all-reduce.** `VLLM_ALLREDUCE_USE_FLASHINFER=1` with `VLLM_FLASHINFER_ALLREDUCE_BACKEND=mnnvl`.
  Keep `VLLM_USE_NCCL_SYMM_MEM=0` alongside it — symmetric memory breaks CUDA-graph capture on this build.
- **Pod networking.** DGD pods use CNI networking, so `NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME`
  are pinned to `eth0`.

## Known issues

- Every worker replica loads its own copy of the checkpoint from the PVC on every cold start, with no
  shared or streamed loading — pod restarts are expensive.
