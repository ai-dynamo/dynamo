<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek-V4-Flash Recipe

Serving recipes for **DeepSeek-V4-Flash** on Dynamo — a MoE model (284B total / 13B active) with hybrid CSA + HCA attention and a Blackwell FP4 indexer cache. B200 recipes serve the `nvidia/DeepSeek-V4-Flash-NVFP4` checkpoint; H200 serves the public `deepseek-ai/DeepSeek-V4-Flash` (see [`model-cache/model-download.yaml`](model-cache/model-download.yaml)). Two families today — see [Recipes](#recipes):

- **Agentic (vLLM)** — the benchmarked, workload-tuned picks (B200 & H200, AGG + disaggregated); numbers in [Performance](#performance).
- **Day-0** — the original single-node aggregated recipes (B200/GB200, vLLM + SGLang).

Shared operational details — [workloads](../README.md#optimization-targets), [per-rank NIC mapping](../README.md#per-rank-nic-mapping-b200--h200-disaggregated) (disaggregated GDR), and [known limitations](../README.md#known-limitations) — live in the [top-level DeepSeek-V4 README](../README.md).

## Recipes

### Agentic (vLLM)

Prefill/decode parallelism + spec-decode differ in disaggregated variants (`prefill / decode`).

| Variant | GPUs | Prefill / Decode | MoE backend | Spec. decode | `max_model_len` | Disagg fabric |
|---|---|---|---|---|---|---|
| [`agg-b200-agentic`](vllm/agg-b200-agentic/deploy.yaml) | 4× B200 | TP4 | FLASHINFER_TRTLLM | none | 1,048,576 | — |
| [`agg-h200-agentic`](vllm/agg-h200-agentic/deploy.yaml) | 4× H200 | DP4 + TP1 + EP | MARLIN (FLASHINFER_MLA attn) | MTP-1 | 1,048,576 | — |
| [`disagg-b200-agentic`](vllm/disagg-b200-agentic/deploy.yaml) | 24× B200 (4P·2D) | TP4 / TP4 | FLASHINFER_TRTLLM | none | 1,048,576 | NIXL GDR |
| [`disagg-h200-agentic`](vllm/disagg-h200-agentic/deploy.yaml) | 28× H200 (4P·3D) | DP4+TP1+EP / DP4+TP1+EP | MARLIN (FLASHINFER_MLA attn) | none / MTP-1 | 1,048,576 | NIXL GDR ¹ |

B200 = `nvidia/DeepSeek-V4-Flash-NVFP4`; H200 = `deepseek-ai/DeepSeek-V4-Flash`. Common: FP8 KV, block 256, KV-aware routing, prefix caching. Modality: text; reasoning + tool calling supported.

¹ Disaggregated uses NIXL over RDMA/GDR — see [Per-rank NIC mapping](../README.md#per-rank-nic-mapping-b200--h200-disaggregated).

### Day-0

Original single-node aggregated recipes — all single-replica. **Experimental (Day-0). Text only.**

| Variant | Backend | GPUs | Parallelism | MoE backend | Spec. decode |
|---|---|---|---|---|---|
| [`vllm-agg-b200`](vllm/agg_b200/deploy.yaml) | vLLM | 4× B200 | DP4 + TP1 + EP | vLLM V4 expert (FP4) | none |
| [`vllm-agg-gb200`](vllm/agg_gb200/deploy.yaml) | vLLM | 4× GB200 | TP4 + EP | `deep_gemm_mega_moe` | none |
| [`sglang-agg`](sglang/agg/deploy.yaml) | SGLang | 4× B200 | TP4 | `flashinfer_mxfp4` | EAGLE MTP 3/4 |
| [`sglang-agg-gb200`](sglang/agg-gb200/deploy.yaml) | SGLang | 4× GB200 | TP4 | `flashinfer_mxfp4` | EAGLE MTP 3/4 |

The B200 variants fill 4 of 8 GPUs on a B200 node; the GB200 variants fill all 4 GPUs of a single GB200 NVL4 tray. All use prebuilt NGC images ².

² Day-0 container images — vLLM (both): `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0-deepseek-v4-cuda13-dev.3` (multi-arch); SGLang B200: `…/sglang-runtime:1.2.0-deepseek-v4-cuda12-dev.3`; SGLang GB200: `…/sglang-runtime:1.2.0-deepseek-v4-cuda13-dev.3` (arm64). To rebuild from source (custom Dynamo branch, different engine base, etc.), see [`../container/README.md`](../container/README.md).

## Performance

Floor-picks (max system tok/s/GPU at user_p50 ≥ 50), default temperature. Agentic variants only; Day-0 recipes are functional (unbenchmarked). Workload definitions: [Optimization targets](../README.md#optimization-targets).

### Agentic workload

| Variant | Concurrency | User tok/s | System tok/s/GPU |
|---|---:|---:|---:|
| [`agg-b200-agentic`](vllm/agg-b200-agentic/deploy.yaml) | 38 | 50.6 | 362.1 |
| [`disagg-b200-agentic`](vllm/disagg-b200-agentic/deploy.yaml) | 320 | 50.0 | 388.3 |
| [`agg-h200-agentic`](vllm/agg-h200-agentic/deploy.yaml) | 16 | 50.7 | 145.4 |
| [`disagg-h200-agentic`](vllm/disagg-h200-agentic/deploy.yaml) | 128 | 51.8 | 201.9 |

### Custom workload

| Variant | Concurrency | User tok/s | System tok/s/GPU |
|---|---:|---:|---:|
| [`agg-b200-agentic`](vllm/agg-b200-agentic/deploy.yaml) | 64 | 50.1 | 741.6 |
| [`disagg-b200-agentic`](vllm/disagg-b200-agentic/deploy.yaml) | 128 | 51.9 | 738.2 ³ |
| [`agg-h200-agentic`](vllm/agg-h200-agentic/deploy.yaml) | 16 | 58.9 | 222.0 |
| [`disagg-h200-agentic`](vllm/disagg-h200-agentic/deploy.yaml) | 416 | 51.8 | 513.6 |

³ Measured at the **1P1D** prefill:decode ratio (optimal for this decode-heavy workload); the shipped `disagg-b200-agentic` deploy.yaml is **4P2D** (agentic-tuned; → 577 tok/s/GPU here) — same config, only the replica count differs.

**AGG figures are single-replica floor-picks.** Deploy AGG as independent single-replica DGDs for linear scaling — KV-routed *multi*-replica AGG does **not** improve per-GPU throughput ([Known limitations](../README.md#known-limitations)).

## Prerequisites

1. **Dynamo Platform installed** — see the [Kubernetes Deployment Guide](../../../docs/kubernetes/README.md).
2. **GPU cluster.** At least 4 GPUs of the matching arch available on one node:
   - **B200 variants**: 4 B200 GPUs (x86_64).
   - **GB200 variants**: 4 GB200 GPUs (single NVL4 tray, arm64). Nodes must be labeled `nvidia.com/gpu.product=NVIDIA-GB200` and tainted `kubernetes.io/arch=arm64:NoSchedule` (the manifests carry the matching `nodeSelector` + `toleration`).
3. **HuggingFace token** with access to `deepseek-ai/DeepSeek-V4-Flash`.

## Quick Start

Common setup (run once — applies to all variants):

```bash
export NAMESPACE=dynamo-demo
kubectl create namespace ${NAMESPACE}

# HuggingFace token secret (consumed by the download Job and, as a convenience, by the worker)
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="your-token-here" \
  -n ${NAMESPACE}

# Download model into the model-cache PVC.
# Edit model-cache/model-cache.yaml and set storageClassName to a RWX class in your cluster.
# The PVC requests 400Gi; DeepSeek-V4-Flash is ~160GB on disk (46 safetensors shards,
# FP4+FP8 mixed) and typically takes 30-60 min to download on first apply.
kubectl apply -f model-cache/model-cache.yaml -n ${NAMESPACE}
kubectl apply -f model-cache/model-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=7200s
```

### Deploy

Same flow for **every** variant (Agentic and Day-0): apply its `deploy.yaml`, then wait on its DGD.
First worker launch loads weights and warms CUDA graphs / kernels — up to ~60 min (the manifests' startup
probes allow for it).

```bash
# RECIPE = the deploy.yaml path from the tables above, e.g.:
#   agentic: vllm/{agg,disagg}-{b200,h200}-agentic
#   Day-0:   vllm/agg_b200 | vllm/agg_gb200 | sglang/agg | sglang/agg-gb200
RECIPE=vllm/agg-b200-agentic

kubectl apply -f ${RECIPE}/deploy.yaml -n ${NAMESPACE}

# Wait on the deployment (DGD name = metadata.name in that deploy.yaml):
DGD=$(awk '/^metadata:/{m=1} m && /name:/{print $2; exit}' ${RECIPE}/deploy.yaml)
kubectl wait --for=condition=Ready pod \
  -l nvidia.com/dynamo-graph-deployment-name=${DGD} \
  -n ${NAMESPACE} --timeout=3600s
```

**Disaggregated variants:** first set `VLLM_GPU_NIC_PCIE_MAPPING` in the manifest — see
[Per-rank NIC mapping](../README.md#per-rank-nic-mapping-b200--h200-disaggregated).

### Test the Deployment

Port-forward the deployment's frontend (`<DGD>-frontend`), then send an OpenAI-compatible request
(same model name / endpoints for vLLM and SGLang):

```bash
kubectl port-forward svc/${DGD}-frontend 8000:8000 -n ${NAMESPACE}

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-V4-Flash",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 512
  }'
```

Flash reasons by default, so the chain-of-thought fills `message.reasoning_content` and the final
answer lands in `message.content`. With too small a `max_tokens` the budget is spent on reasoning
before any `content` is emitted (`content: null`, `finish_reason: "length"`) — that's expected, not a
failure; raise `max_tokens` (or see [Verifying Reasoning](#verifying-reasoning)).

### Verifying Reasoning

Same flow on both variants — same model, same `--dyn-reasoning-parser deepseek_v4`:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-V4-Flash",
    "messages": [{"role": "user", "content": "What is 2+2? Answer briefly."}],
    "max_tokens": 200
  }' | python3 -m json.tool
```

Expected:

- `choices[0].message.reasoning_content` contains the model's chain-of-thought.
- `choices[0].message.content` contains only the final answer.
- No raw `</think>` tags in either field.

If `reasoning_content` is `null` and `</think>` appears in `content`, the reasoning parser isn't wired up — confirm `--dyn-reasoning-parser deepseek_v4` is on the worker command.

### Verifying Tool Calling

Same flow on both variants:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-V4-Flash",
    "messages": [{"role": "user", "content": "What is the weather in San Francisco?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string", "description": "City name"}
          },
          "required": ["location"]
        }
      }
    }],
    "max_tokens": 300
  }' | python3 -m json.tool
```

Expected:

- `choices[0].message.tool_calls` is a structured array with `function.name`, `function.arguments`, and `id`.
- `choices[0].finish_reason` is `"tool_calls"`.
- `choices[0].message.reasoning_content` may contain the model's reasoning about tool selection.

If `tool_calls` is missing and raw tool-call markers appear in `content`, confirm `--dyn-tool-call-parser deepseek_v4` is on the worker command.
