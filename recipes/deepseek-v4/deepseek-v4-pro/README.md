<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek-V4-Pro Recipe

Serving recipes for **DeepSeek-V4-Pro** on Dynamo — a MoE model (1.6T total / 49B active) with hybrid CSA + HCA attention, a Blackwell FP4 indexer cache, and mHC residual connections. B200 recipes serve the `nvidia/DeepSeek-V4-Pro-NVFP4` checkpoint at the full **1M** context; H200 serves the public `deepseek-ai/DeepSeek-V4-Pro` (FP8), capped at `max_model_len=86016` by HBM (see [`model-cache/model-download.yaml`](model-cache/model-download.yaml)). Two families today — see [Recipes](#recipes):

- **Agentic (vLLM)** — the benchmarked, workload-tuned picks (B200 & H200, AGG + disaggregated); numbers in [Performance](#performance).
- **Day-0** — the original single-node / cross-tray aggregated + disaggregated recipes (B200/GB200, vLLM + SGLang).

**Reasoning modes.** The deployed Pro recipes emit `reasoning_content` by default; reasoning effort is selected via `chat_template_kwargs` — `{"thinking":true,"reasoning_effort":"high"}` or `{"thinking":true,"reasoning_effort":"max"}` (Think Max needs `--max-model-len >= 393216`).

Shared operational details — [workloads](../README.md#optimization-targets), [per-rank NIC mapping](../README.md#per-rank-nic-mapping-b200--h200-disaggregated) (disaggregated GDR), and [known limitations](../README.md#known-limitations) — live in the [top-level DeepSeek-V4 README](../README.md).

## Recipes

### Agentic (vLLM)

Prefill/decode parallelism + spec-decode differ in disaggregated variants (`prefill / decode`).

| Variant | GPUs | Prefill / Decode | MoE backend | Spec. decode | `max_model_len` | Disagg fabric |
|---|---|---|---|---|---|---|
| [`agg-b200-agentic`](vllm/agg-b200-agentic/deploy.yaml) | 8× B200 | TP8 + EP | FLASHINFER_TRTLLM | MTP-2 | 1,048,576 | — |
| [`agg-h200-agentic`](vllm/agg-h200-agentic/deploy.yaml) ‡ | 8× H200 | TP8 + EP | MARLIN | none | 86,016 | — |
| [`disagg-b200-agentic`](vllm/disagg-b200-agentic/deploy.yaml) | 16× B200 (1P·1D) | TP8+EP / TP8+EP | FLASHINFER_TRTLLM | none / MTP-2 | 1,048,576 | NIXL GDR ¹ |
| [`disagg-h200-agentic`](vllm/disagg-h200-agentic/deploy.yaml) ‡ | 32× H200 (1P·3D) | TP8+EP / TP8+EP | MARLIN | none | 86,016 | NIXL GDR ¹ |

B200 = `nvidia/DeepSeek-V4-Pro-NVFP4` (1M ctx); H200 = `deepseek-ai/DeepSeek-V4-Pro` (public FP8, 86k cap). Common: FP8 KV, block 256, KV-aware routing, prefix caching. Modality: text; reasoning + tool calling supported.

**Recommended picks:** **B200 → `disagg-b200-agentic`** (1P1D, +6% tps/GPU over AGG at matched user_p50); **H200 → `agg-h200-agentic`** (AGG beats the 1P3D disagg lane — see [Performance](#performance)).

¹ Disaggregated uses NIXL over RDMA/GDR — see [Per-rank NIC mapping](../README.md#per-rank-nic-mapping-b200--h200-disaggregated).
‡ H200 Pro is a **secondary / best-effort** lane on the public FP8 checkpoint, capped at `max_model_len=86016` (HBM) — it cannot serve the longest agentic contexts the B200 1M recipes handle.

### Day-0

Original aggregated + disaggregated recipes. **Experimental (Day-0). Text only.**

| Variant | Backend | GPUs | Parallelism | MoE backend | Spec. decode |
|---|---|---|---|---|---|
| [`vllm-agg-b200`](vllm/agg/b200/deploy.yaml) | vLLM | 8× B200 | TP8 + EP | vLLM V4 expert (FP4) | none |
| [`vllm-agg-gb200`](vllm/agg/gb200/deploy.yaml) | vLLM | 8× GB200 (2 trays) ² | TP8 + EP cross-tray | `deep_gemm_mega_moe` | none |
| [`vllm-disagg-gb200`](vllm/disagg/gb200/deploy.yaml) | vLLM | 16× GB200 (1P·1D, 2 trays/worker) ² | DP8 + EP / DP8 + EP | vLLM V4 expert (FP4) | none |
| [`sglang-agg`](sglang/agg/deploy.yaml) | SGLang | 8× B200 | TP8 | `flashinfer_mxfp4` | EAGLE MTP 3/4 |
| [`sglang-agg-gb200`](sglang/agg-gb200/deploy.yaml) | SGLang | 8× GB200 (2 trays) ² | TP8 cross-tray | `flashinfer_mxfp4` | EAGLE MTP 3/4 |
| [`sglang-disagg-b200`](sglang/disagg-b200/deploy.yaml) | SGLang | 16× B200 (1P·1D) | TP8 / TP8 | `flashinfer_mxfp4` | EAGLE MTP 3/4 |
| [`sglang-disagg-gb200`](sglang/disagg-gb200/deploy.yaml) | SGLang | 16× GB200 (1P·1D, 2 trays/worker) ² | TP8 / TP8 cross-tray | `flashinfer_mxfp4` | EAGLE MTP 3/4 |

All Day-0 variants use prebuilt NGC images ³. The GB200 disagg variant ships a perf Job ([`vllm/disagg/gb200/perf.yaml`](vllm/disagg/gb200/perf.yaml), 8K-input / 1K-output sweep at c=256/512/1024).

² **GB200 cross-tray (MNNVL / ComputeDomain).** V4-Pro (~865 GB) exceeds a single GB200 NVL4 tray (~768 GB HBM / 4 GPUs), so every GB200 variant spans **two trays** over NVLink72 (MNNVL), not RoCE. The DRA **ComputeDomain** controller co-locates the worker pod set on the same NVLink72 clique (2 pods for agg, 4 for disagg) and allocates the MNNVL channel on demand; the cross-node TP/DP all-reduce flows over that fabric. Requires the DRA/ComputeDomain CRD on the cluster (`kubectl get crd | grep computedomain`) and the `NCCL_MNNVL_ENABLE=1` / `UCX_CUDA_IPC_ENABLE_MNNVL=y` / `NCCL_NVLS_ENABLE=1` env set (already in the manifests). See [Prerequisites](#prerequisites).

³ Day-0 container images — vLLM (all): `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0-deepseek-v4-cuda13-dev.3` (multi-arch); SGLang B200: `…/sglang-runtime:1.2.0-deepseek-v4-cuda12-dev.3`; SGLang GB200: `…/sglang-runtime:1.2.0-deepseek-v4-cuda13-dev.3` (arm64). To rebuild from source (custom Dynamo branch, different engine base, etc.), see [`../container/README.md`](../container/README.md).

## Performance

Floor-picks (max system tok/s/GPU at user_p50 ≥ 50), default temperature, **Agentic (MoonTrace) workload** — Pro was not benchmarked on the Custom/synthetic cell. Workload definition: [Optimization targets](../README.md#optimization-targets).

| Variant | Concurrency | User tok/s | System tok/s/GPU |
|---|---:|---:|---:|
| [`agg-b200-agentic`](vllm/agg-b200-agentic/deploy.yaml) | 13 | 51.3 | 72.79 |
| [`disagg-b200-agentic`](vllm/disagg-b200-agentic/deploy.yaml) | 28 | 51.3 | **77.31** |
| [`agg-h200-agentic`](vllm/agg-h200-agentic/deploy.yaml) ‡ | 8 | 53.2 | **45.90** |
| [`disagg-h200-agentic`](vllm/disagg-h200-agentic/deploy.yaml) ‡ | 32 | 50.5 | 43.86 |

Bold = recommended pick per SKU: B200 disagg wins (+6.2%); H200 AGG wins (1P3D disagg subfloors it). ‡ H200 rows are secondary/best-effort on public FP8 (`max_model_len=86016`).

## Prerequisites

1. **Dynamo Platform installed** — see the [Kubernetes Deployment Guide](../../../docs/kubernetes/README.md).
2. **GPU cluster.** Hardware depends on the variant:
   - **B200 variants** (agentic + `vllm-agg-b200`, `sglang-agg`, `sglang-disagg-b200`): 8 B200 GPUs per worker on a single node (x86_64). TP=8 fills the box; disagg places one prefill + one decode worker.
   - **H200 variants** (`agg-h200-agentic`, `disagg-h200-agentic`): 8 H200 GPUs per worker. The public FP8 checkpoint runs at `max_model_len=86016`.
   - **GB200 variants** (`vllm-agg-gb200`, `vllm-disagg-gb200`, `sglang-agg-gb200`, `sglang-disagg-gb200`): **2 GB200 nodes per worker**, each 4 GPUs (one NVL4 tray), on the **same NVLink72 clique**. Label `nvidia.com/gpu.product=NVIDIA-GB200`, taint `kubernetes.io/arch=arm64:NoSchedule`, and install the **DRA / ComputeDomain controller** (`kubectl get crd | grep computedomain`) — each manifest's `ComputeDomain` CR + `resourceClaims` co-locate the pod set on one NVLink72 fabric.
3. **HuggingFace token** with access to `deepseek-ai/DeepSeek-V4-Pro`.

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
# The PVC requests 1500Gi; DeepSeek-V4-Pro is ~865 GB on disk (64 safetensors shards,
# FP4+FP8 mixed) and typically takes 1.5-3 hours to download on first apply.
kubectl apply -f model-cache/model-cache.yaml -n ${NAMESPACE}
kubectl apply -f model-cache/model-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=14400s
```

### Deploy

Same flow for **every** variant (Agentic and Day-0): apply its `deploy.yaml`, then wait on its DGD.
First worker launch loads weights across 8 TP ranks and warms CUDA graphs / kernels — up to ~90 min
(the manifests' startup probes allow for it).

```bash
# RECIPE = the deploy.yaml path from the tables above, e.g.:
#   agentic: vllm/{agg,disagg}-{b200,h200}-agentic
#   Day-0:   vllm/agg/b200 | vllm/agg/gb200 | vllm/disagg/gb200
#            sglang/agg | sglang/agg-gb200 | sglang/disagg-b200 | sglang/disagg-gb200
RECIPE=vllm/disagg-b200-agentic

kubectl apply -f ${RECIPE}/deploy.yaml -n ${NAMESPACE}

# Wait on the deployment (DGD name = metadata.name in that deploy.yaml):
DGD=$(awk '/^metadata:/{m=1} m && /name:/{print $2; exit}' ${RECIPE}/deploy.yaml)
kubectl wait --for=condition=Ready pod \
  -l nvidia.com/dynamo-graph-deployment-name=${DGD} \
  -n ${NAMESPACE} --timeout=5400s
```

- **Disaggregated B200/H200 variants:** first set `VLLM_GPU_NIC_PCIE_MAPPING` in the manifest — see [Per-rank NIC mapping](../README.md#per-rank-nic-mapping-b200--h200-disaggregated).
- **GB200 variants:** require the DRA/ComputeDomain controller and span two trays over MNNVL — see [Day-0 note ²](#day-0) and [Prerequisites](#prerequisites).

### Test the Deployment

Port-forward the deployment's frontend (`<DGD>-frontend`), then send an OpenAI-compatible request
(same model name / endpoints for vLLM and SGLang):

```bash
kubectl port-forward svc/${DGD}-frontend 8000:8000 -n ${NAMESPACE}

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-V4-Pro",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 512
  }'
```

Pro emits chain-of-thought into `message.reasoning_content` and the final answer into
`message.content`. With too small a `max_tokens` the budget can be spent on reasoning before any
`content` is emitted (`content: null`, `finish_reason: "length"`) — that's expected, not a failure;
raise `max_tokens` (or see [Verifying Reasoning](#verifying-reasoning)).

### Verifying Reasoning

Pro emits `reasoning_content` by default; this example passes `chat_template_kwargs` to request high effort explicitly (same model, same `--dyn-reasoning-parser deepseek_v4`):

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-V4-Pro",
    "messages": [{"role": "user", "content": "What is 2+2? Answer briefly."}],
    "max_tokens": 200,
    "chat_template_kwargs": {"thinking": true, "reasoning_effort": "high"}
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
    "model": "deepseek-ai/DeepSeek-V4-Pro",
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
