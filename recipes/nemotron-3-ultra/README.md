<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Nemotron-3-Ultra Recipes

Recipes for **nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4** — a ~550B hybrid Mamba/Attention/MoE model (~55B active).

We ship Dynamo + vLLM deployment profiles across B200 and H200, with aggregated and disaggregated serving modes.

Runtime image:

```text
nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0-nemotron-ultra-dev.1
```

The recipes pin `VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel` and pass `--no-enable-flashinfer-autotune` on vLLM workers. These settings select the non-FlashInfer FP8 linear kernel path used for the B200 benchmark rows and avoid the measured vLLM 0.22 FlashInfer FP8 regression.

## Configurations

|                          | B200 chat | H200 chat | B200 agentic | H200 agentic | B200 disaggregated agentic |
|--------------------------|-----------|-----------|--------------|--------------|-----------------------------|
| **GPU**                  | 4× B200 | 8× H200 | 4× B200 | 8× H200 | 4× B200 prefill + 4× B200 decode |
| **Mode**                 | aggregated | aggregated | aggregated | aggregated | disaggregated |
| **Framework**            | Dynamo + vLLM | Dynamo + vLLM | Dynamo + vLLM | Dynamo + vLLM | Dynamo + vLLM |
| **Precision**            | NVFP4 + FP8 | NVFP4 + FP8 | NVFP4 + FP8 | NVFP4 + FP8 | NVFP4 + FP8 |
| **Parallelism**          | TP4 + EP | TP8 + EP | TP4 + EP | TP8 + EP | TP4 prefill + TP4 decode |
| **Routing**              | KV-aware | KV-aware | KV-aware | KV-aware | KV-aware + P/D transfer |
| **Speculative decoding** | MTP, 1 token | MTP, 1 token | MTP, 1 token | MTP, 1 token | no MTP |
| **Max model length**     | 262144 | 262144 | 262144 | 262144 | 262144 |
| **Max sequences**        | 64 | 16 | 24 | 32 | 32 |
| **Max batched tokens**   | 32768 | 32768 | 32768 | 32768 | 32768 |
| **Block size**           | 64 | 64 | 64 | 64 | 64 |
| **Reference concurrency** | 18 | 10 | 20 | 8 | 32 |
| **Manifest**             | `vllm/agg-b200-chat-mtp/deploy.yaml` | `vllm/agg-h200-chat-mtp/deploy.yaml` | `vllm/agg-b200-agentic-mtp/deploy.yaml` | `vllm/agg-h200-agentic-mtp/deploy.yaml` | `vllm/disagg-b200-agentic/deploy.yaml` |

Aggregated no-MTP fallback manifests are also included under `vllm/agg-*-nomtp/deploy.yaml`.

## Supported Features

- Text-only chat
- Reasoning control through `chat_template_kwargs`
- Tool calling with `qwen3_coder`
- Ultra reasoning parser support through the model-local `ultra_v3_reasoning_parser.py`
- Raw Moontrace replay through AIPerf

## Prerequisites

1. **Dynamo Platform installed** on the target cluster with DGD CRDs served.
2. **NGC image pull secret** named `nvcr-secret`.
3. **Hugging Face token secret** named `hf-token-secret` when using the model download Job:
   ```bash
   kubectl create secret generic hf-token-secret \
     --from-literal=HF_TOKEN="$HF_TOKEN" \
     -n ${NAMESPACE}
   ```
4. A `shared-model-cache` PVC containing the tokenizer-patched Ultra model view, or permission to create and populate it with the manifests in `model-cache/`.

## Quick Start

```bash
export NAMESPACE=your-namespace
```

### 1. Create or Validate Model Cache

If the namespace does not already provide `shared-model-cache`, edit the storage class in `model-cache/model-cache.yaml`, then create and populate the PVC:

```bash
kubectl apply -f model-cache/model-cache.yaml -n ${NAMESPACE}
kubectl apply -f model-cache/model-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/nemotron-ultra-model-download -n ${NAMESPACE} --timeout=12h
```

Validate the patched model view before deploying a server:

```bash
kubectl apply -f model-cache/model-validate.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/nemotron-ultra-model-validate -n ${NAMESPACE} --timeout=30m
```

### 2. Deploy the DGD

Pick the SKU, use-case, and speculative decoding mode for an aggregated recipe:

```bash
SKU=b200       # or h200
USECASE=chat   # or agentic
SPEC=mtp       # or nomtp

kubectl apply -f vllm/agg-${SKU}-${USECASE}-${SPEC}/deploy.yaml -n ${NAMESPACE}
```

The DGD name includes the `-nomtp` suffix only for no-MTP recipes:

```bash
DGD=ultra-agg-${SKU}-${USECASE}-${SPEC}
kubectl get dgd ${DGD} -n ${NAMESPACE} -w
```

Disaggregated recipes are currently agentic b200, no-MTP only.

```bash
kubectl apply -f vllm/disagg-b200-agentic/deploy.yaml -n ${NAMESPACE}
```

### 3. Smoke Test

```bash
kubectl port-forward svc/${DGD}-frontend 8000:8000 -n ${NAMESPACE}

MODEL_ID=nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4

curl http://localhost:8000/v1/models
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL_ID}\",
       \"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}],
       \"max_tokens\":64,
       \"chat_template_kwargs\":{\"enable_thinking\":false,\"force_nonempty_content\":true}}"
```

### 4. Benchmark

See [`perf/README.md`](perf/README.md) for the full benchmark workflow — staging Moontrace-format traces on the PVC, running the AIPerf trace-replay Job ([`perf/perf.yaml`](perf/perf.yaml)), running a concurrency sweep, and fetching artifacts.

## Benchmark Results

B200 rows use 15% raw Moontrace replay with `raw_direct_no_filter` trace semantics. H200 rows use 300-sample replay evidence. All rows should be treated together with their matching recipe, image, trace, and server-shape artifacts.

The B200 rows below point at the actual recipe manifests in this tree. `User output tok/s` is Gen TPS/user p50 from AIPerf; `System output tok/s/GPU` is TPS/GPU.

| Recipe | GPU | Topology | Workload | MTP | Concurrency | User output tok/s | System output tok/s/GPU |
|--------|-----|----------|----------|-----|-------------|-------------------|-------------------------|
| `vllm/agg-b200-agentic-mtp/deploy.yaml` | B200 | AGG | agentic | yes | 20 | 80.6 | 310.8 |
| `vllm/agg-b200-agentic-nomtp/deploy.yaml` | B200 | AGG | agentic | no | 8 | 99.5 | 175.9 |
| `vllm/agg-b200-chat-mtp/deploy.yaml` | B200 | AGG | chat | yes | 18 | 52.0 | 201.4 |
| `vllm/agg-b200-chat-nomtp/deploy.yaml` | B200 | AGG | chat | no | 16 | 51.0 | 181.3 |
| `vllm/disagg-b200-agentic/deploy.yaml` | B200 | 1P1D | agentic | no | 32 | 61.6 | 231.1 |
| `vllm/agg-h200-agentic-mtp/deploy.yaml` | H200 | AGG | agentic | yes | 8 | 53.2 | 27.4 |
| `vllm/agg-h200-agentic-nomtp/deploy.yaml` | H200 | AGG | agentic | no | 8 | 52.3 | 26.5 |
| `vllm/agg-h200-chat-mtp/deploy.yaml` | H200 | AGG | chat | yes | 10 | 58.7 | 46.8 |
| `vllm/agg-h200-chat-nomtp/deploy.yaml` | H200 | AGG | chat | no | 8 | 54.2 | 43.0 |


## Reasoning Controls

Ultra no-thinking request control:

```json
{
  "chat_template_kwargs": {
    "enable_thinking": false,
    "force_nonempty_content": true
  }
}
```

Ultra reasoning budget request control:

```json
{
  "nvext": {
    "max_thinking_tokens": 10
  }
}
```

Do not send `force_nonempty_content` as a top-level request parameter.

## Known Issues

1. Optional OpenAI/vLLM/NIM API fields are shared Dynamo API compatibility gaps, not Ultra recipe-specific failures.
2. Top-level reasoning controls such as `include_reasoning`, `thinking_token_budget`, `reasoning_effort`, and `usage.reasoning_tokens` are part of that shared API compatibility work. Use the Ultra-specific `chat_template_kwargs` and `nvext` controls above as the current model-specific workaround.
3. Do not remove `VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel` or `--no-enable-flashinfer-autotune` from the vLLM worker commands unless rerunning the benchmark qualification. These are part of the performance recipe.
4. Raw Moontrace replay may contain over-context or pathological long-generation rows. Do not drop those rows silently; preserve them as HTTP/error evidence or classify the run accordingly.

## File Layout

```text
recipes/nemotron-3-ultra/
  README.md
  model-cache/
    README.md
    model-cache.yaml          # PVC
    model-download.yaml       # Job: populate patched Ultra model view
    model-validate.yaml       # Job: validate model/tokenizer/parser files
  vllm/
    agg-b200-chat-mtp/deploy.yaml
    agg-b200-chat-nomtp/deploy.yaml
    agg-b200-agentic-mtp/deploy.yaml
    agg-b200-agentic-nomtp/deploy.yaml
    agg-h200-chat-mtp/deploy.yaml
    agg-h200-chat-nomtp/deploy.yaml
    agg-h200-agentic-mtp/deploy.yaml
    agg-h200-agentic-nomtp/deploy.yaml
    disagg-b200-agentic/deploy.yaml
  perf/
    README.md                 # benchmark workflow
    perf.yaml                 # AIPerf trace-replay Job
    traces/                   # 15%, 30%, and full Moontrace JSONL assets
```
