<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3.8-Flash-Next Recipes

Recipes for [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) on Dynamo + vLLM.

Qwen3.8-Flash-Next is a multimodal, ultra-sparse Mixture-of-Experts model with 125B total parameters
(including an additional 51B N-gram embedding table) and 6B active parameters per token. The
checkpoint used here is the [Inferact NVFP4 quantization](https://huggingface.co/Inferact/Qwen3.8-Flash-Next-NVFP4)
(~130 GB VRAM minimum). The architecture combines four ideas:

- **GDN + QSA**: three of every four layers use Gated DeltaNet (linear attention with a compact
  recurrent state); the fourth uses Qwen Sparse Attention for precise long-range retrieval.
- **Gated Residual**: four residual branches dynamically control cross-layer reads and writes.
- **N-gram Embedding**: a 51B lookup memory that can be asynchronously offloaded to host RAM,
  adding capacity with little per-token compute.
- **MTP**: built-in Multi-Token Prediction module for speculative decoding.

The checkpoint natively supports 262,144 tokens (extensible to 1M with YaRN). Weights are NVFP4.

## Configurations

Dynamo + vLLM deployment profiles for the B200 agentic workload (64K ISL / 400 OSL, 90% KV cache reuse):

|                          | B200 Aggregated (4-GPU)                      | B200 Aggregated (8-GPU)                         | B200 Disaggregated                              |
| ------------------------ | -------------------------------------------- | ------------------------------------------------ | ----------------------------------------------- |
| **GPU** (per worker)     | 4x B200                                      | 4x B200 (×2 workers)                             | 4x B200 prefill + 4x B200 decode                |
| **Total GPUs**           | 4                                            | 8                                                | 8 (1P1D)                                        |
| **Nodes**                | 1                                            | 1                                                | 1 (colocated via podAffinity)                    |
| **Mode**                 | Aggregated                                   | Aggregated (2 replicas)                           | Prefill/decode disaggregated                    |
| **Framework**            | vLLM                                         | vLLM                                            | vLLM                                            |
| **Precision**            | NVFP4 weights                               | NVFP4 weights                                  | NVFP4 weights                                  |
| **Parallelism**          | TP4 + expert parallel                        | TP4 + expert parallel (×2)                       | TP4 + expert parallel, both roles              |
| **Routing**              | KV-aware                                     | KV-aware                                        | KV-aware                                        |
| **Speculative decoding** | MTP3                                         | MTP3                                            | MTP3                                            |
| **Context length**       | 262,144                                      | 262,144                                          | 262,144                                         |
| **N-gram embedding**     | Offloaded to host RAM                        | Offloaded to host RAM                           | Offloaded to host RAM                           |
| **KV transfer**          | —                                            | —                                                | NIXL over UCX (cuda_ipc + cuda_copy + tcp)     |
| **Prefix caching**       | Enabled                                      | Enabled                                          | Enabled                                         |
| **hostIPC**              | —                                            | —                                                | Required for cross-pod NVLink                   |

## Supported features

- Modalities: **Text + Image** (multimodal)
- Reasoning (`qwen3` reasoning parser)
- Tool calling (`qwen3_coder` tool-call parser)
- MTP3 speculative decoding
- N-gram embedding offload to host memory
- Expert parallelism
- KV-aware routing and prefix caching
- PD disaggregation

## Prerequisites

1. **Dynamo Platform installed** — see [Kubernetes Deployment Guide](../../docs/fern/pages/kubernetes/getting-started/quickstart.mdx).
2. **vLLM image**: `vllm/vllm-openai:qwen38-flash-next` — a model-specific vLLM build with GDN/QSA
   kernels. `ai-dynamo` is pip-installed at pod startup.
3. **Hugging Face access** to `Inferact/Qwen3.8-Flash-Next-NVFP4`.
4. **Host memory**: ≥ 51 GB per worker for N-gram embedding offload (`VLLM_PLE_CPU_OFFLOAD=1`).

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
> available on the target cluster — `kubectl get storageclass` lists the candidates. The default
> value is `"your-storage-class-name"` (a placeholder) and must be replaced before applying.

```bash
kubectl apply -f model-cache/model-cache.yaml -n ${NAMESPACE}
```

The PVC must be `Bound` before proceeding to the model download step:

```bash
kubectl wait --for=jsonpath='{.status.phase}'=Bound pvc/model-cache -n ${NAMESPACE} --timeout=300s
```

### 3. Download the model

> [!IMPORTANT]
> The model-download Job mounts the `model-cache` PVC. Ensure the PVC is `Bound` (step 2) before
> applying the Job — if the PVC is still `Pending`, the Job will fail.

```bash
kubectl apply -f model-cache/model-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=7200s
```

The Job sets `HF_HOME=/model-cache`, so the checkpoint lands in the PVC's Hugging Face cache; each
worker passes the repo id (`Inferact/Qwen3.8-Flash-Next-NVFP4`) to `--model` and resolves the weights
from there.

> [!NOTE]
> The containers run as `runAsUser: 0` because the cached weight files are root-owned while the image
> defaults to a non-root uid.

> [!WARNING]
> The NVFP4 checkpoint is ~130 GiB. First-load time is bounded by the storage backend, not the GPUs.
> The startup probe budgets 60 minutes per worker; raise it if your storage is slower.

### 4. Deploy

```bash
# 4-GPU aggregated (1 worker × TP4)
kubectl apply -f vllm/agg-b200-agentic/deploy.yaml -n ${NAMESPACE}

# 8-GPU aggregated (2 workers × TP4)
kubectl apply -f vllm/agg-b200-agentic-8gpu/deploy.yaml -n ${NAMESPACE}

# 8-GPU disaggregated (1P1D, requires hostIPC)
kubectl apply -f vllm/disagg-b200-agentic/deploy.yaml -n ${NAMESPACE}
```

### 5. Smoke test

```bash
kubectl port-forward svc/qwen38fn-agg-b200-agentic-frontend 8000:8000 -n ${NAMESPACE} &
```

#### Text

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Inferact/Qwen3.8-Flash-Next-NVFP4",
    "messages": [{"role": "user", "content": "Explain how Gated DeltaNet and Qwen Sparse Attention complement each other."}],
    "max_tokens": 256
  }'
```

A correct response should explain that GDN maintains a compact recurrent summary, while QSA
selectively retrieves important regions from the full token history.

#### Tool calling

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Inferact/Qwen3.8-Flash-Next-NVFP4",
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

#### Image input

```bash
# Download a sample image
curl -s -o /tmp/test.jpg https://raw.githubusercontent.com/ai-dynamo/dynamo/main/docs/fern/assets/img/logo.svg

# Send multimodal request
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Inferact/Qwen3.8-Flash-Next-NVFP4",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,'$(base64 -w0 /tmp/test.jpg)'"}},
        {"type": "text", "text": "Describe this image in one sentence."}
      ]
    }],
    "max_tokens": 128
  }'
```

#### Video input

```python
from openai import OpenAI
import base64

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

with open("sample.mp4", "rb") as f:
    video_b64 = base64.b64encode(f.read()).decode("utf-8")

response = client.chat.completions.create(
    model="Inferact/Qwen3.8-Flash-Next-NVFP4",
    messages=[{
        "role": "user",
        "content": [
            {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}},
            {"type": "text", "text": "What do you see in this video? Describe briefly."}
        ]
    }],
    max_tokens=256,
)
print(response.choices[0].message.content)
```

A correct response should describe the visual content of the video frames.

## Optimization targets

| Workload | Median ISL | Median OSL | KV cache hit rate | User output tok/s |
| -------- | ---------- | ---------- | ----------------- | ----------------- |
| Agentic  | 64k        | 400        | 90%               | TBD               |

## Performance results

Benchmarked on B200, AIPerf trace-replay with 64K agentic trace (15% subset, 3,541 requests, 90% KV cache hit). ~130 requests exceeded the 262K context limit and were rejected (400 errors); ~3,411 requests completed successfully. All numbers below are from AIPerf's official `profile_export_aiperf.json` summary.

| Recipe               | SKU  | Workers | GPUs | Concurrency | System output tok/s | Per-GPU tok/s | User output tok/s (P50) | TTFT P50 (ms) | ITL P50 (ms) | Prefix cache hit |
|----------------------|------|---------|------|-------------|---------------------|---------------|-------------------------|---------------|--------------|-----------------|
| Aggregated (15% subset) | B200 | 1 (TP4)  | 4  | 24 | 1,830  | 457.4 | 102.8 | 329 | 9.7 | 67.9% |
| Aggregated (15% subset) | B200 | 2 (TP4×2) | 8  | 24 | 2,474  | 309.3 | 127.6 | 339 | 7.8 | 73.1% |

> **Note:** The 8-GPU aggregated recipe uses 2 workers (2×TP4) on a single node. Per-GPU throughput is lower (309 vs 458 tok/s/GPU) because the ultra-sparse model (6B active) is already compute-light — adding more workers improves aggregate throughput (+35%) and prefix cache hit rate (+5pp) but doesn't scale linearly due to shared memory bandwidth. ITL improves from 9.7 to 7.8 ms with more GPU resources per request.

## Configuration notes

Non-obvious knobs, all already set in the manifest:

- **N-gram embedding offload.** `VLLM_PLE_CPU_OFFLOAD=1` keeps the 51B N-gram lookup memory in host
  RAM and asynchronously prefetches the required rows. Requires ≥ 51 GB host memory per worker.
- **GDN state layout.** `VLLM_SSM_CONV_STATE_LAYOUT=DS` sets the Gated DeltaNet state layout. This
  MUST match across all workers (prefill and decode) in disaggregated mode.
- **Hybrid KV cache manager.** `--no-disable-hybrid-kv-cache-manager` is required because the model
  mixes GDN (linear attention) and QSA (full attention) layers with different KV layouts.
- **Multimodal.** `--enable-multimodal` enables the vision encoder for image inputs. The model
  supports text + image; omit this flag only for text-only workloads to save VRAM.
- **Context length.** `--max-model-len 262144` is the checkpoint's native context. vLLM uses this
  value by default when `--max-model-len` is omitted. To extend to 1M tokens, enable static YaRN:
  add `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` and
  `--hf-overrides '{"rope_parameters":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144}}'`
  with `--max-model-len 1000000`. Evaluate shorter-context quality before using YaRN as the default.
- **Expert parallelism.** `--enable-expert-parallel` distributes MoE experts across TP ranks to
  improve throughput on the 6B-active / 125B-total sparse model.
- **MTP3 speculative decoding.** `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`
  uses the built-in Multi-Token Prediction module for 3 draft tokens per step.
- **Event-driven KV routing.** Workers publish KV events (`--kv-events-config` over ZMQ), and the
  frontend uses `--router-mode kv --router-kv-events` for prefix-cache-aware routing.
- **Image.** `vllm/vllm-openai:qwen38-flash-next` is a model-specific vLLM build with GDN/QSA kernels.
  `ai-dynamo==1.4.2` is pip-installed at pod startup because the upstream image does not ship dynamo.
- **Runtime version override.** `runtimeVersionOverride: "1.4.2"` is required on each component
  when using a non-semver image tag (e.g. `qwen38-flash-next`). The DGD operator uses this to
  determine API compatibility.
- **SYS_RESOURCE capability.** `capabilities.add: ["IPC_LOCK", "SYS_RESOURCE"]` is required for
  `ulimit -l unlimited` to work inside the container (needed for large pinned memory allocations).
- **Disaggregated KV transfer.** The disagg recipe uses `UCX_TLS=cuda_ipc,cuda_copy,tcp` with
  `hostIPC: true` on both worker pods. `hostIPC` shares the IPC namespace so `cuda_ipc` can open
  GPU memory handles across pods on the same node (NVLink). Without `hostIPC`, UCX falls back to
  `cuda_copy` + TCP (~50 MB/s instead of NVLink speeds).
- **kv_role.** Prefill uses `kv_producer`, decode uses `kv_consumer` (not `kv_both`). This is more
  precise than `kv_both` on both workers and matches the GLM-5.3-Flash pattern.
- **Parser compatibility.** `--dyn-tool-call-parser qwen3_coder` is used instead of `qwen3_xml`
  because `qwen3_xml` is not available in `ai-dynamo==1.4.2`. `--reasoning-parser` (vLLM engine)
  is omitted; only `--dyn-reasoning-parser qwen3` (Dynamo frontend) is used, matching the
  Qwen3.8-2.4T pattern.
- **No `--enable-auto-tool-choice`.** This vLLM arg is not recognized by the `dynamo.vllm` wrapper
  and causes a startup error. Tool calling works via `--dyn-tool-call-parser` only.

## Checkpoint variants

This recipe uses the [Inferact NVFP4 quantization](https://huggingface.co/Inferact/Qwen3.8-Flash-Next-NVFP4)
(~130 GB VRAM minimum), which is the default variant in the upstream [vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next).
NVFP4 exploits B200's native FP4 tensor cores for ~2× weight compression vs FP8.

Alternative checkpoints (not covered by this recipe):

| Variant | Checkpoint | VRAM min | Notes |
| ------- | ---------- | -------- | ----- |
| **NVFP4** (default) | `Inferact/Qwen3.8-Flash-Next-NVFP4` | 130 GB | Inferact quantized, per-layer embeddings in NVFP4 |
| FP8 | `Qwen/Qwen3.8-Flash-Next-FP8` | 250 GB | Official dynamic FP8, blockwise weight quantization |
| BF16 | `Qwen/Qwen3.8-Flash-Next` | 423 GB | Official BF16 checkpoint |

To switch to FP8, change `--model` and `model-download.yaml` to `Qwen/Qwen3.8-Flash-Next-FP8`.
No `--quantization` flag is needed — vLLM auto-detects from the checkpoint config.

## Limitations

- **Disaggregated benchmark incomplete** — the disagg recipe was deployed and smoke-tested
  (text, tool, image, video all passed), but the AIPerf benchmark encountered stability
  issues with `hostIPC` + `cuda_ipc`. Partial results (800/3541 requests) are available but
  not included in the performance table. A follow-up PR will address the disagg benchmark.
- **Single-node only** — all recipes use a single B200 node. Multi-node TP is not validated
  for this model on B200.
- **N-gram offload on NVIDIA only** — the asynchronous host-memory offload currently runs on
  NVIDIA devices only.
- **vLLM version** — the `qwen38-flash-next` image is built from a vLLM 0.29.0+ development
  branch; it is not aligned with the vLLM release shipped in dynamo's standard container build.
- **Upstream image** — `ai-dynamo` is pip-installed at pod startup rather than baked into the
  image, which adds ~30s to cold start. Switching to a dynamo-built image will eliminate this.
- **No compile cache** — vLLM compilation and CUDA graph capture are ephemeral (emptyDir). Each
  cold start pays ~5 minutes for torch.compile + CUDA graph capture. A persistent compile cache
  (PVC) can reduce restart time to ~30 seconds but was not included due to stability issues.
