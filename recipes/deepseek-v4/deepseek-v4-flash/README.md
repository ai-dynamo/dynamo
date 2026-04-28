<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek-V4-Flash Recipe

Aggregated-serving recipe for **DeepSeek-V4-Flash** on Dynamo. Two backends are documented side by side: **vLLM** and **SGLang**. Both are single-replica decode-only deployments that fill 4 of 8 GPUs on a B200 node.

| Variant | Backend | Manifest | GPUs | Topology | Container |
|---------|---------|----------|------|----------|-----------|
| **vllm-agg**   | vLLM   | [`vllm/agg/deploy.yaml`](vllm/agg/deploy.yaml)     | 4x B200 | DP=4 + Expert Parallel, TP=1                   | Custom build (vLLM's V4 stack not in a stock release) |
| **sglang-agg** | SGLang | [`sglang/agg/deploy.yaml`](sglang/agg/deploy.yaml) | 4x B200 | TP=4, MXFP4 MoE via FlashInfer, EAGLE MTP 3/4 | Prebuilt NGC image; optional [custom build](../container/) |

Status: **Experimental** (Day-0). Modality: text only.

## Prerequisites

1. **Dynamo Platform installed** — see the [Kubernetes Deployment Guide](../../../docs/kubernetes/README.md).
2. **GPU cluster** with at least 4 B200 GPUs available on one node.
3. **HuggingFace token** with access to `deepseek-ai/DeepSeek-V4-Flash`.
4. **Container image.** Pick the path that matches your variant:

   - **SGLang** (`sglang-agg`): the manifest pulls the prebuilt NGC image `nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.2.0-sglang-deepseek-v4-b200-dev.1` directly — **no build step required.** To rebuild from source (e.g. to pin a custom Dynamo branch or a different SGLang base), see the shared [`recipes/deepseek-v4/container/README.md`](../container/README.md).

   - **vLLM** (`vllm-agg`): DeepSeek-V4 is not in a stock vLLM release yet, so a reference Dockerfile ships in [`recipes/deepseek-v4/container/`](../container/) (shared with the V4-Pro recipe). Build in two steps:

     1. Build the Dynamo vLLM runtime image locally per [`<repo_root>/container/README.md`](../../../container/README.md) (this produces the local tag `dynamo:latest-vllm-runtime`).
     2. Build the dsv4 overlay on top of it using [`vllm/Dockerfile.dsv4.vllm.b200`](../container/vllm/Dockerfile.dsv4.vllm.b200). See [`recipes/deepseek-v4/container/README.md`](../container/README.md) for build args. From the repo root:

        ```bash
        docker build -f recipes/deepseek-v4/container/vllm/Dockerfile.dsv4.vllm.b200 \
          -t <your-registry>/vllm-dsv4:<tag> .
        ```

     Then set the `image:` fields in `vllm/agg/deploy.yaml` (both the Frontend and the decode worker) to `<your-registry>/vllm-dsv4:<tag>`.

## Quick Start

Common setup (run once — applies to both variants):

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

### Deploy — vLLM (`vllm-agg`)

```bash
# Update the `image:` fields in vllm/agg/deploy.yaml to your Dynamo + vLLM build
# (Prerequisite 4 — vLLM path).
kubectl apply -f vllm/agg/deploy.yaml -n ${NAMESPACE}

# First launch of the decode worker takes up to ~60 minutes (weight load +
# FlashInfer autotune + cudagraph warmup). The startup probe is sized for this.
kubectl wait --for=condition=Ready pod \
  -l nvidia.com/dynamo-graph-deployment-name=dsv4-flash-agg \
  -n ${NAMESPACE} --timeout=3600s
```

### Deploy — SGLang (`sglang-agg`)

```bash
# Manifest already points at the prebuilt NGC image — no image edit needed.
kubectl apply -f sglang/agg/deploy.yaml -n ${NAMESPACE}

# First launch of the decode worker takes up to ~60 minutes (weight load +
# DeepGEMM warmup + cudagraph warmup). The startup probe is sized for this.
kubectl wait --for=condition=Ready pod \
  -l nvidia.com/dynamo-graph-deployment-name=sglang-dsv4-flash \
  -n ${NAMESPACE} --timeout=3600s
```

## Test the Deployment

Port-forward the variant you deployed:

```bash
# vLLM
kubectl port-forward svc/dsv4-flash-agg-frontend 8000:8000 -n ${NAMESPACE}

# SGLang
kubectl port-forward svc/sglang-dsv4-flash-frontend 8000:8000 -n ${NAMESPACE}
```

Either way the request shape is the same — same model name, same OpenAI-compatible endpoints:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-V4-Flash",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

## Recipe Details

### vLLM (`vllm/agg/deploy.yaml`)

| Flag | Purpose |
|------|---------|
| `--tokenizer-mode deepseek_v4` | Selects the DeepSeek-V4 tokenizer |
| `--dyn-reasoning-parser deepseek_v4` | Extracts chain-of-thought into `message.reasoning_content` |
| `--dyn-tool-call-parser deepseek_v4` | Emits OpenAI-compatible structured `tool_calls` |
| `--attention-config '{"use_fp4_indexer_cache":true}'` | Blackwell FP4 indexer cache for CSA+HCA attention |
| `--kv-cache-dtype fp8` + `--block-size 256` | FP8 KV cache; block size matches the upstream recipe |
| `--tensor-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel` | DP=4 + EP across the 4 GPUs (TP=1) |
| `--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'` | Single-node DEP compilation config from the upstream recipe |
| `--max-num-seqs 256` | Concurrency cap |

### SGLang (`sglang/agg/deploy.yaml`)

| Flag | Purpose |
|------|---------|
| `--dyn-reasoning-parser deepseek_v4` | Extracts chain-of-thought into `message.reasoning_content` |
| `--dyn-tool-call-parser deepseek_v4` | Emits OpenAI-compatible structured `tool_calls` |
| `--trust-remote-code` | Required for the V4 architecture's custom modeling code |
| `--tp 4` | Tensor-parallel across the 4 GPUs of one node |
| `--moe-runner-backend flashinfer_mxfp4` | MXFP4 MoE kernel via FlashInfer for the V4 expert weights |
| `--speculative-algo EAGLE` + `--speculative-num-steps 3` + `--speculative-eagle-topk 1` + `--speculative-num-draft-tokens 4` | EAGLE MTP speculative decoding (3 draft steps, top-1 over the EAGLE head, 4 draft tokens per step) |
| `--chunked-prefill-size 4096` | Chunk long prompts at 4k tokens for steady-state decode interleaving |
| `--disable-flashinfer-autotune` | Skip per-shape autotuning at startup; the dsv4 base ships pre-tuned defaults |

## Model Details

| | |
|---|---|
| **Model** | `deepseek-ai/DeepSeek-V4-Flash` (MoE, 284B total / 13B active) |
| **Checkpoint** | Mixed FP4 (expert weights) + FP8 (attention, norm, router) |
| **Attention** | Hybrid CSA + HCA with Blackwell FP4 indexer cache |

Recipe-level (per-variant) settings:

| | vLLM (`vllm-agg`) | SGLang (`sglang-agg`) |
|---|---|---|
| **Backend image** | Custom overlay on `vllm/vllm-openai:deepseekv4-cu130` | Prebuilt `nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.2.0-sglang-deepseek-v4-b200-dev.1` |
| **Parallelism** | DP=4 + Expert Parallel, TP=1 | TP=4 |
| **MoE backend** | vLLM's V4 expert kernel (FP4) | FlashInfer MXFP4 |
| **KV cache** | FP8, block size 256 | engine default |
| **Speculative decoding** | — | EAGLE MTP (3 steps / 4 draft tokens) |

## Verifying Reasoning

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

## Verifying Tool Calling

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

## Notes

### Common

- **Storage class.** Update `storageClassName` in `model-cache/model-cache.yaml` to a RWX class that can serve the PVC to Frontend and worker pods.
- **Model size.** `deepseek-ai/DeepSeek-V4-Flash` is ~160 GB on disk (46 safetensors shards in FP4+FP8 mixed form). The 400Gi PVC leaves headroom for HF cache metadata and one alternate revision.
- **Parser flags.** Use the Dynamo variants on the worker (`--dyn-reasoning-parser`, `--dyn-tool-call-parser`). Each engine's native `--reasoning-parser` / `--tool-call-parser` are engine-side and do not feed the Dynamo OpenAI renderer.
- **Offline model cache.** Both workers run with `HF_HUB_OFFLINE=1` so the engine reads cached weights from the PVC and never contacts the HF Hub at startup. The HF token secret is mounted defensively; it isn't required at runtime once the download Job has completed.
- **First launch is slow.** Decode workers load weights and warm CUDA graphs / DeepGEMM kernels on first launch; the manifests' startup probes allow up to ~60 min (`failureThreshold: 360` at `periodSeconds: 10`).

### vLLM-specific

- **Image tag.** `vllm/agg/deploy.yaml` ships with `nvcr.io/nvidia/ai-dynamo/vllm-runtime:my-tag`. Replace it with your built Dynamo + vLLM (DeepSeek-V4) image — see Prerequisite 4.
- **Engine-ready timeout.** `VLLM_ENGINE_READY_TIMEOUT_S=3600` is set to match the startup probe.
- **DP stability.** `VLLM_RANDOMIZE_DP_DUMMY_INPUTS=1` and `VLLM_SKIP_P2P_CHECK=1` mirror the DeepSeek-R1 vLLM recipe and stabilize DP dummy inputs on Blackwell.

### SGLang-specific

- **Prebuilt image.** `sglang/agg/deploy.yaml` already references the public NGC tag `nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.2.0-sglang-deepseek-v4-b200-dev.1`. To rebuild (custom Dynamo branch, different SGLang base, etc.), see [`recipes/deepseek-v4/container/README.md`](../container/README.md).
- **DeepGEMM / FlashInfer warmup.** `SGLANG_JIT_DEEPGEMM_PRECOMPILE=0` + `SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1` skip the slow precompile and use the fast warmup path. `--disable-flashinfer-autotune` skips per-shape FlashInfer autotuning at startup; the dsv4 base ships pre-tuned defaults.
- **NCCL / Gloo.** `NCCL_CUMEM_ENABLE=1` is set for V4 NCCL collectives on Blackwell. `GLOO_SOCKET_IFNAME=eth0` pins Gloo to the standard pod interface.

## Sibling Recipe

[DeepSeek-V4-Pro](../deepseek-v4-pro/) is the larger sibling (1.6T / 49B active, 1M context, 8x B200) and shares the same dsv4 vLLM and SGLang container images.
