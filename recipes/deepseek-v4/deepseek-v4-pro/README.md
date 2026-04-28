<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek-V4-Pro Recipe

Aggregated-serving recipe for **DeepSeek-V4-Pro** on Dynamo. Two backends are documented side by side: **vLLM** and **SGLang**. Both are single-replica decode-only deployments that fill all 8 GPUs of a B200 node.

| Variant | Backend | Manifest | GPUs | Topology | Container |
|---------|---------|----------|------|----------|-----------|
| **vllm-agg**   | vLLM   | [`vllm/agg/deploy.yaml`](vllm/agg/deploy.yaml)     | 8x B200 | TP=8 + Expert Parallel                         | Standard Dynamo vLLM runtime image |
| **sglang-agg** | SGLang | [`sglang/agg/deploy.yaml`](sglang/agg/deploy.yaml) | 8x B200 | TP=8, MXFP4 MoE via FlashInfer, EAGLE MTP 3/4 | Prebuilt NGC image; optional [custom build](../container/) |

Status: **Experimental** (Day-0). Modality: text only.

> **⚠️ Known Day-0 issue (vLLM): thinking modes produce corrupted output.**
> On the **vLLM** variant, requests with `chat_template_kwargs: {"thinking": true, ...}` emit malformed tokens (numeric tokens spliced mid-word, occasional special-token leakage). The bug is in the OSS port of DeepSeek-V4-Pro's sparse-attention path on the dsv4 vLLM stack and does not affect DeepSeek-V4-Flash on the same stack. Until the upstream fix lands, send `chat_template_kwargs: {"thinking": false}` in chat completion requests against the vLLM variant. Tool calling, structured output, and non-thinking responses work normally. The SGLang variant uses a different attention path (MXFP4 MoE + EAGLE MTP); exercise caution and verify output if testing thinking modes there.

## Prerequisites

1. **Dynamo Platform installed** — see the [Kubernetes Deployment Guide](../../../docs/kubernetes/README.md).
2. **GPU cluster** with at least 8 B200 GPUs available on one node (TP=8 fills an 8-GPU box).
3. **HuggingFace token** with access to `deepseek-ai/DeepSeek-V4-Pro`.
4. **Container image.** Pick the path that matches your variant:

   - **SGLang** (`sglang-agg`): the manifest pulls the prebuilt NGC image `nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.2.0-sglang-deepseek-v4-b200-dev.1` directly — **no build step required.** To rebuild from source (e.g. to pin a custom Dynamo branch or a different SGLang base), see the shared [`recipes/deepseek-v4/container/README.md`](../container/README.md).

   - **vLLM** (`vllm-agg`): Build the standard Dynamo vLLM runtime image per [`<repo_root>/container/README.md`](../../../container/README.md):

     ```bash
     container/render.py --framework vllm --target runtime --output-short-filename
     docker build -t dynamo:latest-vllm-runtime -f container/rendered.Dockerfile .
     ```

     Then set the `image:` fields in `vllm/agg/deploy.yaml` (both the Frontend and the decode worker) to your pushed image tag.

   > The Pro and Flash recipes share the same image on each backend. If you've already built the vLLM or SGLang runtime for [deepseek-v4-flash](../deepseek-v4-flash/), reuse the tag here — model selection happens at runtime via `--model` (vLLM) or `--model-path` (SGLang).

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
# The PVC requests 1500Gi; DeepSeek-V4-Pro is ~865 GB on disk (64 safetensors shards,
# FP4+FP8 mixed) and typically takes 1.5-3 hours to download on first apply.
kubectl apply -f model-cache/model-cache.yaml -n ${NAMESPACE}
kubectl apply -f model-cache/model-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=14400s
```

### Deploy — vLLM (`vllm-agg`)

```bash
# Update the `image:` fields in vllm/agg/deploy.yaml to your Dynamo + vLLM build
# (Prerequisite 4 — vLLM path).
kubectl apply -f vllm/agg/deploy.yaml -n ${NAMESPACE}

# First launch of the decode worker takes up to ~90 minutes (TP=8 weight load +
# FlashInfer autotune + cudagraph warmup). The startup probe is sized for this.
kubectl wait --for=condition=Ready pod \
  -l nvidia.com/dynamo-graph-deployment-name=dsv4-pro-agg \
  -n ${NAMESPACE} --timeout=5400s
```

### Deploy — SGLang (`sglang-agg`)

```bash
# Manifest already points at the prebuilt NGC image — no image edit needed.
kubectl apply -f sglang/agg/deploy.yaml -n ${NAMESPACE}

# First launch of the decode worker takes up to ~60 minutes (TP=8 weight load +
# DeepGEMM warmup + cudagraph warmup). The startup probe is sized for this.
kubectl wait --for=condition=Ready pod \
  -l nvidia.com/dynamo-graph-deployment-name=sglang-dsv4-pro \
  -n ${NAMESPACE} --timeout=3600s
```

## Test the Deployment

Port-forward the variant you deployed:

```bash
# vLLM
kubectl port-forward svc/dsv4-pro-agg-frontend 8000:8000 -n ${NAMESPACE}

# SGLang
kubectl port-forward svc/sglang-dsv4-pro-frontend 8000:8000 -n ${NAMESPACE}
```

Either way the request shape is the same. Send `thinking: false` on the vLLM variant per the Day-0 caveat above:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-V4-Pro",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100,
    "chat_template_kwargs": {"thinking": false}
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
| `--tensor-parallel-size 8 --enable-expert-parallel` | TP=8 across 8 GPUs of one node, with EP enabled for the MoE experts |
| `--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'` | Conservative cudagraph mode appropriate for the larger Pro model (matches upstream V4-Pro example) |
| `--max-num-seqs 256` | Concurrency cap |

### SGLang (`sglang/agg/deploy.yaml`)

| Flag | Purpose |
|------|---------|
| `--dyn-reasoning-parser deepseek_v4` | Extracts chain-of-thought into `message.reasoning_content` |
| `--dyn-tool-call-parser deepseek_v4` | Emits OpenAI-compatible structured `tool_calls` |
| `--trust-remote-code` | Required for the V4 architecture's custom modeling code |
| `--tp 8` | Tensor-parallel across all 8 GPUs of one node |
| `--moe-runner-backend flashinfer_mxfp4` | MXFP4 MoE kernel via FlashInfer for the V4 expert weights |
| `--speculative-algo EAGLE` + `--speculative-num-steps 3` + `--speculative-eagle-topk 1` + `--speculative-num-draft-tokens 4` | EAGLE MTP speculative decoding (3 draft steps, top-1 over the EAGLE head, 4 draft tokens per step) |
| `--chunked-prefill-size 4096` | Chunk long prompts at 4k tokens for steady-state decode interleaving |
| `--disable-flashinfer-autotune` | Skip per-shape autotuning at startup; the dsv4 base ships pre-tuned defaults |

### Why TP=8 (not DP=4 like Flash)?

DeepSeek-V4-Pro is ~5.5x larger than Flash on disk (~865 GB vs. ~160 GB). With FP4+FP8 mixed weights it does not fit in 4 ranks at typical batch shapes, so the upstream tested shape for Pro is **TP=8 across all 8 GPUs of one node** on both backends. On vLLM, Expert Parallel is layered on top of TP — TP shards the dense (attention/router/norm) weights, EP shards the experts. On SGLang, the MXFP4 MoE backend handles the expert sharding internally under the same TP=8 process group.

## Model Details

Sourced from the [`deepseek-ai/DeepSeek-V4-Pro` model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) (preview release):

| | |
|---|---|
| **Model** | `deepseek-ai/DeepSeek-V4-Pro` (MoE, 1.6T total / 49B active per token) |
| **Context length** | 1M tokens |
| **Checkpoint** | Mixed precision — MoE expert weights in FP4; most other parameters in FP8 |
| **Attention** | Hybrid Compressed Sparse Attention (CSA) + Heavily Compressed Attention (HCA). The vLLM variant enables the Blackwell FP4 indexer cache via `--attention-config '{"use_fp4_indexer_cache":true}'` |
| **Residual path** | Manifold-Constrained Hyper-Connections (mHC) |
| **Reasoning modes** | Three effort levels exposed via `chat_template_kwargs`: `{}` (Non-think), `{"thinking":true,"reasoning_effort":"high"}` (Think High), `{"thinking":true,"reasoning_effort":"max"}` (Think Max — needs `--max-model-len >= 393216`) |
| **Long-context efficiency** | Per the model card, ~27% of the per-token inference FLOPs and ~10% of the KV cache vs. DeepSeek-V3.2 at 1M context |
| **License** | MIT |

Recipe-level (per-variant) settings:

| | vLLM (`vllm-agg`) | SGLang (`sglang-agg`) |
|---|---|---|
| **Backend image** | Standard Dynamo vLLM runtime | Prebuilt `nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.2.0-sglang-deepseek-v4-b200-dev.1` |
| **Parallelism** | TP=8, Expert Parallel enabled | TP=8 |
| **MoE backend** | vLLM's V4 expert kernel (FP4) | FlashInfer MXFP4 |
| **KV cache** | FP8, block size 256 | engine default |
| **Speculative decoding** | — | EAGLE MTP (3 steps / 4 draft tokens) |

## Verifying Reasoning

Same flow on both variants — same model, same `--dyn-reasoning-parser deepseek_v4`. On the vLLM variant, omit `chat_template_kwargs.thinking` (or set it to `false`) per the Day-0 caveat:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-V4-Pro",
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

## Notes

### Common

- **Storage class.** Update `storageClassName` in `model-cache/model-cache.yaml` to a RWX class that can serve the PVC to Frontend and worker pods.
- **Model size.** `deepseek-ai/DeepSeek-V4-Pro` is ~865 GB on disk (64 safetensors shards in FP4+FP8 mixed form). The 1500Gi PVC leaves ~1.7x headroom for HF cache metadata and one alternate revision.
- **Parser flags.** Use the Dynamo variants on the worker (`--dyn-reasoning-parser`, `--dyn-tool-call-parser`). Each engine's native `--reasoning-parser` / `--tool-call-parser` are engine-side and do not feed the Dynamo OpenAI renderer.
- **Offline model cache.** Both workers run with `HF_HUB_OFFLINE=1` so the engine reads cached weights from the PVC and never contacts the HF Hub at startup. The HF token secret is mounted defensively; it isn't required at runtime once the download Job has completed.
- **First launch is slow.** Decode workers load weights across 8 TP ranks and warm CUDA graphs / DeepGEMM kernels on first launch; the manifests' startup probes allow ~60–90 min before failing readiness.

### vLLM-specific

- **Image tag.** `vllm/agg/deploy.yaml` ships with `nvcr.io/nvidia/ai-dynamo/vllm-runtime:my-tag`. Replace it with your built standard Dynamo vLLM runtime image — see Prerequisite 4.
- **Engine-ready timeout.** `VLLM_ENGINE_READY_TIMEOUT_S=5400` is set to match the startup probe (`failureThreshold: 540` at `periodSeconds: 10`).
- **Day-0 thinking modes.** Send `chat_template_kwargs: {"thinking": false}` until the upstream sparse-attention fix lands — see the callout at the top of this README.

### SGLang-specific

- **Prebuilt image.** `sglang/agg/deploy.yaml` already references the public NGC tag `nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.2.0-sglang-deepseek-v4-b200-dev.1`. To rebuild (custom Dynamo branch, different SGLang base, etc.), see [`recipes/deepseek-v4/container/README.md`](../container/README.md).
- **DeepGEMM / FlashInfer warmup.** `SGLANG_JIT_DEEPGEMM_PRECOMPILE=0` + `SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1` skip the slow precompile and use the fast warmup path. `--disable-flashinfer-autotune` skips per-shape FlashInfer autotuning at startup; the dsv4 base ships pre-tuned defaults.
- **NCCL / Gloo.** `NCCL_CUMEM_ENABLE=1` is set for V4 NCCL collectives on Blackwell. `GLOO_SOCKET_IFNAME=eth0` pins Gloo to the standard pod interface.

## Sibling Recipe

[DeepSeek-V4-Flash](../deepseek-v4-flash/) is the smaller sibling (284B / 13B active, 4x B200) and shares the same dsv4 vLLM and SGLang container images.
