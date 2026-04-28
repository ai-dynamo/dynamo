<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek-V4-Flash Recipe

Aggregated-serving recipe for **DeepSeek-V4-Flash** on Dynamo. Three backends are documented side by side: **vLLM**, **SGLang**, and **TensorRT-LLM**. All are single-replica decode-only deployments that fill 4 of 8 GPUs on a B200 node.

| Variant | Backend | Manifest | GPUs | Topology | Container |
|---------|---------|----------|------|----------|-----------|
| **vllm-agg**   | vLLM        | [`vllm/agg/deploy.yaml`](vllm/agg/deploy.yaml)     | 4x B200 | DP=4 + Expert Parallel, TP=1                  | Standard Dynamo vLLM runtime image |
| **sglang-agg** | SGLang      | [`sglang/agg/deploy.yaml`](sglang/agg/deploy.yaml) | 4x B200 | TP=4, MXFP4 MoE via FlashInfer, EAGLE MTP 3/4 | Prebuilt NGC image; optional [custom build](../container/) |
| **trtllm-agg** | TensorRT-LLM | [`trtllm/agg/deploy.yaml`](trtllm/agg/deploy.yaml) | 4x B200 | TP=4 + EP=4, Attention DP, MoE backend = TRTLLM | Custom Dynamo TRT-LLM runtime image (see Notes) |

Status: **Experimental** (Day-0). Modality: text only.

## What works today

All three backends serve V4-Flash's core path on B200x4: chat completion (incl. streaming), tool calling, and reasoning extraction. The main differences are about **packaging maturity** (which container you can pull vs. build) and a couple of feature caveats.

| Capability | vLLM (`vllm-agg`) | SGLang (`sglang-agg`) | TensorRT-LLM (`trtllm-agg`) |
|---|:---:|:---:|:---:|
| **Use a prebuilt image (no build)** | ❌ build it | ✅ pull it | ❌ build it |
| Chat completion (sync + streaming) | ✅ | ✅ | ✅ |
| Tool calling (`message.tool_calls`) | ✅ | ✅ | ✅ |
| Reasoning extraction (`message.reasoning_content`) | ✅ | ✅ | ✅ |
| Speculative decoding (EAGLE MTP) | — | ✅ | — |
| KV-cache event publishing | ✅ | ✅ | ❌ not yet |

**TL;DR.** Want it running with the least friction? Pick **SGLang** — the manifest pulls a prebuilt NGC image and works as-is. The **vLLM** and **TensorRT-LLM** variants both require a one-time custom container build until V4 support lands in their public images.

### Known limitations per backend

**TensorRT-LLM** (`trtllm-agg`):
- 🔧 *Custom image required for now.* The public `tensorrtllm-runtime` image will bundle V4 once [TensorRT-LLM PR #13568](https://github.com/NVIDIA/TensorRT-LLM/pull/13568) lands (lifts `TOKENIZER_ALIASES` to module level).
- 📌 *Snapshot SHA pinned in the YAML.* `--model-path` points at a specific HuggingFace snapshot directory — a workaround for [`huggingface/transformers#44843`](https://github.com/huggingface/transformers/issues/44843) (offline-mode regression in transformers 4.57.x). If HuggingFace publishes a new commit for `deepseek-ai/DeepSeek-V4-Flash`, update the SHA in `trtllm/agg/deploy.yaml` before applying. See the pre-flight check in [TensorRT-LLM-specific notes](#tensorrt-llm-specific).
- ❌ *KV-cache event publishing not yet supported.* V4's sparse-MLA cache manager asserts the event buffer is off, so `--publish-events-and-metrics` is intentionally omitted from the worker args.

**vLLM** (`vllm-agg`):
- 🔧 *Custom image required for now.* Render and build the standard Dynamo vLLM runtime image — see [Prerequisites](#prerequisites) step 4.
- 🐢 *First launch is slow (~60 min)* — weight load + FlashInfer autotune + cudagraph warmup. The startup probe is sized for this.

**SGLang** (`sglang-agg`):
- ✅ *No image build needed.* The manifest pulls `nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.2.0-sglang-deepseek-v4-b200-dev.1` directly.
- 🐢 *First launch is slow (~60 min)* — weight load + DeepGEMM warmup + cudagraph warmup. The startup probe is sized for this.

## Prerequisites

1. **Dynamo Platform installed** — see the [Kubernetes Deployment Guide](../../../docs/kubernetes/README.md).
2. **GPU cluster** with at least 4 B200 GPUs available on one node.
3. **HuggingFace token** with access to `deepseek-ai/DeepSeek-V4-Flash`.
4. **Container image.** Pick the path that matches your variant:

   - **SGLang** (`sglang-agg`): the manifest pulls the prebuilt NGC image `nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.2.0-sglang-deepseek-v4-b200-dev.1` directly — **no build step required.** To rebuild from source (e.g. to pin a custom Dynamo branch or a different SGLang base), see the shared [`recipes/deepseek-v4/container/README.md`](../container/README.md).

   - **vLLM** (`vllm-agg`): Build the standard Dynamo vLLM runtime image per [`<repo_root>/container/README.md`](../../../container/README.md):

     ```bash
     container/render.py --framework vllm --target runtime --output-short-filename
     docker build -t dynamo:latest-vllm-runtime -f container/rendered.Dockerfile .
     ```

     Then set the `image:` fields in `vllm/agg/deploy.yaml` (both the Frontend and the decode worker) to your pushed image tag.

   - **TensorRT-LLM** (`trtllm-agg`): Until the public `tensorrtllm-runtime` image bundles a TRT-LLM build that includes DeepSeek V4 support (TensorRT-LLM PR [#13568](https://github.com/NVIDIA/TensorRT-LLM/pull/13568) lifts `TOKENIZER_ALIASES` to module level; see also feat/mewtwo + MR 10189), this variant requires a custom build. The build pattern (TRT-LLM wheel + Dynamo runtime overlay) is captured in the working 04/27 build flow. Set the `image:` fields in `trtllm/agg/deploy.yaml` (both the Frontend and the decode worker) to your built image tag.

## Quick Start

Common setup (run once — applies to all three variants):

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

### Deploy — TensorRT-LLM (`trtllm-agg`)

```bash
# Update the `image:` fields in trtllm/agg/deploy.yaml to your custom Dynamo
# TRT-LLM runtime image (Prerequisite 4 — TensorRT-LLM path).
kubectl apply -f trtllm/agg/deploy.yaml -n ${NAMESPACE}

# First launch of the decode worker takes up to ~60 minutes (weight load +
# CUDA graph warmup). The startup probe is sized for this.
kubectl wait --for=condition=Ready pod \
  -l nvidia.com/dynamo-graph-deployment-name=dsv4-flash-trtllm-agg \
  -n ${NAMESPACE} --timeout=3600s
```

## Test the Deployment

Port-forward the variant you deployed:

```bash
# vLLM
kubectl port-forward svc/dsv4-flash-agg-frontend 8000:8000 -n ${NAMESPACE}

# SGLang
kubectl port-forward svc/sglang-dsv4-flash-frontend 8000:8000 -n ${NAMESPACE}

# TensorRT-LLM
kubectl port-forward svc/dsv4-flash-trtllm-agg-frontend 8000:8000 -n ${NAMESPACE}
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

### TensorRT-LLM (`trtllm/agg/deploy.yaml`)

| Flag | Purpose |
|------|---------|
| `--dyn-reasoning-parser deepseek_v4` | Extracts chain-of-thought into `message.reasoning_content` |
| `--dyn-tool-call-parser deepseek_v4` | Emits OpenAI-compatible structured `tool_calls` |
| `--tensor-parallel-size 4` + `--expert-parallel-size 4` | TP and EP across the 4 GPUs (TP must equal world_size) |
| `--max-batch-size 4`, `--max-num-tokens 8192`, `--max-seq-len 2048` | Conservative request bounds for first-launch validation |
| `--kv-block-size 128` | V4 requires `tokens_per_block` ∈ {128, 256}; matches the engine YAML |
| `--extra-engine-args /config/engine.yaml` | Injects the V4 LlmArgs (`custom_tokenizer: deepseek_v4`, `enable_attention_dp: true`, `moe_config.backend: TRTLLM`, etc.) |

`engine.yaml` (mounted from a ConfigMap at `/config`):

| Key | Why |
|---|---|
| `custom_tokenizer: deepseek_v4` | Routes through the V4 tokenizer alias (TRT-LLM MR 10189), avoiding HF AutoTokenizer's offline-mode trap |
| `enable_attention_dp: true` + `attention_dp_config` | Data-parallel attention across the 4 GPUs (per V4 serving recipe) |
| `moe_config.backend: TRTLLM` | trtllm-gen MoE kernel (PYTORCH backend fails on Blackwell) |
| `kv_cache_config.tokens_per_block: 128`, `free_gpu_memory_fraction: 0.3` | V4 KV-cache block size; conservative memory headroom for first launch |
| `cuda_graph_config.enable_padding: true` | Pad to compiled batch sizes to reduce capture overhead |

## Model Details

| | |
|---|---|
| **Model** | `deepseek-ai/DeepSeek-V4-Flash` (MoE, 284B total / 13B active) |
| **Checkpoint** | Mixed FP4 (expert weights) + FP8 (attention, norm, router) |
| **Attention** | Hybrid CSA + HCA with Blackwell FP4 indexer cache |

Recipe-level (per-variant) settings:

| | vLLM (`vllm-agg`) | SGLang (`sglang-agg`) | TRT-LLM (`trtllm-agg`) |
|---|---|---|---|
| **Backend image** | Standard Dynamo vLLM runtime | Prebuilt `nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.2.0-sglang-deepseek-v4-b200-dev.1` | Custom Dynamo TRT-LLM runtime (TRT-LLM wheel from feat/mewtwo + MR 10189) |
| **Parallelism** | DP=4 + Expert Parallel, TP=1 | TP=4 | TP=4 + EP=4 with attention DP |
| **MoE backend** | vLLM's V4 expert kernel (FP4) | FlashInfer MXFP4 | TRT-LLM (trtllm-gen kernel) |
| **KV cache** | FP8, block size 256 | engine default | tokens_per_block=128, free_gpu_memory_fraction=0.3 |
| **Speculative decoding** | — | EAGLE MTP (3 steps / 4 draft tokens) | — |

## Verifying Reasoning

Same flow on all three variants — same model, same `--dyn-reasoning-parser deepseek_v4`:

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

Same flow on all three variants:

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

- **Image tag.** `vllm/agg/deploy.yaml` ships with `nvcr.io/nvidia/ai-dynamo/vllm-runtime:my-tag`. Replace it with your built standard Dynamo vLLM runtime image — see Prerequisite 4.
- **Engine-ready timeout.** `VLLM_ENGINE_READY_TIMEOUT_S=3600` is set to match the startup probe.
- **DP stability.** `VLLM_RANDOMIZE_DP_DUMMY_INPUTS=1` and `VLLM_SKIP_P2P_CHECK=1` mirror the DeepSeek-R1 vLLM recipe and stabilize DP dummy inputs on Blackwell.

### SGLang-specific

- **Prebuilt image.** `sglang/agg/deploy.yaml` already references the public NGC tag `nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.2.0-sglang-deepseek-v4-b200-dev.1`. To rebuild (custom Dynamo branch, different SGLang base, etc.), see [`recipes/deepseek-v4/container/README.md`](../container/README.md).
- **DeepGEMM / FlashInfer warmup.** `SGLANG_JIT_DEEPGEMM_PRECOMPILE=0` + `SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1` skip the slow precompile and use the fast warmup path. `--disable-flashinfer-autotune` skips per-shape FlashInfer autotuning at startup; the dsv4 base ships pre-tuned defaults.
- **NCCL / Gloo.** `NCCL_CUMEM_ENABLE=1` is set for V4 NCCL collectives on Blackwell. `GLOO_SOCKET_IFNAME=eth0` pins Gloo to the standard pod interface.

### TensorRT-LLM-specific

- **Image is custom for now.** No public `tensorrtllm-runtime` tag bundles V4 yet. Build a TRT-LLM wheel from `feat/mewtwo` plus MR 10189, layer onto the Dynamo TRT-LLM runtime, and push to your registry. Until [TensorRT-LLM PR #13568](https://github.com/NVIDIA/TensorRT-LLM/pull/13568) lands, the wheel also needs the `TOKENIZER_ALIASES` module-level fix applied as an in-place patch (or the equivalent cherry-pick).
- **Local snapshot path for `--model-path`.** The deploy uses a local directory under `/models/hub/.../snapshots/<sha>` rather than the canonical `deepseek-ai/DeepSeek-V4-Flash` HF id. This bypasses [`huggingface/transformers#44843`](https://github.com/huggingface/transformers/issues/44843) — the `_patch_mistral_regex → is_base_mistral → model_info` path that ignores `HF_HUB_OFFLINE=1` in transformers 4.57.2/4.57.3. Once TRT-LLM bumps to a transformers release with the fix (PRs #43603 / #45444 / #45359), revert to the HF id. The OpenAI API contract is preserved either way because `--served-model-name` registers the canonical id with the frontend.
- **Snapshot SHA pre-flight.** Before `kubectl apply`, verify the pinned SHA still matches `refs/main` in the cache:
  ```bash
  kubectl exec -n ${NAMESPACE} <download-job-pod> -- \
    cat /model-store/hub/models--deepseek-ai--DeepSeek-V4-Flash/refs/main
  ```
  If HF has published a new commit, update the SHA in `trtllm/agg/deploy.yaml`.
- **`--publish-events-and-metrics` is intentionally omitted.** V4's sparse-MLA `cache_manager.py` asserts `event_buffer_max_size == 0`; the flag would set it `> 0` and crash worker init. Re-enable once V4 KV-cache supports event publishing.
- **Default request plane is TCP (no `--request-plane nats`).** Frontend defaults TCP; if the worker is set to NATS, chat completions fail with "Invalid TCP address …" because the frontend tries to parse the NATS subject as a TCP address.
- **TP=4 + EP=4.** TP must equal `world_size` (= GPU count). The engine YAML's `enable_attention_dp: true` data-parallelizes attention across the 4 ranks for higher concurrency.
- **Memory + UCX/MPI tuning** (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `UCX_TLS=tcp,self,sm,cma`, `UCX_NET_DEVICES=lo`, `OMPI_MCA_btl=self,vader,tcp`, `OMPI_MCA_pml=ob1`) matches the verified 04/27 single-node config. Adjust if your cluster's UCX/MPI setup differs.

## Sibling Recipe

[DeepSeek-V4-Pro](../deepseek-v4-pro/) is the larger sibling (1.6T / 49B active, 1M context, 8x B200) and shares the same dsv4 vLLM and SGLang container images.
