<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek-V4-Pro Recipe

Recipes for **DeepSeek-V4-Pro** on Dynamo across two backends (**vLLM**, **SGLang**) and two hardware targets (**B200**, **GB200**). Single-node aggregated serving fills 8 GPUs of a B200 box; on GB200 the model exceeds a single 4-GPU NVL4 tray, so V4-Pro spans two GB200 trays via NVLink72 (MNNVL) — either **aggregated with TP=8 cross-node** or **disaggregated prefill/decode**.

| Variant | Backend | Hardware | Manifest | Topology | Container |
|---------|---------|----------|----------|----------|-----------|
| **vllm-agg-b200**       | vLLM   | 1 node, 8x B200 | [`vllm/agg/b200/deploy.yaml`](vllm/agg/b200/deploy.yaml)         | TP=8 + Expert Parallel (single node)                                            | Standard Dynamo vLLM runtime image |
| **vllm-agg-gb200**      | vLLM   | 2 nodes, 4x GB200 each (8 total) | [`vllm/agg/gb200/deploy.yaml`](vllm/agg/gb200/deploy.yaml)       | TP=8 + Expert Parallel cross-node, MNNVL via ComputeDomain (NVLink72)           | Standard Dynamo vLLM runtime image (arm64) |
| **vllm-disagg-gb200**   | vLLM   | 2 nodes, 4x GB200 each (8 prefill + 8 decode = 16 total) | [`vllm/disagg/gb200/deploy.yaml`](vllm/disagg/gb200/deploy.yaml) | 1P + 1D, DP=8 + Expert Parallel per worker, MNNVL via ComputeDomain (NVLink72) | Standard Dynamo vLLM runtime image (arm64) |
| **sglang-agg**          | SGLang | 1 node, 8x B200 | [`sglang/agg/deploy.yaml`](sglang/agg/deploy.yaml)               | TP=8, MXFP4 MoE via FlashInfer, EAGLE MTP 3/4                                   | Prebuilt NGC image; optional [custom build](../container/) |

A perf-benchmark Job for the GB200 disagg variant is provided alongside the deploy:

| Perf Job | Notes |
|---|---|
| [`vllm/disagg/gb200/perf.yaml`](vllm/disagg/gb200/perf.yaml) | Runs `aiperf profile` against `dsv4-pro-disagg-frontend:8000` with an 8K-input / 1K-output concurrency sweep (256 / 512 / 1024). Override `CONCURRENCIES` in the Job env for a smaller smoke run. |

Status: **Experimental** (Day-0). Modality: text only.

## Prerequisites

1. **Dynamo Platform installed** — see the [Kubernetes Deployment Guide](../../../docs/kubernetes/README.md).
2. **GPU cluster.** Hardware depends on the variant:
   - **B200 variants** (`vllm-agg-b200`, `sglang-agg`): 8 B200 GPUs available on a single node (x86_64). TP=8 fills the box.
   - **GB200 variants** (`vllm-agg-gb200`, `vllm-disagg-gb200`): **2 GB200 nodes**, each with 4 GPUs (single NVL4 tray each), connected to the **same NVLink72 clique**. Nodes must be labeled `nvidia.com/gpu.product=NVIDIA-GB200` and tainted `kubernetes.io/arch=arm64:NoSchedule`. The cluster must have the **DRA / ComputeDomain controller** installed (verify with `kubectl get crd | grep computedomain`); each manifest's `ComputeDomain` CR + `resourceClaims` are how the operator co-locates the worker pod set on the same NVLink72 fabric (the agg variant places 2 pods, the disagg variant places 4).
3. **HuggingFace token** with access to `deepseek-ai/DeepSeek-V4-Pro`.
4. **Container image.** Pick the path that matches your variant:

   - **SGLang** (`sglang-agg`): the manifest pulls the prebuilt NGC image `nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.2.0-sglang-deepseek-v4-b200-dev.1` directly — **no build step required.** To rebuild from source (e.g. to pin a custom Dynamo branch or a different SGLang base), see the shared [`recipes/deepseek-v4/container/README.md`](../container/README.md).

   - **vLLM** (`vllm-agg-b200` or `vllm-disagg-gb200`): Build the standard Dynamo vLLM runtime image per [`<repo_root>/container/README.md`](../../../container/README.md):

     ```bash
     container/render.py --framework vllm --target runtime --output-short-filename
     docker build -t dynamo:latest-vllm-runtime -f container/rendered.Dockerfile .
     ```

     For the GB200 variant, build with `--platform linux/arm64`. Then set the `image:` fields in your chosen `vllm/.../deploy.yaml` (Frontend + both worker pods on disagg) to your pushed image tag.

   > The Pro and Flash recipes share the same image on each backend and architecture. If you've already built the vLLM or SGLang runtime for [deepseek-v4-flash](../deepseek-v4-flash/), reuse the tag here — model selection happens at runtime via `--model` (vLLM) or `--model-path` (SGLang).

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

### Deploy — vLLM B200 (`vllm-agg-b200`)

```bash
# Update the `image:` fields in vllm/agg/b200/deploy.yaml to your Dynamo + vLLM
# build (Prerequisite 4 — vLLM path).
kubectl apply -f vllm/agg/b200/deploy.yaml -n ${NAMESPACE}

# First launch of the decode worker takes up to ~90 minutes (TP=8 weight load +
# FlashInfer autotune + cudagraph warmup). The startup probe is sized for this.
kubectl wait --for=condition=Ready pod \
  -l nvidia.com/dynamo-graph-deployment-name=dsv4-pro-agg \
  -n ${NAMESPACE} --timeout=5400s
```

### Deploy — vLLM GB200 agg (`vllm-agg-gb200`)

```bash
# Update the `image:` fields in vllm/agg/gb200/deploy.yaml to your arm64
# Dynamo + vLLM build (Prerequisite 4 — vLLM path).
kubectl apply -f vllm/agg/gb200/deploy.yaml -n ${NAMESPACE}

# First launch of the decode worker takes up to ~90 minutes (TP=8 weight load
# + NCCL bring-up over MNNVL + cudagraph capture across 2 nodes).
kubectl wait --for=condition=Ready pod \
  -l nvidia.com/dynamo-graph-deployment-name=dsv4-pro-agg \
  -n ${NAMESPACE} --timeout=5400s
```

### Deploy — vLLM GB200 disagg (`vllm-disagg-gb200`)

```bash
# Update the `image:` fields in vllm/disagg/gb200/deploy.yaml to your arm64
# Dynamo + vLLM build (Prerequisite 4 — vLLM path).
kubectl apply -f vllm/disagg/gb200/deploy.yaml -n ${NAMESPACE}

# First launch of each leader takes up to ~90 minutes (DP=8 weight load +
# NIXL/UCX setup + NCCL bring-up over MNNVL + cudagraph capture).
kubectl wait --for=condition=Ready pod \
  -l nvidia.com/dynamo-graph-deployment-name=dsv4-pro-disagg \
  -n ${NAMESPACE} --timeout=5400s

# Optional: run the perf benchmark Job (8K input / 1K output sweep at c=256/512/1024)
# kubectl apply -f vllm/disagg/gb200/perf.yaml -n ${NAMESPACE}
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
# vLLM B200 agg or GB200 agg (same DGD/service name — only one of these
# variants can be deployed in a given namespace at a time)
kubectl port-forward svc/dsv4-pro-agg-frontend 8000:8000 -n ${NAMESPACE}

# vLLM GB200 disagg
kubectl port-forward svc/dsv4-pro-disagg-frontend 8000:8000 -n ${NAMESPACE}

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

### vLLM B200 agg (`vllm/agg/b200/deploy.yaml`)

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

### vLLM GB200 agg (`vllm/agg/gb200/deploy.yaml`)

V4-Pro at ~865 GB on disk does not fit a single GB200 NVL4 tray (~768 GB HBM across 4 GPUs), so the GB200 agg recipe stretches one tensor-parallel group across **two** trays — the cross-node TP all-reduce / all-gather flows over NVLink72 (MNNVL), not RoCE. The two pods are co-located on the same NVLink72 clique by the DRA `ComputeDomain` controller.

| Flag / env | Purpose |
|---|---|
| `--tensor-parallel-size 8 --enable-expert-parallel` | TP=8 + EP across 2 nodes (4 GPUs/node × 2 nodes) — no DP. |
| `--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"],"pass_config":{"fuse_allreduce_rms":false}}'` | FULL_AND_PIECEWISE cudagraph + all custom ops; `fuse_allreduce_rms:false` avoids a non-fatal FlashInfer trtllm allreduce-norm workspace warning at startup. |
| `--attention-config '{"use_fp4_indexer_cache":true}'` + `--moe-backend deep_gemm_mega_moe` | Blackwell FP4 indexer cache + DeepGEMM "mega MoE" kernel — same kernels as the B200 agg variant. |
| `NCCL_MNNVL_ENABLE=1`, `UCX_CUDA_IPC_ENABLE_MNNVL=y`, `UCX_TLS=cuda_copy,cuda_ipc,tcp`, `NCCL_NVLS_ENABLE=1`, `NCCL_P2P_LEVEL=NVL` | Enable cross-node NVLink72 / MNNVL fabric. Required because the TP=8 process group spans 2 nodes. |
| `ComputeDomain` CR + `resourceClaimTemplate` (top of manifest) | DRA primitive that asks the scheduler to allocate an MNNVL channel on demand and co-locate the 2-pod set on the same NVLink72 clique. |
| (no `--data-parallel-rpc-port`) | TP-only — torch.distributed master binds `MASTER_PORT` (29500) for the cross-node rendezvous, which also satisfies the operator's `wait-for-leader-mp` TCP probe. |

### vLLM GB200 disagg (`vllm/disagg/gb200/deploy.yaml`)

V4-Pro at ~865 GB on disk does not fit a single GB200 NVL4 tray (~768 GB HBM across 4 GPUs), so the GB200 recipe is the **disaggregated** prefill/decode shape: one prefill replica spanning 2 GB200 nodes (DP=8 + EP) and one decode replica spanning 2 GB200 nodes (DP=8 + EP), all four pods placed on the same NVLink72 clique by the DRA `ComputeDomain` controller.

| Flag / env | Purpose |
|---|---|
| `--data-parallel-size 8 --enable-expert-parallel --tensor-parallel-size 1` | DP=8 + EP across 2 nodes per worker (4 GPUs/node × 2 nodes) — TP=1 |
| `--data-parallel-rpc-port 29500` | Binds vLLM's DP coordinator on `:29500`. The dynamo operator's `wait-for-leader-mp` init container does a TCP probe to `<leader>:29500` and blocks worker startup until that port accepts; pinning the DP coord port to 29500 makes the real RPC server satisfy the probe (cleaner than parking a placeholder listener). |
| `--disaggregation-mode prefill` (prefill only) + `--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'` | Prefill writes KV blocks via NIXL; decode reads them. NIXL's UCX active-messages control plane goes over TCP (`UCX_TLS=cuda_ipc,cuda_copy,tcp`) while bulk KV flows over MNNVL. |
| `NCCL_MNNVL_ENABLE=1`, `UCX_CUDA_IPC_ENABLE_MNNVL=y`, `NCCL_NVLS_ENABLE=1`, `NCCL_P2P_LEVEL=NVL` | Enable cross-node NVLink72 / MNNVL fabric. Required because the prefill and decode workers each span 2 nodes. |
| `ComputeDomain` CR + `resourceClaimTemplate` (top of manifest) | DRA primitive that asks the scheduler to allocate an MNNVL channel on demand and co-locate the 4-pod set on the same NVLink72 clique. Without it, NCCL bring-up across pods fails — TCP-only fallback is not viable for DP=8 cross-pod all-reduce. |
| `--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'` (decode), `--enforce-eager` (prefill) | Conservative compile/graph config — matches the B200 agg variant's V4-Pro tuning. |
| `--max-model-len 9280`, `--max-num-seqs 16` (prefill) / `128` (decode) | Capped to the 8K-input / 1K-output benchmark shape. |

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

DeepSeek-V4-Pro is ~5.5x larger than Flash on disk (~865 GB vs. ~160 GB). With FP4+FP8 mixed weights it does not fit in 4 ranks at typical batch shapes, so the upstream tested shape for Pro is **TP=8** on both backends — across all 8 GPUs of one B200 node, or across two GB200 NVL4 trays connected by NVLink72. On vLLM, Expert Parallel is layered on top of TP — TP shards the dense (attention/router/norm) weights, EP shards the experts. On SGLang, the MXFP4 MoE backend handles the expert sharding internally under the same TP=8 process group.

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

- **Image tag.** All three vLLM manifests (`vllm/agg/b200/`, `vllm/agg/gb200/`, `vllm/disagg/gb200/`) ship with `nvcr.io/nvidia/ai-dynamo/vllm-runtime:my-tag`. Replace with your built Dynamo vLLM runtime tag — see Prerequisite 4. Both GB200 manifests expect an arm64 build.
- **Engine-ready timeout.** `VLLM_ENGINE_READY_TIMEOUT_S=5400` is set to match the startup probe budget (`failureThreshold: 540` at `periodSeconds: 10`).
- **GB200: agg vs. disagg.** Both spread V4-Pro across two GB200 NVL4 trays via MNNVL/ComputeDomain. The agg variant runs one TP=8 group across both nodes (lower-latency, simpler topology, 2 pods); the disagg variant runs separate prefill and decode DP=8 workers (higher steady-state throughput at high concurrency, 4 pods). Use the agg variant for general-purpose serving and the disagg variant when prefill/decode separation pays off for the workload.

### SGLang-specific

- **Prebuilt image.** `sglang/agg/deploy.yaml` already references the public NGC tag `nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.2.0-sglang-deepseek-v4-b200-dev.1`. To rebuild (custom Dynamo branch, different SGLang base, etc.), see [`recipes/deepseek-v4/container/README.md`](../container/README.md).
- **DeepGEMM / FlashInfer warmup.** `SGLANG_JIT_DEEPGEMM_PRECOMPILE=0` + `SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1` skip the slow precompile and use the fast warmup path. `--disable-flashinfer-autotune` skips per-shape FlashInfer autotuning at startup; the dsv4 base ships pre-tuned defaults.
- **NCCL / Gloo.** `NCCL_CUMEM_ENABLE=1` is set for V4 NCCL collectives on Blackwell. `GLOO_SOCKET_IFNAME=eth0` pins Gloo to the standard pod interface.

## Sibling Recipe

[DeepSeek-V4-Flash](../deepseek-v4-flash/) is the smaller sibling (284B / 13B active, 4x B200) and shares the same dsv4 vLLM and SGLang container images.
