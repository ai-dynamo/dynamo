# DeepSeek-V4-Pro Recipe

Aggregated-serving recipe for **DeepSeek-V4-Pro** on vLLM with Dynamo.

| Variant | Model | Status | Modality | Manifest | GPUs |
|---------|-------|--------|----------|----------|------|
| **vllm-agg** | `deepseek-ai/DeepSeek-V4-Pro` | Experimental | Text only | [`vllm/agg/vllm-dgd.yaml`](vllm/agg/vllm-dgd.yaml) | 8x B200 |

Aggregated, single-replica: 1 decode pod running TP=8 + Expert Parallel on all 8 GPUs of one node.

## Prerequisites

1. **Dynamo Platform installed** — see the [Kubernetes Deployment Guide](../../docs/kubernetes/README.md).
2. **GPU cluster** with at least 8 B200 GPUs available on one node (TP=8 fills an 8-GPU box).
3. **HuggingFace token** with access to `deepseek-ai/DeepSeek-V4-Pro`.
4. **Dynamo + vLLM image with the DeepSeek-V4 stack.** DeepSeek-V4-Pro is not in a stock vLLM release yet. It is built in two steps:

   1. Build the Dynamo vLLM runtime image locally per [`<repo_root>/container/README.md`](../../container/README.md) (this produces the local tag `dynamo:latest-vllm-runtime`).
   2. Build the DeepSeek-V4-Pro overlay on top of it using [`container/Dockerfile.dsv4`](container/Dockerfile.dsv4). See [`container/README.md`](container/README.md) for build args and troubleshooting. From the repo root:

      ```bash
      docker build -f recipes/deepseek-v4-pro/container/Dockerfile.dsv4 \
        -t <your-registry>/vllm-dsv4:<tag> .
      ```

   Then set the `image:` fields in `vllm/agg/vllm-dgd.yaml` (both the frontend and decode workers) to `<your-registry>/vllm-dsv4:<tag>`.

   > The Pro and Flash recipes share the same dsv4 image. If you've already built it for [deepseek-v4-flash](../deepseek-v4-flash/), reuse the tag here — model selection happens at runtime via `--model`.

## Quick Start

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

# Update the `image:` fields in vllm/agg/vllm-dgd.yaml to your Dynamo + vLLM build.

# Deploy
kubectl apply -f vllm/agg/vllm-dgd.yaml -n ${NAMESPACE}

# First launch of the decode worker takes up to ~90 minutes (TP=8 weight load +
# FlashInfer autotune + cudagraph warmup). The startup probe is sized for this.
kubectl wait --for=condition=Ready pod \
  -l nvidia.com/dynamo-graph-deployment-name=dsv4-pro-agg \
  -n ${NAMESPACE} --timeout=5400s
```

## Test the Deployment

```bash
kubectl port-forward svc/dsv4-pro-agg-frontend 8000:8000 -n ${NAMESPACE}

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-V4-Pro",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

## Recipe Details

The worker command lives in `vllm/agg/vllm-dgd.yaml`. Key flags and why they're there:

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

### Why TP=8 (not DP=4 like Flash)?

DeepSeek-V4-Pro is ~5.5x larger than Flash on disk (~865 GB vs. ~160 GB). With FP4+FP8 mixed weights it does not fit in 4 ranks at typical batch shapes, so the upstream tested shape for Pro is **TP=8 across all 8 GPUs of one node**. Expert Parallel is still enabled on top of TP — TP shards the dense (attention/router/norm) weights, EP shards the experts.

## Model Details

Sourced from the [`deepseek-ai/DeepSeek-V4-Pro` model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) (preview release):

| | |
|---|---|
| **Model** | `deepseek-ai/DeepSeek-V4-Pro` (MoE, 1.6T total / 49B active per token) |
| **Context length** | 1M tokens |
| **Checkpoint** | Mixed precision — MoE expert weights in FP4; most other parameters in FP8 |
| **Attention** | Hybrid Compressed Sparse Attention (CSA) + Heavily Compressed Attention (HCA). Recipe enables the Blackwell FP4 indexer cache via `--attention-config '{"use_fp4_indexer_cache":true}'` |
| **Residual path** | Manifold-Constrained Hyper-Connections (mHC) |
| **Reasoning modes** | Three effort levels exposed via `chat_template_kwargs`: `{}` (Non-think), `{"thinking":true,"reasoning_effort":"high"}` (Think High), `{"thinking":true,"reasoning_effort":"max"}` (Think Max — needs `--max-model-len >= 393216`) |
| **Long-context efficiency** | Per the model card, ~27% of the per-token inference FLOPs and ~10% of the KV cache vs. DeepSeek-V3.2 at 1M context |
| **License** | MIT |

Recipe-level (not model-card) settings in this deployment:

| | |
|---|---|
| **Backend** | vLLM with the DeepSeek-V4 stack (`vllm/vllm-openai:deepseekv4-cu130`) |
| **Parallelism** | TP=8, Expert Parallel enabled |
| **KV cache** | FP8, block size 256 |

## Verifying Reasoning

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

- **Storage class.** Update `storageClassName` in `model-cache/model-cache.yaml` to a RWX class that can serve the PVC to frontend and worker pods.
- **Model size.** `deepseek-ai/DeepSeek-V4-Pro` is ~865 GB on disk (64 safetensors shards in FP4+FP8 mixed form). The 1500Gi PVC leaves ~1.7x headroom for HF cache metadata and one alternate revision.
- **Image tag.** The manifest ships with `nvcr.io/nvidia/ai-dynamo/vllm-runtime:my-tag`. Replace with your built Dynamo + vLLM (DeepSeek-V4) image — see Prerequisite 4.
- **First launch is slow.** The decode worker loads weights across 8 TP ranks and warms CUDA graphs; the startup probe allows up to ~90 min (`failureThreshold: 540` at `periodSeconds: 10`) and `VLLM_ENGINE_READY_TIMEOUT_S=5400` is set to match.
- **Parser flags.** Use the Dynamo variants on the worker (`--dyn-reasoning-parser`, `--dyn-tool-call-parser`). vLLM's native `--reasoning-parser` / `--tool-call-parser` are engine-side and do not feed the Dynamo OpenAI renderer.
- **Offline model cache.** The worker runs with `HF_HUB_OFFLINE=1` so vLLM reads the cached weights from the PVC and never contacts the HF Hub at startup. The HF token secret is mounted defensively; it isn't required at runtime once the download Job has completed.
- **Sibling recipe.** [DeepSeek-V4-Flash](../deepseek-v4-flash/) is the smaller sibling (284B / 13B active, DP=4 + EP on 4 B200 GPUs) and uses the same dsv4 container image.
