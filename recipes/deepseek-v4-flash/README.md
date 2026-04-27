# DeepSeek-V4-Flash Recipe

Aggregated-serving recipe for **DeepSeek-V4-Flash** on vLLM with Dynamo.

| Variant | Model | Status | Modality | Manifest | GPUs |
|---------|-------|--------|----------|----------|------|
| **vllm-agg** | `deepseek-ai/DeepSeek-V4-Flash` | Experimental | Text only | [`vllm/agg/vllm-dgd.yaml`](vllm/agg/vllm-dgd.yaml) | 4x B200 |

Aggregated, single-replica: 1 decode pod running DP=4 + Expert Parallel on 4 B200 GPUs (TP=1). Tested on 4 of 8 GPUs per B200 node.

## Prerequisites

1. **Dynamo Platform installed** — see the [Kubernetes Deployment Guide](../../docs/kubernetes/README.md).
2. **GPU cluster** with at least 4 B200 GPUs available on one node.
3. **HuggingFace token** with access to `deepseek-ai/DeepSeek-V4-Flash`.
4. **Dynamo + vLLM image with the DeepSeek-V4 stack.** DeepSeek-V4-Flash is not in a stock vLLM release yet. It is built in two steps:

   1. Build the Dynamo vLLM runtime image locally per [`<repo_root>/container/README.md`](../../container/README.md) (this produces the local tag `dynamo:latest-vllm-runtime`).
   2. Build the DeepSeek-V4-Flash overlay on top of it using [`container/Dockerfile.dsv4`](container/Dockerfile.dsv4). See [`container/README.md`](container/README.md) for build args and troubleshooting. From the repo root:

      ```bash
      docker build -f recipes/deepseek-v4-flash/container/Dockerfile.dsv4 \
        -t <your-registry>/vllm-dsv4:<tag> .
      ```

   Then set the `image:` fields in `vllm/agg/vllm-dgd.yaml` (both the frontend and decode workers) to `<your-registry>/vllm-dsv4:<tag>`.

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
# The PVC requests 400Gi; DeepSeek-V4-Flash is ~160GB on disk (46 safetensors shards,
# FP4+FP8 mixed) and typically takes 30-60 min to download on first apply.
kubectl apply -f model-cache/model-cache.yaml -n ${NAMESPACE}
kubectl apply -f model-cache/model-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=7200s

# Update the `image:` fields in vllm/agg/vllm-dgd.yaml to your Dynamo + vLLM build.

# Deploy
kubectl apply -f vllm/agg/vllm-dgd.yaml -n ${NAMESPACE}

# First launch of the decode worker takes up to ~60 minutes (weight load +
# FlashInfer autotune + cudagraph warmup). The startup probe is sized for this.
kubectl wait --for=condition=Ready pod \
  -l nvidia.com/dynamo-graph-deployment-name=dsv4-flash-agg \
  -n ${NAMESPACE} --timeout=3600s
```

## Test the Deployment

```bash
kubectl port-forward svc/dsv4-flash-agg-frontend 8000:8000 -n ${NAMESPACE}

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-V4-Flash",
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
| `--tensor-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel` | DP=4 + EP across the 4 GPUs (TP=1) |
| `--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'` | Single-node DEP compilation config from the upstream recipe |
| `--max-num-seqs 256` | Concurrency cap |

## Model Details

| | |
|---|---|
| **Model** | `deepseek-ai/DeepSeek-V4-Flash` (MoE, 284B total / 13B active) |
| **Checkpoint** | Mixed FP4 (expert weights) + FP8 (attention, norm, router) |
| **Backend** | vLLM with the DeepSeek-V4 stack (`vllm/vllm-openai:deepseekv4-cu130`) |
| **Parallelism** | TP=1, DP=4, Expert Parallel enabled |
| **KV cache** | FP8, block size 256 |
| **Attention** | Hybrid CSA + HCA with Blackwell FP4 indexer cache |

## Verifying Reasoning

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

- **Storage class.** Update `storageClassName` in `model-cache/model-cache.yaml` to a RWX class that can serve the PVC to frontend and worker pods.
- **Model size.** `deepseek-ai/DeepSeek-V4-Flash` is ~160 GB on disk (46 safetensors shards in FP4+FP8 mixed form). The 400Gi PVC leaves headroom for HF cache metadata and one alternate revision.
- **Image tag.** The manifest ships with `nvcr.io/nvidia/ai-dynamo/vllm-runtime:my-tag`. Replace with your built Dynamo + vLLM (DeepSeek-V4) image — see Prerequisite 4.
- **First launch is slow.** The decode worker loads weights and warms CUDA graphs; the startup probe allows up to ~60 min (`failureThreshold: 360` at `periodSeconds: 10`) and `VLLM_ENGINE_READY_TIMEOUT_S=3600` is set to match.
- **Parser flags.** Use the Dynamo variants on the worker (`--dyn-reasoning-parser`, `--dyn-tool-call-parser`). vLLM's native `--reasoning-parser` / `--tool-call-parser` are engine-side and do not feed the Dynamo OpenAI renderer.
- **DP stability.** `VLLM_RANDOMIZE_DP_DUMMY_INPUTS=1` and `VLLM_SKIP_P2P_CHECK=1` mirror the DeepSeek-R1 vLLM recipe and stabilize DP dummy inputs on Blackwell.
- **Offline model cache.** The worker runs with `HF_HUB_OFFLINE=1` so vLLM reads the cached weights from the PVC and never contacts the HF Hub at startup. The HF token secret is mounted defensively; it isn't required at runtime once the download Job has completed.
