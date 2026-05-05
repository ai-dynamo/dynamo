# Qwen3-Omni: Aggregated (one AsyncOmni instance, text + audio out)

Single-deployment aggregated reference for the DYN-2581 agg-vs-disagg perf
benchmark. One `dynamo.vllm.omni` worker hosts all stages
(thinker -> talker -> code2wav) inside one AsyncOmni and serves both text
output (chat) and audio output (TTS / multimodal-in -> audio) from the same
HTTP endpoint.

## Hardware

- **1x NVIDIA H200 (141 GB HBM3e)** is sufficient for Qwen3-Omni-30B-A3B in
  aggregated mode (one AsyncOmni process; thinker / talker / code2wav share
  the same device and partition GPU memory internally). On 80 GB H100 you'll
  want 2 GPUs and `--enforce-eager` to avoid OOM.
- ≥ 150 GiB host memory.

## Prerequisites

1. Dynamo Platform installed — see [Kubernetes Deployment Guide](../../../../docs/kubernetes/README.md).
2. Pre-existing `model-cache` and `compilation-cache` PVCs.
3. `hf-token-secret` Secret in the target namespace.
4. `nvcr-imagepullsecret` Secret to pull the Dynamo vLLM runtime image used by
   `deploy.yaml`: `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0-deepseek-v4-cuda13-dev.2`.
   The worker installs `vllm-omni` from the pinned `v0.20.0rc1` tag at startup.

```bash
export NAMESPACE=your-namespace

kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="$HF_TOKEN" \
  -n ${NAMESPACE}

# NGC API key (https://org.ngc.nvidia.com/setup/api-key) — username is literally `$oauthtoken`.
kubectl create secret docker-registry nvcr-imagepullsecret \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password="$NGC_API_KEY" \
  -n ${NAMESPACE}
```

## Deploy

```bash
kubectl apply -f deploy.yaml -n ${NAMESPACE}
kubectl wait --for=condition=ready pod \
  -l nvidia.com/dynamo-graph-deployment-name=agg-qwen3-omni \
  -n ${NAMESPACE} --timeout=1800s
```

## Verify

```bash
kubectl port-forward svc/agg-qwen3-omni-frontend 8000:8000 -n ${NAMESPACE}

# text output (multimodal-in -> text)
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
  "messages": [{"role":"user","content":"Hello in one sentence."}],
  "max_tokens": 64
}' | jq

# audio output
curl -s http://localhost:8000/v1/audio/speech -H 'Content-Type: application/json' -d '{
  "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
  "input": "Hello, this is Qwen3-Omni speaking through Dynamo.",
  "voice": "ethan"
}' -o dynamo-audio.wav
```

## Benchmarking

The mixed-modality sweep that drives this deployment lives at
[`benchmarks/omni/qwen3/`](../../../../benchmarks/omni/qwen3/).
