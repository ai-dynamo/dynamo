# Qwen3-Omni: Aggregated (one AsyncOmni instance, text + audio out)

Single-deployment aggregated reference for the DYN-2581 agg-vs-disagg perf
benchmark. One `dynamo.vllm.omni` worker hosts all stages
(thinker -> talker -> code2wav) inside one AsyncOmni and serves both text
output (chat) and audio output (TTS / multimodal-in -> audio) from the same
HTTP endpoint.

## Hardware

- **2x NVIDIA H100-80GB** (Hopper). Verified by upstream vllm-omni on the same
  SKU; talker + code2wav share the second GPU, thinker takes the first. If you
  only have a single H100, drop to `gpu: "1"` and enable
  `--enforce-eager`/`--enable-cpu-offload` for the talker — expect throughput
  to fall significantly.
- ≥ 150 GiB host memory.

## Prerequisites

1. Dynamo Platform installed — see [Kubernetes Deployment Guide](../../../../docs/kubernetes/README.md).
2. Pre-existing `model-cache` and `compilation-cache` PVCs.
3. `hf-token-secret` Secret in the target namespace.

```bash
export NAMESPACE=your-namespace
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="$HF_TOKEN" \
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
