# Qwen3-Omni: Upstream `vllm-omni serve` baseline

Single-pod baseline running upstream `vllm-omni serve` for the DYN-2581
agg-vs-disagg perf benchmark. No Dynamo frontend, no router — this is the
reference point the Dynamo agg + disagg topologies are compared against.

## Hardware

- **2x NVIDIA H200 (141 GB HBM3e)**. Matches the disagg recipe so we measure
  the Dynamo overhead at the same hardware envelope.

## Prerequisites

1. Pre-existing `model-cache` PVC.
2. `hf-token-secret` Secret in the target namespace.
3. `nvcr-imagepullsecret` Secret to pull `nvcr.io/nvstaging/ai-dynamo/vllm-runtime:nightly-<YYYYMMDD>-<sha>`
   (nightly Dynamo runtime; includes `vllm-omni` on `$PATH`):

   ```bash
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
  -l app=vllm-serve-qwen3-omni \
  -n ${NAMESPACE} --timeout=1800s
```

## Verify

```bash
kubectl port-forward svc/vllm-serve-qwen3-omni 8000:8000 -n ${NAMESPACE}

curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
  "messages": [{"role":"user","content":"Hello in one sentence."}],
  "max_tokens": 64
}' | jq

curl -s http://localhost:8000/v1/audio/speech -H 'Content-Type: application/json' -d '{
  "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
  "input": "Hello, this is vllm-omni standalone.",
  "voice": "ethan"
}' -o vllm-omni-audio.wav
```
