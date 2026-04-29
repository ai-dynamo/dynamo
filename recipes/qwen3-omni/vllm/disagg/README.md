# Qwen3-Omni: Disaggregated (3 stage workers + router)

Disaggregated reference for the DYN-2581 agg-vs-disagg perf benchmark. The
3-stage Qwen3-Omni-MoE pipeline is split across separate pods so each stage
gets its own GPU and scheduler:

| Stage | Component   | GPU | Final output     |
|-------|-------------|-----|------------------|
| 0     | OmniThinker | 1   | text (chat path) |
| 1     | OmniTalker  | 1   | (intermediate)   |
| 2     | OmniCode2Wav| 1   | audio            |

`OmniRouter` orchestrates the DAG; chat requests stop at stage 0, audio
requests run the full pipeline through stage 2.

## Hardware

- **3x NVIDIA H100-80GB** (one per stage worker), plus a router/frontend pod
  on CPU. The thinker pod owns the heavy LLM (~MoE weights), talker is a
  smaller AR head, code2wav is a small generation head. If GPU count is the
  constraint, the talker and code2wav pods can be coscheduled on the same
  node and share a GPU — set `nvidia.com/gpu` to `"0"` on the smaller one and
  rely on `CUDA_VISIBLE_DEVICES` from the larger pod (advanced; out of scope
  for this recipe).
- ≥ 150 GiB host memory on the thinker / talker nodes.

## Prerequisites

1. Dynamo Platform installed.
2. Pre-existing `model-cache` and `compilation-cache` PVCs.
3. `hf-token-secret` in the target namespace.
4. The `qwen3-omni-stage-config` ConfigMap shipped inline in `deploy.yaml`
   (it inlines the same stage YAML colocated at
   `examples/backends/vllm/launch/stage_configs/qwen3_omni_moe.yaml`).

## Deploy

```bash
kubectl apply -f deploy.yaml -n ${NAMESPACE}
kubectl wait --for=condition=ready pod \
  -l nvidia.com/dynamo-graph-deployment-name=disagg-qwen3-omni \
  -n ${NAMESPACE} --timeout=1800s
```

## Verify

```bash
kubectl port-forward svc/disagg-qwen3-omni-frontend 8000:8000 -n ${NAMESPACE}

# text output — stops at thinker
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
  "messages": [{"role":"user","content":"Hello in one sentence."}],
  "max_tokens": 64
}' | jq

# audio output — full thinker -> talker -> code2wav pipeline
curl -s http://localhost:8000/v1/audio/speech -H 'Content-Type: application/json' -d '{
  "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
  "input": "Hello, this is Qwen3-Omni speaking through disaggregated Dynamo.",
  "voice": "ethan"
}' -o dynamo-audio.wav
```

## Benchmarking

Run the mixed-modality sweep at
[`benchmarks/omni/qwen3/`](../../../../benchmarks/omni/qwen3/) against this
deployment to compare against the [agg](../agg/) and
[vllm-serve](../vllm-serve/) recipes.
