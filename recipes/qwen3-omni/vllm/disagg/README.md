# Qwen3-Omni: Disaggregated (3 stage workers + router)

Disaggregated reference for the DYN-2581 agg-vs-disagg perf benchmark. The
3-stage Qwen3-Omni-MoE pipeline is split across separate pods so each stage
gets its own GPU and scheduler:

| Stage | Process     | CUDA_VISIBLE_DEVICES | Final output     |
|-------|-------------|----------------------|------------------|
| 0     | thinker     | 0                    | text (chat path) |
| 1     | talker      | 1                    | (intermediate)   |
| 2     | code2wav    | 1                    | audio            |

The omni router orchestrates the DAG; chat requests stop at stage 0, audio
requests run the full pipeline through stage 2. The frontend listens on
port 8000.

## Single-pod layout (why)

The Qwen3-Omni stage YAML uses the **SharedMemoryConnector** for stage-to-stage
hand-off, which depends on `/dev/shm`. K8s Pods don't share `/dev/shm` across
each other, so all stage workers + router + frontend are colocated in one
container backed by an `emptyDir{medium: Memory}` volume mounted at `/dev/shm`.
This mirrors `examples/backends/vllm/launch/disagg_omni_qwen3.sh` exactly.

The deploy ships as plain k8s `Deployment` + `Service` (not a
`DynamoGraphDeployment`) for the same reason: a DynamoGraphDeployment would
spawn one Pod per service entry, breaking SHM.

## Hardware

- **2x NVIDIA H200 (141 GB HBM3e)** in a single Pod. Thinker takes GPU 0;
  talker + code2wav share GPU 1 (per the stage config). On 80 GB H100 you
  may need a third GPU and adjusted `gpu_memory_utilization`.
- ≥ 200 GiB host memory.

## Prerequisites

1. Dynamo Platform installed (operator only — this deploy doesn't depend on it,
   but the cluster's PVCs/secrets do).
2. Pre-existing `model-cache` and `compilation-cache` PVCs.
3. `hf-token-secret` Secret in the target namespace.
4. `nvcr-imagepullsecret` Secret to pull the Dynamo vLLM runtime image used by
   `deploy.yaml`: `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0-deepseek-v4-cuda13-dev.2`.
   The launcher installs `vllm-omni` from the pinned `v0.20.0rc1` tag at startup:

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
  -l app=disagg-qwen3-omni \
  -n ${NAMESPACE} --timeout=1800s
```

## Verify

```bash
kubectl port-forward svc/disagg-qwen3-omni 8000:8000 -n ${NAMESPACE}

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
