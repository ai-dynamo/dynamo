# Qwen3-Omni Async Disaggregated Single-Node H200

This recipe runs `Qwen/Qwen3-Omni-30B-A3B-Instruct` as a vLLM-Omni async
disaggregated audio pipeline in one Kubernetes pod with file discovery, TCP
request plane, and ZMQ event plane.

The primary reproduced comparison is:

- `1h1t1c`: 1 thinker, 1 talker, 1 code2wav on 3 H200 GPUs.
- `1h2t1c`: 1 thinker, 2 talkers, 1 code2wav on 4 H200 GPUs.

Validated runtime image:

`nvcr.io/nvidian/dynamo-dev/dynamo:vllm-runtime-ptarasiewicz-dyn3068-qwen3-async-38584b4-r20-current-omni-20260602`

## Files

- `deploy-baseline-1h1t1c.yaml`: baseline deployment.
- `deploy-scaled-1h2t1c.yaml`: scaled deployment with one additional talker.
- `bench-baseline-1h1t1c-heavy-c80-c128.yaml`: baseline benchmark sweep.
- `bench-scaled-1h2t1c-heavy-c256-c384.yaml`: scaled benchmark sweep.
- `report.md`: validation notes and benchmark comparison.

## Cluster Assumptions

The manifests assume the Nebius `ptarasiewicz-test` namespace conventions:

- H200 node selector `nvidia.com/gpu.product=NVIDIA-H200`
- model cache PVC `model-cache`
- image pull secrets `nvcr-dynamo-dev-pullsecret` and `nvcr-imagepullsecret`
- Hugging Face secret `hf-token-secret`

## Workload

The benchmark sends streaming OpenAI-compatible chat completion requests with
`modalities=["audio"]`, WAV output, voice `Chelsie`, and `max_tokens=512`.

Prompt:

```text
Read the following paragraph clearly and naturally: Independent stage scaling lets a disaggregated multimodal serving stack add capacity to the stage that is saturated. In this workload the audio talker must produce a longer spoken response, while the thinker and code to waveform stages should remain shared. This benchmark repeats the same paragraph to measure sustained audio generation throughput under concurrent requests.
```

## Expected Results

The recorded validation artifacts are under `results/`.

| Recipe | GPUs | Artifact | Best concurrency | OK/fail | Req/s | Audio s/s | Audio s/s/GPU |
|---|---:|---|---:|---:|---:|---:|---:|
| `1h1t1c` | 3 | `baseline-1t1t1c-r20-retry-c80-c128-benchmark.json` | 128 | 256/0 | 3.015 | 96.578 | 32.193 |
| `1h2t1c` | 4 | `rate-match-1h2t1c-heavy-c256-c384.json` | 256 | 512/0 | 5.424 | 173.212 | 43.303 |

`1h2t1c` improves absolute generated-audio throughput by 79.4% and per-GPU
generated-audio throughput by 34.5% versus the recorded `1h1t1c` baseline.

## Reproduce

Set the namespace:

```bash
export NS=ptarasiewicz-test
```

Create or refresh the benchmark script ConfigMap:

```bash
kubectl create configmap qwen3-omni-audio-benchmark-script \
  --from-file=chat_audio_benchmark.py=benchmarks/omni/audio/chat_audio_benchmark.py \
  -n "${NS}" \
  --dry-run=client -o yaml | kubectl apply -f -
```

### Baseline: `1h1t1c`

Deploy:

```bash
kubectl apply -f recipes/qwen3-omni/vllm/async-disagg-single-node/deploy-baseline-1h1t1c.yaml -n "${NS}"
kubectl rollout status deploy/qwen3-omni-async-1h1t1c -n "${NS}" --timeout=1800s
```

Optional readiness check:

```bash
kubectl port-forward svc/qwen3-omni-async-1h1t1c 18000:8000 -n "${NS}"
curl -fsS http://127.0.0.1:18000/v1/models
```

Run the benchmark job:

```bash
kubectl apply -f recipes/qwen3-omni/vllm/async-disagg-single-node/bench-baseline-1h1t1c-heavy-c80-c128.yaml -n "${NS}"
kubectl wait --for=condition=Complete job/qwen3-omni-async-1h1t1c-bench-heavy-c80-c128 -n "${NS}" --timeout=1800s
kubectl logs job/qwen3-omni-async-1h1t1c-bench-heavy-c80-c128 -n "${NS}" > recipes/qwen3-omni/vllm/async-disagg-single-node/results/repro-baseline-1h1t1c-heavy-c80-c128.log
```

Clean up:

```bash
kubectl delete job qwen3-omni-async-1h1t1c-bench-heavy-c80-c128 -n "${NS}" --ignore-not-found=true
kubectl scale deploy/qwen3-omni-async-1h1t1c --replicas=0 -n "${NS}"
```

### Scaled: `1h2t1c`

Deploy:

```bash
kubectl apply -f recipes/qwen3-omni/vllm/async-disagg-single-node/deploy-scaled-1h2t1c.yaml -n "${NS}"
kubectl rollout status deploy/qwen3-omni-async-1h2t1c -n "${NS}" --timeout=1800s
```

Optional readiness check:

```bash
kubectl port-forward svc/qwen3-omni-async-1h2t1c 18001:8000 -n "${NS}"
curl -fsS http://127.0.0.1:18001/v1/models
```

Run the benchmark job:

```bash
kubectl apply -f recipes/qwen3-omni/vllm/async-disagg-single-node/bench-scaled-1h2t1c-heavy-c256-c384.yaml -n "${NS}"
kubectl wait --for=condition=Complete job/qwen3-omni-async-1h2t1c-bench-heavy-c256-c384 -n "${NS}" --timeout=2400s
kubectl logs job/qwen3-omni-async-1h2t1c-bench-heavy-c256-c384 -n "${NS}" > recipes/qwen3-omni/vllm/async-disagg-single-node/results/repro-scaled-1h2t1c-heavy-c256-c384.log
```

Clean up:

```bash
kubectl delete job qwen3-omni-async-1h2t1c-bench-heavy-c256-c384 -n "${NS}" --ignore-not-found=true
kubectl scale deploy/qwen3-omni-async-1h2t1c --replicas=0 -n "${NS}"
```

Each benchmark log prints `__RESULT_JSON__` followed by the full JSON summary.
Use the `best` object in that JSON for the table values above.
