# DeepSeek-R1 NVFP4, TensorRT-LLM Aggregated on 8x B200

Aggregated serving of DeepSeek-R1 (NVFP4) on a single 8x B200 node with
attention-DP and TP=EP=8.

## Configuration

| Field          | Value                                  |
|----------------|----------------------------------------|
| Hardware       | 1 node, 8x B200                        |
| Parallelism    | TP=8, EP=8, PP=1, attention DP enabled |
| Spec decoding  | None                                   |
| MoE backend    | CUTLASS                                |
| KV cache dtype | FP8                                    |
| Max batch size | 8192                                   |
| Concurrency    | 4096 (512 per GPU)                     |
| ISL / OSL      | 1024 / 1024                            |

## Files

- `deploy.yaml`: `DynamoGraphDeployment` and engine `ConfigMap` for one 8x B200 worker.
- `perf.yaml`: AIPerf `Job` that benchmarks the deployment at concurrency 4096, ISL/OSL 1024/1024.

## Deploy

```bash
kubectl apply -f ../../../model-cache/model-cache.yaml -n <namespace>
kubectl apply -f ../../../model-cache/model-download.yaml -n <namespace>
kubectl apply -f deploy.yaml -n <namespace>
```

Once the worker is `Ready`, run the benchmark:

```bash
kubectl apply -f perf.yaml -n <namespace>
```

## Notes

- The model is loaded from a local PVC at `/model-cache/deepseek-r1-fp4`. To use
  the HuggingFace ID directly, replace `--model-path` and set `HF_HOME` on the worker.
- `TRTLLM_ENABLE_PDL=1` enables Programmatic Dependent Launch on Blackwell.
- For lower-latency operating points reduce `CONCURRENCY_PER_GPU` in `perf.yaml`
  (e.g. 64 or 128).
