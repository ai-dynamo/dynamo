# Kimi-K2.5 NVFP4, TensorRT-LLM Disaggregated WideEP on 10x GB200

Disaggregated serving of `nvidia/Kimi-K2.5-NVFP4` on 10x GB200 nodes (40 GPUs),
tuned for an 8K input / 1K output workload.

## Configuration

| Field                | Value                                                     |
|----------------------|-----------------------------------------------------------|
| Hardware             | 10 nodes, 40x GB200                                       |
| Topology             | 2x CTX (DEP4) + 1x GEN (DEP32 WideEP)                     |
| Spec decoding        | None (Kimi-K2.5 has no MTP layers)                        |
| MoE backend          | CUTEDSL (with `use_low_precision_moe_combine`)            |
| KV cache dtype       | FP8                                                       |
| Max batch size (gen) | 256                                                       |
| Total concurrency    | 8192                                                      |
| ISL / OSL            | 8192 / 1024                                               |

## Files

- `deploy.yaml`: `DynamoGraphDeployment`, prefill/decode `ConfigMap`s, and `ComputeDomain` (`numNodes: 10`).
- `perf.yaml`: AIPerf `Job` benchmarking the deployment at concurrency 8192, ISL/OSL 8192/1024.

## Deploy

```bash
kubectl apply -f ../../../../model-cache/model-cache.yaml -n <namespace>
kubectl apply -f ../../../../model-cache/nvidia/model-download.yaml -n <namespace>
kubectl wait --for=condition=Complete job/model-download -n <namespace> --timeout=6000s

kubectl apply -f deploy.yaml -n <namespace>

kubectl apply -f perf.yaml -n <namespace>
```

## Notes

- Multi-node networking: GB200 NVLink is required between the GEN worker's
  8 nodes; InfiniBand or RoCE is required for the prefill-to-decode KV cache
  transfer (UCX/NIXL/MPI; this recipe uses the default backend).
- For a single-node deployment use the aggregated recipe at
  [`../../../agg/nvidia/`](../../../agg/nvidia/).
