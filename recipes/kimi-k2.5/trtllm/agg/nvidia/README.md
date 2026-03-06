# Kimi-K2.5 Aggregated Deployment with KVBM on Kubernetes

## Prerequisites

- A Kubernetes cluster with the [Dynamo Operator](https://docs.nvidia.com/dynamo/) installed
- 8× GPU nodes (e.g. H100/H200)
- A `hf-token-secret` Secret containing your Hugging Face token
- A pre-existing `model-cache` PVC
- Replace the placeholder image tag `nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:my-tag` in `deploy-kvbm.yaml` with your actual image

## Deploy

```bash
kubectl apply -f deploy-kvbm.yaml
```

This creates:
- A **ConfigMap** (`llm-config-kimi-agg-kvbm`) with TRT-LLM engine parameters (TP=8, EP=8, FP8 KV-cache, KVBM connector).
- A **DynamoGraphDeployment** (`kimi-k25-agg-kvbm`) with a Frontend (KV-router mode) and a TrtllmWorker serving `nvidia/Kimi-K2.5-NVFP4`.

Key environment variables on the worker:

| Variable | Default | Description |
|---|---|---|
| `DYN_KVBM_CPU_CACHE_GB` | `10` | CPU cache size in GB for KVBM |
| `DYN_KVBM_METRICS` | `true` | Enable Prometheus metrics endpoint |
| `DYN_KVBM_METRICS_PORT` | `6880` | Base port for the metrics endpoint. With `enable_attention_dp: true`, each DP rank gets its own port: `base` through `base + dp_ranks - 1` (e.g. 6880–6887 for 8 DP ranks). |

## Enable Prometheus Metrics Scraping

If you have the [Prometheus Operator](https://github.com/prometheus-operator/prometheus-operator) installed, apply the PodMonitor:

```bash
kubectl apply -f podmonitor-kvbm.yaml -n monitoring
```

This scrapes `/metrics` on ports `6880`–`6887` (named `kvbm-0` through `kvbm-7`) every 5 seconds from worker pods labeled with:
- `nvidia.com/dynamo-component-type: worker`
- `nvidia.com/metrics-enabled: "true"`

> **Note:** If your Prometheus Operator watches a namespace other than `monitoring` for PodMonitors, change `metadata.namespace` in `podmonitor-kvbm.yaml` accordingly.
