# Kimi-K2.5 nvidia/Kimi-K2.5-NVFP4 — Aggregated Deployments on Kubernetes

> **Note:** The `nvidia/Kimi-K2.5-NVFP4` model requires a patched TensorRT-LLM container image because
> upstream TRT-LLM support for Kimi K2.5 has not yet been released. You must build the patched image before
> deploying either configuration below. See [`patch/`](patch/) for the script and instructions.

> **Text only:** The patch registers `KimiK25ForConditionalGeneration` by loading the DeepSeek-V3
> text backbone (`text_config`) only. The vision encoder is not loaded, so image inputs are not
> processed. Full multimodal support requires native upstream TRT-LLM support for Kimi K2.5.

This directory contains two aggregated deployment configurations for the `nvidia/Kimi-K2.5-NVFP4` model:

| Deployment | Manifest | Description |
|-----------|----------|-------------|
| **Standard Aggregated** | [`deploy.yaml`](deploy.yaml) | Basic aggregated serving with KV-aware routing |
| **Aggregated + KVBM** | [`deploy-kvbm.yaml`](deploy-kvbm.yaml) | Aggregated serving with CPU-offloaded KV cache (KV Block Manager) |

## Prerequisites

- A Kubernetes cluster with the [Dynamo Operator](https://docs.nvidia.com/dynamo/) installed
- 8x B200 GPUs
- A `hf-token-secret` Secret containing your Hugging Face token
- A pre-existing `model-cache` PVC with the downloaded model
- A **patched container image** -- the deploy manifests ship with a placeholder `nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:my-tag-patched`. You must build a patched image and update the `image:` fields before deploying. See [patch instructions](patch/) for details.

---

## Standard Aggregated Deployment

Uses [`deploy.yaml`](deploy.yaml). This is the simpler configuration -- aggregated serving with KV-aware routing, no CPU-offloaded KV cache.

```bash
# Update the image in deploy.yaml to your patched image, then:
kubectl apply -f deploy.yaml -n ${NAMESPACE}
```

This creates:
- A **ConfigMap** (`llm-config`) with TRT-LLM engine parameters (TP=8, EP=8, FP8 KV-cache).
- A **DynamoGraphDeployment** (`kimi-k25-agg`) with a Frontend (KV-router mode) and a TrtllmWorker serving `nvidia/Kimi-K2.5-NVFP4`.

---

## Aggregated Deployment with KVBM

Uses [`deploy-kvbm.yaml`](deploy-kvbm.yaml). This configuration adds CPU-offloaded KV cache via the KV Block Manager (KVBM), which allows larger effective context by spilling KV cache to host memory.

```bash
# Update the image in deploy-kvbm.yaml to your patched image, then:
kubectl apply -f deploy-kvbm.yaml -n ${NAMESPACE}
```

This creates:
- A **ConfigMap** (`llm-config-kimi-agg-kvbm`) with TRT-LLM engine parameters (TP=8, EP=8, FP8 KV-cache, KVBM connector).
- A **DynamoGraphDeployment** (`kimi-k25-agg-kvbm`) with a Frontend (KV-router mode) and a TrtllmWorker serving `nvidia/Kimi-K2.5-NVFP4`.

### KVBM Configuration

Key environment variables on the worker:

| Variable | Default | Description |
|---|---|---|
| `DYN_KVBM_CPU_CACHE_GB` | `10` | CPU cache size in GB for KVBM |
| `DYN_KVBM_METRICS` | `true` | Enable Prometheus metrics endpoint |
| `DYN_KVBM_METRICS_PORT` | `6880` | Port for the metrics endpoint |

### Enable Prometheus Metrics Scraping

If you have the [Prometheus Operator](https://github.com/prometheus-operator/prometheus-operator) installed, apply the PodMonitor:

```bash
kubectl apply -f podmonitor-kvbm.yaml -n monitoring
```

This scrapes `/metrics` on port `6880` (named `kvbm`) every 5 seconds from worker pods labeled with:
- `nvidia.com/dynamo-component-type: worker`
- `nvidia.com/metrics-enabled: "true"`

> **Note:** If your Prometheus Operator watches a namespace other than `monitoring` for PodMonitors, change `metadata.namespace` in `podmonitor-kvbm.yaml` accordingly.
