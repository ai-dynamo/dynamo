# Kimi-K2.5 Aggregated Deployment with KVBM on Kubernetes

## Build

### Build for nScale cluster with cuda 13.1 and trtllm 1.3.0rc5.post1:
```bash
python container/render.py --framework trtllm --target runtime --output-short-filename --cuda-version 13.1

docker build -t dynamo:cuda-13.1-trtllm-runtime-rc5post1-kimi-k2.5 -f container/rendered.Dockerfile --build-context trtllm_wheel=/tmp/trtllm_wheel .

./recipes/kimi-k2.5/trtllm/agg/nvidia/patch/patch-container.sh dynamo:cuda-13.1-trtllm-runtime-rc5post1-kimi-k2.5

# tag and push to nvcr.io/nvidian/dynamo-dev/
```

### Build for Weka cluster with cuda 13.0 and trtllm 1.3.0rc3:
```bash
1. download the pip wheel which contains the trtllm branch to support nvidia kimi k2.5 from https://drive.google.com/file/d/1aBB1UA8ceyZkVe_LHw79MfgbYZFgEBfq/view

2. unzip and put to /tmp/trtllm_wheel/tensorrt_llm-1.3.0rc3-cp312-cp312-linux_x86_64.whl

3. change has_trtllm_context to "1" in container/context.yaml

4. python container/render.py --framework trtllm --target runtime --output-short-filename --cuda-version 13.0

5. docker build -t dynamo:cuda-13.0-trtllm-runtime-kimi-k2.5 -f container/rendered.Dockerfile --build-context trtllm_wheel=/tmp/trtllm_wheel .

# tag and push to nvcr.io/nvidian/dynamo-dev/
```

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
| `DYN_KVBM_METRICS_PORT` | `6880` | Port for the metrics endpoint |

## Test
```bash
kubectl port-forward -n ziqif-kvbm svc/kimi-k25-agg-kvbm-frontend 8000:8000 &

curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dummy" \
  -d '{
    "model": "nvidia/Kimi-K2.5-NVFP4",
    "messages": [{"role": "user", "content": "Say hello!"}],
    "max_tokens": 64
  }' | jq
```

## Enable Prometheus Metrics Scraping

If you have the [Prometheus Operator](https://github.com/prometheus-operator/prometheus-operator) installed, apply the PodMonitor:

```bash
kubectl apply -f podmonitor-kvbm.yaml -n monitoring
```

This scrapes `/metrics` on port `6880` (named `kvbm`) every 5 seconds from worker pods labeled with:
- `nvidia.com/dynamo-component-type: worker`
- `nvidia.com/metrics-enabled: "true"`

> **Note:** If your Prometheus Operator watches a namespace other than `monitoring` for PodMonitors, change `metadata.namespace` in `podmonitor-kvbm.yaml` accordingly.

To view metrics via Grapana UI via http://localhost:3000, do below first:
```bash
kubectl port-forward svc/kube-prometheus-stack-grafana 3000:80 -n monitoring
```

Sample cmd to scrap metrics manually:
```bash
get pods -n ziqif-kvbm -l nvidia.com/dynamo-component-type=worker -o jsonpath='{.items[0].metadata.name}')
for port in 6880 6881 6882 6883 6884 6885 6886 6887; do
  echo -n "port $port: "
  kubectl exec -n ziqif-kvbm $POD -- curl -s http://127.0.0.1:$port/metrics 2>/dev/null | grep "^kvbm_offload_blocks_d2h"
done

# output:
port 6880: kvbm_offload_blocks_d2h{dp_rank="0"} 1166
port 6881: kvbm_offload_blocks_d2h{dp_rank="1"} 1359
port 6882: kvbm_offload_blocks_d2h{dp_rank="2"} 1015
port 6883: kvbm_offload_blocks_d2h{dp_rank="3"} 1019
port 6884: kvbm_offload_blocks_d2h{dp_rank="4"} 1175
port 6885: kvbm_offload_blocks_d2h{dp_rank="5"} 1529
port 6886: kvbm_offload_blocks_d2h{dp_rank="6"} 848
port 6887: kvbm_offload_blocks_d2h{dp_rank="7"} 685
```

To check TRTLLM related metrics:

```bash
POD=$(kubectl get pods -n ziqif-kvbm -l nvidia.com/dynamo-component-type=worker -o jsonpath='{.items[0].metadata.name}')

# check trtllm_kv_cache_hit_rate only
kubectl exec -n ziqif-kvbm $POD -- curl -s http://127.0.0.1:9090/metrics 2>/dev/null | grep "^trtllm_kv_cache_hit_rate"

# all trtllm related metrics
kubectl exec -n ziqif-kvbm $POD -- curl -s http://127.0.0.1:9090/metrics 2>/dev/null | grep "^trtllm_"
```