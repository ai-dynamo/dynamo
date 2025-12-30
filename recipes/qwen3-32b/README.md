# Qwen3-32B: Aggregated Round Robin vs Disaggregated KV Routing Comparison

This recipe demonstrates the performance difference between **aggregated (round-robin)** and **disaggregated (KV-aware)** routing using a real-world conversation trace dataset from the [Mooncake FAST25 paper](https://github.com/kvcache-ai/Mooncake).

## Results

https://github.com/user-attachments/assets/c425002b-4459-47c4-bfca-fd1e2620500c


## Experiment Overview

We compare two deployment modes on **16x H200 GPUs across 2 nodes**:

| Mode | Routing | Configuration |
|------|---------|---------------|
| **Aggregated** | Round-robin | 8x TP2 workers |
| **Disaggregated** | KV-aware | 6x prefill + 2x decode (TP2) |

## Dataset: Mooncake Conversation Trace

The benchmark uses a production conversation trace with significant prefix sharing potential:

| Metric | Value |
|--------|-------|
| Requests | 12,031 over ~59 minutes (3.4 req/s) |
| Input tokens/sec | 40,937 tok/s |
| Input length | avg 12,035 tokens (range: 891 - 126,195) |
| Output length | avg 343 tokens |

**Cache Reuse Analysis:**

| Metric | Value | What It Measures |
|--------|-------|------------------|
| Blocks reused | 24.2% | Of 182,790 unique blocks, 44,144 appeared in more than one request |
| Cache efficiency | 36.64% | Of 288,500 total block references, 105,710 were repeats (reusable with infinite cache) |

*Why these differ:* Block reuse counts unique blocks that repeat, ignoring how often they repeat. Cache efficiency weights by frequency—a block reused 12,031 times contributes more than one reused once.

This workload is ideal for KV-aware routing—with 36.64% cache efficiency, requests can be routed to workers that already have relevant KV blocks cached, significantly reducing TTFT.

## Prerequisites

1. **Dynamo Platform installed** - See [Kubernetes Deployment Guide](../../docs/kubernetes/README.md)
2. **16x H200 GPUs** across 2 nodes
3. **HuggingFace token** configured:
   ```bash
   export NAMESPACE=your-namespace
   kubectl create secret generic hf-token-secret \
     --from-literal=HF_TOKEN="your-token" \
     -n ${NAMESPACE}
   ```

## Quick Start

### 1. Create Storage

> **Note:** Edit `model-cache/cache.yaml` first and update `storageClassName` to match your cluster (run `kubectl get storageclass` to find available options).

```bash
kubectl apply -f model-cache/cache.yaml -n ${NAMESPACE}
```

### 2. Download Model

```bash
kubectl apply -f model-cache/model-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=600s
```

### 3. Deploy & Benchmark

**Option A: Aggregated (Round-Robin Baseline)**

```bash
# Deploy
kubectl apply -f vllm/agg-round-robin/deploy.yaml -n ${NAMESPACE}

# Wait for ready
kubectl wait --for=condition=ready pod -l nvidia.com/dynamo-graph-deployment-name=agg-8xtp2 \
  -n ${NAMESPACE} --timeout=1200s

# Run benchmark
kubectl apply -f vllm/agg-round-robin/perf.yaml -n ${NAMESPACE}
```

**Option B: Disaggregated (KV-Aware Routing)**

```bash
# Deploy
kubectl apply -f vllm/disagg-kv-router/deploy.yaml -n ${NAMESPACE}

# Wait for ready
kubectl wait --for=condition=ready pod -l nvidia.com/dynamo-graph-deployment-name=disagg-router-6p-2d \
  -n ${NAMESPACE} --timeout=1200s

# Run benchmark
kubectl apply -f vllm/disagg-kv-router/perf.yaml -n ${NAMESPACE}
```

### 4. Monitor Benchmark Progress

The benchmark runs inside a tmux session for easy monitoring:

```bash
# Find the benchmark pod
kubectl get pods -n ${NAMESPACE} | grep benchmark

# Attach to the tmux session to see intermediate results
kubectl exec -it -n ${NAMESPACE} <benchmark-pod-name> -- tmux a -t benchmark

# Detach from tmux: Ctrl+B, then D
```

### 5. View Results

Results are saved to the `perf-cache` PVC:

```bash
# Check artifact directory
kubectl exec -it -n ${NAMESPACE} <benchmark-pod-name> -- ls -la /perf-cache/artifacts/

# Copy results to local machine
kubectl cp ${NAMESPACE}/<benchmark-pod-name>:/perf-cache/artifacts ./benchmark-results
```

## Expected Results

Since the benchmark uses `--fixed-schedule` (replaying requests at their original timestamps), **throughput metrics are fixed by the trace**—latency metrics are what we're comparing:

| Metric | Why It Matters |
|--------|----------------|
| **TTFT** (Time to First Token) | KV-aware routing reduces prefill compute via prefix cache hits |
| **ITL** (Inter-Token Latency) | Disaggregated serving isolates decode from prefill interference |
| **Total Request Latency** | Combined benefit of both optimizations |

**Why disaggregated + KV-aware routing helps this workload:**

1. **KV-aware routing** leverages the 36% cache efficiency to route requests to workers that already have relevant KV cache blocks, reducing redundant prefill computation and lowering TTFT.

2. **Disaggregated serving** separates prefill and decode workers. With long input sequences (avg 12K tokens) and short outputs (avg 343 tokens), dedicated decode workers avoid "prefill injection"—where a new long-context request interrupts ongoing decode operations, causing ITL spikes.

## Cleanup

```bash
# Delete benchmark pods
kubectl delete pod -l app=benchmark -n ${NAMESPACE}

# Delete deployments
kubectl delete dynamographdeployment agg-8xtp2 -n ${NAMESPACE}
kubectl delete dynamographdeployment disagg-router-6p-2d-n ${NAMESPACE}
```

## References

- [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving](https://github.com/kvcache-ai/Mooncake) - FAST25 paper and trace data

