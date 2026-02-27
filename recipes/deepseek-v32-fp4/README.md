# Deepseek V3.2 NVFP4: Aggregated Round Robin vs Disaggregated KV Routing with WideEP

This recipe demonstrates the performance difference between **aggregated (round-robin)** and **disaggregated (KV-aware)** routing using a synthetic trace dataset adapted from the [Mooncake FAST25 paper](https://github.com/kvcache-ai/Mooncake).

## Results

https://github.com/user-attachments/assets/fcdb703c-7c1a-4109-a7ca-54196fcef885

## Experiment Overview

We compare two deployment modes on **32x GB200 GPUs across 8 nodes**:

| Mode | Routing | Configuration |
|------|---------|---------------|
| **Aggregated** | Round-robin | 4x DEP8 workers |
| **Disaggregated** | KV-aware | 2x prefill + 2x decode w/ WideEP (DEP8) |

## Dataset: Mooncake-based Synthetic Coding Trace

The benchmark uses a synthetic trace which simulates coding workloads. We synthesize the data by increasing the input sequence length and prefix reuse rate of the original [Mooncake conversation trace](https://github.com/kvcache-ai/Mooncake/blob/main/FAST25-release/traces/conversation_trace.jsonl).

To reproduce our benchmark, run Dynamo's [prefix data generator tool](https://github.com/ai-dynamo/dynamo/tree/main/benchmarks/prefix_data_generator) on the Mooncake `conversation_trace.jsonl`:
```bash
datagen synthesize \
    --input-file conversation_trace.jsonl \
    --prefix-len-multiplier 16 \
    --prompt-len-multiplier 10 \
    --max-isl 110000 \
    --num-requests 10000
# synthesizes `conversation_trace_synth_16.00x1+10.00_speedup1_maxisl110000.jsonl`
```

The ISL/OSL/cache hit statistics of the resulting trace is below.

<details>
<summary>Dataset statistics: Mooncake-based Synthetic Trace</summary>

```
============================================================
  DATASET ANALYSIS: Mooncake-based Synthetic Trace
  ============================================================
  OVERVIEW
  ----------------------------------------
    Total Requests:      10,000
    Unique Hash Blocks:  430,838
    Total Hash Blocks:   770,934
  INPUT SEQUENCE LENGTH (ISL)
  ----------------------------------------
    Average:             39,186 tokens
    Maximum:             109,459 tokens
    Minimum:             12,801 tokens
  OUTPUT SEQUENCE LENGTH (OSL)
  ----------------------------------------
    Average:             344 tokens
    Maximum:             2,000 tokens
    Minimum:             1 tokens
  KV CACHE / PREFIX REUSE
  ----------------------------------------
    Block-level Hit Rate: 44.1%
    Token-level Hit Rate: 44.0%
    Avg Context (shared): 22,400 tokens/req
    Avg Unique Prompt:    16,786 tokens/req
    Shared Prefix Ratio:  57.2%
  ============================================================

  Summary:
  • ~44% KV cache hit rate (block/token level) based on hash_id overlap across requests
  • ~57% of input tokens come from shared context prefixes
  • Long-context workload: avg 39K input tokens, up to 109K max
```

</details>


## Prerequisites

1. **Dynamo Platform installed** - See [Kubernetes Deployment Guide](../../docs/pages/kubernetes/README.md)
2. **32x GB200 GPUs** across 8 nodes
3. **HuggingFace token** configured:
   ```bash
   export NAMESPACE=your-namespace
   kubectl create secret generic hf-token-secret \
     --from-literal=HF_TOKEN="your-token" \
     -n ${NAMESPACE}
   ```

## Quick Start

### 1. Create Storage

> **Note:** Edit `model-cache/model-cache.yaml` first and update `storageClassName` to match your cluster (run `kubectl get storageclass` to find available options).

```bash
kubectl apply -f model-cache/model-cache.yaml -n ${NAMESPACE}
```

### 2. Configure K8 Benchmarking Environment
For multinode kubernetes deployments, your cluster may require a ComputeDomain to exist in your namespace such that the DRA scheduler can co-locate worker pods on MNNVL-connected nodes. (Otherwise, internode GPU peer memory access would fail.)
```bash
kubectl apply -f model-cache/compute-domain.yaml -n ${NAMESPACE}
```
Make sure to apply any name modifications to this file to the deployment yamls, under `extraPodSpec.resourceClaims` and `mainContainer.resources.claims`.


### 3. Setup Model and Data
We use NVIDIA's official NVFP4-quantized checkpoint ([Huggingface](https://huggingface.co/nvidia/DeepSeek-V3.2-NVFP4)). Copy it into the PVC storage:

```bash
kubectl apply -f model-cache/model-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=600s
```
Similarly, copy the trace file for the benchmark () into the PVC:
```bash
# conversation_trace_synth_16.00x1+10.00_speedup1_maxisl110000.jsonl in our case
kubectl cp <local_trace.jsonl> your-namespace/<helper-pod>:/model-cache/traces/
```

### 4. Deploy & Benchmark

**Option A: Aggregated (Round-Robin Baseline)**

```bash
# Deploy
kubectl apply -f trtllm/agg-round-robin/deploy.yaml -n ${NAMESPACE}

# Wait for ready
kubectl wait --for=condition=ready pod -l nvidia.com/dynamo-graph-deployment-name=agg-round-robin-dsv32-nvfp4 \
  -n ${NAMESPACE} --timeout=1200s

# Run benchmark
kubectl apply -f trtllm/agg-round-robin/perf.yaml -n ${NAMESPACE}
```

**Option B: Disaggregated (KV-Aware Routing)**

```bash
# Deploy
kubectl apply -f trtllm/disagg-kv-router/deploy.yaml -n ${NAMESPACE}

# Wait for ready
kubectl wait --for=condition=ready pod -l nvidia.com/dynamo-graph-deployment-name=disagg-kv-dsv32-nvfp4 \
  -n ${NAMESPACE} --timeout=1200s

# Run benchmark
kubectl apply -f trtllm/disagg-kv-router/perf.yaml -n ${NAMESPACE}
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

For production contexts, we can further evaluate the deployments with **goodput**, i.e. the rate of requests which satisfy a predetermined service level agreement (SLA). For our experiments, we set the SLA as TTFT=20s and ITL=50ms.

## Cleanup

```bash
# Delete benchmark pods
kubectl delete pod -l app=benchmark -n ${NAMESPACE}

# Delete deployments
kubectl delete dynamographdeployment -n agg-round-robin-dsv32-nvfp4 -n ${NAMESPACE}
kubectl delete dynamographdeployment disagg-kv-dsv32-nvfp4 -n ${NAMESPACE}
```

## References

- [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving](https://github.com/kvcache-ai/Mooncake) - FAST25 paper and trace data
- [Optimizing DeepSeek-V3.2 on NVIDIA Blackwell GPUs](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog15_Optimizing_DeepSeek_V32_on_NVIDIA_Blackwell_GPUs.html) - TRTLLM tech blog on available optimizations for DSV3.2 on GB200

