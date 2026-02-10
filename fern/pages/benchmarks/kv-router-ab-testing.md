---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---

This guide walks you through setting up and running A/B benchmarks to compare Dynamo's KV Smart Router against standard round-robin routing on a Kubernetes cluster.

## Overview
Dynamo's KV Smart Router intelligently routes requests based on KV cache affinity, improving performance for workloads with shared prompt prefixes. This guide helps you:

1. Deploy two identical Dynamo configurations:
   a. A vllm server for Qwen3-32B with 8 workers (aggregated) **WITHOUT** KV Smart Router enabled
   b. A vllm server for Qwen3-32B with 8 workers (aggregated) **WITH** KV Smart Router enabled
2. Run controlled benchmarks using AIPerf
3. Compare performance metrics to evaluate KV router effectiveness

**Prerequisites:** Kubernetes cluster with GPUs, kubectl, helm

---

## Prerequisites

### Required Tools

- `kubectl` (configured with cluster access)
- `helm` (v3+)
- HuggingFace account and token (if model downloads are gated)
- Kubernetes cluster with:
  - GPU nodes (H100, H200, or similar)
  - Sufficient GPU capacity (16+ GPUs recommended for this example)
  - Dynamo platform installed globally OR ability to install per-namespace

### Knowledge Requirements

- Basic Kubernetes concepts (namespaces, pods, services)
- Familiarity with LLM inference concepts
- Command-line proficiency

---

## Architecture

This guide sets up two parallel deployments, as well as a benchmarking pod that can test each deployment:

```text
┌─────────────────────────────────────┐
│  Deployment A: Router OFF           │
│  Namespace: router-off-test          │
│  ├─ Frontend (Standard Routing)     │
│  └─ 8x Decode Workers (1 GPU each)  │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  Deployment B: Router ON             │
│  Namespace: router-on-test           │
│  ├─ Frontend (KV Smart Router)      │
│  └─ 8x Decode Workers (1 GPU each)  │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  Benchmark Pod                       │
│  Namespace: benchmark                │
│  └─ AIPerf + Dataset                 │
└─────────────────────────────────────┘
```

**Key Difference:** Deployment B sets `DYN_ROUTER_MODE=kv` on the frontend to enable KV cache-aware routing.

---

## Phase 1: Namespace and Infrastructure Setup

### Step 1.1: Create Namespaces

```bash
# Create namespaces for both deployments
kubectl create namespace router-off-test
kubectl create namespace router-on-test
kubectl create namespace benchmark
```

### Step 1.2: Create HuggingFace Token Secret (optional)

If the model you're seeking to deploy requires HF token to download (Llama family models require this), replace `YOUR_HF_TOKEN` with your actual HuggingFace token:

```bash
# Router-OFF namespace
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="YOUR_HF_TOKEN" \
  -n router-off-test

# Router-ON namespace
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="YOUR_HF_TOKEN" \
  -n router-on-test
```

### Step 1.3: Install Dynamo Platform (Per-Namespace)

If your cluster uses namespace-restricted Dynamo operators, you'll need to install the Dynamo platform in each namespace. Follow the [Dynamo Kubernetes Installation Guide](https://github.com/ai-dynamo/dynamo/blob/main/docs/kubernetes/installation-guide.md) to install the platform in both namespaces:

- `router-off-test`
- `router-on-test`

**Key Configuration Notes:**
- If your cluster uses namespace restrictions, ensure `dynamo-operator.namespaceRestriction.enabled=true` is set during installation
- Adjust version tags to match your cluster's available Dynamo versions
- If you encounter operator compatibility issues (e.g., unsupported MPI arguments), consult your cluster administrator or the Dynamo troubleshooting documentation

### Step 1.4: Verify Infrastructure

Wait for operators and infrastructure to be ready:

```bash
# Check router-off-test
kubectl get pods -n router-off-test

# Check router-on-test
kubectl get pods -n router-on-test
```

You should see:
- `dynamo-platform-dynamo-operator-controller-manager` (2/2 Running)
- `dynamo-platform-etcd-0` (1/1 Running)
- `dynamo-platform-nats-0` (2/2 Running)

---

## Phase 2: Deploy Model Serving

### Step 2.1: Create Deployment YAMLs

Create `router-off-deployment.yaml`:

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: vllm-agg-no-router
spec:
  services:
    Frontend:
      dynamoNamespace: vllm-agg-no-router
      componentType: frontend
      replicas: 1
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.5.0
    VllmDecodeWorker:
      envFromSecret: hf-token-secret
      dynamoNamespace: vllm-agg-no-router
      componentType: worker
      replicas: 8
      resources:
        limits:
          gpu: "1"
      extraPodSpec:
        affinity:
          nodeAffinity:
            requiredDuringSchedulingIgnoredDuringExecution:
              nodeSelectorTerms:
                - matchExpressions:
                    - key: node.kubernetes.io/instance-type
                      operator: In
                      values:
                        - gpu-h200-sxm  # Adjust to your GPU node type
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.5.0
          workingDir: /workspace/examples/backends/vllm
          command:
            - /bin/sh
            - -c
          args:
            - python3 -m dynamo.vllm --model Qwen/Qwen3-32B --quantization fp8
          startupProbe:
            httpGet:
              path: /health
              port: 9090
            initialDelaySeconds: 120
            periodSeconds: 30
            timeoutSeconds: 10
            failureThreshold: 60  # 32 minutes total (120s + 60*30s)
          livenessProbe:
            httpGet:
              path: /live
              port: 9090
            initialDelaySeconds: 300
            periodSeconds: 30
            timeoutSeconds: 10
            failureThreshold: 10
          readinessProbe:
            httpGet:
              path: /live
              port: 9090
            initialDelaySeconds: 300
            periodSeconds: 30
            timeoutSeconds: 10
            failureThreshold: 10
```

Create `router-on-deployment.yaml`:

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: vllm-agg-router
spec:
  services:
    Frontend:
      dynamoNamespace: vllm-agg-router
      componentType: frontend
      replicas: 1
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.5.0
      envs:
        - name: DYN_ROUTER_MODE
          value: kv  # KEY DIFFERENCE: Enable KV Smart Router
    VllmDecodeWorker:
      envFromSecret: hf-token-secret
      dynamoNamespace: vllm-agg-router
      componentType: worker
      replicas: 8
      resources:
        limits:
          gpu: "1"
      extraPodSpec:
        affinity:
          nodeAffinity:
            requiredDuringSchedulingIgnoredDuringExecution:
              nodeSelectorTerms:
                - matchExpressions:
                    - key: node.kubernetes.io/instance-type
                      operator: In
                      values:
                        - gpu-h200-sxm  # Adjust to your GPU node type
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.5.0
          workingDir: /workspace/examples/backends/vllm
          command:
            - /bin/sh
            - -c
          args:
            - python3 -m dynamo.vllm --model Qwen/Qwen3-32B --quantization fp8
          startupProbe:
            httpGet:
              path: /health
              port: 9090
            initialDelaySeconds: 120
            periodSeconds: 30
            timeoutSeconds: 10
            failureThreshold: 60  # 32 minutes total (120s + 60*30s)
          livenessProbe:
            httpGet:
              path: /live
              port: 9090
            initialDelaySeconds: 300
            periodSeconds: 30
            timeoutSeconds: 10
            failureThreshold: 10
          readinessProbe:
            httpGet:
              path: /live
              port: 9090
            initialDelaySeconds: 300
            periodSeconds: 30
            timeoutSeconds: 10
            failureThreshold: 10
```

### Step 2.2: Deploy Both Configurations

```bash
# Deploy router-OFF
kubectl apply -f router-off-deployment.yaml -n router-off-test

# Deploy router-ON
kubectl apply -f router-on-deployment.yaml -n router-on-test
```

**💡 Optimization Tip:** Each worker will download the model independently (~20 minutes per pod). For faster initialization, add a shared PVC with `ReadWriteMany` access mode to cache the model.

First, create the PVC separately:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-cache
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: "your-shared-storage-class"  # e.g., nfs, efs, nebius-shared-fs
  resources:
    requests:
      storage: 100Gi
```

Then reference it in your DynamoGraphDeployment:

```yaml
spec:
  pvcs:
    - create: false
      name: model-cache
      size: "0"
  services:
    VllmDecodeWorker:
      volumeMounts:
        - mountPoint: /root/.cache/huggingface
          name: model-cache
          useAsCompilationCache: false
```

With this configuration, only the first worker downloads the model; others use the cached version, reducing startup time from 20+ minutes to ~2 minutes per pod.

### Step 2.3: Monitor Deployment Progress

```bash
# Watch router-OFF pods
kubectl get pods -n router-off-test -w

# Watch router-ON pods
kubectl get pods -n router-on-test -w
```

Wait for all pods to reach `Running` status and pass readiness probes.

**Expected Timeline:**
- **With shared PVC** (ReadWriteMany): ~5-10 minutes total (first worker downloads, others reuse cache)
- **Without shared PVC**: 20-30 minutes per worker (workers download independently)
  - For 8 workers: Budget **1-2 hours** for full deployment (workers start in parallel but are limited by node scheduling)

The startup probe allows 32 minutes per pod (failureThreshold: 60), which accommodates model download and initialization.

### Step 2.4: Verify All Workers Are Healthy

> ⚠️ **CRITICAL CHECKPOINT**: Before running benchmarks, you **MUST** verify equal worker health in both deployments. Unequal worker counts will invalidate your comparison results.

```bash
# Quick health check - both should show "8/8"
echo "Router OFF: $(kubectl get pods -n router-off-test -l nvidia.com/dynamo-component-type=worker --field-selector=status.phase=Running -o json | jq '[.items[] | select(.status.conditions[] | select(.type=="Ready" and .status=="True"))] | length')/8 ready"
echo "Router ON:  $(kubectl get pods -n router-on-test -l nvidia.com/dynamo-component-type=worker --field-selector=status.phase=Running -o json | jq '[.items[] | select(.status.conditions[] | select(.type=="Ready" and .status=="True"))] | length')/8 ready"

# Detailed view
kubectl get pods -n router-off-test -l nvidia.com/dynamo-component-type=worker
kubectl get pods -n router-on-test -l nvidia.com/dynamo-component-type=worker
```

**Both must show 8/8 workers in Ready state (1/1 Running).** If workers are not ready:
- Check logs: `kubectl logs -n <namespace> <pod-name>`
- Common issues: model download in progress, startup probe timeout, insufficient GPU resources

**Do not proceed with benchmarks until all 16 workers (8 per deployment) are healthy.**

---

## Phase 3: Prepare Benchmark Dataset

### Understanding the Mooncake Trace Dataset

For this A/B comparison, we use the **Mooncake Trace Dataset**, published by [Mooncake AI](https://github.com/kvcache-ai/Mooncake). This is a privacy-preserving dataset of real-world LLM inference traffic from production arxiv workloads.

**What's in the dataset?** Each trace entry contains:
- **Timestamp:** When the request arrived (for realistic request timing)
- **Input/output lengths:** Number of tokens in prompts and responses
- **Block hash IDs:** Cryptographic hashes representing KV cache blocks (explained below)

**Sample trace entry:**
```json
{
  "timestamp": 27482,
  "input_length": 6955,
  "output_length": 52,
  "hash_ids": [46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 2353, 2354]
}
```

### Why Mooncake Traces Matter for KV Cache Benchmarking

**The Challenge:** Traditional LLM benchmarks use synthetic or random data, which are often insufficient to capture real-world optimizations like KV Smart Router. To properly evaluate this feature, we need realistic traffic patterns with **prefix repetition** - but this creates a privacy problem: how do we measure realistic KV cache hit patterns without exposing actual user conversations?

**Mooncake's Solution: Privacy-Preserving Block Hashes**

Instead of storing actual prompt text, the Mooncake dataset uses cryptographic hashes to represent KV cache blocks. Each hash ID represents a **512-token block**, and the hash includes both the current block and all preceding blocks. This preserves the **pattern of prefix reuse** while completely protecting user privacy.

### How it works - Multi-turn conversation example

```text
Turn 1 (initial request - long document analysis):
  Input: ~8,000 tokens (e.g., research paper + question)
  Hash IDs: [46][47][48][49][50][51][52][53][54][55][56][57][58][59][60][61]
            └─ 16 blocks × 512 tokens/block = ~8,192 tokens

Turn 2 (follow-up question on same document):
  Input: Same document + new question (~8,500 tokens)
  Hash IDs: [46][47][48][49][50][51][52][53][54][55][56][57][58][59][60][61][62]
            └──────────── Reuses first 16 blocks (~8,192 tokens) ───────────────┘

            ✅ Cache hit: First 8,192 tokens don't need recomputation!

Turn 3 (another follow-up):
  Input: Same document + different question (~9,000 tokens)
  Hash IDs: [46][47][48][49][50][51][52][53][54][55][56][57][58][59][60][61][62][63]
            └──────────── Reuses first 16 blocks (~8,192 tokens) ───────────────┘
```

When requests share the same hash IDs (e.g., blocks 46-61), it means they share those 512-token blocks - indicating **significant prefix overlap** (in this case, 8,192 tokens). The **KV Smart Router** routes requests with matching hash IDs to the same worker, maximizing cache hits and avoiding redundant computation for those shared prefix tokens.

**Key Dataset Properties:**
- ✅ **Realistic timing:** Request arrival patterns from production workloads
- ✅ **Real prefix patterns:** Up to 50% cache hit ratio ([Mooncake technical report](https://github.com/kvcache-ai/Mooncake))
- ✅ **Privacy-preserving:** No actual text - only hash-based cache block identifiers
- ✅ **Reproducible:** Public dataset enables fair comparisons across different systems

**Why this matters:** With random synthetic data, the KV Smart Router would show no benefit because there's no prefix reuse to exploit. Mooncake traces provide realistic workload patterns that demonstrate the router's real-world performance gains while respecting user privacy.

---

### Download and Prepare the Dataset

```bash
# Download the Mooncake arxiv trace dataset
curl -sL https://raw.githubusercontent.com/kvcache-ai/Mooncake/refs/heads/main/FAST25-release/arxiv-trace/mooncake_trace.jsonl -o mooncake_trace.jsonl

# Trim to 1000 requests for faster benchmarking
head -n 1000 mooncake_trace.jsonl > mooncake_trace_small.jsonl

# Speed up timestamps 4x (reduces benchmark time from ~12 min to ~3 min)
python3 - <<'PY'
import json

with open("mooncake_trace_small.jsonl") as src, open("mooncake_trace_4x.jsonl", "w") as dst:
    for line in src:
        rec = json.loads(line)
        rec["timestamp"] = int(rec["timestamp"] / 4)
        dst.write(json.dumps(rec) + "\n")
PY

echo "Dataset ready: mooncake_trace_4x.jsonl (1000 requests, 4x speed)"
```

---

## Phase 4: Set Up Benchmark Environment

### Step 4.1: Deploy Benchmark Pod

Create `benchmark-job.yaml`:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: aiperf-benchmark
  namespace: benchmark
spec:
  backoffLimit: 1
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: benchmark
        image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.5.0
        command: ["/bin/sh", "-c", "sleep infinity"]
        imagePullPolicy: IfNotPresent
        resources:
          limits:
            nvidia.com/gpu: 0
```

Deploy:

```bash
kubectl apply -f benchmark-job.yaml
```

Wait for pod to be ready:

```bash
kubectl get pods -n benchmark
```

### Step 4.2: Copy Dataset to Benchmark Pod

```bash
POD_NAME=$(kubectl get pods -n benchmark -l job-name=aiperf-benchmark -o jsonpath='{.items[0].metadata.name}')

kubectl -n benchmark cp mooncake_trace_4x.jsonl ${POD_NAME}:/tmp/mooncake_trace_4x.jsonl
```

### Step 4.3: Install AIPerf

```bash
kubectl -n benchmark exec ${POD_NAME} -- bash -lc '. /opt/dynamo/venv/bin/activate && pip install -q aiperf'
```

---

## Phase 5: Run Benchmarks

### Step 5.1: Benchmark Router-OFF (Baseline)

```bash
kubectl -n benchmark exec ${POD_NAME} -- bash -lc '
  . /opt/dynamo/venv/bin/activate
  aiperf profile \
    --model "Qwen/Qwen3-32B" \
    --url "http://vllm-agg-no-router-frontend.router-off-test.svc.cluster.local:8000" \
    --endpoint-type chat \
    --input-file /tmp/mooncake_trace_4x.jsonl \
    --custom-dataset-type mooncake_trace \
    --tokenizer "Qwen/Qwen3-32B" \
    --streaming \
    --request-count 1000 \
    --fixed-schedule \
    --output-artifact-dir /tmp/router_off_results
'
```

**Note:** This will take 3-5 minutes. The terminal output includes a summary table.

### Step 5.2: Benchmark Router-ON (KV Smart Router)

```bash
kubectl -n benchmark exec ${POD_NAME} -- bash -lc '
  . /opt/dynamo/venv/bin/activate
  aiperf profile \
    --model "Qwen/Qwen3-32B" \
    --url "http://vllm-agg-router-frontend.router-on-test.svc.cluster.local:8000" \
    --endpoint-type chat \
    --input-file /tmp/mooncake_trace_4x.jsonl \
    --custom-dataset-type mooncake_trace \
    --tokenizer "Qwen/Qwen3-32B" \
    --streaming \
    --request-count 1000 \
    --fixed-schedule \
    --output-artifact-dir /tmp/router_on_results
'
```

### Step 5.3: Collect Results

```bash
# Copy results to local machine
kubectl -n benchmark cp ${POD_NAME}:/tmp/router_off_results/profile_export_aiperf.csv ./router_off_results.csv
kubectl -n benchmark cp ${POD_NAME}:/tmp/router_on_results/profile_export_aiperf.csv ./router_on_results.csv
```

---

## Phase 6: Analyze Results

### Key Metrics to Compare

| Metric | Description | What to Look For |
|--------|-------------|------------------|
| **Time to First Token (TTFT)** | Latency until first token arrives | Lower is better; KV router may reduce with prefix reuse |
| **Inter Token Latency (ITL)** | Average time between tokens | Lower is better; indicates generation speed |
| **Request Latency** | Total end-to-end latency | Lower is better; overall user experience |
| **Output Token Throughput** | Tokens generated per second (system-wide) | Higher is better; system efficiency |
| **Request Throughput** | Requests completed per second | Higher is better; capacity |

### Interpreting Results

**Your Results May Vary**: The improvement from KV Smart Router depends heavily on your workload characteristics:

**Factors that increase KV router benefit:**
- **High prefix overlap** (shared system prompts, templates, document contexts)
- **Long prompts** (>2000 tokens) where caching saves significant compute
- **Multi-turn conversations** with context carryover
- **Batch workloads** with similar queries

**Factors that reduce KV router benefit:**
- **Unique prompts** with no prefix reuse
- **Short prompts** (\<1000 tokens) where routing overhead exceeds benefit
- **Evenly distributed load** where round-robin is already optimal
- **Low request rate** where cache eviction negates benefits

**Expected Performance:**
- **High prefix overlap workloads**: 20-50% TTFT improvement
- **Moderate prefix overlap**: 10-20% improvement
- **Low prefix overlap**: \<5% improvement (may not be worth enabling)

**KV Smart Router is beneficial when:**
- TTFT improvements > 20%
- No significant degradation in other metrics
- Workload demonstrates measurable prefix reuse patterns

**Standard routing is better when:**
- KV router shows \<10% improvement
- Increased latency variance is observed
- Load distribution across workers is more important than cache affinity

### Example Comparison

From the terminal output, compare the summary tables:

```
Router-OFF (Baseline):
  TTFT avg: 12,764 ms    p99: 45,898 ms
  Request Latency avg: 32,978 ms
  Output Token Throughput: 1,614 tokens/sec
  Request Throughput: 8.61 req/sec

Router-ON (KV Router):
  TTFT avg: 8,012 ms     p99: 28,644 ms  (37% faster ✅)
  Request Latency avg: 28,972 ms  (12% faster ✅)
  Output Token Throughput: 1,746 tokens/sec  (8% higher ✅)
  Request Throughput: 9.33 req/sec  (8% higher ✅)
```

In this example with all 8 workers healthy, the **KV router significantly outperformed** the baseline:
- **37% faster TTFT** - Users see first token much sooner
- **8% higher throughput** - System processes more requests per second
- **12% lower latency** - Faster end-to-end completion

The Mooncake arxiv dataset has sufficient prefix overlap (long input sequences with similar patterns) to benefit from KV cache-aware routing. Workloads with explicit shared prefixes (system prompts, templates) may see even greater improvements.

---

## Phase 7: Cleanup

```bash
# Delete deployments
kubectl delete dynamographdeployment vllm-agg-no-router -n router-off-test
kubectl delete dynamographdeployment vllm-agg-router -n router-on-test

# Delete namespaces (removes all resources)
kubectl delete namespace router-off-test
kubectl delete namespace router-on-test
kubectl delete namespace benchmark
```

---

## Troubleshooting

### Issue: Pods Stuck in Pending

**Cause:** Insufficient GPU resources

**Solution:**
```bash
# Check GPU availability
kubectl describe nodes | grep -A 10 "Allocated resources"

# Reduce worker replicas if needed
kubectl edit dynamographdeployment -n <namespace>
```

### Issue: ImagePullBackOff Errors

**Cause:** Version mismatch or missing credentials

**Solution:**
```bash
# Check available versions
kubectl get pods -n dynamo-system -o yaml | grep image:

# Update deployment YAML to match cluster version
```

### Issue: Operator Not Processing Deployment

**Cause:** Namespace restrictions

**Solution:**
- Ensure Dynamo platform is Helm-installed in the namespace
- Verify operator has `--restrictedNamespace=<your-namespace>` argument
- Check operator logs: `kubectl logs -n <namespace> deployment/dynamo-platform-dynamo-operator-controller-manager`

### Issue: Workers Not Becoming Ready

**Cause:** Model download failures or probe configuration

**Solution:**
```bash
# Check worker logs
kubectl logs -n <namespace> <worker-pod-name>

# Common issues:
# - Invalid HuggingFace token
# - Network connectivity
# - Insufficient disk space for model
```

### Issue: Workers Restarting in CrashLoopBackOff

**Cause:** Startup probe timeout - workers killed before finishing initialization

**Symptoms:**
- Pods show "Container main failed startup probe, will be restarted"
- Logs show model still downloading or loading when pod is killed
- Large models (>30GB) take longer than default 22-minute timeout

**Solution:**
Increase the startup probe `failureThreshold`:

```bash
# Patch the deployment to allow 32 minutes instead of 22
kubectl patch dynamographdeployment <deployment-name> -n <namespace> --type='json' \
  -p='[{"op": "replace", "path": "/spec/services/VllmDecodeWorker/extraPodSpec/mainContainer/startupProbe/failureThreshold", "value": 60}]'
```

Or update your YAML before deploying:
```yaml
startupProbe:
  httpGet:
    path: /health
    port: 9090
  initialDelaySeconds: 120
  periodSeconds: 30
  timeoutSeconds: 10
  failureThreshold: 60  # 32 minutes total (120s + 60*30s)
```

**Model Loading Times (approximate):**
- Qwen3-32B: ~20-25 minutes (first download)
- Llama-70B: ~25-30 minutes (first download)
- With cached model on node: ~2-5 minutes

### Issue: Unequal Worker Health

**Cause:** Resource constraints, image pull issues, or configuration errors

**Solution:**
```bash
# Check all worker status
kubectl get pods -n <namespace> -l nvidia.com/dynamo-component-type=worker

# Describe problematic pods
kubectl describe pod <pod-name> -n <namespace>

# Fix issues before benchmarking or results will be skewed
```

---

## Advanced Configuration

### Testing Different Models

Replace `Qwen/Qwen3-32B` with your model in:
- Deployment YAML `args` section
- AIPerf `--model` and `--tokenizer` parameters

### Adjusting Worker Count

Change `replicas: 8` in the deployment YAMLs. Ensure both deployments use the same count for fair comparison.

### Using Custom Datasets

Replace mooncake dataset with your own JSONL file:
- Format: One request per line with `timestamp` field
- AIPerf supports various formats via `--custom-dataset-type`

### Disaggregated Prefill/Decode

For advanced testing, add separate prefill workers:

```yaml
VllmPrefillWorker:
  componentType: worker
  replicas: 2
  # ... configuration
```

---

## Best Practices

1. **Equal Conditions:** Ensure both deployments have identical worker counts and health before benchmarking
2. **Warm-Up:** Run a small test (100 requests) before the full benchmark to warm up caches
3. **Multiple Runs:** Run benchmarks 3+ times and average results for statistical significance
4. **Monitor Workers:** Watch for any pod restarts or issues during benchmark runs
5. **Document Conditions:** Record cluster state, worker health, and any anomalies
6. **Test Relevant Workloads:** Use datasets that match your actual use case for meaningful results

---

## Conclusion

This guide provides a complete methodology for A/B testing Dynamo's KV Smart Router. The KV router's effectiveness depends heavily on workload characteristics—datasets with high prefix overlap will show the most benefit.

For questions or issues, consult the [Dynamo documentation](https://github.com/ai-dynamo/dynamo) or open an issue on GitHub.

---

## Appendix: Files Reference

- `router-off-deployment.yaml`: Standard routing deployment
- `router-on-deployment.yaml`: KV router enabled deployment
- `benchmark-job.yaml`: AIPerf benchmark pod
- `prepare-dataset.sh`: Dataset preparation script
- Results CSVs: Detailed metrics from AIPerf

**Repository:** [https://github.com/ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo)

