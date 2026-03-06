---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: KV Router A/B Testing
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
  - Sufficient GPU capacity (8+ GPUs recommended for this example)
  - Dynamo platform installed globally OR ability to install per-namespace

### Knowledge Requirements

- Basic Kubernetes concepts (namespaces, pods, services)
- Familiarity with LLM inference concepts
- Command-line proficiency

---

## Architecture

This guide uses a single namespace. We deploy one configuration (e.g. router-ON), run the benchmark, tear it down, then deploy the other (router-OFF) and run the same benchmark. 

```text
┌──────────────────────────────────────────────┐
│ Namespace: dynamo-bench                       │
│ (one of A or B active at a time)              │
│                                              │
│  Deployment A: Router OFF                     │
│    ├─ Frontend (Standard Routing)              │
│    └─ 8x Decode Workers (1 GPU each)          │
│                                              │
│  Deployment B: Router ON                      │
│    ├─ Frontend (KV Smart Router)               │
│    └─ 8x Decode Workers (1 GPU each)          │
│                                              │
│  Benchmark Pod (AIPerf + Dataset)             │
└──────────────────────────────────────────────┘
```

**Key Difference:** Deployment B sets `DYN_ROUTER_MODE=kv` on the frontend to enable KV cache-aware routing.

---

## Phase 1: Namespace and Infrastructure Setup

### Step 1.1: Create Namespace

```bash
kubectl create namespace dynamo-bench
```

### Step 1.2: Create HuggingFace Token Secret (optional)

If the model you're seeking to deploy requires HF token to download (Llama family models require this), replace `YOUR_HF_TOKEN` with your actual HuggingFace token:

```bash
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="YOUR_HF_TOKEN" \
  -n dynamo-bench
```

### Step 1.3: Install Dynamo Platform

If your cluster uses namespace-restricted Dynamo operators, you'll need to install the Dynamo platform in the workload namespace. Follow the [Dynamo Kubernetes Installation Guide](https://github.com/ai-dynamo/dynamo/blob/main/docs/kubernetes/installation-guide.md) to install the platform in `dynamo-bench`.

**Key Configuration Notes:**
- If your cluster uses namespace restrictions, ensure `dynamo-operator.namespaceRestriction.enabled=true` is set during installation
- Adjust version tags to match your cluster's available Dynamo versions
- If you encounter operator compatibility issues (e.g., unsupported MPI arguments), consult your cluster administrator or the Dynamo troubleshooting documentation

### Step 1.4: Verify Infrastructure

```bash
kubectl get pods -n dynamo-bench
```

Expect operator, etcd, and nats pods Running before deploying the graph.

---

## Phase 2: Deploy Model Serving

### Step 2.1: Create Deployment YAMLs

Create `router-off-deployment.yaml` (baseline):

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
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.9.0
          env:
            - name: POD_UID
              valueFrom:
                fieldRef:
                  fieldPath: metadata.uid
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
                        - Standard_NC80adis_H100_v5  # Adjust to your GPU node type
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.9.0
          workingDir: /workspace
          command:
            - /bin/sh
            - -c
          args:
            - >-
              python3 -m dynamo.vllm
              --model Qwen/Qwen3-32B
              --quantization fp8
              --kv-cache-dtype fp8
              --max-model-len 131072
              --hf-overrides '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768},"max_position_embeddings":131072}'
              --gpu-memory-utilization 0.90
              --block-size 64
              --async-scheduling
              --disable-log-requests
          env:
            - name: DYN_HEALTH_CHECK_ENABLED
              value: "false"
            - name: POD_UID
              valueFrom:
                fieldRef:
                  fieldPath: metadata.uid
          startupProbe:
            httpGet:
              path: /health
              port: 9090
            initialDelaySeconds: 120
            periodSeconds: 30
            timeoutSeconds: 10
            failureThreshold: 60
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
      subComponentType: decode
```

Create `router-on-deployment.yaml` (KV router ON):

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
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.9.0
          env:
            - name: POD_UID
              valueFrom:
                fieldRef:
                  fieldPath: metadata.uid
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
                        - Standard_NC80adis_H100_v5  # Adjust to your GPU node type
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.9.0
          workingDir: /workspace
          command:
            - /bin/sh
            - -c
          args:
            - >-
              python3 -m dynamo.vllm
              --model Qwen/Qwen3-32B
              --quantization fp8
              --kv-cache-dtype fp8
              --max-model-len 131072
              --hf-overrides '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768},"max_position_embeddings":131072}'
              --gpu-memory-utilization 0.90
              --block-size 64
              --async-scheduling
              --disable-log-requests
          env:
            - name: DYN_HEALTH_CHECK_ENABLED
              value: "false"
            - name: POD_UID
              valueFrom:
                fieldRef:
                  fieldPath: metadata.uid
          startupProbe:
            httpGet:
              path: /health
              port: 9090
            initialDelaySeconds: 120
            periodSeconds: 30
            timeoutSeconds: 10
            failureThreshold: 60
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
      subComponentType: decode
```

### Step 2.2: Deploy Router-ON First

```bash
kubectl apply -f router-on-deployment.yaml -n dynamo-bench
```

**💡 Optimization Tip:** Each worker will download the model independently (~20 minutes per pod). For faster initialization, add a shared PVC with `ReadWriteMany` access mode to cache the model.

First, create the PVC in the same namespace as your deployment (e.g. `dynamo-bench`). Use a storage class that supports ReadWriteMany:

```bash
kubectl get storageclass   # choose one with ReadWriteMany (e.g. azurefile-csi-premium, nfs, efs)
```

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-cache
  namespace: dynamo-bench
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: "azurefile-csi-premium"   # Adjust to your cluster
  resources:
    requests:
      storage: 100Gi
```

Apply it: `kubectl apply -f pvc-model-cache.yaml`

Then reference the existing PVC in your DynamoGraphDeployment by adding the following under `spec` (and under `VllmDecodeWorker`, add `volumeMounts`):

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

With this configuration, the first run has one worker download; the rest load from cache. The main benefit is on redeploy: the model stays on the PVC, so new pods load from cache and come up in ~5–10 minutes instead of downloading again.

### Step 2.3: Monitor Deployment Progress

```bash
kubectl get pods -n dynamo-bench -w
```

Wait for all pods to reach `Running` status and pass readiness probes.

**Expected Timeline:**
- **With shared PVC** (ReadWriteMany): ~5-10 minutes total (first worker downloads, others reuse cache)
- **Without shared PVC**: 20-30 minutes per worker (workers download independently)
  - For 8 workers: Budget **1-2 hours** for full deployment (workers start in parallel but are limited by node scheduling)

The deployment's startup probe (`initialDelaySeconds: 120`, `periodSeconds: 30`, `failureThreshold: 60`) allows up to 32 minutes per pod for model download and initialization.

### Step 2.4: Verify Workers Are Healthy

> ⚠️ **CRITICAL CHECKPOINT**: Before running benchmarks, you **MUST** verify equal worker health. Unequal worker counts will invalidate your comparison results.

```bash
# Quick health check - should show "8/8"
echo "Workers: $(kubectl get pods -n dynamo-bench -l nvidia.com/dynamo-component-type=worker --field-selector=status.phase=Running -o json | jq '[.items[] | select(.status.conditions[] | select(.type=="Ready" and .status=="True"))] | length')/8 ready"

# Detailed view
kubectl get pods -n dynamo-bench -l nvidia.com/dynamo-component-type=worker
```

**All 8 must show `1/1 Running` and Ready.** Do not proceed until this is confirmed. Repeat this check after you tear down router-ON and deploy router-OFF (Phase 5).

---

## Phase 3: Prepare Benchmark Dataset

### Understanding the Mooncake Toolagent Trace

For this A/B comparison, we use the [**Mooncake FAST'25 Toolagent Trace**](https://github.com/kvcache-ai/Mooncake/blob/main/FAST25-release/traces/toolagent_trace.jsonl), published by [Mooncake AI](https://github.com/kvcache-ai/Mooncake) (USENIX FAST'25 Best Paper). This is a privacy-preserving dataset of real-world LLM inference traffic from production **tool-agent workloads** — AI agents that iteratively call tools and APIs while maintaining a growing conversation context. The trace contains **23,608 requests** spanning ~59 minutes of real-time traffic.

**Why the toolagent trace?** Tool-agent workloads are ideal for evaluating KV cache routing because each agent session involves repeated LLM calls that share a long, growing prefix (system prompt + conversation history + tool results), producing high natural prefix overlap between requests. The Mooncake toolagent trace captures these realistic patterns, letting us demonstrate the router's real-world performance gains.

**What's in the dataset?** Each trace entry contains:
- **Timestamp:** When the request arrived (for realistic request timing)
- **Input/output lengths:** Number of tokens in prompts and responses
- **Block hash IDs:** Cryptographic hashes representing KV cache blocks (no user text; explained below)

**Sample trace entries (showing prefix reuse):**
```json
{"timestamp": 0, "input_length": 9013, "output_length": 3, "hash_ids": [46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]}
{"timestamp": 0, "input_length": 6506, "output_length": 3, "hash_ids": [46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 64]}
```

These two requests share blocks 46–57 (12 blocks × 512 tokens = ~6,144 tokens of shared prefix) — a tool agent continuing the same session with accumulated context. Each hash ID represents a **512-token block**, and the hash includes both the current block and all preceding blocks, preserving the pattern of prefix reuse while protecting user privacy. The **KV Smart Router** routes requests with matching hash IDs to the same worker, maximizing cache hits.

**Key Dataset Properties:**
- ✅ **Realistic timing:** Request arrival patterns from production tool-agent workloads
- ✅ **High prefix overlap:** 59% cache ratio ([Mooncake FAST'25 paper](https://github.com/kvcache-ai/Mooncake/blob/main/FAST25-release/Mooncake-FAST25.pdf)); iterative tool calls within sessions produce natural prefix reuse
- ✅ **Privacy-preserving:** No actual text — only hash-based cache block identifiers
- ✅ **Reproducible:** Public dataset enables fair comparisons across different systems

### Download and Prepare the Dataset

```bash
# Download the Mooncake FAST'25 toolagent trace
curl -sL https://raw.githubusercontent.com/kvcache-ai/Mooncake/refs/heads/main/FAST25-release/traces/toolagent_trace.jsonl -o toolagent_trace.jsonl

# Slow down timestamps to 0.80× replay speed (~5.3 req/s instead of ~6.7 req/s)
python3 - <<'PY'
import json

with open("toolagent_trace.jsonl") as src, open("toolagent_trace_080x.jsonl", "w") as dst:
    for line in src:
        rec = json.loads(line)
        rec["timestamp"] = int(rec["timestamp"] / 0.80)
        dst.write(json.dumps(rec) + "\n")
PY

echo "Dataset ready: toolagent_trace_080x.jsonl (23,608 requests, 0.80x speed)"
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
  namespace: dynamo-bench
spec:
  backoffLimit: 1
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: benchmark
        image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.9.0
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
kubectl get pods -n dynamo-bench
```

### Step 4.2: Copy Dataset to Benchmark Pod

```bash
POD_NAME=$(kubectl get pods -n dynamo-bench -l job-name=aiperf-benchmark -o jsonpath='{.items[0].metadata.name}')
kubectl -n dynamo-bench cp toolagent_trace_080x.jsonl ${POD_NAME}:/tmp/toolagent_trace_080x.jsonl
```

### Step 4.3: Install AIPerf

```bash
kubectl -n dynamo-bench exec ${POD_NAME} -- bash -lc '
  . /opt/dynamo/venv/bin/activate
  pip install -q aiperf==0.5.0
'
```

---

## Phase 5: Run Benchmarks

### Step 5.1: Benchmark Router-ON

Run the benchmark against the **router-ON** deployment:

```bash
kubectl -n dynamo-bench exec ${POD_NAME} -- bash -lc '
  . /opt/dynamo/venv/bin/activate
  AIPERF_HTTP_CONNECTION_LIMIT=200 aiperf profile \
    -m "Qwen/Qwen3-32B" \
    --tokenizer "Qwen/Qwen3-32B" \
    --input-file "/tmp/toolagent_trace_080x.jsonl" \
    --custom-dataset-type "mooncake_trace" \
    --fixed-schedule \
    --url "http://vllm-agg-router-frontend.dynamo-bench.svc.cluster.local:8000" \
    --streaming \
    --random-seed 42 \
    --workers-max 200 \
    --request-timeout-seconds 1000 \
    --profile-export-level "records" \
    --record-processors 8 \
    --artifact-dir "/tmp/aiperf_router_on" \
    --goodput "time_to_first_token:5000 inter_token_latency:100"
'
```

AIPerf writes the run to `/tmp/aiperf_router_on` on the pod (summary JSON and `profile_export.jsonl`).

### Step 5.2: Switch to Router-OFF and Benchmark

Tear down router-ON and deploy the baseline:

```bash
kubectl delete dynamographdeployment vllm-agg-router -n dynamo-bench
kubectl apply -f router-off-deployment.yaml -n dynamo-bench
```

Wait for 8/8 workers to be Ready again, then run the same benchmark against the baseline frontend:

```bash
kubectl -n dynamo-bench exec ${POD_NAME} -- bash -lc '
  . /opt/dynamo/venv/bin/activate
  AIPERF_HTTP_CONNECTION_LIMIT=200 aiperf profile \
    -m "Qwen/Qwen3-32B" \
    --tokenizer "Qwen/Qwen3-32B" \
    --input-file "/tmp/toolagent_trace_080x.jsonl" \
    --custom-dataset-type "mooncake_trace" \
    --fixed-schedule \
    --url "http://vllm-agg-no-router-frontend.dynamo-bench.svc.cluster.local:8000" \
    --streaming \
    --random-seed 42 \
    --workers-max 200 \
    --request-timeout-seconds 1000 \
    --profile-export-level "records" \
    --record-processors 8 \
    --artifact-dir "/tmp/aiperf_router_off" \
    --goodput "time_to_first_token:5000 inter_token_latency:100"
'
```

### Step 5.3: Collect Results

Copy the artifact directories (or the summary/export files inside them) to your machine:

```bash
kubectl -n dynamo-bench cp ${POD_NAME}:/tmp/aiperf_router_on ./aiperf_router_on
kubectl -n dynamo-bench cp ${POD_NAME}:/tmp/aiperf_router_off ./aiperf_router_off
```

AIPerf writes a summary and per-request records; use these to compare TTFT, latency, and throughput between router-ON and router-OFF.

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
- **Short prompts** (<1000 tokens) where routing overhead exceeds benefit
- **Evenly distributed load** where round-robin is already optimal
- **Low request rate** where cache eviction negates benefits

**KV Smart Router is beneficial when:**
- TTFT improvements > 20%
- No significant degradation in other metrics
- Workload demonstrates measurable prefix reuse patterns

**Standard routing is better when:**
- KV router shows <10% improvement
- Increased latency variance is observed
- Load distribution across workers is more important than cache affinity

### Example Comparison

From our Dynamo Operator benchmark with the full toolagent trace at 0.80× replay speed:

| Metric | Router-OFF (Baseline) | Router-ON (KV Router) | Improvement | Speedup |
|--------|----------------------|----------------------|-------------|---------|
| TTFT avg | 63,652 ms | 2,586 ms | **96% faster** | 24.6x ✅ |
| TTFT p99 | 332,974 ms | 17,871 ms | **95% faster** | 18.6x ✅ |
| E2E Latency avg | 92,856 ms | 19,112 ms | **79% faster** | 4.9x ✅ |
| E2E Latency p99 | 411,252 ms | 88,274 ms | **79% faster** | 4.7x ✅ |

In this example with all 8 workers healthy, the **KV router dramatically outperformed** the baseline:
- **96% faster TTFT** — Users see first token in ~2.6s instead of ~64s
- **79% lower E2E latency** — Requests complete in ~19s instead of ~93s
- **95% faster TTFT p99** — Tail latency drops from ~333s to ~18s

The toolagent trace has heavy prefix overlap from tool-agent sessions with repeated context. Without the KV router, requests with overlapping prefixes are scattered across workers, causing redundant recomputation and unbounded queue growth at high utilization. With the KV router, matching prefixes are routed to the same worker, maximizing cache hits and keeping latencies stable under load.

---

## Phase 7: Cleanup

```bash
kubectl delete dynamographdeployment --all -n dynamo-bench
kubectl delete job aiperf-benchmark -n dynamo-bench
kubectl delete namespace dynamo-bench
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
kubectl edit dynamographdeployment -n dynamo-bench
```

### Issue: ImagePullBackOff Errors

**Cause:** Version mismatch or missing credentials

**Solution:**
```bash
# Check available versions
kubectl get pods -n dynamo-bench -o yaml | grep image:

# Update deployment YAML to match cluster version
```

### Issue: Operator Not Processing Deployment

**Cause:** Namespace restrictions

**Solution:**
- Ensure Dynamo platform is Helm-installed in the namespace
- Verify operator has `--restrictedNamespace=dynamo-bench` argument
- Check operator logs: `kubectl logs -n dynamo-bench deployment/dynamo-platform-dynamo-operator-controller-manager`

### Issue: Workers Not Becoming Ready

**Cause:** Model download failures or probe configuration

**Solution:**
```bash
# Check worker logs
kubectl logs -n dynamo-bench <worker-pod-name>

# Common issues:
# - Invalid HuggingFace token
# - Network connectivity
# - Insufficient disk space for model
```

### Issue: Workers Restarting in CrashLoopBackOff

**Cause:** Startup probe timeout — workers killed before finishing initialization

**Symptoms:**
- Pods show "Container main failed startup probe, will be restarted"
- Logs show model still downloading or loading when pod is killed
- Large models (>30GB) take longer than default 22-minute timeout

**Solution:**
Increase the startup probe `failureThreshold`:

```bash
# Patch the deployment to allow 32 minutes instead of 22
kubectl patch dynamographdeployment <deployment-name> -n dynamo-bench --type='json' \
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
- With cached model on node: ~2-5 minutes

### Issue: Unequal Worker Health

**Cause:** Resource constraints, image pull issues, or configuration errors

**Solution:**
```bash
# Check all worker status
kubectl get pods -n dynamo-bench -l nvidia.com/dynamo-component-type=worker

# Describe problematic pods
kubectl describe pod <pod-name> -n dynamo-bench

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

Replace the Mooncake trace with your own JSONL file:
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
6. **Consistent Configuration:** Use the same trace file and AIPerf options for both runs

---

## Conclusion

This guide provides a complete methodology for A/B testing Dynamo's KV Smart Router. The KV router's effectiveness depends heavily on workload characteristics—datasets with high prefix overlap will show the most benefit. For further details on tuning the KV router, see the [Tuning Guidelines](../components/router/router-guide.md#tuning-guidelines).

For questions or issues, consult the [Dynamo documentation](https://github.com/ai-dynamo/dynamo) or open an issue on GitHub.

---

## Appendix: Files Reference

- `router-off-deployment.yaml`: Standard routing deployment
- `router-on-deployment.yaml`: KV router enabled deployment
- `benchmark-job.yaml`: AIPerf benchmark pod
- AIPerf artifact dirs: summary JSON and `profile_export.jsonl` per run

**Repository:** [https://github.com/ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo)
