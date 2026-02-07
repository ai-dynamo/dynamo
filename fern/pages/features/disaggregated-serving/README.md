---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---

# Disaggregated Serving Guide

[AIConfigurator](https://github.com/ai-dynamo/aiconfigurator/tree/main) is a performance optimization tool that helps you find the optimal configuration for deploying LLMs with Dynamo. It automatically determines the best number of prefill and decode workers, parallelism settings, and deployment parameters to meet your SLA targets while maximizing throughput.

## Why Use AIConfigurator?

When deploying LLMs with Dynamo, you need to make several critical decisions:
- **Aggregated vs Disaggregated**: Which architecture gives better performance for your workload?
- **Worker Configuration**: How many prefill and decode workers to deploy?
- **Parallelism Settings**: What tensor/pipeline parallel configuration to use?
- **SLA Compliance**: How to meet your TTFT and TPOT targets?

AIConfigurator answers these questions in seconds, providing:
- Recommended configurations that meet your SLA requirements
- Ready-to-deploy Dynamo configuration files (including Kubernetes manifests)
- Performance comparisons between different deployment strategies
- Up to 1.7x better throughput compared to manual configuration

### End-to-End Workflow

![AIConfigurator end-to-end workflow](/assets/img/e2e-workflow.svg)

### Aggregated vs Disaggregated Architecture

AIConfigurator evaluates two deployment architectures and recommends the best one for your workload:

![Aggregated vs Disaggregated architecture comparison](/assets/img/arch-comparison.svg)

### When to Use Each Architecture

![Decision flowchart for choosing aggregated vs disaggregated](/assets/img/decision-flowchart.svg)

## Quick Start

```bash
# Install
pip3 install aiconfigurator

# Find optimal configuration for vLLM backend
aiconfigurator cli default \
  --model_path Qwen/Qwen3-32B-FP8 \
  --total_gpus 8 \
  --system h200_sxm \
  --backend vllm \
  --backend_version 0.12.0 \
  --isl 4000 \
  --osl 500 \
  --ttft 600 \
  --tpot 16.67 \
  --save_dir ./results_vllm

# Deploy on Kubernetes
kubectl apply -f ./results_vllm/agg/top1/agg/k8s_deploy.yaml
```

## Complete Walkthrough: vLLM on H200

This section walks through a validated example deploying Qwen3-32B-FP8 on 8× H200 GPUs using vLLM.

### Step 1: Run AIConfigurator

```bash
aiconfigurator cli default \
  --model_path Qwen/Qwen3-32B-FP8 \
  --system h200_sxm \
  --total_gpus 8 \
  --isl 4000 \
  --osl 500 \
  --ttft 600 \
  --tpot 16.67 \
  --backend vllm \
  --backend_version 0.12.0 \
  --save_dir ./results_vllm
```

**Parameters explained:**
- `--model_path`: HuggingFace model ID or local path (e.g., `Qwen/Qwen3-32B-FP8`)
- `--system`: GPU system type (`h200_sxm`, `h100_sxm`, `a100_sxm`)
- `--total_gpus`: Number of GPUs available for deployment
- `--isl` / `--osl`: Input/Output sequence lengths in tokens
- `--ttft` / `--tpot`: SLA targets - Time To First Token (ms) and Time Per Output Token (ms)
- `--backend`: Inference backend (`vllm`, `trtllm`, or `sglang`)
- `--backend_version`: Backend version (e.g., `0.12.0` for vLLM)
- `--save_dir`: Directory to save generated deployment configs

### Step 2: Review the Results

AIConfigurator outputs a comparison of aggregated vs disaggregated deployment strategies:

```text
********************************************************************************
*                     Dynamo aiconfigurator Final Results                      *
********************************************************************************
  ----------------------------------------------------------------------------
  Input Configuration & SLA Target:
    Model: Qwen/Qwen3-32B-FP8 (is_moe: False)
    Total GPUs: 8
    Best Experiment Chosen: disagg at 521.77 tokens/s/gpu
  ----------------------------------------------------------------------------
  Overall Best Configuration:
    - Best Throughput: 4,174.16 tokens/s
    - Per-GPU Throughput: 521.77 tokens/s/gpu
    - Per-User Throughput: 76.96 tokens/s/user
    - TTFT: 388.11ms
    - TPOT: 12.99ms
  ----------------------------------------------------------------------------
```

AIC evaluates both aggregated and disaggregated architectures and outputs ranked configurations for each:

```text
agg Top Configurations: (Sorted by tokens/s/gpu)
+------+--------------+---------------+--------+-----------------+-------------+-------------------+----------+----------+----+
| Rank | tokens/s/gpu | tokens/s/user |  TTFT  | request_latency | concurrency | total_gpus (used) | replicas | parallel | bs |
+------+--------------+---------------+--------+-----------------+-------------+-------------------+----------+----------+----+
|  1   |    397.31    |     60.66     | 509.14 |     8734.68     | 56 (=14x4)  |    8 (8=4x2)      |    4     |  tp2pp1  | 14 |
|  2   |    349.90    |     60.98     | 412.58 |     8596.28     | 48 (=24x2)  |    8 (8=2x4)      |    2     |  tp4pp1  | 24 |
|  3   |    235.62    |     62.71     | 482.57 |     8439.41     | 32 (=32x1)  |    8 (8=1x8)      |    1     |  tp8pp1  | 32 |
+------+--------------+---------------+--------+-----------------+-------------+-------------------+----------+----------+----+

disagg Top Configurations: (Sorted by tokens/s/gpu)
+------+--------------+---------------+--------+-----------------+-------------+-------------------+----------+-------------+----------+
| Rank | tokens/s/gpu | tokens/s/user |  TTFT  | request_latency | concurrency | total_gpus (used) | replicas | (p)parallel | (d)parallel |
+------+--------------+---------------+--------+-----------------+-------------+-------------------+----------+-------------+----------+
|  1   |    521.77    |     76.96     | 388.11 |     6871.61     | 60 (=60x1)  |    8 (8=1x8)      |    1     |   tp2pp1    |  tp4pp1  |
|  2   |    521.77    |     63.29     | 388.11 |     8272.31     | 80 (=40x2)  |    8 (8=2x4)      |    2     |   tp2pp1    |  tp2pp1  |
|  3   |    260.89    |     62.81     | 388.11 |     8332.18     | 42 (=42x1)  |    8 (8=1x8)      |    1     |   tp2pp1    |  tp1pp1  |
+------+--------------+---------------+--------+-----------------+-------------+-------------------+----------+-------------+----------+
```

**Reading the output:**
- **tokens/s/gpu**: Overall throughput efficiency — higher is better
- **tokens/s/user**: Per-request generation speed (inverse of TPOT)
- **TTFT**: Predicted time to first token
- **concurrency**: Total concurrent requests across all replicas (e.g., `56 (=14x4)` means batch size 14 × 4 replicas)
- **agg Rank 1** recommends TP2 with 4 replicas — simpler to deploy
- **disagg Rank 1** recommends 2 prefill workers (TP2) + 1 decode worker (TP4) — higher throughput but requires RDMA

### Step 3: Deploy on Kubernetes

The `--save_dir` generates ready-to-use Kubernetes manifests:

```
results_vllm/
├── agg/
│   └── top1/
│       └── agg/
│           ├── k8s_deploy.yaml    # Kubernetes DynamoGraphDeployment
│           └── agg_config.yaml    # Engine configuration
├── disagg/
│   └── top1/
│       └── disagg/
│           ├── k8s_deploy.yaml
│           ├── prefill_config.yaml
│           └── decode_config.yaml
└── pareto_frontier.png
```

#### Prerequisites

Before deploying, ensure you have:

1. **HuggingFace Token Secret** (for gated models):
   ```bash
   kubectl create secret generic hf-token-secret \
     -n your-namespace \
     --from-literal=HF_TOKEN="your-huggingface-token"
   ```

2. **Model Cache PVC** (recommended for faster restarts):
   ```yaml
   apiVersion: v1
   kind: PersistentVolumeClaim
   metadata:
     name: model-cache
     namespace: your-namespace
   spec:
     accessModes:
       - ReadWriteMany
     resources:
       requests:
         storage: 100Gi
   ```

#### Deploy the Configuration

The generated `k8s_deploy.yaml` provides a starting point. You'll typically need to customize it for your environment:

```bash
kubectl apply -f ./results_vllm/agg/top1/agg/k8s_deploy.yaml
```

**Complete deployment example** with model cache and production settings:

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: dynamo-agg
  namespace: your-namespace
spec:
  backendFramework: vllm
  pvcs:
    - name: model-cache
      create: false           # Use existing PVC
  services:
    Frontend:
      componentType: frontend
      replicas: 1
      volumeMounts:
        - name: model-cache
          mountPoint: /opt/models
      envs:
        - name: HF_HOME
          value: /opt/models
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.8.0
          imagePullPolicy: IfNotPresent

    VLLMWorker:
      envFromSecret: hf-token-secret
      componentType: worker
      replicas: 4
      resources:
        limits:
          gpu: "2"
      sharedMemory:
        size: 16Gi            # Required for vLLM
      volumeMounts:
        - name: model-cache
          mountPoint: /opt/models
      envs:
        - name: HF_HOME
          value: /opt/models
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.8.0
          workingDir: /workspace
          imagePullPolicy: IfNotPresent
          command:
            - python3
            - -m
            - dynamo.vllm
          args:
            - --model
            - "Qwen/Qwen3-32B-FP8"
            - "--no-enable-prefix-caching"
            - "--tensor-parallel-size"
            - "2"
            - "--pipeline-parallel-size"
            - "1"
            - "--data-parallel-size"
            - "1"
            - "--kv-cache-dtype"
            - "fp8"
            - "--max-model-len"
            - "6000"
            - "--max-num-seqs"
            - "1024"
```

**Key deployment settings:**

| Setting | Purpose | Notes |
|---------|---------|-------|
| `backendFramework: vllm` | Tells Dynamo which runtime to use | Required at spec level |
| `pvcs` + `volumeMounts` | Caches model weights across restarts | Mount at `/opt/models` (not `/root/`) |
| `HF_HOME` env var | Points HuggingFace to cache location | Must match `mountPoint` |
| `sharedMemory.size: 16Gi` | IPC memory for vLLM | 16Gi for vLLM, 80Gi for TRT-LLM |
| `envFromSecret` | Injects HF_TOKEN | Required for gated models |

### Step 4: Validate with AIPerf

After deployment, validate the predictions against actual performance using [AIPerf](https://github.com/ai-dynamo/aiperf).

<Tip>
Run AIPerf **inside the cluster** to avoid network latency affecting measurements. Use a Kubernetes Job:
</Tip>

#### Deriving AIPerf Parameters from AIC Output

To use AIPerf to benchmark an AIC-recommended configuration, you'll need to translate AIC parameters into AIPerf profiling arguments (we are working to automate this):

![AIC-to-AIPerf parameter mapping](/assets/img/param-mapping.svg)

| AIC Output | AIPerf Parameter | Notes |
|------------|-----------------|-------|
| `concurrency: 56 (=14x4)` | `--concurrency 56` | Use total concurrency when benchmarking via the frontend |
| ISL/OSL targets | `--isl 4000 --osl 500` | Match your AIC inputs |
| - | `--num-requests 800` | Use `concurrency × 40` minimum for statistical stability |
| - | `--extra-inputs "ignore_eos:true"` | Ensures exact OSL tokens generated |

> **Note on concurrency**: AIC reports concurrency as `total (=bs × replicas)`. When benchmarking through the frontend (which routes to all replicas), use the total value. If benchmarking a single replica directly, use the per-replica `bs` value instead.

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: aiperf-benchmark
  namespace: your-namespace
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: aiperf
        image: python:3.10
        command:
        - /bin/bash
        - -c
        - |
          pip install aiperf
          aiperf profile \
            -m Qwen/Qwen3-32B-FP8 \
            --endpoint-type chat \
            -u http://dynamo-agg-frontend:8000 \
            --isl 4000 --isl-stddev 0 \
            --osl 500 --osl-stddev 0 \
            --num-requests 800 \
            --concurrency 56 \
            --streaming \
            --extra-inputs "ignore_eos:true" \
            --num-warmup-requests 40 \
            --ui-type simple
```

```bash
kubectl apply -f aiperf-job.yaml
kubectl logs -f -l job-name=aiperf-benchmark
```

**Validated results** (Qwen3-32B-FP8, 8× H200, TP2×4 replicas, aggregated):

| Metric | AIC Prediction | Actual (avg) | Status |
|--------|---------------|--------------|--------|
| TTFT (ms) | 509 | 209 | Better than target |
| ITL/TPOT (ms) | 16.49 | 15.06 | Within 10% |
| Throughput (req/s) | ~6.3 | 6.9 | Within 10% |
| Total Output TPS | ~3,178 | 3,462 | Within 10% |

<Note>
Actual throughput typically reaches ~85-90% of AIC predictions, with ITL/TPOT being the most accurate metric. Expect some variance between benchmark runs; running multiple times is recommended. Enable prefix caching (`--enable-prefix-caching`) for additional TTFT improvements with repeated prompts.
</Note>

## Fine-Tuning Your Deployment

AIConfigurator provides a strong starting point. Here's how to iterate for production:

### Adjusting for Actual Workload

If your real workload differs from the benchmark parameters:

```bash
# For longer outputs (chat/code generation):
# increase OSL, relax TTFT target
aiconfigurator cli default \
  --model_path Qwen/Qwen3-32B-FP8 \
  --total_gpus 8 \
  --system h200_sxm \
  --backend vllm \
  --backend_version 0.12.0 \
  --isl 2000 \
  --osl 2000 \
  --ttft 1000 \
  --tpot 10 \
  --save_dir ./results_long_output
```

### Exploring Alternative Configurations

Use `exp` mode to compare custom configurations:

```yaml
# custom_exp.yaml
exps:
  - exp_tp2
  - exp_tp4

exp_tp2:
  mode: "patch"
  serving_mode: "agg"
  model_path: "Qwen/Qwen3-32B-FP8"
  total_gpus: 8
  system_name: "h200_sxm"
  backend_name: "vllm"
  backend_version: "0.12.0"
  isl: 4000
  osl: 500
  ttft: 600
  tpot: 16.67
  config:
    agg_worker_config:
      tp_list: [2]

exp_tp4:
  mode: "patch"
  serving_mode: "agg"
  model_path: "Qwen/Qwen3-32B-FP8"
  total_gpus: 8
  system_name: "h200_sxm"
  backend_name: "vllm"
  backend_version: "0.12.0"
  isl: 4000
  osl: 500
  ttft: 600
  tpot: 16.67
  config:
    agg_worker_config:
      tp_list: [4]
```

```bash
aiconfigurator cli exp --yaml_path custom_exp.yaml --save_dir ./results_custom
```

> **Critical**: Disaggregated deployments **require RDMA** for KV cache transfer. Without RDMA, performance degrades by **40x** (TTFT increases from 355ms to 10+ seconds). See the Disaggregated Deployment section below.

### Deploying Disaggregated (RDMA Required)

Disaggregated deployments transfer KV cache between prefill and decode workers. **Without RDMA, this transfer becomes a severe bottleneck**, causing 40x performance degradation.

#### Prerequisites for Disaggregated

1. **RDMA-capable network** (InfiniBand or RoCE)
2. **RDMA device plugin** installed on the cluster (provides `rdma/ib` resources)
3. **ETCD and NATS** deployed (for coordination)

#### Disaggregated DGD with RDMA

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: dynamo-disagg
  namespace: your-namespace
spec:
  backendFramework: vllm
  pvcs:
    - name: model-cache
      create: false
  services:
    Frontend:
      componentType: frontend
      replicas: 1
      volumeMounts:
        - name: model-cache
          mountPoint: /opt/models
      envs:
        - name: HF_HOME
          value: /opt/models
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.8.0
          imagePullPolicy: IfNotPresent

    VLLMPrefillWorker:
      envFromSecret: hf-token-secret
      componentType: worker
      subComponentType: prefill
      replicas: 2
      resources:
        limits:
          gpu: "2"
      sharedMemory:
        size: 16Gi
      volumeMounts:
        - name: model-cache
          mountPoint: /opt/models
      envs:
        - name: HF_HOME
          value: /opt/models
        - name: UCX_TLS
          value: "rc_x,rc,dc_x,dc,cuda_copy,cuda_ipc"  # Enable RDMA transports
        - name: UCX_RNDV_SCHEME
          value: "get_zcopy"
        - name: UCX_RNDV_THRESH
          value: "0"
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.8.0
          workingDir: /workspace
          imagePullPolicy: IfNotPresent
          securityContext:
            capabilities:
              add: ["IPC_LOCK"]  # Required for RDMA memory registration
          resources:
            limits:
              rdma/ib: "2"      # Request RDMA resources
            requests:
              rdma/ib: "2"
          command: ["python3", "-m", "dynamo.vllm"]
          args:
            - --model
            - "Qwen/Qwen3-32B-FP8"
            - "--tensor-parallel-size"
            - "2"
            - "--kv-cache-dtype"
            - "fp8"
            - "--max-num-seqs"
            - "1"               # Prefill workers use batch size 1
            - --is-prefill-worker

    VLLMDecodeWorker:
      envFromSecret: hf-token-secret
      componentType: worker
      subComponentType: decode
      replicas: 1
      resources:
        limits:
          gpu: "4"
      sharedMemory:
        size: 16Gi
      volumeMounts:
        - name: model-cache
          mountPoint: /opt/models
      envs:
        - name: HF_HOME
          value: /opt/models
        - name: UCX_TLS
          value: "rc_x,rc,dc_x,dc,cuda_copy,cuda_ipc"
        - name: UCX_RNDV_SCHEME
          value: "get_zcopy"
        - name: UCX_RNDV_THRESH
          value: "0"
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.8.0
          workingDir: /workspace
          imagePullPolicy: IfNotPresent
          securityContext:
            capabilities:
              add: ["IPC_LOCK"]
          resources:
            limits:
              rdma/ib: "4"
            requests:
              rdma/ib: "4"
          command: ["python3", "-m", "dynamo.vllm"]
          args:
            - --model
            - "Qwen/Qwen3-32B-FP8"
            - "--tensor-parallel-size"
            - "4"
            - "--kv-cache-dtype"
            - "fp8"
            - "--max-num-seqs"
            - "1024"            # Decode workers handle high concurrency
            - --is-decode-worker
```

**Critical RDMA settings:**

| Setting | Purpose |
|---------|---------|
| `rdma/ib: "N"` | Request N RDMA resources (match TP size) |
| `IPC_LOCK` capability | Required for RDMA memory registration |
| `UCX_TLS` env var | Enables RDMA transports (rc_x, dc_x) |
| `UCX_RNDV_SCHEME=get_zcopy` | Zero-copy RDMA transfers |

#### Verifying RDMA is Active

After deployment, check the worker logs for UCX initialization:

```bash
kubectl logs <prefill-worker-pod> | grep -i "UCX\|NIXL"
```

You should see:
```
NIXL INFO Backend UCX was instantiated
```

If you see only TCP transports, RDMA is not active - check your RDMA device plugin and resource requests.

### Tuning vLLM-Specific Parameters

Override vLLM engine parameters with `--generator-set`:

```bash
aiconfigurator cli default \
  --model_path Qwen/Qwen3-32B-FP8 \
  --total_gpus 8 \
  --system h200_sxm \
  --backend vllm \
  --backend_version 0.12.0 \
  --isl 4000 --osl 500 \
  --ttft 600 --tpot 16.67 \
  --save_dir ./results_tuned \
  --generator-set Workers.agg.kv_cache_free_gpu_memory_fraction=0.85 \
  --generator-set Workers.agg.max_num_seqs=2048
```

Run `aiconfigurator cli default --generator-help` to see all available parameters.

### Prefix Caching Considerations

For workloads with repeated prefixes (e.g., system prompts):

- **Enable prefix caching** when you have high prefix hit rates
- **Disable prefix caching** (`--no-enable-prefix-caching`) for diverse prompts

AIConfigurator's default predictions assume no prefix caching. Enable it post-deployment if your workload benefits.

## Supported Configurations

### Backends and Versions

| Backend | Versions | Status |
|---------|----------|--------|
| TensorRT-LLM | 1.0.0rc3, 1.2.0rc5 | Production |
| vLLM | 0.12.0 | Production |
| SGLang | 0.5.6.post2 | Production |

### Systems

| GPU System | TensorRT-LLM | vLLM | SGLang |
|------------|--------------|------|--------|
| H200 SXM | Yes | Yes | Yes |
| H100 SXM | Yes | Yes | Yes |
| A100 SXM | Yes | Yes | -- |
| B200 SXM | Yes | -- | Yes |
| GB200 SXM | Yes | -- | -- |

### Models

- **Dense**: GPT, LLAMA2/3, QWEN2.5/3
- **MoE**: Mixtral, DEEPSEEK_V3

## Common Use Cases

```bash
# Strict latency SLAs (real-time chat)
aiconfigurator cli default \
  --model_path meta-llama/Llama-3.1-70B \
  --total_gpus 16 \
  --system h200_sxm \
  --backend vllm \
  --backend_version 0.12.0 \
  --ttft 200 --tpot 8

# High throughput (batch processing)
aiconfigurator cli default \
  --model_path Qwen/Qwen3-32B-FP8 \
  --total_gpus 32 \
  --system h200_sxm \
  --backend trtllm \
  --ttft 2000 --tpot 50

# Request latency constraint (end-to-end SLA)
aiconfigurator cli default \
  --model_path Qwen/Qwen3-32B-FP8 \
  --total_gpus 16 \
  --system h200_sxm \
  --backend vllm \
  --backend_version 0.12.0 \
  --request_latency 12000 \
  --isl 4000 --osl 500
```

## Additional Options

```bash
# Web interface for interactive exploration
pip3 install aiconfigurator[webapp]
aiconfigurator webapp  # Visit http://127.0.0.1:7860

# Quick config generation (no parameter sweep)
aiconfigurator cli generate \
  --model_path Qwen/Qwen3-32B-FP8 \
  --total_gpus 8 \
  --system h200_sxm \
  --backend vllm

# Check model/system support
aiconfigurator cli support \
  --model_path Qwen/Qwen3-32B-FP8 \
  --system h200_sxm \
  --backend vllm
```

## Troubleshooting

### AIConfigurator Issues

**Model not found**: Use the full HuggingFace path (e.g., `Qwen/Qwen3-32B-FP8` not `QWEN3_32B`)

**Backend version mismatch**: Check supported versions with `aiconfigurator cli support --model_path <model> --system <system> --backend <backend>`

### Deployment Issues

**Pods crash with "Permission denied" on cache directory**:
- Mount the PVC at `/opt/models` instead of `/root/.cache/huggingface`
- Set `HF_HOME=/opt/models` environment variable
- Ensure the PVC has `ReadWriteMany` access mode

**Workers stuck in CrashLoopBackOff**:
- Check logs: `kubectl logs <pod-name> --previous`
- Verify `sharedMemory.size` is set (16Gi for vLLM, 80Gi for TRT-LLM)
- Ensure HuggingFace token secret exists and is named correctly

**Model download slow on every restart**:
- Add PVC for model caching (see deployment example above)
- Verify `volumeMounts` and `HF_HOME` are configured on workers

**"Context stopped or killed" errors (disaggregated only)**:
- Deploy ETCD and NATS infrastructure (required for KV cache transfer)
- See [Dynamo Kubernetes Guide](../../kubernetes/README.md) for platform setup

### Performance Issues

**OOM errors**: Reduce `--max-num-seqs` or increase tensor parallelism

**Performance below predictions**:
- Verify warmup requests are sufficient (40+ recommended)
- Check for competing workloads on the cluster
- Ensure KV cache memory fraction is optimized
- Run benchmarks from inside the cluster to eliminate network latency

**Disaggregated TTFT extremely high (10+ seconds)**:
This is almost always caused by **missing RDMA configuration**. Without RDMA, KV cache transfer falls back to TCP and becomes a severe bottleneck.

To diagnose:
```bash
# Check if RDMA resources are allocated
kubectl get pod <worker-pod> -o yaml | grep -A5 "resources:"

# Check UCX transport in logs
kubectl logs <worker-pod> | grep -i "UCX\|transport"
```

To fix:
1. Ensure your cluster has RDMA device plugin installed
2. Add `rdma/ib` resource requests to worker pods
3. Add `IPC_LOCK` capability to security context
4. Add UCX environment variables (see Disaggregated Deployment section)

**Disaggregated working but throughput lower than aggregated**:
For balanced workloads (ISL/OSL ratio between 2:1 and 10:1), aggregated is often better. Disaggregated shines for:
- Very long inputs (ISL > 8000) with short outputs
- Workloads needing independent prefill/decode scaling

## Learn More

- [AIConfigurator CLI Guide](https://github.com/ai-dynamo/aiconfigurator/blob/main/docs/cli_user_guide.md)
- [Dynamo Deployment Guide](https://github.com/ai-dynamo/aiconfigurator/blob/main/docs/dynamo_deployment_guide.md)
- [Dynamo Installation Guide](../../kubernetes/installation-guide.md)
- [Benchmarking Guide](../../benchmarks/benchmarking.md)
