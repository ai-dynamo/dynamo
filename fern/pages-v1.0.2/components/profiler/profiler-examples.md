---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Profiler Examples
---

Complete examples for profiling with DGDRs.

## DGDR Examples

### Dense Model: Rapid

Fast profiling (~30 seconds):

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeploymentRequest
metadata:
  name: qwen-0-6b
spec:
  model: "Qwen/Qwen3-0.6B"
  image: "nvcr.io/nvidia/ai-dynamo/dynamo-frontend:1.1.0"
```

### Dense Model: Thorough

Profiling with real GPU measurements:

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeploymentRequest
metadata:
  name: vllm-dense-online
spec:
  model: "Qwen/Qwen3-0.6B"
  backend: vllm
  image: "nvcr.io/nvidia/ai-dynamo/dynamo-frontend:1.1.0"
  searchStrategy: thorough
```

### MoE Model

Multi-node MoE profiling with SGLang:

<Info>
The PVC referenced by `modelCache.pvcName` must already exist in the same namespace and contain
the model weights at the specified `pvcModelPath`. The DGDR controller does not create or
populate the PVC — it only mounts it into the profiling job and deployed workers.
</Info>

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeploymentRequest
metadata:
  name: sglang-moe
spec:
  model: "deepseek-ai/DeepSeek-R1"
  backend: sglang
  image: "nvcr.io/nvidia/ai-dynamo/dynamo-frontend:1.1.0"

  hardware:
    numGpusPerNode: 8

  modelCache:
    pvcName: "model-cache"
    pvcModelPath: "deepseek-r1"      # path within the PVC
```

### Private Model

For gated or private HuggingFace models, pass your token via an environment variable injected
into the profiling job. Create the secret first:

```bash
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="${HF_TOKEN}" \
  -n ${NAMESPACE}
```

Then reference it in your DGDR:

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeploymentRequest
metadata:
  name: llama-private
spec:
  model: "meta-llama/Llama-3.1-8B-Instruct"
  image: "nvcr.io/nvidia/ai-dynamo/dynamo-frontend:1.1.0"

  overrides:
    profilingJob:
      template:
        spec:
          containers: []    # required placeholder; leave empty to inherit defaults
          initContainers:
            - name: profiler
              env:
                - name: HF_TOKEN
                  valueFrom:
                    secretKeyRef:
                      name: hf-token-secret
                      key: HF_TOKEN
```

### Custom SLA Targets

Control how the profiler optimizes your deployment by specifying latency targets and workload
characteristics.

**Explicit TTFT + ITL targets** (default mode):

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeploymentRequest
metadata:
  name: low-latency-dense
spec:
  model: "Qwen/Qwen3-0.6B"
  image: "nvcr.io/nvidia/ai-dynamo/dynamo-frontend:1.1.0"

  sla:
    ttft: 500      # Time To First Token target in milliseconds
    itl: 20        # Inter-Token Latency target in milliseconds

  workload:
    isl: 2000      # expected input sequence length (tokens)
    osl: 500       # expected output sequence length (tokens)
```

**End-to-end latency target** (alternative to ttft+itl):

```yaml
spec:
  ...
  sla:
    e2eLatency: 10000    # total request latency budget in milliseconds
```

### Overrides

Use `overrides` to customize the profiling job pod spec — for example to add tolerations for
GPU node taints or inject environment variables.

**GPU node toleration** (common on GKE and shared clusters):

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeploymentRequest
metadata:
  name: dense-with-tolerations
spec:
  model: "Qwen/Qwen3-0.6B"
  image: "nvcr.io/nvidia/ai-dynamo/dynamo-frontend:1.1.0"

  overrides:
    profilingJob:
      template:
        spec:
          containers: []    # required placeholder; leave empty to inherit defaults
          tolerations:
            - key: nvidia.com/gpu
              operator: Exists
              effect: NoSchedule
```

**Override the generated DynamoGraphDeployment** (e.g., to use a custom worker image):

```yaml
spec:
  ...
  overrides:
    dgd:
      apiVersion: nvidia.com/v1alpha1
      kind: DynamoGraphDeployment
      spec:
        services:
          VllmWorker:
            extraEnvs:
              - name: CUSTOM_ENV
                value: "my-value"
```

## SGLang Runtime Profiling

Profile SGLang workers at runtime via HTTP endpoints:

```bash
# Start profiling
curl -X POST http://localhost:9090/engine/start_profile \
  -H "Content-Type: application/json" \
  -d '{"output_dir": "/tmp/profiler_output"}'

# Run inference requests to generate profiling data...

# Stop profiling
curl -X POST http://localhost:9090/engine/stop_profile
```

A test script is provided at `examples/backends/sglang/test_sglang_profile.py`:

```bash
python examples/backends/sglang/test_sglang_profile.py
```

View traces using Chrome's `chrome://tracing`, [Perfetto UI](https://ui.perfetto.dev/), or TensorBoard.
