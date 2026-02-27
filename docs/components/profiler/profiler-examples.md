---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Profiler Examples
---

Complete examples for profiling with DGDRs.

## DGDR Examples

### Dense Model: AIPerf on Real Engines

Standard online profiling with real GPU measurements:

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeploymentRequest
metadata:
  name: vllm-dense-online
spec:
  model: "Qwen/Qwen3-0.6B"
  backend: vllm
  image: "nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.9.0"

  workload:
    isl: 3000
    osl: 150

  sla:
    ttft: 200.0
    itl: 20.0

  autoApply: true
```

### Dense Model: AI Configurator Simulation

Fast offline profiling (~30 seconds, TensorRT-LLM only):

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeploymentRequest
metadata:
  name: trtllm-aic-offline
spec:
  model: "Qwen/Qwen3-32B"
  backend: trtllm
  image: "nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:0.9.0"

  workload:
    isl: 4000
    osl: 500

  sla:
    ttft: 300.0
    itl: 10.0

  autoApply: true
```

### MoE Model

Multi-node MoE profiling with SGLang:

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeploymentRequest
metadata:
  name: sglang-moe
spec:
  model: "deepseek-ai/DeepSeek-R1"
  backend: sglang
  image: "nvcr.io/nvidia/ai-dynamo/sglang-runtime:0.9.0"

  workload:
    isl: 2048
    osl: 512

  sla:
    ttft: 300.0
    itl: 25.0

  hardware:
    numGpusPerNode: 8

  autoApply: true
```

### Using Existing DGD Config (ConfigMap)

Reference a custom DGD configuration via ConfigMap:

```bash
# Create ConfigMap from your DGD config file
kubectl create configmap deepseek-r1-config \
  --from-file=/path/to/your/disagg.yaml \
  --namespace $NAMESPACE \
  --dry-run=client -o yaml | kubectl apply -f -
```

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeploymentRequest
metadata:
  name: deepseek-r1
spec:
  model: deepseek-ai/DeepSeek-R1
  backend: sglang
  image: "nvcr.io/nvidia/ai-dynamo/sglang-runtime:0.9.0"

  workload:
    isl: 4000
    osl: 500

  sla:
    ttft: 300
    itl: 10

  autoApply: true
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
