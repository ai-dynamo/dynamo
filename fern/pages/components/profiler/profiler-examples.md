---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---

# Profiler Examples

Complete examples for profiling with DGDRs, the interactive WebUI, and direct script usage.

## DGDR Examples

### Dense Model: AIPerf on Real Engines

Standard online profiling with real GPU measurements:

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeploymentRequest
metadata:
  name: vllm-dense-online
spec:
  model: "Qwen/Qwen3-0.6B"
  backend: vllm

  profilingConfig:
    profilerImage: "nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.9.0"
    config:
      sla:
        isl: 3000
        osl: 150
        ttft: 200.0
        itl: 20.0

      hardware:
        minNumGpusPerEngine: 1
        maxNumGpusPerEngine: 8

      sweep:
        useAiConfigurator: false

  deploymentOverrides:
    workersImage: "nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.9.0"

  autoApply: true
```

### Dense Model: AI Configurator Simulation

Fast offline profiling (~30 seconds, TensorRT-LLM only):

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeploymentRequest
metadata:
  name: trtllm-aic-offline
spec:
  model: "Qwen/Qwen3-32B"
  backend: trtllm

  profilingConfig:
    profilerImage: "nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:0.9.0"
    config:
      sla:
        isl: 4000
        osl: 500
        ttft: 300.0
        itl: 10.0

      sweep:
        useAiConfigurator: true
        aicSystem: h200_sxm  # Also supports h100_sxm, b200_sxm, gb200_sxm, a100_sxm
        aicHfId: Qwen/Qwen3-32B
        aicBackendVersion: "0.20.0"

  deploymentOverrides:
    workersImage: "nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:0.9.0"

  autoApply: true
```

### MoE Model

Multi-node MoE profiling with SGLang:

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeploymentRequest
metadata:
  name: sglang-moe
spec:
  model: "deepseek-ai/DeepSeek-R1"
  backend: sglang

  profilingConfig:
    profilerImage: "nvcr.io/nvidia/ai-dynamo/sglang-runtime:0.9.0"
    config:
      sla:
        isl: 2048
        osl: 512
        ttft: 300.0
        itl: 25.0

      hardware:
        numGpusPerNode: 8
        maxNumGpusPerEngine: 32

      engine:
        isMoeModel: true

  deploymentOverrides:
    workersImage: "nvcr.io/nvidia/ai-dynamo/sglang-runtime:0.9.0"

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
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeploymentRequest
metadata:
  name: deepseek-r1
spec:
  model: deepseek-ai/DeepSeek-R1
  backend: sglang

  profilingConfig:
    profilerImage: "nvcr.io/nvidia/ai-dynamo/sglang-runtime:0.9.0"
    configMapRef:
      name: deepseek-r1-config
      key: disagg.yaml
    config:
      sla:
        isl: 4000
        osl: 500
        ttft: 300
        itl: 10
      sweep:
        useAiConfigurator: true
        aicSystem: h200_sxm
        aicHfId: deepseek-ai/DeepSeek-V3
        aicBackendVersion: "0.20.0"

  deploymentOverrides:
    workersImage: "nvcr.io/nvidia/ai-dynamo/sglang-runtime:0.9.0"

  autoApply: true
```

## Interactive WebUI

Launch an interactive configuration selection interface:

```bash
python -m benchmarks.profiler.profile_sla \
  --backend trtllm \
  --config path/to/disagg.yaml \
  --pick-with-webui \
  --use-ai-configurator \
  --model Qwen/Qwen3-32B-FP8 \
  --aic-system h200_sxm \
  --ttft 200 --itl 15
```

The WebUI launches on port 8000 by default (configurable with `--webui-port`).

### Features

- **Interactive Charts**: Visualize prefill TTFT, decode ITL, and GPU hours analysis with hover-to-highlight synchronization between charts and tables
- **Pareto-Optimal Analysis**: The GPU Hours table shows pareto-optimal configurations balancing latency and throughput
- **DGD Config Preview**: Click "Show Config" on any row to view the corresponding DynamoGraphDeployment YAML
- **GPU Cost Estimation**: Toggle GPU cost display to convert GPU hours to cost ($/1000 requests)
- **SLA Visualization**: Red dashed lines indicate your TTFT and ITL targets

### Selection Methods

1. **GPU Hours Table** (recommended): Click any row to select both prefill and decode configurations at once based on the pareto-optimal combination
2. **Individual Selection**: Click one row in the Prefill table AND one row in the Decode table to manually choose each

### Example DGD Config Output

When you click "Show Config", you see a DynamoGraphDeployment configuration:

```yaml
# DynamoGraphDeployment Configuration
# Prefill: 1 GPU(s), TP=1
# Decode: 4 GPU(s), TP=4
# Model: Qwen/Qwen3-32B-FP8
# Backend: trtllm
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
spec:
  services:
    PrefillWorker:
      subComponentType: prefill
      replicas: 1
      extraPodSpec:
        mainContainer:
          args:
          - --tensor-parallel-size=1
    DecodeWorker:
      subComponentType: decode
      replicas: 1
      extraPodSpec:
        mainContainer:
          args:
          - --tensor-parallel-size=4
```

Once you select a configuration, the full DGD CRD is saved as `config_with_planner.yaml`.

## Direct Script Examples

### Basic Profiling

```bash
python -m benchmarks.profiler.profile_sla \
  --backend vllm \
  --config path/to/disagg.yaml \
  --model meta-llama/Llama-3-8B \
  --ttft 200 --itl 15 \
  --isl 3000 --osl 150
```

### With GPU Constraints

```bash
python -m benchmarks.profiler.profile_sla \
  --backend sglang \
  --config examples/backends/sglang/deploy/disagg.yaml \
  --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --ttft 200 --itl 15 \
  --isl 3000 --osl 150 \
  --min-num-gpus 2 \
  --max-num-gpus 8
```

### AI Configurator (Offline)

```bash
python -m benchmarks.profiler.profile_sla \
  --backend trtllm \
  --config path/to/disagg.yaml \
  --use-ai-configurator \
  --model Qwen/Qwen3-32B-FP8 \
  --aic-system h200_sxm \
  --ttft 200 --itl 15 \
  --isl 4000 --osl 500
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
