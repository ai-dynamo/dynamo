<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Profiler

The Dynamo Profiler is an automated performance analysis tool that measures model inference characteristics to optimize deployment configurations. It determines optimal tensor parallelism (TP) settings for prefill and decode phases, generates performance interpolation data, and enables SLA-driven autoscaling through the Planner.

## Feature Matrix

| Feature | vLLM | SGLang | TensorRT-LLM |
|---------|------|--------|--------------|
| Dense Model Profiling | ✅ | ✅ | ✅ |
| MoE Model Profiling | 🚧 | ✅ | 🚧 |
| AI Configurator (Offline) | ❌ | ❌ | ✅ |
| Online Profiling (AIPerf) | ✅ | ✅ | ✅ |
| Interactive WebUI | ✅ | ✅ | ✅ |
| Runtime Profiling Endpoints | ❌ | ✅ | ❌ |

## Quick Start

### Prerequisites

- Dynamo platform installed (see [Installation Guide](/docs/kubernetes/installation_guide.md))
- Kubernetes cluster with GPU nodes (for DGDR-based profiling)
- kube-prometheus-stack installed (required for SLA planner)

### Using DynamoGraphDeploymentRequest (Recommended)

The recommended way to profile models is through DGDRs, which automate the entire profiling and deployment workflow.

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeploymentRequest
metadata:
  name: my-model-profiling
spec:
  model: "Qwen/Qwen3-0.6B"
  backend: vllm

  profilingConfig:
    profilerImage: "nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.9.0"
    config:
      sla:
        isl: 3000      # Average input sequence length
        osl: 150       # Average output sequence length
        ttft: 200.0    # Target Time To First Token (ms)
        itl: 20.0      # Target Inter-Token Latency (ms)

  deploymentOverrides:
    workersImage: "nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.9.0"

  autoApply: true
```

```bash
kubectl apply -f my-profiling-dgdr.yaml -n $NAMESPACE
```

### Using AI Configurator (Fast Offline Profiling)

For TensorRT-LLM, use AI Configurator for rapid profiling (~30 seconds):

```yaml
profilingConfig:
  config:
    sweep:
      useAiConfigurator: true
      aicSystem: h200_sxm
      aicHfId: Qwen/Qwen3-32B
      aicBackendVersion: "0.20.0"
```

### Direct Script Usage (Advanced)

For advanced scenarios, run the profiler directly:

```bash
python -m benchmarks.profiler.profile_sla \
  --backend vllm \
  --config path/to/disagg.yaml \
  --model meta-llama/Llama-3-8B \
  --ttft 200 --itl 15 \
  --isl 3000 --osl 150
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sla.isl` | - | Average input sequence length (tokens) |
| `sla.osl` | - | Average output sequence length (tokens) |
| `sla.ttft` | - | Target Time To First Token (milliseconds) |
| `sla.itl` | - | Target Inter-Token Latency (milliseconds) |
| `sweep.useAiConfigurator` | `false` | Use offline simulation instead of real profiling |
| `hardware.minNumGpusPerEngine` | auto | Minimum GPUs per engine (auto-detected from model size) |
| `hardware.maxNumGpusPerEngine` | 8 | Maximum GPUs per engine |

## Profiling Methods

| Method | Duration | Accuracy | GPU Required | Backends |
|--------|----------|----------|--------------|----------|
| Online (AIPerf) | 2-4 hours | Highest | Yes | All |
| Offline (AI Configurator) | 20-30 seconds | Estimated | No | TensorRT-LLM |

## Output

The profiler generates:

1. **Optimal Configuration**: Recommended TP sizes for prefill and decode engines
2. **Performance Data**: Interpolation models for the SLA Planner
3. **Generated DGD**: Complete deployment manifest with optimized settings

Example recommendations:
```text
Suggested prefill TP:4 (TTFT 48.37 ms, throughput 15505.23 tokens/s/GPU)
Suggested decode TP:4 (ITL 4.83 ms, throughput 51.22 tokens/s/GPU)
```

## Next Steps

| Document | Description |
|----------|-------------|
| [Profiler Guide](profiler_guide.md) | Configuration, methods, and troubleshooting |
| [Profiler Examples](profiler_examples.md) | Complete DGDR YAMLs, WebUI, script examples |
| [SLA Planner Guide](/docs/components/planner/planner_guide.md) | End-to-end deployment workflow |
| [SLA Planner Architecture](/docs/components/planner/planner_guide.md) | How the Planner uses profiling data |

```{toctree}
:hidden:

profiler_guide
profiler_examples
```
