---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---

# Planner Examples

Practical examples for deploying the SLA Planner with different configurations. For deployment concepts, see the [Planner Guide](planner-guide.md). For a quick overview, see the [Planner README](README.md).

## Basic Examples

### Minimal DGDR with AIC (Fastest)

The simplest way to deploy with the SLA planner. Uses AI Configurator for offline profiling (20-30 seconds instead of hours):

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeploymentRequest
metadata:
  name: sla-aic
spec:
  model: Qwen/Qwen3-32B
  backend: vllm

  profilingConfig:
    profilerImage: "nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.6.1"
    config:
      sla:
        isl: 3000
        osl: 150
        ttft: 200
        itl: 20
      sweep:
        useAiConfigurator: true
        aicSystem: h200_sxm
        aicHfId: Qwen/Qwen3-32B
        aicBackendVersion: "0.20.0"

  deploymentOverrides:
    workersImage: "nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.6.1"

  autoApply: true
```

Deploy:
```bash
export NAMESPACE=your-namespace
kubectl apply -f benchmarks/profiler/deploy/profile_sla_aic_dgdr.yaml -n $NAMESPACE
```

### Online Profiling (Real Measurements)

Standard online profiling runs real GPU measurements for more accurate results. Takes 2-4 hours:

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeploymentRequest
metadata:
  name: sla-online
spec:
  model: meta-llama/Llama-3.3-70B-Instruct
  backend: vllm

  profilingConfig:
    profilerImage: "nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.6.1"
    config:
      sla:
        isl: 3000
        osl: 150
        ttft: 200
        itl: 20
      sweep:
        useAiConfigurator: false
        prefillInterpolationGranularity: 16
        decodeInterpolationGranularity: 6

  deploymentOverrides:
    workersImage: "nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.6.1"

  autoApply: true
```

Deploy:
```bash
kubectl apply -f benchmarks/profiler/deploy/profile_sla_dgdr.yaml -n $NAMESPACE
```

Available sample DGDRs in `benchmarks/profiler/deploy/`:
- **`profile_sla_dgdr.yaml`**: Standard online profiling for dense models
- **`profile_sla_aic_dgdr.yaml`**: Fast offline profiling using AI Configurator
- **`profile_sla_moe_dgdr.yaml`**: Online profiling for MoE models (SGLang)

> **Profiling Config Cases**: Prior to 0.8.1, fields under `profilingConfig.config` use snake_case. Starting 0.8.1, fields use camelCase. There is backwards compatibility to snake_case, but example DGDRs use camelCase.

## Kubernetes Examples

### MoE Models (SGLang)

For Mixture-of-Experts models like DeepSeek-R1, use SGLang backend:

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeploymentRequest
metadata:
  name: sla-moe
spec:
  model: deepseek-ai/DeepSeek-R1
  backend: sglang

  profilingConfig:
    profilerImage: "nvcr.io/nvidia/ai-dynamo/sglang-runtime:0.6.1"
    config:
      sla:
        isl: 4000
        osl: 500
        ttft: 300
        itl: 10
      sweep:
        useAiConfigurator: false

  deploymentOverrides:
    workersImage: "nvcr.io/nvidia/ai-dynamo/sglang-runtime:0.6.1"

  autoApply: true
```

Deploy:
```bash
kubectl apply -f benchmarks/profiler/deploy/profile_sla_moe_dgdr.yaml -n $NAMESPACE
```

### Using Existing DGD Configs (Custom Setups)

Reference an existing DynamoGraphDeployment config via ConfigMap:

**Step 1: Create ConfigMap from your DGD config:**

```bash
kubectl create configmap deepseek-r1-config \
  --from-file=disagg.yaml=/path/to/your/disagg.yaml \
  --namespace $NAMESPACE \
  --dry-run=client -o yaml | kubectl apply -f -
```

**Step 2: Reference it in your DGDR:**

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeploymentRequest
metadata:
  name: deepseek-r1
spec:
  model: deepseek-ai/DeepSeek-R1
  backend: sglang

  profilingConfig:
    profilerImage: "nvcr.io/nvidia/ai-dynamo/sglang-runtime:0.6.1"
    configMapRef:
      name: deepseek-r1-config
      key: disagg.yaml  # Must match the key used in --from-file
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
    workersImage: "nvcr.io/nvidia/ai-dynamo/sglang-runtime:0.6.1"

  autoApply: true
```

The profiler uses the DGD config from the ConfigMap as a **base template**, then optimizes it based on your SLA targets. The controller automatically injects `spec.model` and `spec.backend` into the final configuration.

### Inline Configuration (Simple Use Cases)

For simple use cases without a custom DGD config, provide profiler configuration directly. The profiler auto-generates a basic DGD configuration:

```yaml
profilingConfig:
  config:
    sla:
      isl: 8000
      osl: 200
      ttft: 200.0
      itl: 10.0

    hardware:
      minNumGpusPerEngine: 2
      maxNumGpusPerEngine: 8
      gpuType: h200_sxm

    sweep:
      prefillInterpolationGranularity: 16
      decodeInterpolationGranularity: 6
```

### Mocker Deployment (Testing)

Deploy a mocker backend that simulates GPU timing behavior without real GPUs. Useful for:
- Large-scale experiments without GPU resources
- Testing planner behavior and infrastructure
- Validating deployment configurations

```yaml
spec:
  model: <model-name>
  backend: trtllm  # Real backend for profiling
  useMocker: true  # Deploy mocker instead of real backend

  profilingConfig:
    profilerImage: "nvcr.io/nvidia/dynamo/trtllm-runtime:<image-tag>"
    config:
      sla:
        isl: 3000
        osl: 150
        ttft: 200
        itl: 20
      sweep:
        useAiConfigurator: true
        aicSystem: h100_sxm
  autoApply: true
```

Profiling runs against the real backend (via GPUs or AIC). The mocker deployment then uses profiling data to simulate realistic timing.

### Model Cache PVC (0.8.1+)

For large models, use a pre-populated PVC instead of downloading from HuggingFace:

See [SLA-Driven Profiling](../profiler/profiler-guide.md) for configuration details.

## Advanced Examples

### Custom Load Predictors

#### Warm-starting with Trace Data

Pre-load predictors with historical request patterns before live traffic:

```yaml
# In planner arguments
args:
  - --load-predictor arima
  - --load-predictor-warmup-trace /data/trace.jsonl
  - --load-predictor-log1p
```

The trace file should be in mooncake-style JSONL format with request-count, ISL, and OSL samples.

#### Kalman Filter Tuning

For workloads with rapid changes, tune the Kalman filter:

```yaml
args:
  - --load-predictor kalman
  - --kalman-q-level 2.0      # Higher = more responsive to level changes
  - --kalman-q-trend 0.5      # Higher = trend changes faster
  - --kalman-r 5.0            # Lower = trusts new measurements more
  - --kalman-min-points 3     # Fewer points before forecasting starts
  - --load-predictor-log1p    # Often helps with request-rate series
```

#### Prophet for Seasonal Workloads

For workloads with daily/weekly patterns:

```yaml
args:
  - --load-predictor prophet
  - --prophet-window-size 100   # Larger window for seasonal detection
  - --load-predictor-log1p
```

### Virtual Connector

For non-Kubernetes environments, use the VirtualConnector to communicate scaling decisions:

```python
from dynamo._core import DistributedRuntime, VirtualConnectorClient

# Initialize client
client = VirtualConnectorClient(distributed_runtime, namespace)

# Main loop: watch for planner decisions and execute them
while True:
    # Block until the planner makes a new scaling decision
    await client.wait()

    # Read the decision
    decision = await client.get()
    print(f"Scale to: prefill={decision.num_prefill_workers}, "
          f"decode={decision.num_decode_workers}, "
          f"id={decision.decision_id}")

    # Execute scaling in your environment
    scale_prefill_workers(decision.num_prefill_workers)
    scale_decode_workers(decision.num_decode_workers)

    # Report completion
    await client.complete(decision)
```

See `components/planner/test/test_virtual_connector.py` for a full working example.

### Planner Configuration Passthrough

Pass planner-specific settings through the DGDR:

```yaml
profilingConfig:
  config:
    planner:
      plannerMinEndpoint: 2
```

### Review Before Deploy (autoApply: false)

Disable auto-deployment to inspect the generated DGD:

```yaml
spec:
  autoApply: false
```

After profiling completes:

```bash
# Extract and review generated DGD
kubectl get dgdr sla-aic -n $NAMESPACE \
  -o jsonpath='{.status.generatedDeployment}' > my-dgd.yaml

# Review and modify as needed
vi my-dgd.yaml

# Deploy manually
kubectl apply -f my-dgd.yaml -n $NAMESPACE
```

### Profiling Artifacts with PVC

Save detailed profiling artifacts (plots, logs, raw data) to a PVC:

```yaml
spec:
  profilingConfig:
    outputPVC: "dynamo-pvc"
    config:
      sla:
        isl: 3000
        osl: 150
        ttft: 200
        itl: 20
```

Setup:
```bash
export NAMESPACE=your-namespace
deploy/utils/setup_benchmarking_resources.sh
```

Access results:
```bash
kubectl apply -f deploy/utils/manifests/pvc-access-pod.yaml -n $NAMESPACE
kubectl wait --for=condition=Ready pod/pvc-access-pod -n $NAMESPACE --timeout=60s
kubectl cp $NAMESPACE/pvc-access-pod:/data ./profiling-results
kubectl delete pod pvc-access-pod -n $NAMESPACE
```

## Related Documentation

- [Planner README](README.md) -- Overview and quick start
- [Planner Guide](planner-guide.md) -- Deployment, configuration, integration
- [Planner Design](../../design-docs/planner-design.md) -- Architecture deep-dive
- [DGDR Configuration Reference](../profiler/profiler-guide.md#dgdr-configuration-structure)
- [SLA-Driven Profiling](../profiler/profiler-guide.md)
