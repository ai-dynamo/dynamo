---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: DGDR Examples
subtitle: Practical DynamoGraphDeploymentRequest examples covering AIC estimates, online profiling, and SLA-driven generation.
---

Practical examples for deploying with `DynamoGraphDeploymentRequest` (DGDR).
The DGDR workflow can use native AIC estimates, optional bootstrap profiling
data, or live FPM warmup depending on the model/backend combination. For DGDR
concepts, see the [DGDR Reference](dgdr.md). For profiling concepts, see the
[Profiler Guide](../components/profiler/profiler-guide.md).

## DGDR Examples

### Minimal DGDR with AIC (Fastest)

The simplest way to generate a deployment from native AIC estimates. Uses AI
Configurator for offline profiling (20-30 seconds instead of hours):

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeploymentRequest
metadata:
  name: sla-aic
spec:
  model: Qwen/Qwen3-32B
  backend: vllm
  image: "nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.2.1"  # dynamo-frontend for Dynamo < 1.1.0
```

Deploy:
```bash
export NAMESPACE=your-namespace
# Save the manifest above as sla-aic.yaml first.
kubectl apply -f sla-aic.yaml -n $NAMESPACE
```

### Online Profiling (Real Measurements)

Standard online profiling runs real GPU measurements for more accurate results. Takes 2-4 hours:

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeploymentRequest
metadata:
  name: sla-online
spec:
  model: meta-llama/Llama-3.3-70B-Instruct
  backend: vllm
  image: "nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.2.1"  # dynamo-frontend for Dynamo < 1.1.0
  searchStrategy: thorough
```

Deploy:
```bash
# Save the manifest above as sla-online.yaml first.
kubectl apply -f sla-online.yaml -n $NAMESPACE
```

> **Note**: Starting with Dynamo 1.0.0 (DGDR API version v1beta1), DGDR fields use structured spec fields (e.g., `spec.workload`, `spec.sla`, `spec.hardware`) instead of the nested `profilingConfig.config` blob used in v1alpha1.

### Planner-Enabled DGDR

Set `spec.features.planner` to enable Planner generation in the final DGD. DGDR
passes this object as PlannerConfig to the Planner service; see the
[Planner Guide](../components/planner/planner-guide.md#plannerconfig-reference)
for available fields.

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeploymentRequest
metadata:
  name: qwen3-planner
spec:
  model: Qwen/Qwen3-0.6B
  backend: vllm
  image: "nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.2.1"  # dynamo-frontend for Dynamo < 1.1.0
  features:
    planner:
      mode: disagg
      backend: vllm
```

`spec.overrides.dgd` is not required to enable Planner; use it only when the
generated DGD needs additional customization.

### KV-Aware Router Override

DGDR-generated deployments default to `round-robin` routing. Use
`spec.overrides.dgd` to configure the generated `Frontend` service with a
different router mode without editing the DGD directly.

**Approximate KV routing** (no worker-side event publishing required):

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeploymentRequest
metadata:
  name: qwen3-kv-router
spec:
  model: Qwen/Qwen3-0.6B
  backend: vllm
  image: "nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.2.1"
  overrides:
    dgd:
      apiVersion: nvidia.com/v1alpha1  # v1beta1 not yet supported for overrides
      kind: DynamoGraphDeployment
      spec:
        services:
          Frontend:
            envs:
              - name: DYN_ROUTER_MODE
                value: kv
              - name: DYN_ROUTER_USE_KV_EVENTS
                value: "false"
```

**Event-driven KV routing for vLLM disaggregated serving** (enable worker
KV-event publishing on the prefill worker):

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeploymentRequest
metadata:
  name: qwen3-disagg-kv-router
spec:
  model: Qwen/Qwen3-0.6B
  backend: vllm
  image: "nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.2.1"
  overrides:
    dgd:
      apiVersion: nvidia.com/v1alpha1  # v1beta1 not yet supported for overrides
      kind: DynamoGraphDeployment
      spec:
        services:
          Frontend:
            envs:
              - name: DYN_ROUTER_MODE
                value: kv
          VllmPrefillWorker:
            extraPodSpec:
              mainContainer:
                args:
                  - --enable-prefix-caching
                  - --kv-events-config
                  - '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}'
```

**Note**: Service names depend on the backend and topology selected by the
profiler. Use `autoApply: false` and inspect the generated DGD before adding
per-service overrides. For aggregated serving, override the single worker
service (not a prefill worker). For cross-backend KV event flags, see
[Router Operations](../components/router/router-operations.md#additional-notes)
and the [Routing section in the DGDR Reference](dgdr.md#routing).

## Additional DGDR Patterns

### MoE Models (SGLang)

For Mixture-of-Experts models like DeepSeek-R1, use SGLang backend:

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeploymentRequest
metadata:
  name: sla-moe
spec:
  model: deepseek-ai/DeepSeek-R1
  backend: sglang
  image: "nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.2.1"  # dynamo-frontend for Dynamo < 1.1.0
```

Deploy:
```bash
# Save the manifest above as sla-moe.yaml first.
kubectl apply -f sla-moe.yaml -n $NAMESPACE
```

### Customizing the Generated DGD

Use `spec.overrides.dgd` to provide a partial `DynamoGraphDeployment` that is
merged into the profiler-generated deployment. Use a `v1beta1` override for
new DGDRs:

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeploymentRequest
metadata:
  name: deepseek-r1
spec:
  model: deepseek-ai/DeepSeek-R1
  backend: sglang
  image: "nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.2.1"  # dynamo-frontend for Dynamo < 1.1.0
  overrides:
    dgd:
      apiVersion: nvidia.com/v1beta1
      kind: DynamoGraphDeployment
      spec:
        env:
          - name: CUSTOM_WORKER_ENV
            value: "enabled"
```

DGDR merges the override into the generated DGD after profiling selects a
configuration. Both `.status.profilingResults.selectedConfig` and the DGD
created when `autoApply: true` use `nvidia.com/v1beta1`.

Existing `v1alpha1` overrides remain supported. Their field shape follows the
`v1alpha1` DGD schema:

```yaml
spec:
  overrides:
    dgd:
      apiVersion: nvidia.com/v1alpha1
      kind: DynamoGraphDeployment
      spec:
        envs:
          - name: CUSTOM_WORKER_ENV
            value: "enabled"
```

The override's API version controls its merge semantics. In particular, the
`v1beta1` graph-level `spec.env` list and container `args` replace their
generated lists, while nested container environment variables merge by name.
`v1alpha1` worker arguments append for compatibility. See
[Generated DGD Overrides](dgdr.md#generated-dgd-overrides) for the complete
behavior and direct-profiler requirements.

### Inline Configuration (Simple Use Cases)

For simple use cases without a custom DGD config, provide the configuration directly in the v1beta1 DGDR spec fields. The profiler auto-generates a basic DGD configuration:

```yaml
spec:
  workload:
    isl: 8000
    osl: 200

  sla:
    ttft: 200.0
    itl: 10.0

  hardware:
    gpuSku: h200_sxm

  searchStrategy: rapid
```

### Simulation with Mocker

Deploy a mocker backend that simulates GPU timing behavior without real GPUs. Useful for:
- Large-scale experiments without GPU resources
- Testing profiling behavior and infrastructure
- Validating deployment configurations

```yaml
spec:
  model: <model-name>
  backend: trtllm  # Real backend for profiling
  features:
    mocker:
      enabled: true  # Deploy mocker instead of real backend

  image: "nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.2.1"  # dynamo-frontend for Dynamo < 1.1.0
```

Profiling runs against the real backend (via GPUs or AIC). The mocker deployment then uses profiling data to simulate realistic timing.

### Model Cache PVC (0.8.1+)

For large models, use a pre-populated PVC instead of downloading from HuggingFace:

See [SLA-Driven Profiling](../components/profiler/profiler-guide.md) for configuration details.

## Advanced DGDR Patterns

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
  -o jsonpath='{.status.profilingResults.selectedConfig}' > my-dgd.yaml

# Review and modify as needed
vi my-dgd.yaml

# Deploy manually
kubectl apply -f my-dgd.yaml -n $NAMESPACE
```

### Profiling Artifacts with PVC

Save detailed profiling artifacts (plots, logs, raw data) to a PVC:

```yaml
spec:
  workload:
    isl: 3000
    osl: 150

  sla:
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

- [DGDR Reference](dgdr.md) -- DGDR field reference and lifecycle
- [Profiler Guide](../components/profiler/profiler-guide.md) -- Profiling workflow
