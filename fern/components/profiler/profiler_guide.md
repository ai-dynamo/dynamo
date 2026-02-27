<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Profiler Guide

This guide covers deployment, configuration, integration, and troubleshooting for the Dynamo Profiler.

## What is a DynamoGraphDeploymentRequest (DGDR)?

A **DynamoGraphDeploymentRequest (DGDR)** is a Kubernetes Custom Resource that serves as the primary interface for users to request model deployments with specific performance and resource constraints. You specify:

- **What** model you want to deploy (`model`)
- **How** it should perform (SLA targets: `sla.ttft`, `sla.itl`)
- **Where** it should run (optional GPU preferences via `hardware`)
- **Which** backend to use (`backend`: auto, vllm, sglang, or trtllm)
- **Which** image to use (`image`)

The Dynamo Operator watches for DGDRs and automatically:
1. Discovers available GPU resources in your cluster
2. Runs profiling (online or offline) to find optimal configurations
3. Generates an optimized DynamoGraphDeployment (DGD) configuration
4. Deploys the DGD to your cluster

**Relationship to DGD:**
- **DGDR**: High-level "intent" - what you want deployed
- **DGD**: Low-level "implementation" - how it's deployed

## Support Matrix

| Backend | Dense Models | MoE Models |
|---------|-------------|------------|
| vLLM | ✅ | 🚧 |
| SGLang | ✅ | ✅ |
| TensorRT-LLM | ✅ | 🚧 |

The profiler sweeps over the following parallelization mappings for prefill and decode:

| Model Architecture | Prefill Parallelization Mapping | Decode Parallelization Mapping |
|---------|-------------|------------|
| MLA+MoE (DeepseekV3ForCausalLM, DeepseekV32ForCausalLM) | TEP, DEP | TEP, DEP |
| GQA+MoE (Qwen3MoeForCausalLM) | TP, TEP, DEP | TP, TEP, DEP |
| Other Models | TP | TP |

> [!NOTE]
> Exact model x parallelization mapping support is dependent on the backend. The profiler does not guarantee that the recommended P/D engine configuration is supported and bug-free by the backend.

## Deployment

### Kubernetes Deployment (DGDR)

The recommended deployment method is through DGDRs. Sample configurations are provided in `benchmarks/profiler/deploy/`:

| Sample | Description |
|--------|-------------|
| `profile_sla_dgdr.yaml` | Standard online profiling with AIPerf |
| `profile_sla_aic_dgdr.yaml` | Fast offline profiling with AI Configurator |
| `profile_sla_moe_dgdr.yaml` | MoE model profiling (SGLang) |

#### Container Images

Each DGDR requires a container image for profiling and deployment:

- **`image`** (Optional): Container image for the profiling job. Must contain the profiler code and dependencies.

```yaml
spec:
  image: "nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.9.0"
```

#### Quick Start: Deploy with DGDR

**Step 1: Create Your DGDR**

Use a sample configuration or create your own:

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeploymentRequest
metadata:
  name: my-model-profiling
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

**Step 2: Apply the DGDR**

```bash
export NAMESPACE=your-namespace
kubectl apply -f my-profiling-dgdr.yaml -n $NAMESPACE
```

**Step 3: Monitor Progress**

```bash
# View status
kubectl get dgdr -n $NAMESPACE

# Detailed status
kubectl describe dgdr my-model-profiling -n $NAMESPACE

# Watch profiling job logs
kubectl logs -f job/profile-my-model-profiling -n $NAMESPACE
```

**DGDR Status Phases:**
- `Pending`: Initial state, preparing to profile
- `Profiling`: Running profiling job (20-30 seconds for AIC, 2-4 hours for online)
- `Ready`: Profiling complete, generated DGD spec available in status
- `Deploying`: Generating and applying DGD configuration
- `Deployed`: DGD successfully deployed and running
- `Failed`: Error occurred (check events for details)

**Step 4: Access Your Deployment**

```bash
# Find the frontend service
kubectl get svc -n $NAMESPACE | grep frontend

# Port-forward to access locally
kubectl port-forward svc/<deployment>-frontend 8000:8000 -n $NAMESPACE

# Test the endpoint
curl http://localhost:8000/v1/models
```

> [!NOTE]
> DGDRs are **immutable**. To update SLAs or configuration, delete the existing DGDR and create a new one.

## Profiling Method

The profiler follows a 5-step process:

1. **Hardware Setup**: Uses defaults or user-specified hardware configuration. Optionally, cluster-scoped operators can enable automatic GPU discovery to detect specifications from cluster nodes.
2. **Identify Sweep Ranges**: Automatically determine minimum and maximum number of GPUs per engine. Minimum is determined by the model size and GPU VRAM. Maximum is set to one node for dense models and 4 nodes for MoE models.
3. **Parallelization Mapping Sweep**: Test performance of engines with different parallelization mappings using the input ISL and OSL.
   - For dense models, test different TP sizes for both prefill and decode.
   - For MoE models (SGLang), evaluate both TEP and DEP as candidates for prefill and decode.
   - **Prefill**:
     - TP/TEP: Measure TTFT with batch size = 1 (assuming ISL is long enough to saturate compute) without KV reuse.
     - DEP: Attention uses data parallelism. Send a single burst with total concurrency `attention_dp_size × attn_dp_num_req_ratio` (defaults to 4) and compute the reported TTFT as `time_to_first_token.max / attn_dp_num_req_ratio` from the AIPerf summary of that burst.
   ![Prefill Performance](../../images/h100_prefill_performance.png)
   - **Decode**: Measure the ITL under different numbers of in-flight requests, from 1 to the maximum the KV cache can hold. To measure ITL without being affected by piggy-backed prefill requests, the script enables KV-reuse and warms up the engine by issuing the same prompts before measuring.
   ![Decode Performance](../../images/h100_decode_performance.png)
4. **Recommendation**: Select optimal parallelization mapping for prefill and decode that achieves the highest per-GPU throughput while adhering to the SLA on TTFT and ITL.
5. **In-Depth Profiling on the Recommended P/D Engine**: Interpolate TTFT with ISL and ITL with active KV cache and decode context length for more accurate performance estimation.
![ITL Interpolation](../../images/pd_interpolation.png)
   - **Prefill**: Measures TTFT and throughput per GPU across different input lengths with batch size=1.
   - **Decode**: Measures ITL and throughput per GPU under various KV cache loads and decode context lengths.

### AIPerf on Real Engines

Profiles your model by creating real test deployments in Kubernetes and measuring their performance.

- **Duration**: 2-4 hours
- **Accuracy**: Highest (real measurements)
- **GPU Requirements**: Full access to test different parallelization mappings
- **Backends**: SGLang, TensorRT-LLM, vLLM

AIPerf-based profiling is the default behavior. Use `searchStrategy: thorough` for comprehensive real-engine profiling:

```yaml
spec:
  searchStrategy: thorough  # Deep exploration with real engine profiling
```

### AI Configurator Simulation

Uses performance simulation to rapidly estimate optimal configurations without running real deployments.

- **Duration**: 20-30 seconds
- **Accuracy**: Estimated (may have errors for unusual configurations)
- **GPU Requirements**: None
- **Backends**: All backends (vLLM, SGLang, TensorRT-LLM)

AI Configurator simulation is enabled by default via `searchStrategy: rapid`:

```yaml
spec:
  searchStrategy: rapid  # Fast profiling with AI Configurator simulation
```

> [!NOTE]
> `aicBackendVersion` specifies the TensorRT-LLM version that AI Configurator simulates. See the [AI Configurator supported features](https://github.com/ai-dynamo/aiconfigurator#supported-features) for available versions.

**Currently supports:**
- **Backends**: TensorRT-LLM (versions 0.20.0, 1.0.0rc3, 1.0.0rc6)
- **Systems**: H100 SXM, H200 SXM, B200 SXM, GB200 SXM, A100 SXM
- **Models**: Wide range including GPT, Llama, Mixtral, DeepSeek, Qwen, and more

See [AI Configurator documentation](https://github.com/ai-dynamo/aiconfigurator#supported-features) for the full list.

### Automatic GPU Discovery

The operator automatically discovers GPU resources from your Kubernetes cluster nodes when available. GPU discovery provides:

- Hardware information (GPU model, VRAM, GPUs per node)
- Automatic calculation of profiling search space based on model size
- Hardware system identifier for AI Configurator integration

**Permissions**: GPU discovery requires cluster-wide node read permissions. Cluster-scoped operators automatically have these permissions. Namespace-restricted operators can also use GPU discovery if granted node read permissions via RBAC.

If GPU discovery is unavailable (no permissions or no GPU labels), the profiler will use manually specified hardware configuration or defaults.

## Configuration

### DGDR Configuration Structure

All profiler configuration is provided through the v1beta1 DGDR spec fields:

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeploymentRequest
metadata:
  name: my-deployment
spec:
  model: "Qwen/Qwen3-0.6B"
  backend: vllm
  image: "nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.9.0"

  workload: { ... }
  sla: { ... }
  hardware: { ... }
  features: { ... }
  overrides: { ... }
```

### SLA Configuration (Optional)

```yaml
workload:
  isl: 3000      # Average input sequence length (tokens)
  osl: 150       # Average output sequence length (tokens)

sla:
  ttft: 200.0    # Target Time To First Token (milliseconds)
  itl: 20.0      # Target Inter-Token Latency (milliseconds)
```

- **ISL/OSL**: Based on your expected traffic patterns
- **TTFT**: First token latency target (lower = more GPUs needed, affects prefill engine)
- **ITL**: Token generation latency target (lower = more GPUs needed, affects decode engine)
- **Trade-offs**: Tighter SLAs require more GPU resources

### Hardware Configuration (Optional)

```yaml
hardware:
  gpuSku: h200_sxm            # GPU SKU identifier (auto-detected)
  vramMb: 81920               # VRAM per GPU in MiB
  totalGpus: 16               # Total GPUs available in the cluster
  numGpusPerNode: 8           # GPUs per node (for multi-node MoE)
```

- **numGpusPerNode**: Determine the upper bound of GPUs per node for dense models and configure Grove for multi-node MoE engines
- **gpuSku**: GPU SKU identifier, auto-detected by the controller

> [!TIP]
> If you don't specify hardware constraints, the controller auto-detects based on your model size and available cluster resources.

### Search Strategy (Optional)

Controls the profiling search depth:

```yaml
spec:
  searchStrategy: rapid   # "rapid" (default) for fast sweep; "thorough" for deeper exploration
```

- **rapid**: Performs a fast sweep over parallelization mappings (default)
- **thorough**: Explores more configurations for potentially better results

### Planner Configuration (Optional)

Pass arguments to the SLA planner via the features section:

```yaml
features:
  planner:
    planner_min_endpoint: 2                    # Minimum endpoints to maintain
    planner_adjustment_interval: 60            # Adjustment interval (seconds)
    planner_load_predictor: linear             # Load prediction method
```

> [!NOTE]
> Planner arguments use `planner_` prefix. See the AI Configurator documentation for full list.

### Model Cache PVC (Advanced)

For large models, use a pre-populated PVC containing model weights instead of downloading from HuggingFace:

```yaml
modelCache:
  pvcName: "model-cache"
  pvcModelPath: "hub/models--deepseek-ai--DeepSeek-R1"
  pvcMountPath: "/opt/model-cache"
```

Requirements:
- The PVC must exist in the same namespace as the DGDR
- The model weights must be accessible at `{mountPath}/{pvcPath}`

### Engine Configuration (Auto-configured)

The controller automatically handles model and backend configuration from high-level fields:

```yaml
# You specify:
spec:
  model: "Qwen/Qwen3-0.6B"
  backend: vllm

# Controller auto-injects into the profiling job
```

You should **not** manually set model or backend in profiling config overrides.

### Using Existing DGD Configs

Provide a base DGD config via the overrides section:

```yaml
overrides:
  dgd:
    apiVersion: nvidia.com/v1alpha1
    kind: DynamoGraphDeployment
    metadata:
      name: my-dgd
    spec:
      # ... your base DGD spec
```

The profiler uses the DGD config as a **base template**, then optimizes it based on your SLA targets.

## Integration

### With SLA Planner

The Profiler generates interpolation data that the SLA Planner uses for autoscaling decisions.

**Prefill Interpolation** (`selected_prefill_interpolation/raw_data.npz`):
- `prefill_isl`: 1D array of input sequence lengths tested
- `prefill_ttft`: 1D array of TTFTs (ms) at each ISL
- `prefill_thpt_per_gpu`: 1D array of throughput (tokens/s/GPU) at each ISL

**Decode Interpolation** (`selected_decode_interpolation/raw_data.npz`):
- `max_kv_tokens`: Total KV tokens capacity in decode engine
- `x_kv_usage`: 1D array of active KV usage percentages [0, 1]
- `y_context_length`: 1D array of average context lengths tested
- `z_itl`: 1D array of ITLs (ms) at each (KV usage, context length) point
- `z_thpt_per_gpu`: 1D array of throughput (tokens/s/GPU) at each point

### With Dynamo Operator

When using DGDR, the Dynamo Operator:

1. Creates profiling jobs automatically
2. Stores profiling data in ConfigMaps (`planner-profile-data`)
3. Generates optimized DGD configurations
4. Deploys the DGD with SLA Planner integration

The generated DGD is tracked via labels:
```yaml
metadata:
  labels:
    dgdr.nvidia.com/name: my-deployment
    dgdr.nvidia.com/namespace: your-namespace
```

### With Observability

Monitor profiling jobs:

```bash
kubectl logs -f job/profile-<dgdr-name> -n $NAMESPACE
kubectl describe dgdr <name> -n $NAMESPACE
```

## Advanced Topics

### Manual Deployment Control

Disable auto-deployment to review the generated DGD before applying:

```yaml
spec:
  autoApply: false
```

Then manually extract and apply:

```bash
# Extract generated DGD from DGDR status
kubectl get dgdr my-deployment -n $NAMESPACE -o jsonpath='{.status.profilingResults.selectedConfig}' | kubectl apply -f -

# Or save to file for review
kubectl get dgdr my-deployment -n $NAMESPACE -o jsonpath='{.status.profilingResults.selectedConfig}' > my-dgd.yaml
```

### Mocker Deployment

Deploy a mocker deployment that simulates engines without GPUs:

```yaml
spec:
  model: <model-name>
  backend: trtllm
  features:
    mocker:
      enabled: true    # Deploy mocker instead of real backend
  autoApply: true
```

Profiling still runs against the real backend to collect performance data. The mocker uses this data to simulate realistic timing behavior. Useful for large-scale experiments, testing Planner behavior, and validating configurations.

### Accessing Profiling Artifacts

By default, profiling data is stored in ConfigMaps. For detailed artifacts (plots, logs, raw data), attach a PVC via overrides:

```yaml
overrides:
  profilingJob:
    template:
      spec:
        volumes:
        - name: profiling-output
          persistentVolumeClaim:
            claimName: "dynamo-pvc"
```

**ConfigMaps (always created):**
- `dgdr-output-<name>`: Generated DGD configuration
- `planner-profile-data`: Profiling data for Planner (JSON)

**PVC artifacts (optional):**
- Performance plots (PNGs)
- DGD configurations for each profiled deployment
- AIPerf profiling artifacts
- Raw profiling data (`.npz` files)
- Profiler logs

Access PVC results:
```bash
kubectl apply -f deploy/utils/manifests/pvc-access-pod.yaml -n $NAMESPACE
kubectl wait --for=condition=Ready pod/pvc-access-pod -n $NAMESPACE --timeout=60s
kubectl cp $NAMESPACE/pvc-access-pod:/data ./profiling-results
kubectl delete pod pvc-access-pod -n $NAMESPACE
```

### Output Performance Plots

The profiler generates plots to visualize performance data:

**Parallelization Mapping Sweep Plots:**
- `prefill_performance.png`: TTFT vs Parallelization Mapping size
- `decode_performance.png`: ITL vs Parallelization Mapping size and in-flight requests

**In-Depth Profiling Plots:**
- `selected_prefill_interpolation/prefill_ttft_interpolation.png`: TTFT vs ISL
- `selected_prefill_interpolation/prefill_throughput_interpolation.png`: Throughput vs ISL
- `selected_decode_interpolation/decode_itl_interplation.png`: ITL vs KV usage and context length
- `selected_decode_interpolation/decode_throughput_interpolation.png`: Throughput vs KV usage and context length

## Runtime Profiling (SGLang)

SGLang workers expose profiling endpoints for runtime performance analysis:

```bash
# Start profiling
curl -X POST http://localhost:9090/engine/start_profile \
  -H "Content-Type: application/json" \
  -d '{"output_dir": "/tmp/profiler_output"}'

# Run inference requests...

# Stop profiling
curl -X POST http://localhost:9090/engine/stop_profile
```

View traces using Chrome's `chrome://tracing`, [Perfetto UI](https://ui.perfetto.dev/), or TensorBoard.

## Troubleshooting

### Profiling Takes Too Long

**Solution 1**: Use `searchStrategy: rapid` for fast AI Configurator profiling (TensorRT-LLM only):
```yaml
spec:
  searchStrategy: rapid
```

**Solution 2**: Reduce search space by specifying hardware constraints in the DGDR:
```yaml
spec:
  hardware:
    numGpusPerNode: 4
    totalGpus: 8
```

### SLA Cannot Be Met

**Symptoms**: Profiler reports no configuration meets targets

**Solutions:**
1. Relax SLA targets (increase TTFT/ITL)
2. Add more GPU resources
3. Try a different backend
4. Use a smaller model

### AI Configurator: Attention Head Constraint Error

**Symptoms**: Profiling fails with error:
```text
AssertionError: num_heads <N> should be divisible by tp_size <M> and the division result should be >= 4
```

**Cause**: AI Configurator requires **≥4 attention heads per GPU**. Small models with few heads cannot use high TP sizes.

**Affected Models:**
- **Qwen3-0.6B** (16 heads): Max TP = 4
- **GPT-2** (12 heads): Max TP = 3
- Most models **<1B parameters**: May hit this constraint

**Solution**: Limit `maxNumGpusPerEngine`:
```yaml
hardware:
  maxNumGpusPerEngine: 4  # For Qwen3-0.6B (16 heads / 4 = max TP of 4)
```

**Calculate Max TP**: `max_tp = num_attention_heads / 4`

> [!NOTE]
> This is an AI Configurator limitation. Online profiling doesn't have this constraint.

### Image Pull Errors

**Symptoms**: `ErrImagePull` or `ImagePullBackOff`

**Solution**: Ensure image pull secrets are configured:
```bash
kubectl create secret docker-registry nvcr-imagepullsecret \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password=<NGC_API_KEY> \
  --namespace <your-namespace>
```

### Out of Memory During Profiling

**Symptoms**: OOM errors in profiling jobs

**Solutions:**
1. Reduce `gpu_memory_utilization` in engine config
2. Reduce `--max-context-length`
3. Skip larger TP configurations
4. Use fewer GPUs per test

### Unsupported Parallelization Mapping in Backend

**Symptoms**: Startup/runtime error in the backend (e.g., prime number of attention heads constraining TP to 1, or backend not supporting different TP sizes for prefill and decode).

**Solutions:**
1. Contact the backend to add support and bump backend version in Dynamo
2. Constrain the max and min number of GPUs per engine to the supported range

## See Also

- [DGDR Examples](../../../components/src/dynamo/profiler/deploy/) - Complete DGDR YAML examples
- [DGDR API Reference](/docs/kubernetes/api-reference.md) - DGDR specification
- [Profiler Arguments Reference](https://github.com/ai-dynamo/dynamo/blob/main/components/src/dynamo/profiler/utils/dgdr_v1beta1_types.py) - Full Configuration Reference
