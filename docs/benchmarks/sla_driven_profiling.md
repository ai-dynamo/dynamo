# SLA-Driven Profiling with DynamoGraphDeploymentRequest

> [!TIP]
> **New to DGDR and SLA-Driven Profiling?** Start with the [SLA-Driven Profiling and Planner Deployment Quick Start Guide](/docs/planner/sla_planner_quickstart.md) for step-by-step instructions. This document provides deeper technical details about the profiling process.

## Overview

Dynamo provides automated SLA-driven profiling through **DynamoGraphDeploymentRequests (DGDR)**. Instead of manually running profiling scripts, you declare your performance requirements and let the Dynamo Operator handle profiling and deployment automatically.

**Key Benefits:**
- **Declarative**: Specify SLAs, not implementation details
- **Automated**: No manual job setup or result processing
- **Integrated**: Seamlessly works with Dynamo Operator
- **Production-Ready**: Generates optimized configurations with SLA planner

This document covers:
- Technical details of online vs offline profiling
- Profiling process internals (GPU usage, measurements, interpolation)
- Direct script usage for advanced scenarios
- Comprehensive troubleshooting

## Using DGDR for Profiling (Recommended)

The recommended way to profile models is through DGDRs. Sample configurations are provided in `deploy/`:

**Available Samples:**
- **`profile_sla_dgdr.yaml`**: Standard online profiling
- **`profile_sla_aic_dgdr.yaml`**: Fast offline profiling with AI Configurator
- **`profile_sla_moe_dgdr.yaml`**: MoE model profiling

The Dynamo Operator automatically:
1. Discovers GPU resources
2. Runs profiling (online or offline)
3. Generates optimal DGD configuration
4. Deploys to your cluster

See the [Quick Start Guide](/docs/planner/sla_planner_quickstart.md) for detailed instructions.

## Profiling Methods

### Online Profiling (Default)

Profiles your model by creating real test deployments in Kubernetes and measuring their performance.

**Characteristics:**
- **Duration**: 2-4 hours
- **Accuracy**: Highest (real measurements)
- **GPU Requirements**: Full access to test different TP configurations
- **Backends**: vLLM, SGLang, TensorRT-LLM

**DGDR Configuration:**
```yaml
profilingConfig:
  config:
    sweep:
      use_ai_configurator: false  # Default
```

**What It Does:**
1. **GPU Discovery**: Detects available GPUs and their specifications
2. **Identify Sweep Ranges**: Automatically determine minimum and maximum number of GPUs per engine. Minimum is determined by the model size and GPU VRAM. Maximum is set to one node for dense model and 4 nodes for MoE models.
3. **Parallelization Mapping Sweep**: Use the input ISL and OSL, test the performance of the engines with different parallelization mappings.  
   - Dense models: Test different TP sizes for both prefill and decode. 
   - MoE models: Test different TEP sizes for prefill and DEP sizes for decode.
4. **Recommendation**: Selects optimal parallelization mapping for prefill and decode that achieves the highest per GPU throughput while adhering the SLA on TTFT and ITL.
5. **In-Depth Profiling on Recommended P/D Engine**:  Generates performance models for SLA planner. 
   - **Prefill**: Measures TTFT across different input lengths   
   - **Decode**: Measures ITL under various KV cache loads and decode context lengths.

### Offline Profiling (AI Configurator)

Uses performance simulation to rapidly estimate optimal configurations without running real deployments.

**Characteristics:**
- **Duration**: 20-30 seconds
- **Accuracy**: Estimated (may have errors for unusual configurations)
- **GPU Requirements**: None
- **Backends**: TensorRT-LLM only (vLLM/SGLang coming soon)

**DGDR Configuration:**
```yaml
profilingConfig:
  config:
    sweep:
      use_ai_configurator: true
    aic:
      system: h200_sxm          # GPU system type
      model_name: QWEN3_32B     # AIC model identifier
      backend_version: "0.20.0"
```

**Supported Configurations:**

For the current list of supported models, systems, and backend versions, see the [AI Configurator documentation](https://github.com/ai-dynamo/aiconfigurator#supported-features).

To check from the command line: `aiconfigurator cli --help`

**Currently supports:**
- **Backends**: TensorRT-LLM (versions 0.20.0, 1.0.0rc3, 1.0.0rc6)
- **Systems**: H100 SXM, H200 SXM, B200 SXM, GB200 SXM, A100 SXM
- **Models**: Wide range including GPT, Llama, Mixtral, DeepSeek, Qwen, and more

## Support Matrix

| Backend | Dense Models | MoE Models |
|---------|-------------|------------|
| vLLM | ✅ | 🚧 |
| SGLang | ✅ | ✅ |
| TensorRT-LLM | ✅ | 🚧 |

## Profiling Process Details

### GPU Resource Usage

Profiling tests different parallelization configurations **sequentially**, not in parallel:

- **One parallelization mapping at a time**: Finish all tests for one engine before starting the next engine.
- **Full GPU access**: Each configuration gets exclusive access to all GPUs
- **Resource isolation**: No interference between tests
- **Accurate measurements**: Consistent conditions for fair comparison

### Prefill Profiling

For prefill engines, the profiler:
1. Tests each TP configuration with batch size = 1
2. Measures TTFT across different input sequence lengths (ISL)
3. Records throughput per GPU at each ISL
4. Selects the TP that meets TTFT SLA with best throughput/GPU

**For Dense Models**: Profiles different TP sizes
**For MoE Models**: Profiles different TEP sizes (DEP generally not optimal for prefill)

### Decode Profiling

For decode engines, the profiler:
1. Tests each TP configuration under varying KV cache loads
2. Measures ITL at different active KV usage levels and context lengths
3. Records throughput per GPU at each load condition
4. Selects the TP that meets ITL SLA with best throughput/GPU

**For Dense Models**: Profiles different TP sizes
**For MoE Models**: Profiles different DEP sizes (TEP decode for low latency coming soon)

### Output Format

After profiling, the DGDR status contains:

1. **Recommended Configuration**: Optimal TP for prefill and decode
2. **Planner Bounds**: Queue size and KV cache utilization thresholds
3. **Performance Data**: Interpolation models for SLA planner
4. **Generated DGD**: Complete deployment manifest

**Example Recommendations:**
```
Suggested prefill TP:4 (TTFT 48.37 ms, throughput 15505.23 tokens/s/GPU)
Suggested planner upper/lower bound for prefill queue size: 0.24/0.10
Suggested decode TP:4 (ITL 4.83 ms, throughput 51.22 tokens/s/GPU)
Suggested planner upper/lower bound for decode kv cache utilization: 0.20/0.10
```

### Performance Plots (Online Profiling)

Online profiling generates visualization plots:

**Main Performance Plots:**
- `prefill_performance.png`: TTFT vs TP size
- `decode_performance.png`: ITL vs TP size and in-flight requests

**Interpolation Plots:**
- `prefill_ttft_interpolation.png`: TTFT vs ISL with quadratic fit
- `prefill_throughput_interpolation.png`: Throughput vs ISL
- `decode_tp{N}.png`: 3D surface of ITL vs KV usage and context length

**Example:**
![Prefill Performance](/docs/images/h100_prefill_performance.png)
![Decode Performance](/docs/images/h100_decode_performance.png)

### Interpolation Data Format

The profiler generates `.npz` files with performance characteristics:

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

![ITL Interpolation](/docs/images/itl_interpolation.png)

## DGDR Configuration Reference

This section provides detailed explanations of all DGDR `profilingConfig` options. The DGDR controller passes this configuration to the profiler script, which is defined in `benchmarks/profiler/utils/profiler_argparse.py`.

### Configuration Structure

All profiler configuration goes under `spec.profilingConfig.config`:

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeploymentRequest
metadata:
  name: my-deployment
spec:
  modelName: "Qwen/Qwen3-0.6B"    # High-level: model to deploy
  backend: vllm                    # High-level: inference backend

  profilingConfig:
    configMapRef:                  # Optional: base DGD config
      name: my-config
      key: disagg.yaml

    config:                        # Profiler configuration
      sla: { ... }
      hardware: { ... }
      sweep: { ... }
      aic: { ... }
      planner: { ... }
```

### SLA Configuration (Required)

Define your performance requirements and workload characteristics:

```yaml
profilingConfig:
  config:
    sla:
      isl: 3000      # Average input sequence length (tokens)
      osl: 150       # Average output sequence length (tokens)
      ttft: 200.0    # Target Time To First Token (milliseconds)
      itl: 20.0      # Target Inter-Token Latency (milliseconds)
```

**What these control:**
- **ISL/OSL**: Based on your expected traffic patterns
- **TTFT**: First token latency target (lower = more GPUs needed, affects prefill engine)
- **ITL**: Token generation latency target (lower = more GPUs needed, affects decode engine)
- **Trade-offs**: Tighter SLAs require more GPU resources

### Hardware Configuration (Optional)

Control GPU search space and constraints:

```yaml
profilingConfig:
  config:
    hardware:
      min_num_gpus_per_engine: 2     # if not provided, will automatically determine based on model and VRAM size
      max_num_gpus_per_engine: 8     # Maximum GPUs to test
      num_gpus_per_node: 8            # GPUs per node (for multi-node MoE)
      gpu_type: h200_sxm              # GPU type hint
```

**When to use:**
- **min_num_gpus_per_engine**: Skip small TP sizes if your model is large
- **max_num_gpus_per_engine**: Limit search space or work around constraints (e.g., [AIC attention heads](#ai-configurator-attention-head-constraint-error))
- **num_gpus_per_node**: Required for MoE models with TEP/DEP sizing
- **gpu_type**: Informational, auto-detected by controller

> [!TIP]
> If you don't specify hardware constraints, the controller auto-detects based on your model size and available cluster resources.

### Sweep Configuration (Optional)

Control profiling behavior:

```yaml
profilingConfig:
  config:
    sweep:
      use_ai_configurator: false              # Use offline profiling (default: false)
      skip_existing_results: false            # Skip TP sizes with existing results
      force_rerun: false                      # Force re-profiling even if results exist
      prefill_interpolation_granularity: 16   # Samples for prefill TTFT curve
      decode_interpolation_granularity: 6     # Samples for decode ITL curve
```

**Use cases:**
- **use_ai_configurator**: Set to `true` for 20-30 second profiling (TensorRT-LLM only)
- **skip_existing_results**: Resume interrupted profiling runs
- **force_rerun**: Override cached results for validation
- **interpolation_granularity**: How many samples to benchmark (lower = faster but may be less accurate)

### AI Configurator Configuration (Required if `use_ai_configurator: true`)

Configure offline profiling mode:

```yaml
profilingConfig:
  config:
    sweep:
      use_ai_configurator: true
      aic_system: h200_sxm              # GPU system: h100_sxm, h200_sxm, b200_sxm, gb200_sxm, a100_sxm
      aic_model_name: QWEN3_32B         # AIC model identifier (see supported list)
      aic_backend_version: "0.20.0"     # TensorRT-LLM version: 0.20.0, 1.0.0rc3, 1.0.0rc6
```

**Supported configurations:** See [AI Configurator documentation](https://github.com/ai-dynamo/aiconfigurator#supported-features)

**Model name mapping examples:**
- `Qwen/Qwen3-32B` → `QWEN3_32B`
- `meta-llama/Llama-3.1-70B` → `LLAMA3.1_70B`
- `deepseek-ai/DeepSeek-V3` → `DEEPSEEK_V3`

### Planner Configuration (Optional)

Pass arguments to the SLA planner:

```yaml
profilingConfig:
  config:
    planner:
      planner_min_endpoint: 2                    # Minimum endpoints to maintain
      planner_adjustment_interval: 60            # Adjustment interval (seconds)
      planner_load_predictor: linear             # Load prediction method
```

**Common planner arguments:**
- **planner_min_endpoint**: Minimum number of prefill/decode endpoints
- **planner_max_endpoint**: Maximum number of prefill/decode endpoints
- **planner_adjustment_interval**: How often planner adjusts replicas
- **planner_load_predictor**: Load prediction algorithm

> [!NOTE]
> Planner arguments use `planner_` prefix. See planner documentation for full list.

### Engine Configuration (Auto-configured)

The controller automatically sets these from high-level fields:

```yaml
# You specify:
spec:
  modelName: "Qwen/Qwen3-0.6B"
  backend: vllm

# Controller auto-injects into config:
profilingConfig:
  config:
    deployment:
      model: "Qwen/Qwen3-0.6B"      # From spec.modelName
    engine:
      backend: vllm                  # From spec.backend
      config: /path/to/configmap     # From spec.profilingConfig.configMapRef (if provided)
```

**You should not manually set** `deployment.model` or `engine.backend` in `profilingConfig.config` - they are automatically injected from the high-level fields.

### Complete Example: Online Profiling

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeploymentRequest
metadata:
  name: vllm-dense-online
spec:
  modelName: "Qwen/Qwen3-0.6B"
  backend: vllm

  profilingConfig:
    config:
      sla:
        isl: 3000
        osl: 150
        ttft: 200.0
        itl: 20.0

      hardware:
        min_num_gpus_per_engine: 1
        max_num_gpus_per_engine: 8

      sweep:
        use_ai_configurator: false
        skip_existing_results: false

  autoApply: true
```

### Complete Example: Offline Profiling (AI Configurator)

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeploymentRequest
metadata:
  name: trtllm-aic-offline
spec:
  modelName: "Qwen/Qwen3-32B"
  backend: trtllm

  profilingConfig:
    config:
      sla:
        isl: 4000
        osl: 500
        ttft: 300.0
        itl: 10.0

      sweep:
        use_ai_configurator: true

      aic:
        system: h200_sxm
        model_name: QWEN3_32B
        backend_version: "0.20.0"

  autoApply: true
```

### Complete Example: MoE Model

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeploymentRequest
metadata:
  name: sglang-moe
spec:
  modelName: "mistralai/Mixtral-8x7B-v0.1"
  backend: sglang

  profilingConfig:
    config:
      sla:
        isl: 2048
        osl: 512
        ttft: 300.0
        itl: 25.0

      hardware:
        num_gpus_per_node: 8     # Required for MoE
        max_num_gpus_per_engine: 32

      engine:
        is_moe_model: true       # Enable MoE profiling mode

  autoApply: true
```

## Troubleshooting

### Profiling Takes Too Long

**Solution 1**: Use AI Configurator for rapid profiling (TensorRT-LLM only):
```yaml
sweep:
  use_ai_configurator: true
```

**Solution 2**: Reduce search space:
```yaml
config:
  sweep:
    min_num_gpus: 4  # Skip TP1, TP2
    max_num_gpus: 8  # Don't test beyond TP8
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
```
AssertionError: num_heads <N> should be divisible by tp_size <M> and the division result should be >= 4
```

**Cause**: AI Configurator requires **≥4 attention heads per GPU**. Small models with few heads cannot use high TP sizes.

**Affected Models:**
- **Qwen3-0.6B** (16 heads): Max TP = 4 ❌ Fails at TP=8
- **GPT-2** (12 heads): Max TP = 3
- Most models **<1B parameters**: May hit this constraint

**Solution**: Limit `max_num_gpus_per_engine` in your DGDR:

```yaml
profilingConfig:
  config:
    hardware:
      max_num_gpus_per_engine: 4  # For Qwen3-0.6B (16 heads / 4 = max TP of 4)
    sweep:
      use_ai_configurator: true
    aic:
      system: h200_sxm
      model_name: QWEN3_0_6B
```

**Calculate Max TP**: `max_tp = num_attention_heads / 4`

> **Note**: This is an AI Configurator limitation. Online profiling doesn't have this constraint.

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

## Next Steps

- **Deploy with DGDR**: See [Quick Start Guide](/docs/planner/sla_planner_quickstart.md)
- **Understand SLA Planner**: Read [SLA Planner Deep Dive](/docs/planner/sla_planner.md)
- **Monitor Deployments**: Set up [Observability](/docs/kubernetes/observability/metrics.md)
- **Optimize Performance**: See [Performance Tuning](/docs/performance/tuning.md)

## Related Documentation

- [DGDR API Reference](/docs/kubernetes/api_reference.md)
- [SLA Planner Quick Start](/docs/planner/sla_planner_quickstart.md)
- [SLA Planner Architecture](/docs/planner/sla_planner.md)
- [Profiler Arguments Reference](/benchmarks/profiler/utils/profiler_argparse.py)
