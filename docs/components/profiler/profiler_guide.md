# Profiler Guide

This guide covers deployment, configuration, and integration for the Dynamo Profiler.

## Deployment

### Kubernetes Deployment (DGDR)

The recommended deployment method is through DynamoGraphDeploymentRequests. Sample configurations are provided in `benchmarks/profiler/deploy/`:

| Sample | Description |
|--------|-------------|
| `profile_sla_dgdr.yaml` | Standard online profiling with AIPerf |
| `profile_sla_aic_dgdr.yaml` | Fast offline profiling with AI Configurator |
| `profile_sla_moe_dgdr.yaml` | MoE model profiling (SGLang) |

```bash
# Apply a sample DGDR
kubectl apply -f benchmarks/profiler/deploy/profile_sla_dgdr.yaml -n $NAMESPACE

# Monitor progress
kubectl get dgdr -n $NAMESPACE
kubectl describe dgdr <name> -n $NAMESPACE
```

### Direct Script Execution

For advanced use cases or local development:

```bash
python -m benchmarks.profiler.profile_sla \
  --backend vllm \
  --config path/to/disagg.yaml \
  --model meta-llama/Llama-3-8B \
  --ttft 200 --itl 15 \
  --isl 3000 --osl 150 \
  --min-num-gpus 1 \
  --max-num-gpus 8
```

## Configuration

### SLA Configuration

Define performance requirements and workload characteristics:

```yaml
sla:
  isl: 3000      # Average input sequence length (tokens)
  osl: 150       # Average output sequence length (tokens)
  ttft: 200.0    # Target Time To First Token (milliseconds)
  itl: 20.0      # Target Inter-Token Latency (milliseconds)
```

**Choosing SLA Values:**

- **ISL/OSL**: Based on expected traffic patterns
- **TTFT**: First token latency target (lower = more GPUs needed)
- **ITL**: Token generation latency target (lower = more GPUs needed)
- **Trade-offs**: Tighter SLAs require more GPU resources

### Hardware Configuration

Control GPU search space and constraints:

```yaml
hardware:
  minNumGpusPerEngine: 2    # Skip small TP sizes for large models
  maxNumGpusPerEngine: 8    # Maximum GPUs to test
  numGpusPerNode: 8         # GPUs per node (for multi-node MoE)
  gpuType: h200_sxm         # GPU type hint
```

### Sweep Configuration

Control profiling behavior:

```yaml
sweep:
  useAiConfigurator: false              # Use real profiling (default)
  prefillInterpolationGranularity: 16   # Samples for prefill TTFT curve
  decodeInterpolationGranularity: 6     # Samples for decode ITL curve
```

### AI Configurator Configuration

For offline profiling with TensorRT-LLM:

```yaml
sweep:
  useAiConfigurator: true
  aicSystem: h200_sxm              # GPU system type
  aicHfId: Qwen/Qwen3-32B          # HuggingFace model ID
  aicBackendVersion: "0.20.0"      # TensorRT-LLM version
```

**Supported Systems:** H100 SXM, H200 SXM, B200 SXM, GB200 SXM, A100 SXM

See [AI Configurator documentation](https://github.com/ai-dynamo/aiconfigurator#supported-features) for the full list of supported models and configurations.

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--backend` | string | - | Inference backend: vllm, sglang, trtllm |
| `--config` | string | - | Path to DGD YAML config file |
| `--model` | string | - | HuggingFace model ID |
| `--ttft` | float | - | Target TTFT in milliseconds |
| `--itl` | float | - | Target ITL in milliseconds |
| `--isl` | int | - | Average input sequence length |
| `--osl` | int | - | Average output sequence length |
| `--min-num-gpus` | int | auto | Minimum GPUs per engine |
| `--max-num-gpus` | int | 8 | Maximum GPUs per engine |
| `--use-ai-configurator` | flag | false | Use offline AI Configurator |
| `--pick-with-webui` | flag | false | Launch interactive WebUI |
| `--webui-port` | int | 8000 | Port for WebUI |

## Integration

### With SLA Planner

The Profiler generates interpolation data that the SLA Planner uses for autoscaling decisions:

**Prefill Interpolation** (`selected_prefill_interpolation/raw_data.npz`):
- `prefill_isl`: Input sequence lengths tested
- `prefill_ttft`: TTFTs at each ISL
- `prefill_thpt_per_gpu`: Throughput per GPU at each ISL

**Decode Interpolation** (`selected_decode_interpolation/raw_data.npz`):
- `max_kv_tokens`: KV token capacity
- `x_kv_usage`: Active KV usage percentages
- `y_context_length`: Context lengths tested
- `z_itl`: ITLs at each (KV usage, context length)
- `z_thpt_per_gpu`: Throughput per GPU

### With Dynamo Operator

When using DGDR, the Dynamo Operator:

1. Creates profiling jobs automatically
2. Stores profiling data in ConfigMaps (`planner-profile-data`)
3. Generates optimized DGD configurations
4. Deploys the DGD with SLA Planner integration

### With Observability

Monitor profiling jobs:

```bash
# View profiling job logs
kubectl logs -f job/profile-<dgdr-name> -n $NAMESPACE

# Check DGDR status
kubectl describe dgdr <name> -n $NAMESPACE
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

**Features:**
- Interactive charts for prefill TTFT and decode ITL
- Pareto-optimal configuration analysis
- GPU cost estimation
- DGD config preview and export

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

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Profiling takes too long | Large search space | Use AI Configurator or reduce GPU range |
| SLA cannot be met | Insufficient resources | Relax SLA targets or add GPUs |
| AI Configurator attention head error | Small model + high TP | Limit `maxNumGpusPerEngine` |
| Image pull errors | Missing secrets | Configure `nvcr-imagepullsecret` |
| OOM during profiling | Memory pressure | Reduce `gpu_memory_utilization` |

### AI Configurator Attention Head Constraint

AI Configurator requires ≥4 attention heads per GPU. For small models:

```yaml
hardware:
  maxNumGpusPerEngine: 4  # For Qwen3-0.6B (16 heads / 4 = max TP of 4)
```

**Calculate Max TP**: `max_tp = num_attention_heads / 4`

### Debug Mode

Enable verbose logging:

```bash
python -m benchmarks.profiler.profile_sla \
  --backend vllm \
  --config path/to/disagg.yaml \
  --verbose
```

## Output Artifacts

### ConfigMaps (Always Created)

- `dgdr-output-<name>`: Generated DGD configuration
- `planner-profile-data`: Profiling data for Planner (JSON)

### PVC Artifacts (Optional)

When using `outputPVC`:

- Performance plots (PNGs)
- DGD configurations for each profiled deployment
- AIPerf profiling artifacts
- Raw profiling data (`.npz` files)
- Profiler logs

```yaml
profilingConfig:
  outputPVC: "dynamo-pvc"
```

## See Also

| Document | Description |
|----------|-------------|
| [SLA-Driven Profiling](/docs/benchmarks/sla_driven_profiling.md) | Technical deep dive |
| [SLA Planner Quick Start](/docs/planner/sla_planner_quickstart.md) | End-to-end workflow |
| [SLA Planner Architecture](/docs/planner/sla_planner.md) | Planner design |
| [DGDR API Reference](/docs/kubernetes/api_reference.md) | DGDR specification |
