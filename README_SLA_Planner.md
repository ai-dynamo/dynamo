# SLA Planner

The SLA (Service Level Agreement) planner is an intelligent autoscaling system that monitors system performance and adjusts the number of prefill and decode workers to meet specified latency targets. Unlike the load-based planner that scales based on resource utilization thresholds, the SLA planner uses predictive modeling and performance interpolation to proactively scale workers to maintain Time to First Token (TTFT) and Inter-Token Latency (ITL) SLAs.

## Features

* **SLA-driven scaling**: Automatically scales prefill/decode workers to meet TTFT and ITL targets
* **Predictive load forecasting**: Uses ARIMA, Prophet, or constant predictors to forecast future load
* **Performance interpolation**: Leverages pre-profiled performance data for accurate scaling decisions
* **Correction factors**: Adapts to real-world performance deviations from profiled data
* **Multi-backend support**: Works with both local and Kubernetes environments
* **Prometheus integration**: Collects real-time metrics for decision making

## Architecture

The SLA planner consists of several key components:

1. **Load Predictors**: Forecast future request patterns (number of requests, input/output sequence lengths)
2. **Performance Interpolators**: Estimate TTFT and ITL based on profiled performance data
3. **Correction Factors**: Adjust predictions based on observed vs. expected performance
4. **Scaling Logic**: Calculate optimal number of prefill/decode replicas to meet SLA targets

## Prerequisites

### Performance Profiling

Before using the SLA planner, you must profile your model's performance to generate interpolation data:

```bash
python -m utils.profile_sla \
  --config <path-to-dynamo-config-file> \
  --output-dir <path-to-profile-results-dir> \
  --isl <target-input-sequence-length> \
  --osl <target-output-sequence-length> \
  --ttft <target-ttft-ms> \
  --itl <target-itl-ms>
```

This script will:
- Profile prefill performance across different tensor parallelism (TP) sizes
- Profile decode performance under various concurrency levels
- Generate interpolation data saved to `<output-dir>/selected_{prefill,decode}_interpolation/raw_data.npz`
- Recommend optimal TP configurations and scaling thresholds

### Prometheus Setup

The SLA planner requires Prometheus to collect real-time metrics. Ensure Prometheus is running and accessible at the configured endpoint (default: `http://localhost:9090`).

## Configuration

### YAML Configuration

```yaml
Planner:
  environment: local                    # or "kubernetes"
  no-operation: false                   # set to true for observation-only mode
  
  # SLA targets
  ttft: 0.5                            # Time to First Token target (seconds)
  itl: 0.05                            # Inter-Token Latency target (seconds)
  isl: 3000                            # Expected Input Sequence Length (tokens)
  osl: 150                             # Expected Output Sequence Length (tokens)
  
  # Infrastructure
  max-gpu-budget: 8                    # Maximum GPUs available
  min-endpoint: 1                      # Minimum replicas per worker type
  prefill-engine-num-gpu: 1            # GPUs per prefill engine
  decode-engine-num-gpu: 1             # GPUs per decode engine
  
  # Timing
  adjustment-interval: 180             # Seconds between scaling decisions
  
  # Data sources
  prometheus-endpoint: "http://localhost:9090"
  profile-results-dir: "profiling_results"
  
  # Load prediction
  load-predictor: "arima"              # "constant", "arima", or "prophet"
  load-prediction-window-size: 50      # Number of samples for prediction
```

### Command-line Configuration

```bash
dynamo serve graphs.disagg:Frontend -f config.yaml \
  --Planner.ttft=0.5 \
  --Planner.itl=0.05 \
  --Planner.profile-results-dir=./profiling_results \
  --Planner.no-operation=false
```

## Usage

### 1. Profile Your Model

First, generate performance profiles for your specific model and hardware:

```bash
# Example profiling for a disaggregated setup
python -m utils.profile_sla \
  --config examples/llm/configs/disagg.yaml \
  --output-dir ./profiling_results \
  --isl 3000 \
  --osl 150 \
  --ttft 500 \
  --itl 50
```

### 2. Configure SLA Planner

Create or modify your configuration file:

```yaml
# config.yaml
Common:
  model: your-model-name
  # ... other common configs

Planner:
  environment: local
  no-operation: false
  ttft: 0.5                           # 500ms TTFT target
  itl: 0.05                           # 50ms ITL target
  profile-results-dir: "./profiling_results"
  max-gpu-budget: 8
  adjustment-interval: 180
  load-predictor: "arima"

# ... other component configs
```

### 3. Start the System

```bash
# Start with SLA planner enabled
dynamo serve graphs.disagg:Frontend -f config.yaml
```

### 4. Monitor Performance

The SLA planner logs detailed information about:
- Current and predicted load metrics
- Correction factors applied to performance models
- Scaling decisions and GPU allocation
- SLA compliance status

## Load Prediction Models

The SLA planner supports three load prediction models:

### Constant Predictor
- **Use case**: Stable, predictable workloads
- **Behavior**: Assumes next load equals current load
- **Configuration**: `load-predictor: "constant"`

### ARIMA Predictor
- **Use case**: Time-series data with trends and seasonality
- **Behavior**: Uses auto-ARIMA to fit optimal model parameters
- **Configuration**: `load-predictor: "arima"`
- **Requirements**: Minimum 5 data points for prediction

### Prophet Predictor
- **Use case**: Complex seasonal patterns and trend changes
- **Behavior**: Facebook's Prophet model for time-series forecasting
- **Configuration**: `load-predictor: "prophet"`
- **Requirements**: Minimum 5 data points for prediction

## Scaling Algorithm

The SLA planner uses a sophisticated scaling algorithm:

### 1. Metric Collection
Every adjustment interval, collect:
- Average Time to First Token (TTFT)
- Average Inter-Token Latency (ITL)
- Request count and duration
- Input/Output sequence lengths

### 2. Correction Factor Calculation
- **Prefill correction**: `actual_ttft / expected_ttft`
- **Decode correction**: `actual_itl / expected_itl`

### 3. Load Prediction
Forecast next interval's:
- Number of requests
- Input sequence length
- Output sequence length

### 4. Replica Calculation

**Prefill replicas**:
```
predicted_load = next_requests * next_isl / interval * min(1, prefill_correction)
prefill_replicas = ceil(predicted_load / interpolated_throughput / gpus_per_engine)
```

**Decode replicas**:
```
corrected_itl_sla = target_itl / decode_correction
optimal_throughput = find_best_throughput(corrected_itl_sla, context_length)
decode_replicas = ceil(next_requests * next_osl / interval / optimal_throughput / gpus_per_engine)
```

### 5. GPU Budget Enforcement
If total GPU requirement exceeds budget, scale down proportionally while maintaining minimum replicas.

## Monitoring and Debugging

### Observation Mode
Run with `no-operation: true` to observe scaling decisions without making changes:

```yaml
Planner:
  no-operation: true  # Only log decisions, don't scale
```

### Logging
The SLA planner provides detailed logging:
- Load predictions and actual values
- Correction factors and their impact
- Scaling decisions and reasoning
- GPU allocation and budget utilization

### Key Metrics to Monitor
- **Correction factors**: Should stabilize over time
  - Prefill correction << 1 indicates queueing delays
  - Decode correction ≈ 1 indicates accurate modeling
- **Prediction accuracy**: Compare predicted vs. actual load
- **SLA compliance**: Monitor actual TTFT/ITL vs. targets

## Limitations and Considerations

### Current Limitations
- **Fixed ISL/OSL assumption**: Profiling assumes consistent sequence lengths
- **No KV cache reuse**: Performance models don't account for cache reuse benefits
- **Piggy-backed prefill**: Decode ITL may be affected by short prefill requests
- **Single-node profiling**: Multi-node engine profiling not yet supported

### Best Practices
1. **Profile regularly**: Re-profile when changing models or hardware
2. **Monitor correction factors**: Large deviations indicate model drift
3. **Adjust prediction windows**: Tune `load-prediction-window-size` for your workload
4. **Conservative SLA targets**: Set targets with some buffer for real-world variance
5. **Gradual rollout**: Start with `no-operation: true` to validate behavior

## Troubleshooting

### Common Issues

**High correction factors**:
- Check if actual workload matches profiled conditions
- Verify Prometheus metrics are accurate
- Consider re-profiling with current workload patterns

**Frequent scaling oscillations**:
- Increase `adjustment-interval` for more stability
- Reduce prediction sensitivity by increasing window size
- Check for noisy or inconsistent metrics

**SLA violations**:
- Verify profiling data quality and coverage
- Check if GPU budget is sufficient
- Monitor for resource contention or external factors

**Prediction errors**:
- Try different load predictors (`arima` vs `prophet`)
- Adjust `load-prediction-window-size`
- Ensure sufficient historical data for complex predictors

### Debug Commands

```bash
# Check profiling results
ls -la profiling_results/selected_*_interpolation/

# Verify Prometheus connectivity
curl http://localhost:9090/api/v1/query?query=up

# Monitor planner logs
tail -f dynamo.log | grep -i planner
```

## Comparison with Load-based Planner

| Aspect | Load-based Planner | SLA Planner |
|--------|-------------------|-------------|
| **Scaling trigger** | Resource thresholds | SLA targets |
| **Prediction** | Reactive | Proactive |
| **Setup complexity** | Low | High (requires profiling) |
| **Accuracy** | Good for stable loads | Better for variable loads |
| **SLA guarantees** | None | Target-based |
| **Resource efficiency** | Moderate | Higher |

## Future Enhancements

- Multi-node engine profiling support
- KV cache reuse modeling
- Dynamic ISL/OSL adaptation
- Advanced prediction models (neural networks)
- Integration with external monitoring systems
- Automated profiling and model updates 