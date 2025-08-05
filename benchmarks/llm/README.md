# LLM Benchmarking Tools

This directory contains tools for benchmarking LLM inference performance in Dynamo deployments.

## Overview

The benchmarking suite includes:
- **`perf.sh`** - Automated performance benchmarking script using GenAI-Perf
- **`plot_pareto.py`** - Results analysis and Pareto efficiency visualization 
- **`nginx.conf`** - Load balancer configuration for multi-backend setups

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--tensor-parallelism, --tp` | Tensor parallelism for aggregated mode | 0 |
| `--data-parallelism, --dp` | Data parallelism for aggregated mode | 0 |
| `--prefill-tp` | Prefill tensor parallelism for disaggregated mode | 0 |
| `--prefill-dp` | Prefill data parallelism for disaggregated mode | 0 |
| `--decode-tp` | Decode tensor parallelism for disaggregated mode | 0 |
| `--decode-dp` | Decode data parallelism for disaggregated mode | 0 |
| `--model` | HuggingFace model ID | `neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic` |
| `--url` | Target inference endpoint | `http://localhost:8000` |
| `--concurrency` | Comma-separated concurrency levels | `1,2,4,8,16,32,64,128,256` |
| `--isl` | Input sequence length | 3000 |
| `--osl` | Output sequence length | 150 |
| `--mode` | Serving mode (`aggregated` or `disaggregated`) | `aggregated` |


## Best Practices

1. **Warm up services** before benchmarking to ensure stable performance
2. **Match parallelism settings** to your actual deployment configuration
3. **Run multiple benchmark iterations** for statistical confidence
4. **Monitor resource utilization** during benchmarks to identify bottlenecks
5. **Compare configurations** using Pareto plots to find optimal settings

## Requirements

- GenAI-Perf tool installed and available in PATH
- Python 3.7+ with matplotlib, pandas, seaborn, numpy
- nginx (for load balancing scenarios)
- Access to target LLM inference service

## Troubleshooting

- Ensure the target URL is accessible before running benchmarks
- Verify model names match those available in your deployment
- Check that parallelism settings align with your hardware configuration
- Monitor system resources to avoid resource contention during benchmarks


