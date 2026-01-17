<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Finding Best Initial Configs using AIConfigurator

[AIConfigurator](https://github.com/ai-dynamo/aiconfigurator/tree/main) is a performance optimization tool that helps you find the optimal configuration for deploying LLMs with Dynamo. It automatically determines the best number of prefill and decode workers, parallelism settings, and deployment parameters to meet your SLA targets while maximizing throughput.

## Why Use AIConfigurator?

When deploying LLMs with Dynamo, you need to make several critical decisions:
- **Aggregated vs Disaggregated**: Which architecture gives better performance for your workload?
- **Worker Configuration**: How many prefill and decode workers to deploy?
- **Parallelism Settings**: What tensor/pipeline parallel configuration to use?
- **SLA Compliance**: How to meet your TTFT and TPOT targets?

AIConfigurator answers these questions in seconds, providing:
- Optimal configurations that meet your SLA requirements
- Ready-to-deploy Dynamo configuration files
- Performance comparisons between different deployment strategies
- Up to 1.7x better throughput compared to manual configuration

## Quick Start

```bash
# Install
pip3 install aiconfigurator

# Find optimal configuration
aiconfigurator cli default \
  --model QWEN3_32B \        # Model name (QWEN3_32B, LLAMA3.1_70B, etc.)
  --total_gpus 32 \          # Number of available GPUs
  --system h200_sxm \        # GPU type (h100_sxm, h200_sxm, a100_sxm)
  --isl 4000 \               # Input sequence length (tokens)
  --osl 500 \                # Output sequence length (tokens)
  --ttft 300 \               # Target Time To First Token (ms)
  --tpot 10 \                # Target Time Per Output Token (ms)
  --save_dir ./dynamo-configs

# Deploy
kubectl apply -f ./dynamo-configs/disagg/top1/disagg/k8s_deploy.yaml
```

## Example Output

```text
********************************************************************************
*                     Dynamo aiconfigurator Final Results                      *
********************************************************************************
  ----------------------------------------------------------------------------
  Input Configuration & SLA Target:
    Model: QWEN3_32B (is_moe: False)
    Total GPUs: 32
    Best Experiment Chosen: disagg at 812.92 tokens/s/gpu (1.70x better)
  ----------------------------------------------------------------------------
  Overall Best Configuration:
    - Best Throughput: 812.92 tokens/s/gpu
    - User Throughput: 120.23 tokens/s/user
    - TTFT: 276.76ms
    - TPOT: 8.32ms
  ----------------------------------------------------------------------------
  Pareto Frontier:
               QWEN3_32B Pareto Frontier: tokens/s/gpu vs tokens/s/user
      ┌────────────────────────────────────────────────────────────────────────┐
1600.0┤ •• disagg                                                              │
      │ ff agg                                                                 │
      │ xx disagg best                                                         │
      │                                                                        │
1333.3┤   f                                                                    │
      │   ff                                                                   │
      │     ff    •                                                            │
      │       f   ••••••••                                                     │
1066.7┤        f         ••                                                    │
      │         fff       ••••••••                                             │
      │            f              ••                                           │
      │            f                ••••                                       │
 800.0┤            fffff                •••x                                   │
      │                 fff                 ••                                 │
      │                   fff                •                                 │
      │                     fffff             ••                               │
 533.3┤                         ffff            ••                             │
      │                             ffff          ••                           │
      │                                 fffffff     •••••                      │
      │                                        ffffff    ••                    │
 266.7┤                                              fffff •••••••••           │
      │                                                   ffffffffff           │
      │                                                             f          │
      │                                                                        │
   0.0┤                                                                        │
      └┬─────────────────┬─────────────────┬────────────────┬─────────────────┬┘
       0                60                120              180              240
tokens/s/gpu                         tokens/s/user

1. **Performance Comparison**: Shows disaggregated vs aggregated serving performance
2. **Optimal Configuration**: The best configuration that meets your SLA targets
3. **Deployment Files**: Ready-to-use Dynamo configuration files

## Key Features

### Fast Profiling Integration
```bash
# Use with Dynamo's SLA planner (20-30 seconds vs hours)
python3 -m benchmarks.profiler.profile_sla \
   --config ./examples/backends/trtllm/deploy/disagg.yaml \
   --backend trtllm \
   --use-ai-configurator \
   --aic-system h200_sxm \
   --aic-model-name QWEN3_32B
```

### Custom Configuration
```bash
# For advanced users: define custom search space
aiconfigurator cli exp --yaml_path custom_config.yaml
```

## Common Use Cases

```bash
# Strict SLAs (low latency)
aiconfigurator cli default --model QWEN2.5_7B --total_gpus 8 --system h200_sxm --ttft 100 --tpot 5

# High throughput (relaxed latency)
aiconfigurator cli default --model QWEN3_32B --total_gpus 32 --system h200_sxm --ttft 1000 --tpot 50
```

## Supported Configurations

**Models**: GPT, LLAMA2/3, QWEN2.5/3, Mixtral, DEEPSEEK_V3
**GPUs**: H100, H200, A100, B200 (preview), GB200 (preview)
**Backend**: TensorRT-LLM (vLLM and SGLang coming soon)

## Additional Options

```bash
# Web interface
aiconfigurator webapp  # Visit http://127.0.0.1:7860

# Docker
docker run -it --rm nvcr.io/nvidia/aiconfigurator:latest \
  aiconfigurator cli default --model LLAMA3.1_70B --total_gpus 16 --system h100_sxm
```

## Troubleshooting

**Model name mismatch**: Use exact model name that matches your deployment
**GPU allocation**: Verify available GPUs match `--total_gpus`
**Performance variance**: Results are estimates - benchmark actual deployment

## Learn More

- [Dynamo Installation Guide](/docs/kubernetes/installation_guide.md)
- [SLA Planner Quick Start Guide](/docs/planner/sla_planner_quickstart.md)
- [Benchmarking Guide](/docs/benchmarks/benchmarking.md)