<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Profiling Method Comparison Test Plan

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Model | Qwen/Qwen3-32B |
| TTFT Target | 500 ms |
| ITL Target | 50 ms |
| ISL | 2048 |
| OSL | 256 |
| GPU Type | H200 SXM |
| TP Sweep | 1, 2, 4, 8 |

## Part 1: Predictive Accuracy

Test at 4 load levels:

| Level | Concurrency | Purpose |
|-------|-------------|---------|
| Idle | 1 | Baseline latency |
| Medium | 50% max BS | Moderate load |
| Saturation | 90% max BS | Near capacity |
| Overload | 110% max BS | Degradation |

## Part 2: Optimization Accuracy

Ground truth = Profile-ALL config's goodput under realistic load.

Workloads:
1. Sinusoidal load (sin_load_generator)
2. Production trace (DeepInfra or Mooncake)

Metric: Regret = `(ground_truth - method) / ground_truth`

## Execution

### Phase 1: Profiling

```bash
# AIC-ALL
python -m benchmarks.profiler.profile_sla \
    --model Qwen/Qwen3-32B --backend trtllm \
    --use-ai-configurator --aic-system h200_sxm \
    --isl 2048 --osl 256 --ttft 500 --itl 50 \
    --output-dir results/aic_all

# Profile-ALL
python -m benchmarks.profiler.profile_sla \
    --model Qwen/Qwen3-32B --backend vllm \
    --namespace hannahz \
    --isl 2048 --osl 256 --ttft 500 --itl 50 \
    --output-dir results/online_all
```

### Phase 2: Predictive Accuracy

```bash
# Deploy recommended config, then:
python -m benchmarks.profiler.comparison.validate_deployment \
    --url localhost:8000 --model Qwen/Qwen3-32B \
    --max-batch-size 64 \
    --predicted-ttft <from_profiler> --predicted-itl <from_profiler> \
    --ttft-target 500 --itl-target 50 \
    --output-dir validation/
```

### Phase 3: Optimization Accuracy

```bash
# Generate load
python benchmarks/sin_load_generator/sin_synth.py \
    --time-duration 1800 \
    --request-rate-min 5 --request-rate-max 45 \
    --isl1 2048 --osl1 256 \
    --output-file sinusoidal.jsonl

# Run benchmark
aiperf profile \
    --model Qwen/Qwen3-32B --url localhost:8000 --streaming \
    --input-file sinusoidal.jsonl --custom-dataset-type mooncake_trace \
    --goodput "time_to_first_token:500 inter_token_latency:50"
```

### Phase 4: Compare

```bash
python -m benchmarks.profiler.comparison.run_comparison \
    --aic-results results/aic_all \
    --online-results results/online_all \
    --model Qwen/Qwen3-32B --ttft 500 --itl 50
```
