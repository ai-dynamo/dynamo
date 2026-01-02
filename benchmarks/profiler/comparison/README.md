<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Profiling Method Comparison Framework

Compare profiling methods across **three dimensions**:
1. **Cost**: Time, GPU-hours, number of deployments
2. **Predictive accuracy**: Does predicted latency match actual latency?
3. **Optimization accuracy**: Did the sweep find the best config?

## Methods

This framework compares two **baselines** plus any number of **experimental methods**:

| Method | Description | Role |
|--------|-------------|------|
| **AIC-ALL** | AI Configurator simulation, no deployments | Baseline (fastest, cheapest) |
| **Online-ALL** | Online AIPerf on all configs | Baseline (most accurate, most expensive) |
| *Additional methods* | Stack rank, Bayesian search, hybrids, etc. | Experimental (1-n methods) |

The goal is to find methods that approach Online-ALL accuracy at AIC-ALL cost.

---

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Model | `Qwen/Qwen3-32B` |
| TTFT | 500 ms |
| ITL | 50 ms |
| ISL | 2048 |
| OSL | 256 |
| TP sweep | 1, 2, 4, 8 |

---

## Step 1: Run Profiling

```bash
export NAMESPACE=hannahz
```

### Baseline 1: AIC-ALL (~30 seconds)

```bash
kubectl delete dgdr qwen3-32b-aic -n $NAMESPACE 2>/dev/null
kubectl apply -n $NAMESPACE -f benchmarks/profiler/comparison/configs/aic_profiling_dgdr.yaml
kubectl logs -f job/profile-qwen3-32b-aic -n $NAMESPACE
```

### Baseline 2: Online-ALL (~30-60 min)

```bash
kubectl delete dgdr qwen3-32b-online -n $NAMESPACE 2>/dev/null
kubectl apply -n $NAMESPACE -f benchmarks/profiler/comparison/configs/online_profiling_dgdr.yaml
kubectl logs -f job/profile-qwen3-32b-online -n $NAMESPACE
```

### Additional Methods

For each experimental method, run its profiling and save results to `results/<method_name>/`.

---

## Step 2: Download Results & Configs

```bash
# Create results directories
mkdir -p benchmarks/profiler/comparison/results/{aic_all,online_all}

# Download AIC results
kubectl get configmap dgdr-output-qwen3-32b-aic -n $NAMESPACE \
  -o jsonpath='{.data.config_with_planner\.yaml}' \
  > benchmarks/profiler/comparison/results/aic_all/config_with_planner.yaml
kubectl logs job/profile-qwen3-32b-aic -n $NAMESPACE \
  > benchmarks/profiler/comparison/results/aic_all/profile_sla.log

# Download Online results
kubectl get configmap dgdr-output-qwen3-32b-online -n $NAMESPACE \
  -o jsonpath='{.data.config_with_planner\.yaml}' \
  > benchmarks/profiler/comparison/results/online_all/config_with_planner.yaml
kubectl logs job/profile-qwen3-32b-online -n $NAMESPACE \
  > benchmarks/profiler/comparison/results/online_all/profile_sla.log

# Repeat for each additional method → results/<method_name>/
```

---

## Step 3: Predictive Accuracy Testing

For each method, deploy its recommended config and measure actual latency at different loads.

### Deploy a method's config

```bash
export METHOD=online_all  # update for each method
kubectl apply -n $NAMESPACE \
  -f benchmarks/profiler/comparison/results/${METHOD}/config_with_planner.yaml
kubectl port-forward svc/trtllm-disagg-frontend 8000:8000 -n $NAMESPACE &
```

### Run AIPerf at 4 load levels

```bash
MAX_BS=128

for LEVEL in idle medium saturation overload; do
  case $LEVEL in
    idle) CONC=1 ;;
    medium) CONC=$((MAX_BS / 2)) ;;
    saturation) CONC=$((MAX_BS * 9 / 10)) ;;
    overload) CONC=$((MAX_BS * 11 / 10)) ;;
  esac
  
  aiperf profile --model Qwen/Qwen3-32B --url localhost:8000 --streaming \
    --concurrency $CONC --request-count 100 \
    --synthetic-input-tokens-mean 2048 --output-tokens-mean 256 \
    --goodput "time_to_first_token:500 inter_token_latency:50" \
    --artifact-dir benchmarks/profiler/comparison/results/${METHOD}/validation/$LEVEL
done
```

**Repeat for each method** (aic_all, online_all, and all experimental methods).

---

## Step 4: Optimization Accuracy Testing

Test all configs under realistic workloads to measure actual goodput.

### Generate sinusoidal load

```bash
python benchmarks/sin_load_generator/sin_synth.py \
  --time-duration 1800 \
  --request-rate-min 5 --request-rate-max 45 \
  --isl1 2048 --osl1 256 \
  --output-file benchmarks/profiler/comparison/results/sinusoidal.jsonl
```

***Note: the benchmarks directory needs to be in your PYTHONPATH.***

### Benchmark each method's config

```bash
# For each method:
aiperf profile --model Qwen/Qwen3-32B --url localhost:8000 --streaming \
  --input-file benchmarks/profiler/comparison/results/sinusoidal.jsonl \
  --custom-dataset-type mooncake_trace \
  --goodput "time_to_first_token:500 inter_token_latency:50" \
  --artifact-dir benchmarks/profiler/comparison/results/${METHOD}/optimization
```

---

## Step 5: Compare All Results

```bash
python -m benchmarks.profiler.comparison.run_comparison \
  --aic-results benchmarks/profiler/comparison/results/aic_all \
  --online-results benchmarks/profiler/comparison/results/online_all \
  --model Qwen/Qwen3-32B \
  --ttft 500 --itl 50 --isl 2048 --osl 256 \
  --output-dir benchmarks/profiler/comparison/results/comparison

  # --additional-results stack_rank:benchmarks/profiler/comparison/results/stack_rank \
  # --additional-results bayesian:benchmarks/profiler/comparison/results/bayesian \
```

---

## Output Structure

```
results/
├── aic_all/                        # Baseline: AIC
│   ├── config_with_planner.yaml
│   ├── profile_sla.log
│   ├── validation/{idle,medium,saturation,overload}/
│   └── optimization/
├── online_all/                     # Baseline: Online
│   ├── config_with_planner.yaml
│   ├── profile_sla.log
│   ├── validation/
│   └── optimization/
├── <method_name>/                  # Each experimental method
│   ├── config_with_planner.yaml
│   ├── profile_sla.log
│   ├── validation/
│   └── optimization/
├── sinusoidal.jsonl
└── comparison/
    ├── comparison_results.json
    ├── summary.txt
    └── *.png
```

See [TEST_PLAN.md](./TEST_PLAN.md) for methodology details.
