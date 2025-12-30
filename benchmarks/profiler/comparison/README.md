<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Profiling Method Comparison

Compare profiling methods across two dimensions:
- **Predictive accuracy**: Does predicted perf match actual perf?
- **Optimization accuracy**: Did the sweep find the best config?

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

### AIC-Search ALL (~30 seconds)

```bash
kubectl apply -n $NAMESPACE -f benchmarks/profiler/comparison/configs/aic_profiling_dgdr.yaml

# Watch progress
kubectl logs -f job/profile-qwen3-32b-aic -n $NAMESPACE
```

### Profile-ALL Online (~30-60 min)

```bash
kubectl apply -n $NAMESPACE -f benchmarks/profiler/comparison/configs/online_profiling_dgdr.yaml

# Watch progress
kubectl logs -f job/profile-qwen3-32b-online -n $NAMESPACE
```

---

## Step 2: Get Results from ConfigMaps

```bash
# List profiling ConfigMaps
kubectl get configmaps -n $NAMESPACE | grep profiling

# Save AIC results
mkdir -p benchmarks/profiler/comparison/results/aic_all
kubectl get configmap qwen3-32b-aic-profiling-results -n $NAMESPACE -o jsonpath='{.data}' \
  > benchmarks/profiler/comparison/results/aic_all/configmap_data.json

# Save Online results
mkdir -p benchmarks/profiler/comparison/results/online_all
kubectl get configmap qwen3-32b-online-profiling-results -n $NAMESPACE -o jsonpath='{.data}' \
  > benchmarks/profiler/comparison/results/online_all/configmap_data.json

# Also save the logs
kubectl logs job/profile-qwen3-32b-aic -n $NAMESPACE \
  > benchmarks/profiler/comparison/results/aic_all/profile_sla.log
kubectl logs job/profile-qwen3-32b-online -n $NAMESPACE \
  > benchmarks/profiler/comparison/results/online_all/profile_sla.log
```

---

## Step 3: Compare Results

```bash
python -m benchmarks.profiler.comparison.run_comparison \
  --aic-results benchmarks/profiler/comparison/results/aic_all \
  --online-results benchmarks/profiler/comparison/results/online_all \
  --model Qwen/Qwen3-32B \
  --ttft 500 --itl 50 \
  --isl 2048 --osl 256 \
  --output-dir benchmarks/profiler/comparison/results/comparison
```

---

## Step 4: Validate Predictions (Optional)

Deploy the recommended config and test at multiple load levels:

```bash
python -m benchmarks.profiler.comparison.validate_deployment \
  --url localhost:8000 \
  --model Qwen/Qwen3-32B \
  --max-batch-size 64 \
  --predicted-ttft <from_profiler> \
  --predicted-itl <from_profiler> \
  --ttft-target 500 --itl-target 50 \
  --output-dir benchmarks/profiler/comparison/results/validation
```

---

## Output Files

Each profiling run creates:
- `profile_sla.log` - Timing and recommendations
- `selected_prefill_interpolation/raw_data.npz` - Prefill perf data
- `selected_decode_interpolation/raw_data.npz` - Decode perf data
- `prefill_performance.png`, `decode_performance.png` - Plots
- `config_with_planner.yaml` - Recommended DGD config

See [TEST_PLAN.md](./TEST_PLAN.md) for full methodology.
