<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Profiling Method Comparison Test Plan

## Objective

Compare profiling methods to find the best trade-off between:
- **Cost** (time, GPU-hours)
- **Predictive accuracy** (predicted vs actual latency)
- **Optimization accuracy** (finding the best config)

## Methods Under Test

### Baselines (always included)

| Method | Description | Expected Cost | Expected Accuracy |
|--------|-------------|---------------|-------------------|
| **AIC-ALL** | AI Configurator simulation | ~30 sec, 0 GPU-hrs | Lower (simulation) |
| **Online-ALL** | Online AIPerf on all configs | ~30-60 min, high GPU-hrs | Highest (real measurements) |

### Experimental Methods (1-n)

Methods being explored to achieve online profiling (profiling with any amount of online deployments) accuracy at lower cost:

| Method | Approach |
|--------|----------|
| Stack Rank | Rank configs by predicted performance, only deploy top candidates |
| Bayesian Search | Use Bayesian optimization to explore config space efficiently |
| Stack Rank + Bayesian | Hybrid: rank to narrow, then Bayesian to refine |
| *Others* | Any new approach can be added |

---

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

---

## Metrics

### 1. Cost Metrics

Collected from profiling logs:

| Metric | Description |
|--------|-------------|
| Duration (sec) | Total profiling time |
| GPU-hours | Estimated compute cost |
| Deployments | Number of real k8s deployments |

### 2. Predictive Accuracy

How closely does predicted TTFT/ITL match actual?

| Metric | Formula |
|--------|---------|
| TTFT Error % | `\|predicted - actual\| / actual × 100` |
| ITL Error % | `\|predicted - actual\| / actual × 100` |
| SLA Hit Rate % | Requests meeting both TTFT & ITL targets |

Tested at 4 load levels:

| Level | Concurrency | Purpose |
|-------|-------------|---------|
| Idle | 1 | Baseline (no contention) |
| Medium | 50% max BS | Moderate load |
| Saturation | 90% max BS | Near capacity |
| Overload | 110% max BS | Degradation behavior |

### 3. Optimization Accuracy

Did the method find the best possible config?

| Metric | Formula |
|--------|---------|
| Goodput (req/s) | Requests meeting SLA per second |
| Goodput/GPU | Efficiency metric |
| Regret % | `(ground_truth - method) / ground_truth × 100` |

**Ground truth** = Online-ALL's config goodput under realistic load.

---

## Execution Phases

### Phase 1: Profiling

Run each method and collect:
- `config_with_planner.yaml` (recommended deployment)
- `profile_sla.log` (timing, recommendations)

### Phase 2: Predictive Accuracy

For each method:
1. Deploy its `config_with_planner.yaml`
2. Run AIPerf at idle/medium/saturation/overload
3. Compare predicted vs actual TTFT/ITL

### Phase 3: Optimization Accuracy

For each method:
1. Deploy its config
2. Run AIPerf with sinusoidal load (30 min)
3. Measure goodput, compare to ground truth

### Phase 4: Compare & Report

Generate:
- `comparison_results.json` (all metrics)
- `summary.txt` (human-readable)
- Plots (cost, accuracy, regret)

---

## Key Questions

| Question | How We Answer |
|----------|---------------|
| How fast is each method? | Compare profiling duration |
| How accurate are predictions? | Compare predicted vs actual at multiple loads |
| Does it find good configs? | Compare goodput under realistic load |
| What's the cost/accuracy trade-off? | Plot cost vs accuracy for all methods |
| Which method should I use? | Recommendation based on use case |

---

## Success Criteria

An experimental method is successful if it:
1. **Faster than Online-ALL** by at least 50%
2. **Predictive accuracy** within 20% of Online-ALL at medium load
3. **Optimization regret** under 10% vs ground truth
