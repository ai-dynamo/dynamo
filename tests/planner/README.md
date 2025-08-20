<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SLA Planner Load Test

This directory contains comprehensive testing tools for validating the SLA planner's scaling behavior.
The SLA planner monitors metrics every 60 seconds (default adjustment interval) and scales
prefill/decode workers based on TTFT, ITL, and request patterns.

## Pre-Requisite: Pre-Deployment Profiling Data

You have two options to obtain the pre-deployment profiling data:

### Option A: Use Test Configuration (Quickstart)

Use the pre-configured test deployment with sample profiling data, we provide the results and the deployment configuration for the following models x hardware configurations:
- `nvidia/Llama-3.1-8B-Instruct-FP8` on H200 with max context length 16384, TP1 Prefill, and TP1 Decode. At ISL/OSL 3000/150, it achieves 40k tokens/s/gpu prefill with 80ms TTFT and 10k tokens/s/gpu decode with 10ms ITL. See `profiling_results/H200_TP1P_TP1D/`.

### Option B: Use Your Own Profiling Results

1. Run pre-deployment profiling for your specific setup. See the [pre-deployment profiling documentation](../../docs/architecture/pre_deployment_profiling.md) for detailed instructions.

## Interpolator Testing

SLA planner uses two interpolators to estimate the performance of prefill and decode. You can test the interpolators with the following command:

```bash
python components/planner/src/dynamo/planner/utils/perf_interpolation.py \
  --profile_results_dir <path_to_profile_results> \
  --isl <ISL> \
  --osl <OSL> \
  --ttft <TTFT(s)> \
  --itl <ITL(s)>
```

The script will perform the interpolation based on ISL, OSL, and TTFT and ITL SLAs and advise the load that can saturate the engine.

For example, to test the interpolator for `nvidia/Llama-3.1-8B-Instruct-FP8` on H200,

```bash
python components/planner/src/dynamo/planner/utils/perf_interpolation.py \
  --profile_results_dir tests/planner/profiling_results/H200_TP1P_TP1D/ \
  --isl 3000 \
  --osl 150 \
  --ttft 0.1 \
  --itl 0.01

> ISL=3000, OSL=150
> TTFT=0.1s, ITL=0.01s
> Using profile results from tests/planner/profiling_results/H200_TP1P_TP1D/
>
> Interpolating prefill performance ...
>         Estimated TTFT=0.027s <= target TTFT=0.100s. Requests can queue 0.073s maximally while meeting TTFT SLA.
>         Estimated throughput: 110893.48 tokens/s/gpu. Request rate at 36.96 requests/s will saturate one GPU.
>
> Interpolating decode performance ...
>         Average context length: isl + osl/2 = 3075.
>         Estimated ITL=0.0098s <= target ITL=0.0100s at 33.33% active kv usage.
>         Estimated throughput: 10226.60 token/s/gpu. Request rate at 68.18 requests/s will saturate one GPU.
```

## Scaling Tests

This directory contains comprehensive tests for validating the SLA planner's scaling behavior. The tests validate both the replica calculation logic and end-to-end scaling behavior.

### Test Types

1. **Unit Tests** (`test_replica_calculation.py`) - Test the mathematical formulas for calculating prefill and decode replicas in isolation
2. **End-to-End Tests** (`run_scaling_test.sh`) - Test complete workflow including Kubernetes deployment, load generation, and pod scaling validation

### Quick Start

#### Run Unit Tests Only
Test the replica calculation logic without requiring Kubernetes:

```bash
python -m pytest test_replica_calculation.py -v
```

#### Run Full End-to-End Test
Test complete scaling behavior including Kubernetes deployment and load generation:

```bash
./run_scaling_test.sh
```

With custom namespace:
```bash
./run_scaling_test.sh --namespace production
```

### Test Scenario

The main test scenario validates scaling for **H200 with 1P1D configuration**:
- **Phase 1**: 10 req/s (maintains 1P1D)
- **Phase 2**: 20 req/s (scales to 2P1D - 2 prefill workers, 1 decode worker)
- **ISL/OSL**: 3000/150 tokens

### Prerequisites for E2E Tests

- Kubernetes cluster with GPU nodes
- kubectl configured and accessible
- genai-perf available in PATH
- Python dependencies installed

For detailed configuration, troubleshooting, and architecture information, see [README_scaling_tests.md](README_scaling_tests.md).
