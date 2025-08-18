<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SLA Planner Load Test

This directory contains comprehensive testing tools for validating the SLA planner's scaling behavior.
The SLA planner monitors metrics every 60 seconds (default adjustment interval) and scales
prefill/decode workers based on TTFT, ITL, and request patterns.

## Setup Instructions

### Step 1: Start a Deployment with Planner

You have two options for setting up the planner:

#### Option A: Use Test Configuration (Quickstart)

Use the pre-configured test deployment with sample profiling data, we provide the results and the deployment configuration for two models x hardware configurations:
- placeholder for A100
- `nvidia/Llama-3.1-8B-Instruct-FP8` on H200 with max context length 16384, TP1 Prefill, and TP1 Decode. At ISL/OSL 3000/150, it achieves 40k tokens/s/gpu prefill with 80ms TTFT and 10k tokens/s/gpu decode with 10ms ITL. See `profiling_results/H200_TP1P_TP1D/`.

```bash
kubectl apply -f tests/planner/disagg_planner.yaml
```

This deployment uses test profiling results from the `profiling_results/` directory.

#### Option B: Use Your Own Profiling Results

1. Run pre-deployment profiling for your specific setup. See the [pre-deployment profiling documentation](../../docs/architecture/pre_deployment_profiling.md) for detailed instructions.

2. Start your own planner instance using one of the components/backends if you want to use planner based on your real profiling results in your repository.

### Step 2: Verify Deployment

Wait for all pods to be ready:

```bash
kubectl get pods -n default -w
```

Check that the frontend is accessible and port-forward the frontend to localhost:

```bash
kubectl port-forward service/vllm-disagg-planner-frontend 8000:8000
```

## Test Types

### Load Pattern Tests

Generate different types of load to stress the system:

```bash
# Sustained load -- simulating continuous concurrent requests from multiple users
python sla_planner_load_test.py --pattern sustained --concurrent-users 10 --duration 300

# Burst load pattern -- period bursts of requests, testing scaling responsiveness
python sla_planner_load_test.py --pattern burst --burst-requests 15 --burst-interval 30 --duration 300

# Ramp-up pattern -- gradually increasing load over time
python sla_planner_load_test.py --pattern ramp --concurrent-users 15 --ramp-duration 60 --duration 300
```

### Scaling Validation Tests

Validate that pods are scaling up and down as expected:

```bash
# Test scale-up behavior
python sla_planner_load_test.py --pattern scale_up_test --kubernetes-namespace default

# Test scale-down behavior
python sla_planner_load_test.py --pattern scale_down_test --kubernetes-namespace default

# Test selective scaling (one worker type scales but not the other)
python sla_planner_load_test.py --pattern selective_scaling_test --kubernetes-namespace default

# Run all scaling validation tests
python sla_planner_load_test.py --pattern all_scaling_tests --kubernetes-namespace default
```

## Configuration Options

| Category | Option | Default Value | Description |
|----------|---------|---------------|-------------|
| **Basic Configuration** | | | |
| | `--frontend-url` | `http://localhost:8000` | Frontend URL |
| | `--pattern` | `sustained` | Load pattern or validation test |
| | `--concurrent-users` | `5` | Number of concurrent users |
| | `--duration` | `180` | Test duration in seconds |
| **Advanced Configuration** | | | |
| | `--request-timeout` | `120` | Request timeout in seconds |
| | `--metrics-interval` | `30` | Metrics reporting interval in seconds |
| | `--kubernetes-namespace` | `default` | K8s namespace for pod monitoring |
| | `--expected-scaling-time` | `120` | Expected time for scaling events in seconds |
| | `--validate-scaling` | `False` | Enable pod scaling validation (auto-enabled for scaling tests) |
| **Pattern-Specific Configuration** | | | |
| | `--ramp-duration` | `60` | Ramp-up duration in seconds (for ramp pattern) |
| | `--burst-interval` | `30` | Interval between bursts in seconds (for burst pattern) |
| | `--burst-requests` | `10` | Number of requests per burst (for burst pattern) |

## Understanding Results

### Load Test Output

The test provides real-time metrics including:
- Active and total request counts
- Error rates and response times
- TTFT (Time to First Token) statistics
- Current pod counts (when validation enabled)

### Scaling Validation

When running scaling tests, look for:
- ✓ PASSED: Scaling behavior detected as expected
- ✗ FAILED: No scaling detected within timeout
- Pod count changes in real-time monitoring

### Expected Scaling Behavior

- **Scale-up**: Triggered by sustained high TTFT/ITL metrics (typically after 60-120 seconds)
- **Scale-down**: Occurs when load decreases and metrics stabilize (typically takes longer)
- **Selective scaling**: May scale only prefill or decode workers based on workload characteristics
