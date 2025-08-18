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
