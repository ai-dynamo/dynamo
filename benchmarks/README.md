<!-- # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. -->

# Benchmarks

This directory contains benchmarking tools and scripts for Dynamo deployments. Benchmarking uses [AIPerf](https://github.com/ai-dynamo/aiperf) directly — a comprehensive tool for measuring generative AI inference performance.

## Quick Start

### Benchmark a Dynamo Deployment
First, deploy your DynamoGraphDeployment using the [deployment documentation](../docs/kubernetes/), then:

```bash
# Port-forward your deployment to http://localhost:8000
kubectl port-forward -n <namespace> svc/<frontend-service-name> 8000:8000 > /dev/null 2>&1 &

# Run a single benchmark
aiperf profile \
    --model <your-model> \
    --url http://localhost:8000 \
    --endpoint-type chat \
    --streaming \
    --concurrency 10 \
    --request-count 100

# Run a concurrency sweep for Pareto analysis
for c in 1 2 5 10 50 100; do
    aiperf profile \
        --model <your-model> \
        --url http://localhost:8000 \
        --endpoint-type chat \
        --streaming \
        --concurrency $c \
        --request-count $(( c * 3 > 10 ? c * 3 : 10 )) \
        --artifact-dir "artifacts/my-benchmark/c$c"
done

# Generate comparison plots
aiperf plot artifacts/my-benchmark
```

## Directory Contents

- **`incluster/`** — Kubernetes Job manifest for running benchmarks inside the cluster
- **`router/`** — KV Router benchmarking scripts (prefix ratio, trace replay, agent, priority queue)
- **`prefix_data_generator/`** — Tools for analyzing and synthesizing prefix-structured data

## Comprehensive Guide

For detailed documentation including server-side benchmarking, Pareto analysis, and advanced AIPerf features, see the [complete benchmarking guide](../docs/benchmarks/benchmarking.md).
