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

This directory contains benchmarking scripts and tools for performance evaluation of Dynamo deployments. The benchmarking framework is a wrapper around aiperf that makes it easy to benchmark DynamoGraphDeployments or other deployments with exposed endpoints.

## Quick Start

### Benchmark a Dynamo Deployment
First, deploy your DynamoGraphDeployment using the [deployment documentation](../docs/kubernetes/), then:

```bash
# Port-forward your deployment to http://localhost:8000
kubectl port-forward -n <namespace> svc/<frontend-service-name> 8000:8000 > /dev/null 2>&1 &

# Run benchmark
python3 -m benchmarks.utils.benchmark \
    --benchmark-name my-benchmark \
    --endpoint-url http://localhost:8000 \
    --model "<your-model>"

# Generate plots
python3 -m benchmarks.utils.plot --data-dir ./benchmarks/results

# Or plot only specific benchmark experiments
python3 -m benchmarks.utils.plot --data-dir ./benchmarks/results --benchmark-name my-benchmark
```

## Features

Benchmark any HTTP endpoints! The benchmarking framework supports:

**Flexible Configuration:**
- User-defined benchmark names using `--benchmark-name` flag
- Support for single endpoint benchmarking with `--endpoint-url` flag
- Customizable concurrency levels (configurable via CONCURRENCIES env var), sequence lengths, and models
- Automated performance plot generation with custom benchmark names

**Supported Backends:**
- DynamoGraphDeployments with port-forwarded endpoints
- External HTTP endpoints (for comparison with non-Dynamo backends or platforms)

## Installation

This is already included as part of the Dynamo container images. To install locally or standalone:

```bash
pip install -e .
```

## Data Generation Tools

This directory also includes lightweight tools for:
- Analyzing prefix-structured data (`datagen analyze`)
- Synthesizing structured data customizable for testing purposes (`datagen synthesize`)

Detailed information is provided in the `prefix_data_generator` directory.

## Comprehensive Guide

For detailed documentation, configuration options, and advanced usage, see the [complete benchmarking guide](../docs/benchmarks/benchmarking.md).
