<!-- # SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

This directory contains benchmarking scripts and tools for performance evaluation of Dynamo deployments. The benchmarking framework is a wrapper around genai-perf that makes it easy to benchmark DynamoGraphDeployments and compare them with external endpoints.

## Quick Start

### Benchmark an Existing Endpoint
```bash
./benchmark.sh --namespace my-namespace --input my-endpoint=http://your-endpoint:8000
```

### Benchmark Dynamo Deployments
```bash
# Benchmark disaggregated vLLM with custom label
./benchmark.sh --namespace my-namespace --input vllm-disagg=components/backends/vllm/deploy/disagg.yaml

# Benchmark TensorRT-LLM disaggregated deployment
./benchmark.sh --namespace my-namespace --input trtllm-disagg=components/backends/trtllm/deploy/disagg.yaml

# Compare multiple Dynamo deployments
./benchmark.sh --namespace my-namespace \
  --input agg=components/backends/vllm/deploy/agg.yaml \
  --input disagg=components/backends/vllm/deploy/disagg.yaml

# Compare Dynamo vs external endpoint
./benchmark.sh --namespace my-namespace \
  --input dynamo=components/backends/vllm/deploy/disagg.yaml \
  --input external=http://localhost:8000
```

**Note**:
- The sample manifests may reference private registry images. Update the `image:` fields to use accessible images from [Dynamo NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/collections/ai-dynamo/artifacts) or your own registry before running.
- Only DynamoGraphDeployment manifests are supported for automatic deployment. To benchmark non-Dynamo backends (vLLM, TensorRT-LLM, SGLang, etc.), deploy them manually using their Kubernetes guides and use the endpoint option.

## Features

The benchmarking framework supports:

**Two Benchmarking Modes:**
- **Endpoint Benchmarking**: Test existing HTTP endpoints without deployment overhead
- **Deployment Benchmarking**: Deploy, test, and cleanup DynamoGraphDeployments automatically

**Flexible Configuration:**
- User-defined labels for each input using `--input label=value` format
- Support for multiple inputs to enable comparisons
- Customizable concurrency levels (configurable via CONCURRENCIES env var), sequence lengths, and models
- Automated performance plot generation with custom labels

**Sequential GPU Usage:**
- Models are deployed and benchmarked **sequentially**, not in parallel
- Each deployment gets exclusive access to all available GPUs during its benchmark run
- Ensures accurate performance measurements and fair comparison across configurations

**Supported Backends:**
- DynamoGraphDeployments
- External HTTP endpoints (for comparison with non-Dynamo backends)

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
