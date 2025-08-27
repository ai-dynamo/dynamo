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

This directory contains benchmarking scripts and tools for performance evaluation of Dynamo deployments.

## Quick Start

### Benchmark an Existing Endpoint
```bash
./benchmark.sh --namespace my-namespace --endpoint "http://your-endpoint:8000"
```

### Benchmark Dynamo Deployments
```bash
# Benchmark disaggregated vLLM
./benchmark.sh --namespace my-namespace --disagg components/backends/vllm/deploy/disagg.yaml

# Benchmark TensorRT-LLM GPT-OSS
./benchmark.sh --namespace my-namespace --disagg components/backends/trtllm/deploy/gpt-oss-disagg.yaml

# Benchmark all deployment types
./benchmark.sh --namespace my-namespace \
  --agg components/backends/vllm/deploy/agg.yaml \
  --disagg components/backends/vllm/deploy/disagg.yaml \
  --vanilla benchmarks/utils/templates/vanilla-vllm.yaml
```

**Note**: The sample manifests may reference private registry images. Update the `image:` fields to use accessible images from [Dynamo NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/collections/ai-dynamo/artifacts) or your own registry before running.

## Features

The benchmarking framework supports:

**Two Benchmarking Modes:**
- **Endpoint Benchmarking**: Test existing HTTP endpoints without deployment overhead
- **Deployment Benchmarking**: Deploy, test, and cleanup Dynamo configurations automatically

**Flexible Configuration:**
- Optional deployment types - specify any combination of `--agg`, `--disagg`, `--vanilla`
- Engine-agnostic vanilla backend support (vLLM, TensorRT-LLM, etc.)
- Customizable concurrency levels, sequence lengths, and models
- Automated performance plot generation

**Supported Backends:**
- Dynamo Aggregated (vLLM)
- Dynamo Disaggregated (vLLM, TensorRT-LLM)
- Vanilla backends (vLLM, TensorRT-LLM, or any OpenAI-compatible API)

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
