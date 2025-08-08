<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# LLM Benchmarks

This directory contains scripts and tools for benchmarking LLM deployments served via OpenAI-compatible APIs.

## Tools

- **perf.sh**: Wrapper script to benchmark aggregated or disaggregated deployment topologies using `genai-perf`.
- **perf.py**: Python version of the benchmarking CLI, offering better integration and configurability.
- **plot_pareto.py**: Generates Pareto plots comparing throughput/user vs throughput/GPU.
- **nginx.conf**: Sample load balancing configuration for directing traffic to multiple LLM backends.

## Benchmarking

### Aggregated

```bash
bash perf.sh \
  --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --url http://localhost:8080 \
  --tp 2 \
  --dp 2 \
  --mode aggregated
```

### Disaggregated

```bash
bash perf.sh \
  --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --url http://localhost:8080 \
  --prefill-tp 1 \
  --prefill-dp 2 \
  --decode-tp 2 \
  --decode-dp 2 \
  --mode disaggregated
```

## Plotting Results

```bash
python plot_pareto.py --artifacts-root-dir artifacts_root --title "Dynamo vs vLLM"
```
