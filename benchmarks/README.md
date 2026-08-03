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

Harnesses, workload generators, and trace converters for measuring Dynamo.

**Not sure which one you need?** Start at Performance Analysis — the
[development-host guide](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/cli/operations/performance-analysis.md)
or the [Kubernetes guide](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/kubernetes/operations/performance-analysis.md) —
which routes a performance question to the tool that answers it. For the protocol that
applies whichever tool you pick, see
[Performance Analysis Method](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/developer-guide/knowledge-base/concepts/performance-analysis-method.md).

To send load at an endpoint, see the AIPerf benchmarking guides in the same Operations sections.

## Harnesses

| Directory | Entry point | Use it for |
| --- | --- | --- |
| [`frontend/`](frontend/README.md) | `scripts/sweep_runner.py`, `scripts/run_perf.sh` | Frontend and router sweeps against mock workers, locally or on Kubernetes, with profiling captures alongside the load |
| `incluster/` | `benchmark_job.yaml` | AIPerf as a Kubernetes Job against a deployment |
| [`router/`](router/README.md) | `prefix_ratio_benchmark.py`, `real_data_benchmark.py`, `agent_benchmark.py` | KV router behavior under varying prefix reuse and replayed traces |
| [`mocker/`](mocker/README.md) | `bench_aic_concurrency.py` | The mocker's own latency-prediction path |
| `multimodal/` | `sweep/`, `jsonl/`, `http/` | Multimodal and encode/prefill/decode benchmarking, plus multimodal request datasets |
| `omni/` | `image/` | Text-to-image generation benchmarks |

## Workload generators and trace converters

All produce Mooncake-format JSONL that AIPerf and the replay tools consume.

| Directory | Entry point | Use it for |
| --- | --- | --- |
| [`prefix_data_generator/`](prefix_data_generator/README.md) | `datagen` console script | Analyzing and synthesizing prefix-structured data, when KV reuse is what you are testing |
| [`sin_load_generator/`](sin_load_generator/README.md) | `sin_synth.py` | Time-varying request rate and input/output length ratio |
| [`burstgpt_loadgen/`](burstgpt_loadgen/README.md) | `convert.py` | Converting BurstGPT traces. Does not model KV reuse |
| [`nat_trace/`](nat_trace/README.md) | `convert.py`, `convert_telemetry.py` | Converting NeMo Agent Toolkit profiler traces, preserving session and hash identity |

## Analysis

| Directory | Entry point | Use it for |
| --- | --- | --- |
| [`request_trace/`](request_trace/README.md) | `convert_to_perfetto.py` | Turning request traces into a Perfetto timeline with per-request prefill and decode stages |

## Conventions

Install the packaged pieces with `uv pip install -e ./benchmarks`, which provides `datagen`
and pins the AIPerf version this tree is tested against.

A new harness is not finished until it appears in the table above and in the routing table
on the Performance Analysis pages. A tool nobody can find does not get used.
