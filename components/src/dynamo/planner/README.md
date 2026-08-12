<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Planner

SLA-driven autoscaling controller for Dynamo inference graphs.

## Scaling Modes

The SLA Planner supports two scaling modes that can be used independently or together:

### Throughput-Based Scaling

Uses traffic prediction and the engine performance model to compute the number
of replicas needed to meet TTFT and ITL SLA targets. The model can bootstrap
from worker self-benchmarks, native AI Configurator estimates, profiler output,
or live FPM regression.

### Load-Based Scaling

Uses ForwardPassMetrics (FPM) from the Dynamo event plane for reactive scaling.
With `optimization_target: sla`, the Planner uses online regression to target
TTFT and ITL. Other optimization targets use static or user-defined queue and
KV-utilization thresholds. Load-based scaling does not require profiling data
or the KV Router and supports vLLM, SGLang, and TensorRT-LLM deployments that
emit the required FPM signals, including per-rank signals for attention-DP.

With `optimization_target: sla` and both modes enabled, throughput-based
scaling provides a lower bound on replicas while load-based scaling handles
real-time adjustments.

### Support Matrix

| Deployment Type | Throughput-Based | Load-Based |
|-----------------|:----------------:|:-------------------------:|
| Disaggregated   | Supported        | Supported                 |
| Aggregated      | Supported        | Supported                 |

## Documentation

- **User docs**: [Planner Guide](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/planner/planner-guide.md) (deployment, configuration, examples)
- **Design docs**: [Planner Design](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/planner/planner-design.md) (architecture, algorithms)
- **Manual workflows**: [tests/manual/README.md](tests/manual/README.md) (dry run helpers, perf configs, and manual scaling scripts)
