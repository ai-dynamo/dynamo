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

Uses pre-deployment profiling data and traffic prediction to compute the number of prefill/decode replicas needed to meet TTFT and ITL SLA targets. Requires profiling data from the Dynamo profiler.

### Load-Based Scaling (Experimental)

Uses real-time per-worker load metrics (active prefill tokens, active KV blocks) from the router to make SLA-aware scaling decisions via online linear regression. Does not require profiling data. Responds quickly to traffic bursts.

When both modes are enabled, throughput-based scaling provides a lower bound on replicas while load-based scaling handles real-time adjustments.

### Support Matrix

| Deployment Type | Throughput-Based | Load-Based (Experimental) |
|-----------------|:----------------:|:-------------------------:|
| Disaggregated   | Supported        | Supported                 |
| Aggregated      | Unsupported      | Supported                 |

## Documentation

- **User docs**: [Planner Guide](../../../../docs/pages/components/planner/planner-guide.md) (deployment, configuration, examples)
- **Design docs**: [Planner Design](../../../../docs/pages/design-docs/planner-design.md) (architecture, algorithms)
