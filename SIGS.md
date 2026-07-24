<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

<!-- HAND-MAINTAINED. This file is not generated (unlike CONTRIBUTORS.md).
     Updates record SIG lifecycle decisions; see "Governance Changes" in GOVERNANCE.md.
     CI checks that every non-program CODEOWNERS area appears in this file. -->

# Dynamo Special Interest Groups

Special Interest Groups (SIGs) are open, standing groups that coordinate work within one domain of the project: roadmap discussion, design review, and cross-area coordination. Anyone may join and participate. The SIG model - what SIGs are, the SIG Lead role, and how SIGs are created, merged, or retired - is defined in [GOVERNANCE.md](GOVERNANCE.md); this file is the living roster.

| SIG | Scope | CODEOWNERS Groups |
| :---- | :---- | :---- |
| sig-core | Core runtime and frontend | `dynamo-runtime-codeowners`, `dynamo-frontend-codeowners`, `dynamo-observability-codeowners` |
| sig-router | Router and the Inference Gateway Endpoint Picker (EPP) | `dynamo-router-codeowners`, `dynamo-epp-codeowners` |
| sig-memory-transport | KV/memory transport and storage | `dynamo-kv-memory-codeowners`, `dynamo-gms-codeowners` |
| sig-agents | Agentic workloads | `dynamo-agents-codeowners` |
| sig-hardware | Non-NVIDIA hardware enablement | `dynamo-xpu-codeowners` |
| sig-deploy | Deploy path | `dynamo-operator-codeowners` |
| sig-scaling | Scaling and lifecycle | `dynamo-planner-codeowners`, `dynamo-fault-tolerance-codeowners` |
| sig-rl | Reinforcement learning integrations | `dynamo-rl-codeowners` |
| sig-engines | Backend engine integrations (vLLM, SGLang, TensorRT-LLM) | `dynamo-backend-vllm-codeowners`, `dynamo-backend-sglang-codeowners`, `dynamo-backend-trtllm-codeowners`, `dynamo-tokenspeed-codeowners` |
| sig-performance | Performance and AIPerf | `dynamo-performance-codeowners` |
| sig-simulation | Simulation and AIConfigurator | `dynamo-performance-codeowners` |
| sig-multimodal | Multimodal workloads | `dynamo-multimodal-codeowners` |
| sig-diffusion | Diffusion workloads | `dynamo-diffusion-codeowners` |

The `docs`, `ops`, and `process` areas are program functions that Maintainers coordinate themselves; they have no SIG.
