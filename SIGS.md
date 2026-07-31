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

Special Interest Groups (SIGs) are open, standing groups that coordinate work within one domain of the project: roadmap discussion, design review, and cross-area coordination. Anyone may join and participate. The SIG model - what SIGs are, the SIG Lead role, and how SIGs are created, merged, or retired - is defined in [GOVERNANCE.md](GOVERNANCE.md); this file is the living roster. SIG Leads are published at ratification.

| SIG | Scope | CODEOWNERS Groups |
| :---- | :---- | :---- |
| sig-core | Core runtime, frontend, and observability | `dynamo-runtime-codeowners`, `dynamo-frontend-codeowners`, `dynamo-observability-codeowners` |
| sig-router | Router and the Inference Gateway Endpoint Picker (EPP) | `dynamo-router-codeowners`, `dynamo-epp-codeowners` |
| sig-memory-transport | KV/memory transport and storage, including network fabric enablement | `dynamo-kv-memory-codeowners`, `dynamo-efa-codeowners` |
| sig-agents | Agentic workloads | `dynamo-agents-codeowners` |
| sig-rl | Reinforcement learning integrations | `dynamo-rl-codeowners` |
| sig-hardware | Hardware platform enablement and optimization, across NVIDIA and non-NVIDIA accelerators | `dynamo-xpu-codeowners` |
| sig-deploy | Deploy path, including cloud-provider platform enablement | `dynamo-operator-codeowners`, `dynamo-efa-codeowners` |
| sig-scaling | Scaling and lifecycle | `dynamo-planner-codeowners`, `dynamo-fault-tolerance-codeowners`, `dynamo-gms-codeowners` |
| sig-engines | Backend engine integrations (vLLM, SGLang, TensorRT-LLM) | `dynamo-backend-vllm-codeowners`, `dynamo-backend-sglang-codeowners`, `dynamo-backend-trtllm-codeowners`, `dynamo-tokenspeed-codeowners` |
| sig-perf-sim | Performance, AIPerf, and simulation (AIConfigurator) | `dynamo-performance-codeowners` |
| sig-multimodal | Multimodal and diffusion workloads | `dynamo-multimodal-codeowners`, `dynamo-diffusion-codeowners` |

The `docs`, `ops`, and `process` areas are program functions that Maintainers coordinate themselves; they have no SIG.

## Ecosystem Projects

Several projects in the [ai-dynamo](https://github.com/ai-dynamo) organization live in their own repositories. Each is coordinated by the SIG whose domain it serves: the SIG is where its roadmap and its integration with Dynamo are discussed. Code review and merge authority stay with each repository's own owners.

| Project | Repository | SIG |
| :---- | :---- | :---- |
| AIPerf | [ai-dynamo/aiperf](https://github.com/ai-dynamo/aiperf) | sig-perf-sim |
| AIConfigurator | [ai-dynamo/aiconfigurator](https://github.com/ai-dynamo/aiconfigurator) | sig-perf-sim |
| NIXL | [ai-dynamo/nixl](https://github.com/ai-dynamo/nixl) | sig-memory-transport |
| FlexTensor | [ai-dynamo/flextensor](https://github.com/ai-dynamo/flextensor) | sig-memory-transport |
| ModelExpress | [ai-dynamo/modelexpress](https://github.com/ai-dynamo/modelexpress) | sig-rl, sig-scaling |
| Grove | [ai-dynamo/grove](https://github.com/ai-dynamo/grove) | sig-deploy, sig-scaling |
| Snapshot | [ai-dynamo/snapshot](https://github.com/ai-dynamo/snapshot) | sig-scaling, sig-deploy |
| OpenEngine | [ai-dynamo/openengine](https://github.com/ai-dynamo/openengine) | sig-engines, sig-core |
| AITune | [ai-dynamo/aitune](https://github.com/ai-dynamo/aitune) | sig-multimodal |
