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

# Dynamo Examples

This directory holds the source for Dynamo's runnable examples: Kubernetes manifests, local launch
scripts, clients, and supporting assets.

To browse, configure, and deploy the examples, use the
[Examples catalog](../docs/fern/pages/recipes/examples/overview.mdx) on the Dynamo docs site. Each
example page renders the manifests and launch scripts from this directory and links to the setup
guide it needs. To get started, see the [Kubernetes quickstart](../docs/fern/pages/kubernetes/getting-started/quickstart.mdx)
or the [CLI quickstart](../docs/fern/pages/cli/getting-started/quickstart.mdx).

> [!IMPORTANT]
> All DynamoGraphDeployment manifests use `nvidia.com/v1beta1`. Other custom resources that do not
> have a `v1beta1` API, such as `DynamoModel`, continue to use their supported version.
> To migrate an existing `v1alpha1` manifest, follow the
> [API version converter instructions](../deploy/utils/README.md).

## Layout

| Path | Contents |
|---|---|
| [`backends/`](backends/) | Per-backend Kubernetes manifests (`deploy/`) and local launch scripts (`launch/`) for vLLM, SGLang, TensorRT-LLM, TokenSpeed, Mocker, and the sample backend |
| [`deployments/`](deployments/) | Cloud-provider manifests (EKS, AKS, GKE, ECS), DGDR examples, and the offline batch integration |
| [`diffusers/`](diffusers/) | FastVideo video diffusion runtime, manifests, and local launcher |
| [`global_planner/`](global_planner/) | GlobalPlanner manifests for shared GPU budgets and multi-pool endpoints |
| [`gpu-sharing/`](gpu-sharing/) | Manifests and scripts behind the GPU sharing results |
| [`nemotron_speech_cascaded_pipeline/`](nemotron_speech_cascaded_pipeline/) | Realtime ASR and streaming TTS adapters for NVIDIA Speech NIMs |
| [`power-aware-budget/`](power-aware-budget/) | Power-cap and power-budget contract fragment for the Planner |
| [`rl/`](rl/) | Reinforcement learning integrations, such as Slime external rollouts |
| [`router/`](router/) | Custom router policy examples and policy-class queue configuration |
| [`custom_backend/`](custom_backend/) | Runtime examples for developers using the Python bindings, such as Hello World and request cancellation |
| [`custom_encoder/`](custom_encoder/) | Custom vision encoder example |
| [`chat_templates/`](chat_templates/) | Vendored Jinja chat templates for tool-call and reasoning parsers |
| [`common/`](common/) | Shared helper scripts used by the launch scripts |

## Contributing

To request an example, open a [GitHub issue](https://github.com/ai-dynamo/dynamo/issues). To
contribute one, open a [pull request](https://github.com/ai-dynamo/dynamo/pulls) and follow the
[example authoring guide](../docs/fern/pages/recipes/examples/_catalog/README.md) to surface it in the
Examples catalog.
