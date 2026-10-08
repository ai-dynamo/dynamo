---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Examples
subtitle: Choose a runnable example by environment, backend, and feature.
---

Use this page to pick a starting point. For the complete, maintained inventory of
the user-facing tree under [`examples/`](https://github.com/ai-dynamo/dynamo/tree/main/examples),
including cloud, component, and custom-backend examples, see
[Examples](../../recipes/examples/overview.mdx) in the Recipes tab. The examples in
the repository track `main`; if you run a stable release, check out the matching
release branch before using a script.

## Choose by environment

| Environment | Start here | Full index |
| --- | --- | --- |
| Local or bare-metal host | [CLI Getting Started](../../cli/getting-started/introduction.mdx) and the [vLLM](../../recipes/cli-templates/vllm.mdx), [SGLang](../../recipes/cli-templates/sglang.mdx), and [TensorRT-LLM](../../recipes/cli-templates/tensorrt-llm.mdx) local templates. | [Examples in Recipes](../../recipes/examples/overview.mdx) |
| Kubernetes | [Kubernetes Quickstart](../../kubernetes/getting-started/quickstart.mdx), then the [DGD templates](../../recipes/kubernetes-templates/dgd/vllm.mdx) and [DGDR](../../recipes/kubernetes-templates/dgdr.mdx). | [Examples in Recipes](../../recipes/examples/overview.mdx) |
| Cloud | [Amazon EKS](https://github.com/ai-dynamo/dynamo/tree/main/examples/deployments/EKS), [Google GKE](https://github.com/ai-dynamo/dynamo/tree/main/examples/deployments/GKE), [Azure AKS](https://github.com/ai-dynamo/dynamo/tree/main/examples/deployments/AKS), and [Amazon ECS](https://github.com/ai-dynamo/dynamo/tree/main/examples/deployments/ECS). | [Examples in Recipes](../../recipes/examples/overview.mdx) |
| Custom runtime or component | [Custom backend](https://github.com/ai-dynamo/dynamo/tree/main/examples/custom_backend/hello_world), [router policy examples](../../developer-guide/knowledge-base/modular-components/router/router-examples.md), and [planner](../../developer-guide/knowledge-base/modular-components/planner/planner-examples.md) and [profiler](../../developer-guide/knowledge-base/modular-components/profiler/profiler-examples.md) examples. | [Examples in Recipes](../../recipes/examples/overview.mdx) |

## Backend examples

| Backend | Local templates | Kubernetes templates |
| --- | --- | --- |
| vLLM | [vLLM local deployment examples](../../recipes/cli-templates/vllm.mdx) | [vLLM DGD templates](../../recipes/kubernetes-templates/dgd/vllm.mdx) |
| SGLang | [SGLang local deployment examples](../../recipes/cli-templates/sglang.mdx) | [SGLang DGD templates](../../recipes/kubernetes-templates/dgd/sglang.mdx) |
| TensorRT-LLM | [TensorRT-LLM local deployment examples](../../recipes/cli-templates/tensorrt-llm.mdx) | [TensorRT-LLM DGD templates](../../recipes/kubernetes-templates/dgd/tensorrt-llm.mdx) |

## Component examples

- [Router Examples](../../developer-guide/knowledge-base/modular-components/router/router-examples.md) — Python API usage, Kubernetes examples, and custom routing patterns.
- [Planner Examples](../../developer-guide/knowledge-base/modular-components/planner/planner-examples.md) — custom load predictors and non-Kubernetes scaling environments.
- [Profiler Examples](../../developer-guide/knowledge-base/modular-components/profiler/profiler-examples.md) — DGDR YAMLs and profiling script examples.

Browse the full [examples directory](https://github.com/ai-dynamo/dynamo/tree/main/examples)
in the repository, or use the maintained [Examples index](../../recipes/examples/overview.mdx)
when you want every supported example with its purpose and repository source.
