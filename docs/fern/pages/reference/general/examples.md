---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Examples
subtitle: Reference deployments for SGLang, TensorRT-LLM, vLLM, and custom Dynamo backends.
---

Use these examples when you want a concrete starting point instead of a conceptual guide. The examples in the Dynamo repository track `main`; if you are using a stable release, prefer the examples from the matching release branch or the versioned recipes in these docs.

## Start Here

- [Hello World custom backend](https://github.com/ai-dynamo/dynamo/tree/main/examples/custom_backend/hello_world) — a minimal GPU-unaware graph that demonstrates Dynamo's component model.
- [Kubernetes Quickstart](../../kubernetes/getting-started/quickstart.mdx) — run a model on Kubernetes with the current recommended path.
- [CLI Getting Started](../../cli/getting-started/introduction.mdx) — run Dynamo locally from the CLI.

## Backend Examples

- [vLLM local deployment examples](../../recipes/cli-templates/vllm.mdx) — CLI launch patterns for vLLM.
- [SGLang local deployment examples](../../recipes/cli-templates/sglang.mdx) — CLI launch patterns for SGLang.
- [TensorRT-LLM local deployment examples](../../recipes/cli-templates/tensorrt-llm.mdx) — CLI launch patterns for TensorRT-LLM.

## Kubernetes Templates

- [vLLM DGD templates](../../recipes/kubernetes-templates/dgd/vllm.mdx) — aggregated, disaggregated, and multinode Kubernetes manifests.
- [SGLang DGD templates](../../recipes/kubernetes-templates/dgd/sglang.mdx) — aggregated and disaggregated Kubernetes manifests.
- [TensorRT-LLM DGD templates](../../recipes/kubernetes-templates/dgd/tensorrt-llm.mdx) — TensorRT-LLM Kubernetes manifests.
- [DGDR template](../../recipes/kubernetes-templates/dgdr.mdx) — profiler and planner driven deployment request example.

## Component Examples

- [Router Examples](../../developer-guide/knowledge-base/modular-components/router/router-examples.md) — Python API usage, Kubernetes examples, and custom routing patterns.
- [Planner Examples](../../developer-guide/knowledge-base/modular-components/planner/planner-examples.md) — custom load predictors and non-Kubernetes scaling environments.
- [Profiler Examples](../../developer-guide/knowledge-base/modular-components/profiler/profiler-examples.md) — DGDR YAMLs and profiling script examples.

## Repository Examples

Browse the full [examples directory](https://github.com/ai-dynamo/dynamo/tree/main/examples) for source-controlled examples that may not yet have a polished docs page.
