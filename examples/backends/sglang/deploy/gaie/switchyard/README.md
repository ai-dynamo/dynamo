<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Switchyard model routing with SGLang and Dynamo GAIE

This directory holds a Kustomize add-on that runs a Switchyard PreProc on an existing GAIE Gateway
and routes each request to one of two SGLang model pools.

For prerequisites, deployment steps, and configuration, see
[Route Between Models with Switchyard](../../../../../../docs/fern/pages/kubernetes/kv-aware-routing/switchyard.md).

## Contents

| Path | Description |
|---|---|
| `http-routes.yaml` | HTTPRoutes that bind each model name to its `InferencePool` and set the gateway request timeout |
| `kustomization.yaml` | Kustomization for the PreProc add-on: Gateway namespace, PreProc image, and `routes.toml` ConfigMap |
| `preproc-policy.yaml` | agentgateway PreRouting policy that sends Gateway traffic through PreProc |
| `preproc.yaml` | Switchyard PreProc Deployment and Service |
| `routes.toml` | Switchyard route, model target, and routing policy configuration |
