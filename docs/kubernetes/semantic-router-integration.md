<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# vLLM Semantic Router Integration

Route requests to a Dynamo deployment through the [vLLM Semantic Router](https://github.com/vllm-project/semantic-router) — an Envoy `ext_proc` filter that classifies each request (intent/category, PII, jailbreak) and selects a model before forwarding to a backend. Pointing that backend at a Dynamo frontend gives you semantic, Mixture-of-Models routing in front of Dynamo's KV-aware disaggregated serving.

The deployment manifests live in the semantic-router repo under [`deploy/kubernetes/dynamo`](https://github.com/vllm-project/semantic-router/tree/main/deploy/kubernetes/dynamo). This page documents how the pieces fit together with Dynamo and the Dynamo-side prerequisites.

## Architecture

```
client
  │  HTTP /v1/chat/completions
  ▼
Envoy Gateway  ──ext_proc──►  Semantic Router
  │                            • intent/category classification + model selection
  │                            • PII / jailbreak detection
  │                            • semantic response cache
  ▼  (selected model)
Dynamo Frontend (DynamoGraphDeployment)
  │  KV-aware routing
  ▼
Prefill / Decode workers
```

Semantic Router is gateway-agnostic (it is a [Gateway API Inference Extension](inference-gateway.md)-style endpoint processor); the reference integration uses **Envoy Gateway**. The Dynamo frontend is an ordinary `DynamoGraphDeployment` — its OpenAI-compatible service (`<release>-frontend:8000`) is the Envoy upstream.

## Prerequisites

- A Kubernetes cluster with the [Dynamo Kubernetes Platform](README.md) installed (operator + CRDs).
- A Dynamo deployment whose frontend Service is reachable in-cluster (the semantic-router `backend_refs` points at `dynamo-vllm-frontend.<namespace>.svc.cluster.local:8000`).
- Envoy Gateway with the `EnvoyPatchPolicy` extension API enabled (installed by the integration's `install_gaie_crd_agentgateway.sh` helper, or `helm install eg oci://docker.io/envoyproxy/gateway-helm`).
- The semantic-router `extproc` image (`ghcr.io/vllm-project/semantic-router/extproc`), which carries the classifier models.

## Deploy

Follow the semantic-router repo's [Dynamo integration README](https://github.com/vllm-project/semantic-router/tree/main/deploy/kubernetes/dynamo). At a high level:

1. **Deploy a Dynamo backend** — any `DynamoGraphDeployment` exposing a frontend Service on port 8000.
2. **Install Envoy Gateway** (with `EnvoyPatchPolicy` enabled).
3. **Install semantic-router** (`helm install semantic-router deploy/helm/semantic-router -f deploy/kubernetes/dynamo/semantic-router-values/values.yaml`), with the `backend_refs` endpoint set to your Dynamo frontend Service.
4. **Apply the gateway resources** (`gwapi-resources.yaml`): the `Gateway`, an `HTTPRoute` whose `backendRefs` is the Dynamo frontend, and an `EnvoyPatchPolicy` wiring `ext_proc` to the semantic-router service.

Send requests to the Envoy Gateway; semantic-router classifies and forwards to the Dynamo frontend, which returns the `x-vsr-selected-*` headers describing the routing decision.

## What it provides

| Capability | Notes |
|---|---|
| Intent/category classification | `x-vsr-selected-category`; routes per the category → decision map |
| Mixture-of-Models selection | with `model: "auto"`, selects a model per category (`x-vsr-selected-model`) |
| System-prompt injection | per-category system prompts (`x-vsr-injected-system-prompt`) |
| Reasoning routing | per-category `use_reasoning` (`x-vsr-selected-reasoning`) |
| Semantic response cache | identical/similar prompts served from cache (`x-vsr-cache-hit`) without hitting Dynamo |
| PII / jailbreak guardrails | classifier-based; **enforcement (blocking) is config + model dependent** — see Notes |

## Testing on a CPU-only cluster (no GPU)

The routing/classification control plane can be validated **without GPUs** by using a Dynamo **mocker** worker as the backend (`python3 -m dynamo.mocker` in the `dynamo-planner` image), which serves an OpenAI-compatible endpoint without running a real model. This exercises the full `Envoy → semantic-router → Dynamo` path (classification, model selection, system prompts, reasoning, cache) end-to-end; only real token generation requires GPU workers.

## Notes and gotchas

- **Helm chart dependencies**: the semantic-router chart pulls optional subcharts (redis, milvus, jaeger, prometheus, grafana). Add those Helm repos and run `helm dependency update` before installing.
- **Models PVC storage class**: the chart requests storage class `standard` for its models volume. On clusters whose default class has another name (for example microk8s `microk8s-hostpath`), create a `standard` StorageClass alias or override the value, or the PVC stays `Pending`.
- **Guardrail enforcement (PII/jailbreak blocking)** depends on the `extproc` image carrying the detector models *and* on a config schema that the running image understands. The published `extproc` image ships only a subset of models by default, and its config schema may lag the repo's reference `config/config.yaml` (whose `routing.modules` enforcement block can be rejected as an unknown field by older images). To enable blocking, use an `extproc` image whose config schema matches the enforcement config you deploy, and ensure the jailbreak/PII detector models are present.
- **Gateway address**: on clusters without a LoadBalancer, the `Gateway` may report `Programmed: False` (no external address) while the Envoy proxy and listener are healthy — use `kubectl port-forward` to the Envoy service to test.

## See also

- [Inference Gateway (GAIE)](inference-gateway.md) — Dynamo's native Gateway API Inference Extension / EPP integration.
- [vLLM Semantic Router](https://github.com/vllm-project/semantic-router)
