---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Security and Authentication
subtitle: Trust boundaries and a production security checklist for Dynamo on Kubernetes
---

NVIDIA Dynamo is designed to run inside a trusted service network. Its inference frontend and several internal
services do not authenticate end users. Before exposing a Dynamo deployment outside that network, place it behind
a gateway, ingress, or service mesh that terminates Transport Layer Security (TLS) and enforces authentication,
authorization, rate limits, and request policy.

> [!WARNING]
> Do not expose the Dynamo frontend, planner dashboard, standalone router services, NATS, etcd, or ZMQ endpoints
> directly to an untrusted network.

## Security Model

Dynamo provides serving, routing, discovery, and orchestration building blocks. The surrounding platform owns the
user-facing security boundary.

| Boundary | Dynamo provides | Deployment responsibility |
| --- | --- | --- |
| Client to inference API | OpenAI-compatible HTTP APIs and optional [server-side TLS](../components/frontend/configuration.md#tls-and-client-authentication) | Authenticate and authorize clients, terminate TLS or mutual TLS, filter requests, and enforce quotas at a gateway. See the [Gateway API integration](gateway-api/README.mdx). |
| Component to component | [Request-plane](../design-docs/request-plane.md#transport-security) and [event-plane](../design-docs/event-plane.md#transport-security) transports for trusted networks | Keep internal ports private, apply NetworkPolicies, and configure transport authentication and encryption where supported. |
| Pod to Kubernetes API | Kubernetes API access over TLS with ServiceAccounts and role-based access control (RBAC) | Use least-privilege identities and restrict ServiceAccount token access. See [Service Discovery](service-discovery.md#rbac). |
| Workload and data | Execution of selected images, models, backends, and plugins; operational metrics and routing metadata | Approve artifacts, constrain pods, isolate tenants, and control access to telemetry and state. |

> [!NOTE]
> etcd discovery is deprecated for Kubernetes deployments. The Dynamo operator uses Kubernetes-native discovery by
> default. Use etcd only for supported local, bare-metal, or legacy deployments. See
> [Service Discovery](service-discovery.md#kv-store-discovery-etcd).

## Production Security Checklist

- [ ] **Protect the inference API.** Put the frontend behind an authenticating gateway. Configure
  [frontend TLS](../components/frontend/configuration.md#tls-and-client-authentication) when the frontend terminates
  TLS directly, and review the [Gateway API integration](gateway-api/README.mdx) when deploying an inference gateway.
- [ ] **Limit exposed frontend routes.** Keep metrics, health, OpenAPI, Swagger UI, and administration routes private.
  Disable the `/busy_threshold` administration API when it is not needed. See
  [Frontend HTTP Endpoints](../components/frontend/configuration.md#http-endpoints) and
  [Frontend Feature Switches](../components/frontend/configuration.md#frontend-feature-switches).
- [ ] **Isolate internal transports.** Restrict request-plane, event-plane, discovery, and peer ports to the workloads
  that use them. Follow the [Request Plane security guidance](../design-docs/request-plane.md#transport-security) and
  [Event Plane security guidance](../design-docs/event-plane.md#transport-security).
- [ ] **Use least-privilege Kubernetes identities.** Review the operator's cluster scope in the
  [Operator Deployment Guide](dynamo-operator.md), and use role-specific ServiceAccounts as described in
  [Service Discovery](service-discovery.md#use-a-pre-created-serviceaccount).
- [ ] **Constrain pods and secrets.** Set backend-compatible security contexts, use workload identity where available,
  and expose each Secret only to the components that need it. See the [security context](api-reference.md#security-context),
  [ServiceAccounts](api-reference.md#service-accounts), and [image pull secrets](api-reference.md#image-pull-secrets)
  references.
- [ ] **Keep standalone router services private.** The [KV indexer](../components/router/standalone-indexer.md),
  [selection service](../components/router/standalone-selection.md), and
  [slot tracker](../components/router/standalone-slot-tracker.md) do not authenticate callers. Their tenant fields
  partition state; they do not establish caller identity.
- [ ] **Protect KV cache metadata.** KV events contain token IDs and cumulative block hashes. Restrict publishers,
  subscribers, and diagnostic endpoints as described in
  [KV Event Security Considerations](../integrations/kv-events-custom-engines.md#security-considerations).
- [ ] **Secure planner interfaces.** Disable or isolate the live dashboard and explicitly configure plugin
  authentication. See [Planner Diagnostics](../components/planner/planner-guide.md#diagnostics-reports),
  [Plugin Security](../components/planner/planner-guide.md#plugin-security), and the
  [Global Planner namespace policy](../components/planner/global-planner.md#step-2-create-the-control-dgd).
- [ ] **Trust the software supply chain.** Use supported [Dynamo release artifacts](../reference/release-artifacts.md),
  approved container images, and immutable model revisions. Follow the
  [Model and Image Trust](model-deployment-guide.md#production-detail-model-and-image-trust) guidance and review
  model storage and credentials in [Model Caching](model-caching.md).
- [ ] **Control user-derived data.** Review the data captured by [request tracing](../observability/request-tracing.md),
  [audit payload logging](../observability/logging.md#audit-payload-logging-otlp), and
  [multimodal URL fetching](../features/multimodal/README.md#security-url-validation). Apply access and retention
  policies to the most sensitive enabled signal.
- [ ] **Isolate privileged features.** Review the privileged DaemonSet required by
  [Snapshot Restore](snapshot.md#limitations) and any backend-specific host access, such as the
  [TensorRT-LLM EFA security context](../backends/trtllm/trtllm-kv-cache-transfer.md#aws-efa).
- [ ] **Limit resource exhaustion.** Configure frontend request-size limits, gateway rate limits, per-client quotas,
  and [request rejection and load shedding](../fault-tolerance/request-rejection.md).

## Report a Vulnerability

Do not disclose suspected vulnerabilities in a public issue. Follow the private reporting process in the
[Dynamo Security Policy](https://github.com/ai-dynamo/dynamo/blob/main/SECURITY.md).
