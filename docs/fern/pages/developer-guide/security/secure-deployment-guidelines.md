---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Secure Deployment Guidelines
---

Dynamo is a distributed inference framework built from multiple cooperating
components — a frontend, backend workers, a KV-aware router, and a control plane
(service discovery and messaging). This document describes how to deploy Dynamo
securely and the trust boundaries you are responsible for when you operate it.

Dynamo is designed to run **inside a trusted network boundary**. The guidance
below explains where that boundary sits, what Dynamo protects on its own, and
what the deployer must protect.

> [!IMPORTANT]
> The `docker compose` files and example manifests in this repository are
> provided for **local development and demonstration only**. They are not a
> hardened, production deployment mechanism. For secure, production deployments,
> use the Kubernetes deployment path described below.

## Trust Model

Dynamo separates two kinds of traffic:

- The **data plane** — the frontend's OpenAI-compatible HTTP endpoint, which
  serves inference requests to clients.
- The **control plane** — service discovery (etcd or Kubernetes custom
  resources), messaging (NATS), and model/weight distribution
  (ModelExpress/NIXL). Components use the control plane to find each other and
  coordinate.

The security posture rests on two assumptions:

1. The **control plane and infrastructure services** (etcd, NATS, ModelExpress,
   and the RDMA/NIXL data-transfer fabric) are deployed by the operator in a
   secure fashion and reside within a **secure network** that untrusted clients
   cannot reach.
2. **Untrusted clients reach only the frontend**, and only through a gateway or
   proxy that terminates authentication and TLS (see below).

If both hold, the externally reachable surface is limited to the frontend's
inference API. The sections below explain how to satisfy each assumption.

## Deploy Behind a Secure Proxy or Gateway

Do not expose the Dynamo frontend directly to an untrusted network. Deploy it as
a microservice behind a dedicated gateway or proxy that provides:

- **Authentication and authorization** of clients.
- **TLS termination** and encryption in transit.
- **Rate limiting** and request-size limits.
- **Load balancing** across frontend replicas.

On Kubernetes, use the [Dynamo inference gateway](../../kubernetes/installation/gateway-api-routing.mdx)
or a standard ingress/gateway in front of the frontend service. The frontend
itself implements no client authentication; that is the gateway's
responsibility.

## Secure the Control Plane

The control plane is the highest-value part of the trust boundary: components
discover each other and exchange routing metadata through it. Keep it off any
network that untrusted clients can reach.

### Prefer Kubernetes-based discovery

For production, set `DYN_DISCOVERY_BACKEND=kubernetes`. In this mode Dynamo
discovers workers through **RBAC-gated custom resources** rather than etcd. Reads
and writes are authorized by the Kubernetes API server using each pod's
ServiceAccount, so there is no anonymous, network-reachable discovery store to
protect. See the [Discovery Plane](../knowledge-base/concepts/communication-planes/discovery-plane.md)
reference for details.

### If you use etcd, authenticate it

When discovery uses etcd (`DYN_DISCOVERY_BACKEND=etcd`, the default outside
Kubernetes), do not run etcd with anonymous access on a shared network. Dynamo's
etcd client supports authentication:

| Variable | Purpose |
|----------|---------|
| `ETCD_AUTH_USERNAME` / `ETCD_AUTH_PASSWORD` | Username/password authentication |
| `ETCD_AUTH_CA` | CA certificate path for TLS |
| `ETCD_AUTH_CLIENT_CERT` / `ETCD_AUTH_CLIENT_KEY` | Client certificate and key for mutual TLS |

Enable authentication (and, ideally, mutual TLS) on the etcd cluster and provide
the matching credentials to every Dynamo component. Apply the same principle to
NATS (enable authentication and TLS) and to any ModelExpress deployment.

> [!WARNING]
> Running etcd or NATS with authentication disabled is acceptable only on a
> private network segment that untrusted parties cannot reach. Anyone who can
> connect to an unauthenticated control plane can enumerate workers and inject
> or alter routing metadata.

## Restrict or Disable Optional Frontend Features

The frontend exposes optional surfaces beyond plain inference. Disable the ones
you do not need so that only the required capabilities are reachable.

### Client-controlled routing (`nvext`)

By default the frontend honors an `nvext` request extension and routing-override
headers that let a client pin a request to a specific worker instance. In a
multi-tenant or untrusted-client setting, disable this so clients cannot target
individual workers:

```bash
export DYN_DISABLE_FRONTEND_NVEXT=1
```

This drops `request.nvext` at handler entry, ignores the routing-override headers
(`x-dynamo-worker-instance-id`, `x-dynamo-prefill-instance-id`,
`x-dynamo-dp-rank`, `x-dynamo-prefill-dp-rank`), and ignores the response-side
`extra_fields` opt-in.

### Admin API

The frontend's HTTP admin API (for example, `GET`/`POST /busy_threshold`) is
enabled by default. If operators do not need to change runtime tunables through
it, disable it:

```bash
export DYN_DISABLE_FRONTEND_ADMIN_API=1
```

Inference, metrics, models, health, and liveness routes are unaffected.

### Metrics endpoint

The metrics/`/metrics` endpoint is intended for scraping by trusted monitoring
systems. Do not expose it to untrusted networks; scope it to your observability
stack.

## Securing Model and Backend Code

Dynamo loads models, tokenizers, chat templates, and — depending on the backend
— executable model code. Treat all of these as code that runs with the worker's
privileges.

- **Load models only from trusted sources.** Restrict which model repositories
  and registries workers may pull from, and restrict write access to any shared
  model cache or storage so that only trusted principals can publish artifacts.
- **Be deliberate about remote code execution options.** Framework options that
  execute model-supplied Python (for example, `trust_remote_code`) should be
  enabled only for models you trust.
- **Validate request-derived values.** When integrating or extending Dynamo,
  validate values taken from a request before using them in security-sensitive
  operations such as outbound network requests, file paths, or deserialization.

## Running with Least Privilege

### Kubernetes deployments

Follow standard Kubernetes hardening practices:

- Use **RBAC** and grant each component's ServiceAccount only the permissions it
  needs. The Dynamo operator ships with scoped roles; do not broaden them.
- Apply **NetworkPolicies** so that the control plane (etcd/NATS), the workers,
  and the RDMA/NIXL fabric are reachable only by the components that need them,
  and never from outside the cluster boundary.
- Run pods as **non-root** with a read-only root filesystem where possible, drop
  unneeded Linux capabilities, and set resource limits.

### Standalone Docker deployments

When running containers directly, apply standard Docker hardening: restrict
container network access to the trusted segment, set CPU/memory limits, avoid
`--privileged`, drop unneeded capabilities, and run as a non-root user.

## Reporting a Vulnerability

To report a potential security vulnerability in Dynamo, follow the process in
[`SECURITY.md`](https://github.com/ai-dynamo/dynamo/blob/main/SECURITY.md) —
NVIDIA PSIRT via the
[Security Vulnerability Submission Form](https://www.nvidia.com/en-us/support/submit-security-vulnerability/)
or [psirt@nvidia.com](mailto:psirt@nvidia.com). Do not open a public GitHub issue
for security reports.
