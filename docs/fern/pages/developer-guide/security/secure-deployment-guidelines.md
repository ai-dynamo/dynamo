---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Secure Deployment Guidelines
subtitle: Trust model and hardening guidance for production Dynamo deployments
---

NVIDIA Dynamo is a high-performance distributed inference platform designed to be
deployed within a secure, trusted environment. A deployment coordinates many
parts — a frontend, backend workers, a KV router, and internal communication
planes for discovery, events, and requests — across a cluster. This guide
describes Dynamo's trust model and how to secure each part of a deployment.

Dynamo assumes it runs inside a trusted network boundary: untrusted clients reach
only the frontend, through an authenticating gateway, while the internal
communication planes and infrastructure run on a network the operator isolates.
The sections below explain that boundary and how to harden each component and
plane. The [Security Configuration Reference](#security-configuration-reference)
at the end lists every security-related setting in one place.

> [!IMPORTANT]
> The `docker compose` files and example manifests in this repository are
> provided for **local development and demonstration only**. They are not a
> hardened, production deployment mechanism. For secure, production deployments,
> use the Kubernetes deployment path.

## Trust Model

Dynamo separates client-facing traffic from internal coordination:

- The **data plane** — the frontend's OpenAI-compatible HTTP endpoint, which
  serves inference requests to clients.
- The **communication planes** — internal discovery, event, and request planes,
  plus infrastructure services (NATS, ModelExpress, and the NIXL/RDMA
  data-transfer fabric) that components use to find each other and coordinate.

The security posture rests on two assumptions:

1. The **communication planes and infrastructure services** are deployed by the
   operator in a secure fashion and reside within a **trusted network** that
   untrusted clients cannot reach.
2. **Untrusted clients reach only the frontend**, and only through a gateway or
   proxy that terminates authentication and TLS.

If both hold, the externally reachable surface is limited to the frontend's
inference API. The sections below explain how to satisfy each assumption.

## Securing the Data Plane

Do not expose the Dynamo frontend directly to an untrusted network. Deploy it as
a microservice behind a dedicated gateway or proxy that provides:

- **Authentication and authorization** of clients.
- **TLS termination** and encryption in transit.
- **Rate limiting** and request-size limits.
- **Load balancing** across frontend replicas.

On Kubernetes, place a standard ingress or Gateway that you configure for
authentication and TLS in front of the Dynamo Frontend service. The frontend
implements no client authentication; that is the gateway's responsibility. If you
adopt Dynamo's optional [Gateway API routing topology](../../kubernetes/installation/gateway-api-routing.mdx),
note that its Endpoint Picker selects a backend for load and KV-cache reasons and
does not authenticate clients, so it still sits behind your authenticating
gateway.

## Securing the Communication Planes

Dynamo components coordinate over three internal communication planes —
**discovery**, **event**, and **request**. All three are intended to run within
the trusted network and must never be reachable by untrusted clients. Secure each
as follows.

### Discovery plane

Workers register their endpoints and are discovered through the discovery plane.

- **Recommended — Kubernetes-based discovery.** Set
  `DYN_DISCOVERY_BACKEND=kubernetes`. Dynamo discovers workers through RBAC-gated
  custom resources; reads and writes are authorized by the Kubernetes API server
  using each pod's ServiceAccount, so there is no anonymous, network-reachable
  discovery store to protect.
- **Alternative — authenticated etcd.** If you use etcd for discovery, enable
  authentication; never run it with anonymous access on a shared network.
  Dynamo's etcd client supports username/password
  (`ETCD_AUTH_USERNAME`/`ETCD_AUTH_PASSWORD`) and mutual TLS (`ETCD_AUTH_CA`,
  `ETCD_AUTH_CLIENT_CERT`, `ETCD_AUTH_CLIENT_KEY`); provide the matching
  credentials to every component.

**Why it matters:** an unauthenticated discovery plane lets any peer on the
network enumerate workers and inject or alter routing metadata. See the
[Discovery Plane](../knowledge-base/concepts/communication-planes/discovery-plane.md)
reference.

### Event plane

Components exchange coordination events over NATS.

- Enable **NATS authentication and TLS**. Running NATS without authentication is
  acceptable only on a private segment that untrusted parties cannot reach.

**Why it matters:** an unauthenticated event plane allows forged events and
eavesdropping on cluster coordination. See the
[Event Plane](../knowledge-base/concepts/communication-planes/event-plane.md)
reference.

### Request plane

Requests and KV-cache data move between components over the request plane (TCP,
and the NIXL/RDMA fabric for data transfer).

> [!NOTE]
> Mutual TLS for the TCP request plane and NATS is available once transport mTLS
> support lands; the settings below apply from that release onward. Until then,
> keep the request plane on the trusted network.

- Enable **mutual TLS (mTLS)** on the TCP request plane and NATS so both ends of
  every connection authenticate. A server configured with a client CA
  (`DYN_TCP_TLS_CLIENT_CA_CERT_PATH`) rejects any client that does not present a
  valid certificate; clients present an identity via
  `DYN_TCP_TLS_CLIENT_CERT_PATH`/`_KEY_PATH` (and
  `NATS_TLS_CLIENT_CERT_PATH`/`_KEY_PATH` for NATS).
- Keep the **NIXL/RDMA** data-transfer fabric on the trusted network.

**Why it matters:** without mutual authentication, any peer that can reach a
worker on the request plane can deliver requests or data-transfer payloads to it.
See the [Request Plane](../knowledge-base/concepts/communication-planes/request-plane.md)
reference.

## Restrict or Disable Optional Surfaces

Dynamo exposes optional control and extension surfaces beyond plain inference.
Disable the ones you do not need so that only the required capabilities are
reachable.

### Frontend extensions and admin API

- **Client-controlled routing (`nvext`).** By default the frontend honors an
  `nvext` request extension and routing-override headers that let a client pin a
  request to a specific worker instance. In a multi-tenant or untrusted-client
  setting, set `DYN_DISABLE_FRONTEND_NVEXT=1` so clients cannot target individual
  workers. This drops `request.nvext` at handler entry and ignores the
  routing-override headers.
- **Admin API.** The frontend's HTTP admin API (for example,
  `GET`/`POST /busy_threshold`) is enabled by default. If operators do not need to
  change runtime tunables through it, set `DYN_DISABLE_FRONTEND_ADMIN_API=1`.
  Inference, metrics, models, health, and liveness routes are unaffected.
- **Metrics endpoint.** The `/metrics` endpoint is intended for scraping by
  trusted monitoring systems; scope it to your observability stack rather than
  exposing it to untrusted networks.

### Worker control surface

Each worker runs a system server (the `/engine/*` routes on `DYN_SYSTEM_PORT`)
that exposes advanced control operations — profiling, memory management, and
weight updates. Keep this port on the trusted network only, and expose only the
routes a deployment actually uses.

## Securing Model and Backend Code

Dynamo loads models, tokenizers, chat templates, and (depending on the backend)
executable model code. Treat all of these as code that runs with the worker's
privileges.

- **Load models only from trusted sources.** Restrict which model repositories and
  registries workers may pull from, and restrict write access to any shared model
  cache or storage so that only trusted principals can publish artifacts.
- **Be deliberate about remote code execution options.** Framework options that
  execute model-supplied Python (for example, `trust_remote_code`) should be
  enabled only for models you trust.
- **Validate request-derived values.** When integrating or extending Dynamo,
  validate values taken from a request before using them in security-sensitive
  operations such as outbound network requests, file paths, or deserialization.

## Running with Least Privilege

### Kubernetes deployments

- Use **RBAC** and grant each component's ServiceAccount only the permissions it
  needs. The Dynamo operator ships with scoped roles; do not broaden them.
- Apply **NetworkPolicies** so the communication planes, the workers, and the
  NIXL/RDMA fabric are reachable only by the components that need them, and never
  from outside the cluster boundary.
- Run pods as **non-root** with a read-only root filesystem where possible, drop
  unneeded Linux capabilities, and set resource limits.

### Standalone Docker deployments

Apply standard Docker hardening: restrict container network access to the trusted
segment, set CPU/memory limits, avoid `--privileged`, drop unneeded capabilities,
and run as a non-root user.

## Security Configuration Reference

The security-related settings, grouped by the part of the deployment they harden.
Set restrictive values in your production manifests.

**Frontend (data plane)**

| Setting | Purpose | Default |
|---------|---------|---------|
| `DYN_DISABLE_FRONTEND_NVEXT=1` | Strip `nvext` client-controlled routing and override headers | off (`nvext` enabled) |
| `DYN_DISABLE_FRONTEND_ADMIN_API=1` | Disable the `/busy_threshold` admin API | off (admin enabled) |

**Discovery plane**

| Setting | Purpose | Default |
|---------|---------|---------|
| `DYN_DISCOVERY_BACKEND=kubernetes` | RBAC-gated custom-resource discovery (no external store) | — |
| `ETCD_AUTH_USERNAME` / `ETCD_AUTH_PASSWORD` | etcd username/password authentication | unset |
| `ETCD_AUTH_CA` / `ETCD_AUTH_CLIENT_CERT` / `ETCD_AUTH_CLIENT_KEY` | etcd mutual TLS | unset |

**Event and request planes** — *available once transport mTLS support lands*

| Setting | Purpose | Default |
|---------|---------|---------|
| `DYN_TCP_TLS_CLIENT_CA_CERT_PATH` | Require and verify client mTLS on the request plane | unset |
| `DYN_TCP_TLS_CLIENT_CERT_PATH` / `_KEY_PATH` | Client identity for request-plane mTLS | unset |
| `NATS_TLS_CLIENT_CERT_PATH` / `_KEY_PATH` | Client identity for NATS mTLS | unset |

## Reporting a Vulnerability

To report a potential security vulnerability in Dynamo, follow the process in
[`SECURITY.md`](https://github.com/ai-dynamo/dynamo/blob/main/SECURITY.md) —
NVIDIA PSIRT via the
[Security Vulnerability Submission Form](https://www.nvidia.com/en-us/support/submit-security-vulnerability/)
or [psirt@nvidia.com](mailto:psirt@nvidia.com). Do not open a public GitHub issue
for security reports.
