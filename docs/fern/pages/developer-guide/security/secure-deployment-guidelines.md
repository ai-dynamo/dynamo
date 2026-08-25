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

Dynamo assumes it runs inside a trusted network boundary: external clients reach
only the frontend, through an authenticating gateway, while the internal
communication planes and infrastructure run on a network the deployer isolates.
The sections below explain that boundary and how to harden each component and
plane.

Throughout this guide, **deployer** refers to the team responsible for deploying
and operating Dynamo in a cluster — distinct from the **Dynamo Operator**, the
Kubernetes operator component.

> [!WARNING]
> Do not expose the Dynamo frontend, planner dashboard, standalone router
> services, NATS, etcd, or ZMQ endpoints directly to an untrusted network.

> [!IMPORTANT]
> The `docker compose` files and example manifests in this repository are
> provided for **local development and demonstration only**. They are not a
> hardened, production deployment mechanism. For secure, production deployments,
> use the Kubernetes deployment path.

## Trust Model

Dynamo separates client-facing traffic from internal coordination:

- The **external-facing inference API** — the frontend's OpenAI-compatible HTTP
  endpoint, which serves inference requests to clients.
- The **internal communication planes** — discovery, event, and request planes,
  plus infrastructure services (NATS, ModelExpress, and the NIXL/RDMA
  data-transfer fabric) that components use to find each other and coordinate.

The security posture rests on two assumptions:

1. The **internal communication planes and infrastructure services** are deployed by the
   deployer in a secure fashion and reside within a **trusted network** that
   external clients cannot reach.
2. **External clients reach only the frontend**, and only through a gateway or
   proxy that terminates authentication and TLS.

If both hold, the externally reachable surface is limited to the frontend's
inference API. The sections below explain how to satisfy each assumption.

![Dynamo trust boundary — external clients reach the frontends only through an authenticating gateway; the internal communication planes and backend workers run on the trusted network and must be isolated from outside the cluster.](../../../assets/img/secure-deployment-trust-boundary.svg)

The gateway is the only entry point into the cluster for external clients; it forwards to the
frontends (the external-facing inference API). Everything inside — the frontends,
the internal communication planes, and the workers — runs on the trusted network
and must be protected from access outside the cluster.

## Securing the External-Facing Inference API

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

TLS termination at the gateway secures only the client-to-gateway hop. If traffic
from the gateway to the frontend crosses an untrusted segment, re-encrypt that hop
by enabling the frontend's own server-side TLS (`DYN_TLS_CERT_PATH` /
`DYN_TLS_KEY_PATH`); see the
[frontend TLS configuration](../../reference/components/frontend-configuration.mdx).
This is server-side TLS only — it does not authenticate end users, which remains
the gateway's responsibility.

## Securing the Internal Communication Planes

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
- **etcd (legacy — local, bare-metal, or non-Kubernetes only).** etcd discovery is
  **deprecated for Kubernetes**; the Dynamo Operator uses Kubernetes-native
  discovery by default, and a Kubernetes deployment should not add a second
  network-reachable discovery store. Where you do use etcd, enable authentication —
  never anonymous access on a shared network. Dynamo's etcd client supports
  username/password (`ETCD_AUTH_USERNAME`/`ETCD_AUTH_PASSWORD`) and mutual TLS
  (`ETCD_AUTH_CA`, `ETCD_AUTH_CLIENT_CERT`, `ETCD_AUTH_CLIENT_KEY`).

**Why it matters:** an unauthenticated discovery plane lets any peer on the
network enumerate workers and inject or alter routing metadata. See the
[Discovery Plane](../knowledge-base/concepts/communication-planes/discovery-plane.md)
reference.

### Event plane

Components exchange coordination events — including the KV-cache events used for
KV-aware routing — over the event plane. Dynamo supports two event transports,
selected by `DYN_EVENT_PLANE`:

- **NATS** (preferred where the event plane must be authenticated). Enable NATS
  authentication — `NATS_AUTH_USERNAME`/`NATS_AUTH_PASSWORD`, or
  `NATS_AUTH_TOKEN`, `NATS_AUTH_NKEY`, or `NATS_AUTH_CREDENTIALS_FILE` — and TLS
  (`NATS_TLS_CA_CERT_PATH`). Whether the NATS server *requires* these is enforced
  by the NATS server configuration (for example `tls { ca_file: …; verify: true }`),
  not by Dynamo, so harden the server config as well.
- **ZMQ** is the default event transport in configurations without NATS, and some
  backends (for example, vLLM) publish KV-cache events over ZMQ natively. ZMQ has
  no built-in authentication or encryption. **Treat KV-cache events as sensitive
  request-derived data:** they carry token IDs, cumulative block hashes, and the
  cache/LoRA namespace — given the model tokenizer, the token IDs can reconstruct
  prompt text and the hashes reveal shared prefixes. Keep every ZMQ endpoint on the
  trusted network: bind the broker (`ZMQ_BROKER_XSUB_BIND` / `ZMQ_BROKER_XPUB_BIND`)
  and the advertised KV-event host to cluster-internal addresses, keep intra-node
  sockets on loopback, and restrict who can publish or subscribe with NetworkPolicy.
  Authenticated encryption for ZMQ is possible future hardening but requires
  additional key-management infrastructure.

**Why it matters:** the event plane carries **sensitive request-derived data** —
KV events can reconstruct prompt content — so keep it on the trusted network and
restrict publishers and subscribers. Prefer NATS with authentication and TLS
wherever the event plane crosses a trust boundary. See the
[Event Plane](../knowledge-base/concepts/communication-planes/event-plane.md)
reference.

### Request plane

Requests and KV-cache data move between components over the request plane. The
transport is selected by `DYN_REQUEST_PLANE` — `tcp` (with the NIXL/RDMA fabric
for data transfer) or `nats` — and the TLS/mTLS settings below apply to whichever
you use.

- **Encrypt the request plane with TLS.** For the TCP transport, enable TLS with
  `DYN_TCP_TLS_CERT_PATH` + `DYN_TCP_TLS_KEY_PATH` (server side); clients verify the
  server with `DYN_TCP_TLS_CA_CERT_PATH` (and `DYN_TCP_TLS_SERVER_NAME` to pin the
  expected name). For the NATS transport, enable TLS with `NATS_TLS_CA_CERT_PATH`.
  This protects the confidentiality and integrity of request-plane traffic in
  transit.
- **Authenticate both ends with mutual TLS (mTLS).** Set
  `DYN_TCP_TLS_CLIENT_CA_CERT_PATH` on the server so it requires clients to present
  a certificate — an unauthenticated client is then rejected at the handshake;
  clients present an identity with `DYN_TCP_TLS_CLIENT_CERT_PATH` /
  `DYN_TCP_TLS_CLIENT_KEY_PATH`. For the NATS transport, use
  `NATS_TLS_CLIENT_CERT_PATH` / `NATS_TLS_CLIENT_KEY_PATH`.
- Keep the **NIXL/RDMA** data-transfer fabric on the trusted network.

**Why it matters:** TLS encrypts request-plane traffic, and mTLS additionally
authenticates the client so an unauthenticated peer cannot deliver requests or
data-transfer payloads to a worker. Keep the plane on the trusted network as
defense in depth. See the
[Request Plane](../knowledge-base/concepts/communication-planes/request-plane.md)
reference.

## Restrict or Disable Optional Surfaces

Dynamo exposes optional control and extension surfaces beyond plain inference.
Disable the ones you do not need so that only the required capabilities are
reachable.

### Frontend extensions and admin API

- **Client-controlled routing (`nvext`).** By default the frontend honors an
  `nvext` request extension and routing-override headers that let a client pin a
  request to a specific worker instance. In a multi-tenant or untrusted-client
  setting, the deployer may want to set `DYN_DISABLE_FRONTEND_NVEXT=1` so clients cannot
  target individual
  workers. This drops `request.nvext` at handler entry and ignores the
  routing-override headers.
- **Admin API.** The frontend's HTTP admin API (for example,
  `GET`/`POST /busy_threshold`) is enabled by default. If the deployer does not need to
  change runtime tunables through it, set `DYN_DISABLE_FRONTEND_ADMIN_API=1`.
  Inference, metrics, models, health, and liveness routes are unaffected.
- **Metrics endpoint.** The `/metrics` endpoint is intended for scraping by
  trusted monitoring systems; scope it to your observability stack rather than
  exposing it to untrusted networks.

### Internal control and diagnostic interfaces

Several of Dynamo's internal control and diagnostic interfaces **do not
authenticate callers** — including the worker system server (`/engine/*` on
`DYN_SYSTEM_PORT`), the planner live dashboard, and the standalone KV router
services (indexer, selection, and slot tracker). Several bind on all interfaces.

**Keep every such interface on the trusted network**, expose only the routes a
deployment uses, and disable the ones you don't need. Some surfaces add their own
authentication on top of that isolation — for example, planner plugin registration
authenticates callers and is fail-closed by default (it refuses to start until you
configure a `trusted_sources` auth source, so avoid `allow_unauthenticated` outside
development). Treat that as defense in depth, not a replacement for network
isolation. See each component's documentation for its specific listeners and
controls.

## Securing Model and Backend Code

Dynamo loads models, tokenizers, chat templates, and (depending on the backend)
executable model code. Treat all of these as code that runs with the worker's
privileges.

- **Load models only from trusted sources.** Restrict which model repositories and
  registries workers may pull from, and restrict write access to any shared model
  cache or storage so that only trusted principals can publish artifacts.
- **Be deliberate about remote code execution options.** Framework options that
  execute model-supplied Python (for example, `trust_remote_code`) should be
  enabled only for models you trust. A pre-existing `trust_remote_code` /
  `--trust-remote-code` flag in a deployment template or worker config does **not**
  by itself indicate that a deployer reviewed and approved it — some stock
  templates ship it — so audit templates and pin the provenance of any such flag.
- **Pin what you run.** Pin container images by digest and models to an immutable
  commit/revision (or a reviewed local snapshot), and review generated deployment
  configuration rather than trusting a templated flag to reflect your intent.
- **Validate request-derived values.** When integrating or extending Dynamo,
  validate values taken from a request before using them in security-sensitive
  operations such as outbound network requests, file paths, or deserialization.

## Running with Least Privilege

### Kubernetes deployments

- Use **RBAC** and grant each component's ServiceAccount only the permissions it
  needs. The Dynamo Operator ships with scoped roles; do not broaden them.
- Apply **NetworkPolicies** so the internal communication planes, the workers, and the
  NIXL/RDMA fabric are reachable only by the components that need them, and never
  from outside the cluster boundary.
- Run pods as **non-root** with a read-only root filesystem where possible, drop
  unneeded Linux capabilities, and set resource limits.

### Standalone Docker deployments

Apply standard Docker hardening: restrict container network access to the trusted
segment, set CPU/memory limits, avoid `--privileged`, drop unneeded capabilities,
and run as a non-root user.

## Reporting a Vulnerability

To report a potential security vulnerability in Dynamo, follow the process in
[`SECURITY.md`](https://github.com/ai-dynamo/dynamo/blob/main/SECURITY.md) —
NVIDIA PSIRT via the
[Security Vulnerability Submission Form](https://www.nvidia.com/en-us/support/submit-security-vulnerability/)
or [psirt@nvidia.com](mailto:psirt@nvidia.com). Do not open a public GitHub issue
for security reports.
