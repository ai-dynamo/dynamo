---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Secure Deployment Guidelines
subtitle: Trust model and production hardening checklist for Dynamo deployments
---

NVIDIA Dynamo is designed to run inside a trusted service network. The deployer—the
team that operates Dynamo, distinct from the Dynamo Operator—owns the external
security boundary and the isolation of Dynamo's internal services.

> [!WARNING]
> Do not expose the Dynamo Frontend, Planner dashboard, standalone router services,
> NATS, etcd, or ZMQ endpoints directly to an untrusted network.

> [!IMPORTANT]
> The Docker Compose files and example manifests in this repository are for local
> development and demonstration. Use the Kubernetes deployment path and your
> platform's production security controls for a hardened deployment.

## Trust Model

Dynamo separates client-facing traffic from internal coordination:

- External clients reach the inference API through a gateway that authenticates
  and authorizes them, enforces request policy and quotas, and protects the
  client-to-gateway connection with TLS.
- Frontends, workers, and the discovery, event, and request planes run on a trusted
  network that external clients cannot reach.
- Infrastructure services such as NATS, ModelExpress, and the NIXL/RDMA fabric are
  part of that internal boundary and must be isolated and hardened by the deployer.

![Dynamo trust boundary—external clients reach the frontends only through an authenticating gateway; internal communication planes and backend workers run on the trusted network.](../../../assets/img/secure-deployment-trust-boundary.svg)

The Frontend does not authenticate or authorize end users. A gateway is therefore
the public security boundary, not merely a load balancer. TLS termination at the
gateway protects only the client-to-gateway hop; protect the gateway-to-Frontend
hop separately when it crosses an untrusted segment.

## Security Principles

1. **Expose the smallest possible edge.** Route clients through an authenticating
   gateway and keep Frontend administrative and observability routes private.
2. **Treat every internal control surface as privileged.** Network reachability is
   not authorization; a ClusterIP remains reachable across the cluster. Restrict
   listeners to authorized workloads even when they provide their own authentication.
3. **Encrypt and authenticate transport peers where supported.** Use TLS or mTLS
   for request traffic and authenticated NATS when a brokered plane crosses a trust
   boundary. Keep ZMQ and NIXL/RDMA on the trusted network.
4. **Treat models and operational data as executable or sensitive inputs.** Pin
   images and model revisions, review remote model code, and control access to
   traces, KV events, metrics, logs, and caches.

## Production Checklist

- [ ] **Protect the public request path.** Configure authentication,
  authorization, TLS, rate limits, and request policy at the gateway. If the
  gateway-to-Frontend hop is not already protected by the trusted network or a
  service mesh, enable Frontend TLS and configure the gateway to verify it. See
  [Frontend client authentication and exposure](../../reference/components/frontend-configuration.mdx#client-authentication-and-exposure)
  and [Secure the public request path](../../kubernetes/model-deployment/expose-the-frontend.md#secure-the-public-request-path).
- [ ] **Limit Frontend routes.** Keep metrics, health, model discovery, OpenAPI,
  and Swagger endpoints private. Disable client-controlled routing extensions and
  the `/busy_threshold` administration API when they are not required. See
  [Frontend HTTP endpoints](../../reference/components/frontend-configuration.mdx#http-endpoints)
  and [Frontend feature switches](../../reference/components/frontend-configuration.mdx#frontend-feature-switches).
- [ ] **Secure the request plane.** Keep request and response-stream ports private.
  Configure TCP TLS or mTLS on every component, or configure authenticated TLS to
  NATS when using the NATS request plane. See [Request-plane transport security](../knowledge-base/concepts/communication-planes/request-plane.md#transport-security)
  and the [TLS reference](../../reference/components/tls-configuration.mdx).
- [ ] **Secure the event plane and KV metadata.** Use authenticated TLS to NATS
  when events cross a trust boundary. ZMQ has no built-in authentication or
  encryption and must remain on the trusted network. Treat token IDs, cumulative
  block hashes, and cache namespaces as sensitive request-derived data. See
  [Event-plane transport security](../knowledge-base/concepts/communication-planes/event-plane.md#transport-security)
  and [KV event security considerations](../advanced-customizations/writing-custom-backends/publish-kv-events.md#security-considerations).
- [ ] **Use Kubernetes-native discovery.** Use the operator's default Kubernetes
  discovery and namespace-scoped RBAC. etcd discovery is deprecated on Kubernetes;
  authenticate and encrypt it in supported local, bare-metal, or legacy deployments.
  See [Service discovery](../knowledge-base/kubernetes/kubernetes-operator/service-discovery.md).
- [ ] **Isolate Planner control surfaces.** Disable the unauthenticated live
  dashboard when unused. Configure plugin registration authentication explicitly,
  keep the default Unix-domain registration socket where possible, and do not treat
  the Global Planner's caller namespace allowlist as caller authentication. See
  [Planner security](../knowledge-base/modular-components/planner/planner-guide.md#security),
  [Planner plugin configuration](../../reference/components/planner-configuration.mdx#scheduling-and-plugin-pipeline),
  and [Global Planner management modes](../knowledge-base/modular-components/planner/global-planner-guide.md#step-2-create-the-control-dgd).
- [ ] **Isolate component system servers.** These servers use plaintext HTTP and
  do not authenticate callers. A ClusterIP does not authorize callers; restrict
  health, metrics, metadata, `/engine/*`, and LoRA management routes to authorized
  workloads and namespaces, and verify the NetworkPolicy. See
  [System and metrics configuration](../../reference/observability/environment-variables.mdx#system-and-metrics).
- [ ] **Keep standalone router services private.** The standalone
  [KV indexer](../knowledge-base/modular-components/router/standalone-indexer.md#model-and-routing-group-support),
  [selection service](../knowledge-base/modular-components/router/standalone-selection.md#build-and-launch),
  and [slot tracker](../knowledge-base/modular-components/router/standalone-slot-tracker.md#build-and-launch)
  do not authenticate callers. Their `routing_group` values partition state; they
  do not establish identity.
- [ ] **Pin executable artifacts.** Pin container images by digest and models to
  immutable revisions or reviewed local snapshots. Enable model-supplied remote
  code only after review, and inspect generated DGDs before applying them. See
  [Secure model and image provenance](../../kubernetes/model-deployment/introduction.mdx#secure-model-and-image-provenance).
- [ ] **Control operational data.** Apply access and retention policies to
  [request traces](../../reference/observability/request-traces.mdx),
  [request payload exports](../../reference/observability/logging.mdx#request-payload-export),
  metrics, Planner reports, and model or KV caches. Validate request-derived URLs,
  paths, and serialized values before using them in privileged operations.
- [ ] **Run with least privilege.** Use dedicated ServiceAccounts, narrow RBAC,
  NetworkPolicies, non-root containers, read-only filesystems where compatible,
  and minimal Linux capabilities. Review privileged features such as
  [snapshot restore](../knowledge-base/kubernetes/kubernetes-operator/snapshot.md#limitations),
  and configure gateway quotas plus [request rejection and load shedding](../../kubernetes/fault-tolerance/request-rejection.md).

## Report a Vulnerability

Do not disclose suspected vulnerabilities in a public issue. Follow the private
reporting process in the [Dynamo Security Policy](https://github.com/ai-dynamo/dynamo/blob/main/SECURITY.md).
