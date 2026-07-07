---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Security and Authentication
subtitle: Security boundaries, authentication responsibilities, and production hardening for Dynamo on Kubernetes
---

NVIDIA Dynamo is designed to run inside a trusted service network. Its inference frontend and several internal
services do not authenticate end users. Before exposing a Dynamo deployment outside that network, place it behind
a gateway, ingress, or service mesh that terminates Transport Layer Security (TLS) and enforces authentication,
authorization, rate limits, and request policy.

> [!WARNING]
> Do not expose the Dynamo frontend, planner dashboard, standalone router services, NATS, etcd, or ZMQ endpoints
> directly to an untrusted network.

## Security Responsibilities

Dynamo provides serving, routing, discovery, and orchestration building blocks. The platform around Dynamo owns
the user-facing security boundary.

| Boundary | Dynamo behavior | Deployment responsibility |
| --- | --- | --- |
| Client to inference API | Serves OpenAI-compatible HTTP APIs; optional server-side TLS is available | Authenticate and authorize clients, terminate TLS or mutual TLS (mTLS), enforce quotas, and filter requests at a gateway |
| Component to component | Uses Kubernetes discovery, NATS, etcd, TCP, or ZMQ depending on configuration | Keep internal ports private, apply NetworkPolicies, and configure transport authentication where available |
| Pod to Kubernetes API | Uses ServiceAccounts and role-based access control (RBAC) for discovery and operator functions | Use least-privilege ServiceAccounts and workload identity; restrict token access to pods that need it |
| Workload execution | Runs selected containers, models, backends, and plugins | Approve and pin artifacts, constrain pods, and isolate privileged workloads |
| Telemetry and state | Emits metrics, logs, traces, routing metadata, and optional audit data | Restrict access, define retention, and prevent sensitive payload collection unless required |

## Production Hardening Checklist

- Put every client-facing endpoint behind an authenticating gateway or service mesh.
- Restrict component, discovery, event, metrics, and administration ports with Kubernetes NetworkPolicies and
  infrastructure firewalls.
- Configure NATS or etcd authentication and transport encryption when those services cross a pod or node trust
  boundary.
- Use dedicated, least-privilege ServiceAccounts and narrowly scoped Secrets for each workload role.
- Apply non-root execution, seccomp, dropped capabilities, read-only filesystems where supported, and admission
  policies appropriate for the selected backend.
- Pin Dynamo images by digest and models by an immutable revision; review model code and plugins as executable
  dependencies.
- Treat KV events, traces, audit records, dashboards, and configuration dumps as sensitive operational data.
- Isolate privileged features and mutually untrusted tenants into separate security boundaries.

## Protect External Access

The frontend listens on `0.0.0.0` by default. It can terminate server-side TLS when a certificate and key are
configured, but it does not provide end-user authentication or authorization. Its inference, health, metrics,
OpenAPI, Swagger UI, and administration routes share the same listener.

Use the following controls at the external boundary:

- Route traffic through a gateway that validates credentials before forwarding inference requests. The
  [Dynamo Gateway API integration](gateway-api/README.mdx) is one deployment option; configure the chosen gateway's
  authentication and authorization features separately.
- Expose only required paths. Keep `/metrics`, `/openapi.json`, `/docs`, and health endpoints on private routes or
  explicitly protect them at the gateway.
- Set `DYN_DISABLE_FRONTEND_ADMIN_API=true` unless runtime updates to `/busy_threshold` are required. If the route
  remains enabled, authorize it separately from inference traffic.
- Enforce request-size, concurrency, timeout, and rate limits before requests consume tokenizer, CPU, GPU, or
  storage resources.
- Preserve the original client identity in trusted proxy metadata, and discard client-supplied identity or
  routing headers that the gateway did not create.

See the [Frontend Configuration Reference](../components/frontend/configuration.md) for listener, TLS, endpoint,
and feature-switch settings.

## Protect Internal Services

Internal Dynamo traffic is not a substitute for a network security boundary. Use default-deny ingress and egress
policies, then allow only the required communication between workload roles, discovery services, and monitoring
systems. Dynamo does not create topology-specific NetworkPolicies automatically because allowed flows vary by
backend and deployment mode.

### NATS

When NATS is used for the request or event plane, configure one supported credential mechanism:

- `NATS_AUTH_USERNAME` and `NATS_AUTH_PASSWORD`
- `NATS_AUTH_TOKEN`
- `NATS_AUTH_NKEY`
- `NATS_AUTH_CREDENTIALS_FILE`

If none of these variables is set, the client attempts the default username and password `user` / `user`. This is
not a production authentication control. The bundled platform chart also disables NATS TLS by default. Enable TLS
in NATS, distribute unique credentials through Kubernetes Secrets, and restrict the service to Dynamo clients.

### etcd

When using etcd discovery, configure username/password authentication with `ETCD_AUTH_USERNAME` and
`ETCD_AUTH_PASSWORD`, or configure mutual TLS with `ETCD_AUTH_CA`, `ETCD_AUTH_CLIENT_CERT`, and
`ETCD_AUTH_CLIENT_KEY`. The bundled etcd chart does not enable role-based authentication by default. Enable etcd
authentication and TLS, limit key-prefix permissions, and isolate the client port.

For Kubernetes deployments, [Kubernetes-native service discovery](service-discovery.md) avoids operating a
separate discovery datastore, but its pods need appropriate Kubernetes API permissions.

### Standalone and Auxiliary Services

The standalone [KV indexer](../components/router/standalone-indexer.md),
[worker selector](../components/router/standalone-selection.md), and
[slot tracker](../components/router/standalone-slot-tracker.md) bind to all interfaces in documented examples and
do not provide an authentication layer. The planner live dashboard and internal ZMQ transports also assume a
trusted network. Bind them to private interfaces where possible and restrict access with Services,
NetworkPolicies, a service mesh, or an authenticated proxy.

## Isolate KV Cache Metadata and Tenants

KV event streams and router state can contain token IDs, cumulative block hashes, model names, worker topology,
and request correlation data. These values can reveal workload characteristics even when raw prompts are absent.
Restrict event-plane subscribers and router administration endpoints, encrypt cross-node traffic, and use short
retention periods for captured events. See [Integrating Custom Engines with KV Events](../integrations/kv-events-custom-engines.md)
for the event schema.

`tenant_id` and similar request metadata select a routing or cache partition; they do not authenticate the caller.
Set tenant identity only after authentication, prevent clients from overriding trusted metadata, and verify that
every selected backend preserves the intended partition. Likewise, cache salts are not a universal isolation
control across all engines. Use separate deployments, namespaces, credentials, and network policies when tenants
do not trust one another.

## Constrain Kubernetes Identities and Pods

The cluster-wide operator installation is the recommended and default mode, so the operator has permissions
across its managed cluster scope. Protect its ServiceAccount and admission webhooks, restrict who can create or
modify Dynamo custom resources, and review the installed RBAC before use in a shared cluster. The
[Operator Deployment Guide](dynamo-operator.md) and [webhook documentation](webhooks.md) describe these
components.

Normal frontend and worker components can use a pre-created ServiceAccount through
`podTemplate.spec.serviceAccountName`. Use this to attach a cloud workload identity or a role-specific Kubernetes
identity, as shown in [Service Discovery](service-discovery.md). Grant only the discovery and resource access that
the component needs. Disable automatic ServiceAccount token mounting only for pods that do not need Kubernetes
API access.

Dynamo applies filesystem ownership defaults to generated workloads, but it does not select every Pod Security
setting on behalf of each backend. Set a security context in the component's `podTemplate` and test it with the
chosen runtime image. A starting point for images that support non-root execution is:

```yaml
podTemplate:
  spec:
    securityContext:
      runAsNonRoot: true
      runAsUser: 1000
      runAsGroup: 1000
      fsGroup: 1000
      seccompProfile:
        type: RuntimeDefault
    containers:
      - name: main
        securityContext:
          allowPrivilegeEscalation: false
          capabilities:
            drop:
              - ALL
```

Where supported, also use a read-only root filesystem, resource limits, restricted volume types, and Kubernetes
Pod Security Admission. Some GPU, RDMA, shared-memory, or backend configurations require additional devices,
memory locking, or writable paths; grant only the specific exception they require.

## Handle Secrets and Workload Credentials

- Create separate Secrets for model registry credentials, NATS or etcd credentials, TLS keys, and cloud access.
  Limit each Secret to the ServiceAccounts and workload roles that need it.
- A component's `envFromSecret` setting exposes every key in the referenced Secret as an environment variable.
  Prefer a narrowly scoped Secret rather than reusing a namespace-wide bundle. See the
  [API Reference](api-reference.md) for the current component schema.
- Use workload identity instead of static cloud keys where the platform supports it.
- Review automatic image pull secret discovery. Set the
  `nvidia.com/disable-image-pull-secret-discovery: "true"` component annotation and provide explicit
  `imagePullSecrets` when automatic propagation is broader than intended.
- Do not place credentials in command-line arguments, model names, labels, or ordinary configuration values.
  Protect configuration dumps and support bundles, and verify their contents before sharing them.

## Trust Models, Images, and Plugins

Container images, model repositories, tokenizers, custom engines, and planner plugins can all execute code in or
alongside the serving workload. Treat them as software supply-chain inputs:

- Use supported Dynamo releases and pin production images by digest. The
  [release artifacts](../reference/release-artifacts.md) page lists published artifacts.
- Pin remote models to an immutable commit or revision, verify their source and license, and control who can
  replace models in object storage or persistent volumes. Model caching improves availability and startup time;
  it does not establish model trust. See [Model Caching](model-caching.md) and the
  [Model Deployment Guide](model-deployment-guide.md).
- Do not assume one Dynamo-level option disables remote model code for every backend. Review the trust and
  remote-code settings of the selected backend, tokenizer, and multimodal processor.
- Run third-party plugins out of process where practical. An in-process planner plugin has the planner process's
  privileges and access to its memory.

Planner plugin authentication must be explicitly configured. Set
`plugin_registration.auth.trusted_sources: [static_secret]` and inject the corresponding static secrets from a
mounted Kubernetes Secret. Legacy or omitted configuration can fall back to unauthenticated registration. External
gRPC plugin transport is plaintext and fails closed unless insecure TCP is explicitly allowed; use a pod-local
Unix socket or place the connection inside an authenticated, encrypted channel. See the
[Planner Guide](../components/planner/planner-guide.md) for plugin configuration.

For the Global Planner, restrict `--managed-namespaces` to an allowlist and protect its transport endpoint. An
omitted namespace allowlist accepts caller-provided namespaces; it is a routing scope, not an authorization check.
See [Global Planner](../components/planner/global-planner.md).

## Protect Multimodal Fetching

Multimodal requests can ask the frontend to fetch remote media. Dynamo allows HTTPS and data URLs by default,
blocks private, loopback, and link-local destinations, and revalidates redirects. Keep those protections enabled
and add gateway-level URL and payload policy for public services.

Never set `DYN_MM_ALLOW_INTERNAL=1` on a service that accepts untrusted requests. It permits access to internal
addresses and weakens server-side request forgery protections. Review all controls in
[Multimodal Security](../features/multimodal/README.md#security-url-validation).

## Protect Observability Data

Request tracing omits prompts, responses, and tool arguments, but trace records still contain request IDs,
session-derived hashes, model names, timing, token counts, and routing decisions. Audit sinks can intentionally
capture request or response payloads. Apply access controls and retention limits according to the most sensitive
enabled signal.

- Send telemetry only to approved collectors over protected channels.
- Keep planner dashboards, metrics endpoints, and operator metrics private. Operator metrics authentication uses
  Kubernetes TokenReview and SubjectAccessReview; see [Operator Metrics](observability/operator-metrics.md).
- Avoid secrets and raw user content in logs, labels, metric dimensions, and custom trace attributes.
- Review [Request Tracing](../observability/request-tracing.md) and [Logging](../observability/logging.md) before
  enabling production exports.

## Isolate Privileged Features

Some optional features require broader host access. Snapshot support deploys a privileged DaemonSet to prepare
host resources; review the security considerations in [Snapshot Restore](snapshot.md). Power-management agents
and some high-performance networking configurations can require host PID access, host paths, root, capabilities,
or privileged containers.

Run these components only on dedicated nodes or in tightly controlled namespaces. Pin their images, restrict
scheduling, minimize host mounts and Linux capabilities, and enforce exceptions with admission policy rather than
relaxing the cluster-wide baseline.

## Limit Resource Exhaustion

Authentication does not prevent an authorized client from exhausting resources. Configure
`DYN_HTTP_BODY_LIMIT_MB`, gateway request limits, per-identity quotas, timeouts, and concurrency controls. Dynamo's
router can reject or queue work based on load, but external rate and cost limits remain the responsibility of the
serving platform. See [Request Rejection and Load Shedding](../fault-tolerance/request-rejection.md).

## Report a Vulnerability

Do not disclose suspected vulnerabilities in a public issue. Follow the private reporting process in the
[Dynamo Security Policy](https://github.com/ai-dynamo/dynamo/blob/main/SECURITY.md).
