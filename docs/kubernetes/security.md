---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Security and Authentication
subtitle: Dynamo's security model, shared responsibility boundaries, and production hardening guidance.
---

Dynamo is designed to run inside a **trusted internal service network**. Most internal services and the inference frontend do not authenticate end users. Client authentication, authorization, rate limiting, and network isolation must be enforced at the gateway, ingress, service mesh, or platform boundary before traffic reaches Dynamo.

This page documents Dynamo's security surfaces, their defaults, and the controls available for production hardening.

<Warning>
Dynamo does not provide built-in end-user API authentication. TLS and authentication are not enabled on most services by default. Review every section of this page before exposing a Dynamo deployment outside a trusted network.
</Warning>

## Security model and shared responsibility

| Responsibility | Dynamo | Gateway / Platform |
|---|---|---|
| Inference routing and scheduling | ✅ | — |
| Internal service authentication (NATS, etcd) | Optional | — |
| End-user authentication and authorization | ❌ | ✅ |
| TLS termination for external clients | Optional | ✅ Recommended |
| Rate limiting and request-size policy | Partial (body limit, admission control) | ✅ |
| Network isolation between tenants | ❌ | ✅ |
| Secrets management | ❌ | ✅ |

## External and internal endpoints

### Frontend listener

The frontend binds to `0.0.0.0` on port `8000` by default. All inference, metrics, OpenAPI, health, and admin surfaces share this single listener.

| Endpoint | Default state | Notes |
|---|---|---|
| `POST /v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/responses` | Open | No authentication |
| `POST /inference/v1/generate` | Open | Experimental — no authentication; currently returns HTTP 501 (not implemented) |
| `GET /v1/models`, `GET /v1/models/{model}`, `GET /v1/models/{model}/ready` | Open | Lists discovered models; exposes model metadata and per-namespace worker readiness without authentication |
| `GET /metrics` | Open | Prometheus metrics |
| `GET /docs`, `GET /openapi.json` | Open | API schema |
| `GET /health`, `GET /live` | Open | Health probes |
| `GET /busy_threshold`, `POST /busy_threshold` | Open | Runtime admission tuning — **disable in production** |

**TLS:** TLS is optional. To enable it, provide `--tls-cert-path` / `DYN_TLS_CERT_PATH` and `--tls-key-path` / `DYN_TLS_KEY_PATH`. When TLS is not configured the frontend accepts plain HTTP.

**Admin API:** The `GET /busy_threshold` and `POST /busy_threshold` endpoints allow runtime mutation of admission thresholds. Disable them in production by setting `DYN_DISABLE_FRONTEND_ADMIN_API=1`.

**Request body limit:** The default maximum request body is 45 MB (`DYN_HTTP_BODY_LIMIT_MB`). Adjust this value based on your workload and rate-limiting policy.

See [Frontend Configuration Reference](../components/frontend/configuration.md) for the full list of CLI arguments and environment variables.

### Internal-only services

The following services are for internal use and must not be exposed to untrusted networks:

| Service | Transport | Notes |
|---|---|---|
| NATS | TCP 4222 | Event plane; no authentication by default in the bundled Helm chart |
| etcd | TCP 2379 | Discovery (KV store backend); RBAC disabled by default in the bundled Helm chart |
| ZMQ endpoints | TCP (dynamic) | Request plane; no authentication |
| Standalone KV indexer | TCP | Accepts cache salt without mixing it into hashes — treat as sensitive |
| Slot tracker | HTTP | Internal load metrics |
| Planner live dashboard | HTTP | Exposes scheduling state |
| Selection service | HTTP | Topology placement |
| Shared-cache service | HTTP | Inter-worker coordination |

Use Kubernetes [NetworkPolicy](https://kubernetes.io/docs/concepts/services-networking/network-policies/) to restrict access to these services to the Dynamo namespace. Dynamo does not currently ship production NetworkPolicy manifests.

## NATS authentication and TLS

The Dynamo runtime supports the following NATS authentication methods, evaluated in priority order:

| Method | Environment variables |
|---|---|
| Username + password | `NATS_AUTH_USERNAME`, `NATS_AUTH_PASSWORD` |
| Token | `NATS_AUTH_TOKEN` |
| NKey | `NATS_AUTH_NKEY` |
| Credentials file | `NATS_AUTH_CREDENTIALS_FILE` |

The bundled NATS chart has no authorization configured by default. For production deployments, configure NATS authentication via `nats.config.nats.merge.authorization` in your Helm values and set the matching `NATS_AUTH_*` environment variables on Dynamo client pods.

NATS TLS is not enabled by default. Configure TLS at the NATS server level and provide the appropriate credentials via the environment variables above.

## etcd authentication and TLS

The runtime connects to etcd with no authentication by default. Two authentication modes are supported:

**Username and password:**
```bash
ETCD_AUTH_USERNAME=<username>
ETCD_AUTH_PASSWORD=<password>
```

**Mutual TLS:**
```bash
ETCD_AUTH_CA=<PEM-encoded CA certificate>
ETCD_AUTH_CLIENT_CERT=<PEM-encoded client certificate>
ETCD_AUTH_CLIENT_KEY=<PEM-encoded client key>
```

The bundled etcd Helm chart disables RBAC by default. For production, enable etcd RBAC and configure one of the authentication modes above.

## Kubernetes RBAC and pod security

### Operator permissions

The Dynamo operator is cluster-wide by default and requires permissions to create and manage CRDs, deployments, services, and roles across namespaces. Review the operator's RBAC footprint before installing into a shared cluster.

### Per-deployment discovery permissions

When Kubernetes discovery is enabled (the default), the operator creates the following RBAC resources in the `DynamoGraphDeployment` namespace for each deployment:

- `ServiceAccount`: `<DGD-NAME>-k8s-service-discovery`
- `Role`: `<DGD-NAME>-k8s-service-discovery-role` — grants `get`, `list`, `watch` on `Endpoints`, `Pods`, and `EndpointSlices`; and `create`, `get`, `list`, `watch`, `update`, `patch`, `delete` on `DynamoWorkerMetadata` resources
- `RoleBinding`: `<DGD-NAME>-k8s-service-discovery-binding`

These are namespace-scoped. Workers only discover peers within the same namespace.

To use a custom `ServiceAccount`, supply it via `podTemplate` in your DGD spec. The custom `ServiceAccount` must be bound in the same namespace to `<DGD-NAME>-k8s-service-discovery-role` to retain Kubernetes discovery permissions. Set `automountServiceAccountToken: false` if your workload does not need API server access.

### Pod and container security contexts

Dynamo workloads receive `fsGroup: 1000` by default. Pods are not forced to run as non-root. To enforce non-root execution, add a `securityContext` via `podTemplate`:

```yaml
podTemplate:
  spec:
    securityContext:
      runAsNonRoot: true
      runAsUser: 1000
      fsGroup: 1000
    containers:
      - name: main
        securityContext:
          allowPrivilegeEscalation: false
          capabilities:
            drop: ["ALL"]
```

<Warning>
Providing any `podTemplate.spec.securityContext` disables the operator's default security context injection, including `fsGroup: 1000`. Include all required fields explicitly when supplying a custom pod security context.
</Warning>

Enable [Pod Security Admission](https://kubernetes.io/docs/concepts/security/pod-security-admission/) at the namespace level to enforce security standards across all workloads in a namespace.

## Secrets and workload identity

**`envFromSecret`:** When you reference a Kubernetes Secret via `envFromSecret`, every key in that Secret is exposed as an environment variable in the pod. Scope your secrets narrowly and avoid putting unrelated credentials in the same Secret.

**Image pull secrets:** The operator automatically discovers `kubernetes.io/dockerconfigjson` secrets in the component's namespace and injects matching pull secrets for each container image. To opt out, set `nvidia.com/disable-image-pull-secret-discovery: "true"` on the component's `annotations` field in the DGD spec or on the component's pod-template annotations. Prefer IRSA, Workload Identity, or equivalent cloud-provider mechanisms over long-lived registry credentials.

**Config dumps:** The `--dump-config-to` / `DYN_DUMP_CONFIG_TO` flag writes the resolved frontend configuration to a file. Protect this file; it may contain connection strings and endpoint addresses. NATS passwords and tokens are redacted in logs but the dump file behavior should be verified for your deployment.

**Workload identity:** Where supported by your cloud provider, prefer IAM roles bound to Kubernetes ServiceAccounts (IRSA on AWS, Workload Identity on GCP, etc.) over long-lived credentials stored in Secrets.

## KV cache privacy and multi-tenancy

KV events carry token IDs and cumulative block hashes. These signals can leak information about prefixes processed by a worker.

- Treat NATS, ZMQ, the standalone KV indexer, indexer recovery dumps, request traces, and routing hashes as **sensitive internal data**.
- The `cache_salt` / `nvext.cache_salt` field is intended for tenant isolation but is not currently mixed into block hashes in the standalone indexer. **Do not rely on `cache_salt` alone for end-to-end tenant isolation.**
- For mutually untrusting tenants, use **deployment-level isolation**: separate DGDs, namespaces, or clusters.

## Model and container supply chain

**`--trust-remote-code`:** Behavior differs across backends and paths:
- vLLM and SGLang backends: auto-injected by the profiler when `config.json` contains an `auto_map` section (custom Python code mapping). The flag is not universally applied across all paths.
- TensorRT-LLM backend and multimodal paths: verify your deployment configuration explicitly.

**A local or PVC-cached model is operator-trusted input, not automatically safe to execute.** Always source models from known-good registries or verified artifact stores.

**Image pinning:** Use immutable image digests (`image@sha256:...`) instead of mutable tags in production. Pin model revisions rather than using branches such as `main`.

## Observability data handling

Request traces, planner dashboards, config dumps, audit payloads, and log files may contain request metadata, routing decisions, and system state. Treat these as access-controlled data:

- Apply appropriate RBAC to Prometheus, trace exporters, and log aggregation systems.
- Set explicit retention policies for traces and audit logs.
- Do not expose the planner live dashboard or `/metrics` to untrusted clients.

See [Request Tracing](../observability/request-tracing.md) for details on what trace data is collected and retained.

## Multimodal SSRF protections

The multimodal pipeline fetches external URLs for image and video inputs. An SSRF guard is enabled by default.

<Warning>
`DYN_MM_ALLOW_INTERNAL=1` disables the SSRF guard and allows requests to internal network addresses. Do not set this in public-facing deployments.
</Warning>

See [Multimodal README](../features/multimodal/README.md) for full details.

## Planner plugin authentication

The planner supports static-secret authentication for external plugins. When a static secret is configured, the planner validates it on each request. When no secret is configured, the legacy unauthenticated fallback is used.

The external plugin transport is currently plaintext-only. Do not expose plugin endpoints outside a trusted network.

For the Global Planner, the `--managed-namespaces` flag controls which Dynamo namespaces (`caller_namespace` values from scale requests) the planner is authorized to act on; it does not filter by Kubernetes namespace names. See [Global Planner](../components/planner/global-planner.md) for details.

## Privileged features

The following optional components deploy with elevated privileges:

| Feature | Privilege level | Notes |
|---|---|---|
| Snapshot agent | Privileged DaemonSet | Required for CRIU checkpoint/restore operations |
| Power Agent | Privileged DaemonSet | Required for GPU power management |

These components have cluster-wide blast radius. Only deploy them when required, and review their RBAC and host-mount footprint before installation.

See [Snapshot](snapshot.md) for deployment modes and limitations.

## Availability hardening

- **Request body limit:** Set `DYN_HTTP_BODY_LIMIT_MB` to the smallest value your workload requires (default: 45 MB).
- **Admission control:** Set `--admission-control token-capacity` to activate busy-threshold-based request rejection.
- **Router load shedding:** `--router-queue-threshold` delays dispatch and backpressures routing when the threshold is exceeded; it does not cap or drop queued requests. For hard load shedding, rely on your router policy's queue limits or gateway/admission controls upstream of Dynamo.
- **Timeouts:** Set `DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS` and backend-specific timeouts.
- **External rate limiting:** Request rate limiting remains the gateway or platform's responsibility. Dynamo does not implement per-client rate limiting.

## Production hardening checklist

- [ ] Place an authenticating gateway or ingress in front of the Dynamo frontend
- [ ] Enable TLS on the frontend (`DYN_TLS_CERT_PATH` / `DYN_TLS_KEY_PATH`) or terminate TLS at the gateway
- [ ] Disable the admin API in production (`DYN_DISABLE_FRONTEND_ADMIN_API=1`)
- [ ] Configure NATS authentication (`NATS_AUTH_USERNAME` + `NATS_AUTH_PASSWORD`, token, NKey, or credentials file)
- [ ] Enable etcd authentication (`ETCD_AUTH_USERNAME` / `ETCD_AUTH_PASSWORD` or mutual TLS)
- [ ] Apply Kubernetes NetworkPolicy to restrict access to NATS, etcd, ZMQ, and internal HTTP services
- [ ] Review operator RBAC footprint and scope `--managed-namespaces` for the Global Planner
- [ ] Use least-privilege `ServiceAccounts` and set `automountServiceAccountToken: false` where not needed
- [ ] Enable Pod Security Admission at the namespace level
- [ ] Run workload pods as non-root with `allowPrivilegeEscalation: false`
- [ ] Use immutable image digests and pinned model revisions
- [ ] Scope `envFromSecret` references to the minimum required credentials
- [ ] Prefer cloud workload identity over long-lived credentials stored in Secrets
- [ ] Do not set `DYN_MM_ALLOW_INTERNAL=1` on public-facing deployments
- [ ] Restrict access to Prometheus `/metrics`, traces, config dumps, and planner dashboards
- [ ] Only deploy Snapshot and Power Agent when required; review their host-mount and privilege footprint
- [ ] For multi-tenant workloads, use deployment-level isolation (separate DGDs, namespaces, or clusters)

## Vulnerability reporting

To report a security vulnerability in Dynamo or any NVIDIA product, see [SECURITY.md](https://github.com/ai-dynamo/dynamo/blob/main/SECURITY.md).
