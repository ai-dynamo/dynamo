---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Customize Health Probes
subtitle: Understand the operator's default liveness, readiness, and startup probes and override them in a DynamoGraphDeployment when defaults do not fit.
# TODO: either add or drop this page
---

The Dynamo operator attaches sensible health probes to every component, so most deployments need no probe configuration at all. Override them when a large model needs a longer startup window, or when your engine exposes health differently. This page covers the defaults and how to override them in a DynamoGraphDeployment (DGD).

This is a [how-to](dgd-guide.md) for an existing deployment. For the health endpoints themselves and local-process health signals, see [Health Checks](../observability/health-checks.md).

## What the operator sets by default

You do not author these — the operator injects them. Knowing them tells you when an override is needed.

**Frontend readiness** — exec `curl` against `/health`, initial delay 60s, period 60s, timeout 30s, failure threshold 10.

**Worker probes** (served on the `system` port, 9090):

| Probe | Path | Period | Timeout | Failure threshold |
|---|---|---|---|---|
| Liveness | `/live` | 5s | 30s | 1 |
| Readiness | `/health` | 10s | 30s | 60 |
| Startup | `/live` | 10s | 5s | 720 (≈2 hours) |

The startup probe's 2-hour budget (10s × 720) is what tolerates long model downloads and engine warm-up. User-specified probes always take precedence over these defaults.

## When to override

- **Model startup exceeds 2 hours** (very large models on slow storage). Raise the startup probe's `failureThreshold`: `failureThreshold = expected_startup_seconds / periodSeconds`.
- **Your engine exposes health on a different path or port.** Point the probe at it.
- **You want faster failure detection** for a small, fast-loading model. Lower the thresholds.

> [!TIP]
> Prefer [Model Caching](model-caching.md) over a longer startup probe when the bottleneck is download time — caching removes the per-pod download instead of just waiting longer for it.

## Override in a DGD

Set `livenessProbe` and `readinessProbe` on the `main` container in the component's `podTemplate`. They are standard Kubernetes [Probe](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.28/#probe-v1-core) objects:

```yaml
spec:
  components:
  - name: VllmDecodeWorker
    type: worker
    podTemplate:
      spec:
        containers:
        - name: main
          readinessProbe:
            httpGet:
              path: /health
              port: 9090
            periodSeconds: 10
            failureThreshold: 120      # double the default startup tolerance
          livenessProbe:
            httpGet:
              path: /live
              port: 9090
            periodSeconds: 10
            failureThreshold: 3
```

## Multinode note

For multinode deployments the operator adjusts probes by backend and node role — for example, it removes worker-node probes for vLLM Ray workers and replaces the TensorRT-LLM worker readiness probe with a TCP check on the SSH port. See [Multinode Deployments](deployment/multinode-deployment.md) for the per-backend behavior before overriding probes on a multinode service.

## Related pages

- [Health Checks](../observability/health-checks.md) — health endpoints and local-process signals.
- [API Reference](api-reference.md) — full probe defaults per component type.
- [Model Caching](model-caching.md) — remove startup download time instead of extending probes.
