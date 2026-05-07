<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SMG (Shepherd Model Gateway) - Production Addon

SMG is the client-facing OpenAI-compatible HTTP entry point in front of the Dynamo Frontend Service. In this production profile it is limited to gateway behavior: forwarding requests, health checking the Dynamo backend, retries, circuit breaking, metrics, tracing, logging, and dashboard export.

The full boundary is in [`docs/smg-integration.md`](../../docs/smg-integration.md).

## What's Wired On By Default

| Capability | Where | Knob |
|---|---|---|
| OpenAI-compatible HTTP surface | SMG router :30000, Service :80 | `router.port`, `router.service` |
| Round-robin to Dynamo Frontend | Single backend, no SMG prefix tree | `router.policy: round_robin`, `router.workerUrls` |
| Retry and circuit breaker | Per request, against Dynamo Frontend | `router.retry`, `router.circuitBreaker` |
| Health checks | Dynamo Frontend `/health` | `router.healthCheck` |
| OpenTelemetry tracing | Otel collector in `observability` namespace | `router.tracing.enabled`, `otlpEndpoint` |
| Prometheus metrics | ServiceMonitor for kube-prometheus-stack | `router.metrics.serviceMonitor.enabled` |
| Grafana dashboard | ConfigMap with sidecar discovery label | `grafana.dashboard.enabled` |
| Structured logging | Router stdout for Fluentd/Loki | `router.logging.json: true` |

## What SMG Does Not Own Here

- Tokenization or detokenization.
- Reasoning parsing or tool-call parsing.
- Multimodal request processing.
- MCP tool orchestration.
- Chat history or audit-log storage.
- KV-aware routing or prefix-cache ownership.

Those capabilities must remain in Dynamo, SGLang, or external application layers when they are needed. The DeepSeek REAP manifest keeps parser configuration on the Dynamo/SGLang path instead of the SMG chart.

## What This Addon Does Not Change

- The 4-GPU prefill plus 4-GPU decode disaggregation in [`examples/deepseek-v32-reap-sglang.yaml`](../../examples/deepseek-v32-reap-sglang.yaml).
- Dynamo Frontend's `--router-mode kv --router-kv-events --router-reset-states` configuration.
- HiSparse, IndexCache, TurboQuant, and SMC-SD worker configuration.
- SGLang worker ownership of GPU inference processes.

## Why `policy: round_robin` And Not `cache_aware`

SMG's `cache_aware` policy and Dynamo Frontend's KV router both maintain prefix state. Dynamo sees the actual SGLang KV-cache state through ZMQ kv-events, so this profile keeps SMG as a simple gateway and leaves prefix routing to Dynamo.

If the cluster grows to multiple Dynamo Frontends, `policy: power_of_two` across Dynamo Frontend services may be useful. Each Dynamo Frontend should still own prefix routing within its own fleet.

## Pulling In SMG Updates

Renovate at [`.github/renovate.json5`](../../../../.github/renovate.json5) handles upstream sync for Argo CD applications under `gitops/apps/` and `gitops/optional/`, plus the SMG image tag in this `values.yaml`.

| Update type | Behavior |
|---|---|
| Patch and minor | Renovate opens a PR and auto-merges after CI. |
| Major | Renovate opens a PR with `needs-human-review` label. |

To bump SMG manually outside Renovate, edit `targetRevision` in `gitops/apps/70-smg.yaml` and the matching image `tag:` in this `values.yaml` together.

## Auth

`auth.apiKey` is empty by default. Production authentication should be added only after the expected client path is defined; this profile does not currently wire a secret into SMG.

## Verify After Deploy

```bash
kubectl -n smg port-forward svc/smg-router 8080:80

curl -sS http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "BlaiseAI/DeepSeek-V3.2-REAP-345B-NVFP4-W4A4KV4-IndexerK8-FP8-GatedNorm-G1",
    "messages": [{"role": "user", "content": "Reply with exactly: pong"}],
    "max_tokens": 8
  }'
```

A 200 with a `chat.completion` response means the HTTP chain from SMG to Dynamo Frontend to SGLang is live. The end-to-end script [`tests/smg-roundtrip.sh`](../../../../tests/smg-roundtrip.sh) attributes failures per layer.

To verify observability:

```bash
kubectl -n observability logs -l app.kubernetes.io/name=opentelemetry-collector | grep smg
kubectl -n monitoring get cm -l grafana_dashboard=1 | grep smg
```
