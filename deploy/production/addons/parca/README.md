<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Parca - Optional Production Addon

Parca is the optional continuous profiling stack. In this profile the Parca server is enabled, but the agent is disabled by default for the current A4/k3s lane.

## Deployment

| Field | Value |
|---|---|
| Argo CD app | [`gitops/optional/parca.yaml`](../../gitops/optional/parca.yaml) |
| Chart | `parca` |
| Source | `https://parca-dev.github.io/helm-charts` |
| Version | `4.19.0` |
| Namespace | `parca` |
| Values | [`values.yaml`](values.yaml) |
| Sync wave | `8` |
| Baseline | No |

## What This Addon Owns

| Capability | Local setting |
|---|---|
| Parca server | `server.enabled: true`. |
| Server ServiceMonitor | `server.serviceMonitor.enabled: true`. |
| Parca Agent | `agent.enabled: false`. |
| Agent ServiceMonitor | Defined but inactive while the agent is disabled. |

## What This Addon Does Not Own

- It does not collect node profiles unless the agent is enabled.
- It does not replace Prometheus metrics, Loki logs, or OpenTelemetry traces.
- It is not deployed by the baseline root app.

## Operating Notes

Enable the agent only after validating kernel, eBPF, and node overhead on the target GPU nodes. Continuous profiling is useful, but the agent is intentionally off until there is a clear profiling requirement.

## Verify

```bash
kubectl -n parca get pods,svc
kubectl get servicemonitor -A | grep -i parca
deploy/pre-deployment/pre-deployment-check.sh --require parca
```

If the agent is expected to be enabled:

```bash
kubectl -n parca get daemonset
```

## Upgrade Notes

Treat server and agent upgrades separately. Agent upgrades need node-level validation; server-only upgrades are lower risk.

Upstream reference: [Parca project documentation](https://www.parca.dev/docs/).
