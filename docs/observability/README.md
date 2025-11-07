<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dynamo Observability

## Quick Start

For a quick start guide to get Prometheus and Grafana running with Dynamo on a single machine, see [Prometheus + Grafana Setup](prometheus-grafana.md).

## Observability Documentations

| Guide | Description | Environment Variables to Control |
|-------|-------------|----------------------------------|
| [Metrics](metrics.md) | Available metrics reference | `DYN_SYSTEM_ENABLED`†, `DYN_SYSTEM_PORT`† |
| [Health Checks](health-checks.md) | Component health monitoring and readiness probes | `DYN_SYSTEM_ENABLED`†, `DYN_SYSTEM_PORT`†, `DYN_SYSTEM_STARTING_HEALTH_STATUS`, `DYN_SYSTEM_HEALTH_PATH`, `DYN_SYSTEM_LIVE_PATH`, `DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS` |
| [Tracing](tracing.md) | Distributed tracing with OpenTelemetry and Tempo | `DYN_LOGGING_JSONL`†, `OTEL_EXPORT_ENABLED`†, `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`†, `OTEL_SERVICE_NAME`† |
| [Logging](logging.md) | Structured logging configuration | `DYN_LOGGING_JSONL`†, `DYN_LOG`, `DYN_LOG_USE_LOCAL_TZ`, `DYN_LOGGING_CONFIG_PATH`, `OTEL_SERVICE_NAME`†, `OTEL_EXPORT_ENABLED`†, `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`† |

**Variables marked with † are shared across multiple observability systems.**

## Developer Guides

| Guide | Description | Environment Variables to Control |
|-------|-------------|----------------------------------|
| [Metrics Developer Guide](metrics-developer-guide.md) | Creating custom metrics in Rust and Python | `DYN_SYSTEM_ENABLED`†, `DYN_SYSTEM_PORT`† |

## Kubernetes

For Kubernetes-specific setup and configuration, see [docs/kubernetes/observability/](../kubernetes/observability/).

