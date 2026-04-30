<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Validation

Run the baseline production checks after the add-ons are healthy:

```bash
deploy/pre-deployment/pre-deployment-check.sh --profile production --output json
```

Run the Dynamo-specific checks after `dynamo-platform` has synced:

```bash
deploy/pre-deployment/pre-deployment-check.sh --require dynamo-crds,dynamo-webhooks,kai-queue --output json
```

Check optional integrations only when they are installed:

```bash
deploy/pre-deployment/pre-deployment-check.sh --require keda,opentelemetry,parca,lws-volcano --output json
```

For the A4 k3s profile, Parca pins `parca-agent` to `v0.20.0`. Newer
chart-compatible agent images currently fail on the Rocky 9.7 kernel while
loading their BPF object with `argument list too long`; the pinned image keeps
the node-level profiling DaemonSet healthy while the Parca server and
ServiceMonitors remain enabled.
