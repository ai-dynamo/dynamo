---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Admission Control
subtitle: Configure and verify the backend admission gate in a Kubernetes deployment.
---

Configure NVIDIA Dynamo backend admission control on each worker component in your
DynamoGraphDeployment (DGD). The gate bounds the requests inside the engine and, among them, the
requests still awaiting their first engine response, and queues excess work outside the engine.

For the design, defaults, tuning guidance, and metrics, see
[Admission Control Architecture](../../developer-guide/knowledge-base/concepts/fault-tolerance/admission-control-architecture.md).

## Configure the Worker

Start with a working [DynamoGraphDeployment](../../reference/kubernetes-api/dynamo-graph-deployment.mdx).
Configure the backend admission gate through these three environment variables on the worker container:

| Environment variable | Purpose | Default |
|---|---|---|
| `DYN_BACKEND_ADMISSION_ENGINE_REQUEST_LIMIT` | Limits admitted requests that have not finished. Each request holds this capacity until its response stream ends. | `10000`. |
| `DYN_BACKEND_ADMISSION_ENGINE_WAIT_LIMIT` | Limits admitted requests still awaiting their first engine response. Each request holds this capacity until its first response; the stream then continues. | `10000`. |
| `DYN_BACKEND_ADMISSION_REQUEST_QUEUE_LIMIT` | Limits requests waiting outside the engine in the admission-gate queue. Set `0` to disable queueing. | `40000`. |

A request enters the engine only when both engine limits have room. The engine limits are fixed; the
gate does not size them from the engine's reported capacity.

The following `nvidia.com/v1beta1` fragment shows all three settings together, with an engine request
limit of `256`, an engine wait limit of `32`, and the default queue limit. Omit an entry to keep its
default.

Merge these entries into the existing worker container's `env` list, retaining its image,
arguments, resources, and other environment entries.

```yaml
spec:
  components:
    - name: worker
      type: worker
      podTemplate:
        spec:
          containers:
            - name: main
              env:
                - name: DYN_BACKEND_ADMISSION_ENGINE_REQUEST_LIMIT
                  value: "256"
                - name: DYN_BACKEND_ADMISSION_ENGINE_WAIT_LIMIT
                  value: "32"
                - name: DYN_BACKEND_ADMISSION_REQUEST_QUEUE_LIMIT
                  value: "40000"
```

For a disaggregated deployment, apply the settings to both the prefill and decode worker components,
choosing limits for each stage. Each worker process has an independent gate. Set these variables on
the workers rather than on the frontend.

The worker's `--engine-request-limit` argument and the legacy `DYN_ENGINE_REQUEST_LIMIT` and
`DYN_DYNAMO_REQUEST_QUEUE_LIMIT` variables configure the same engine request and queue limits. If a
worker already sets them, `--engine-request-limit` takes precedence over both engine request limit
variables, and each variable above takes precedence over its legacy alias. See
[Configuration](../../developer-guide/knowledge-base/concepts/fault-tolerance/admission-control-architecture.md#configuration)
and [Best Practices](../../developer-guide/knowledge-base/concepts/fault-tolerance/admission-control-architecture.md#best-practices)
for accepted values, precedence, and tuning guidance.

## Verify the Configuration

Check the worker's `Backend admission gate created` log entry for the effective engine request,
engine wait, and queue limits.

Use the `dynamo_backend_admission_*` metrics to verify that both engine counts stay within their
limits and to observe queue occupancy and rejections under load. See
[Metrics](../../developer-guide/knowledge-base/concepts/fault-tolerance/admission-control-architecture.md#metrics)
for the names, labels, and meanings.
