---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Admission Control
subtitle: Configure the backend admission gate in a Kubernetes deployment.
---

Backend admission control limits requests inside each engine and queues excess work outside it.
Configure the gate on the worker containers in your DynamoGraphDeployment (DGD). It supports both
Dynamo Python and Dynamo Sidecar backends.

For behavior and tuning guidance, see
[Admission Control Architecture](../../developer-guide/knowledge-base/concepts/fault-tolerance/admission-control-architecture.md).

## Configure the Worker

Start with a working [DynamoGraphDeployment](../../reference/kubernetes-api/dynamo-graph-deployment.mdx).
These three settings apply **per data-parallel (DP) rank**:

| Environment variable | What it limits | Default per rank |
|---|---|---|
| `DYN_BACKEND_ADMISSION_ENGINE_REQUEST_LIMIT` | Admitted requests that have not finished. | Unset: unlimited |
| `DYN_BACKEND_ADMISSION_ENGINE_REQUEST_WAITING_LIMIT` | Admitted requests awaiting their first generated token or completion. | Unset: unlimited |
| `DYN_BACKEND_ADMISSION_REQUEST_QUEUE_LIMIT` | Requests waiting outside the engine. Set `0` to disable queueing. | `0`: no queue |

Setting either engine limit enables admission control; an engine limit left unset stays unlimited.
When either engine limit is reached, new requests wait in the gate queue. If the queue is full, the
worker rejects them; overload reaches the client as HTTP 529 by default.

The following `nvidia.com/v1beta1` fragment enables admission control with the recommended starting
limits for an engine whose `max_num_seqs` is 256: each engine limit at `max_num_seqs`, and a queue of
twice that. Add these entries to your worker container's existing `env` list.

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
                - name: DYN_BACKEND_ADMISSION_ENGINE_REQUEST_WAITING_LIMIT
                  value: "256"
                - name: DYN_BACKEND_ADMISSION_REQUEST_QUEUE_LIMIT
                  value: "512"
```

For a disaggregated deployment, configure prefill and decode workers separately. Set these variables on
the workers, not the frontend. See
[Configuration](../../developer-guide/knowledge-base/concepts/fault-tolerance/admission-control-architecture.md#configuration)
for CLI precedence and aliases, and
[Best Practices](../../developer-guide/knowledge-base/concepts/fault-tolerance/admission-control-architecture.md#best-practices)
for tuning from the starting limits.

## Verify the Configuration

Check the worker logs for the effective limits. Under load, use the `dynamo_backend_admission_*`
metrics to observe engine occupancy, queued requests, rejections, and cancellations. These metrics
stay at zero while admission control is disabled. The `dp_rank` label identifies the rank;
`unspecified` covers unranked requests. Sum over `dp_rank` for worker totals.

See [Metrics](../../developer-guide/knowledge-base/concepts/fault-tolerance/admission-control-architecture.md#metrics)
for metric names and labels.
