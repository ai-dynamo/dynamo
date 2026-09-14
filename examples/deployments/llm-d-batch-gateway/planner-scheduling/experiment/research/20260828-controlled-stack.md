<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Planner-controlled llm-d Async stack

> Historical research note: this records the August 2026 environment. The
> command below is intentionally not a current runnable recipe: it uses the old
> local chart path, hard-coded namespace, omitted companion image, and flags
> from a different Helm generation. Use the current
> [Planner deployment recipe](https://github.com/ai-dynamo/dynamo/blob/3511ee2f203de2556962b48d93b7692cc69b8a8b/examples/deployments/llm-d-batch-gateway/planner-scheduling/README.md#deploy-the-planner-poc) instead.

This is the controlled counterpart to the stock baseline. It keeps the Batch
Gateway, Valkey, queue names, Dynamo frontend, and model workers unchanged. The
only data-plane change is an llm-d Async image with a leased Redis rate gate at
the `dynamo-batch` worker-pool admission boundary.

The controlled overlay also creates a 5-second Async `PodMonitor`. This was
enabled only after verifying that the target cluster has the PodMonitor CRD and
its Prometheus instance selects PodMonitors across namespaces. The Planner
runner still targets the Prometheus HTTP query API through a local port-forward;
it must not target Async's raw metrics port.

## Stable identifiers

- Kubernetes namespace: caller-selected (examples below use `default`)
- Async worker pool: `dynamo-batch`
- Request sorted set: `llm-d-async:requests:dynamo-batch`
- Result-list base: `llm-d-async:results:dynamo-batch`. Pinned Batch Gateway
  v0.3.0 routes requests to that unsuffixed list; newer Gateway code may append
  a processor consumer ID. The controlled queue config must not override the
  per-request result destination, so both forms continue to work.
- Drain-limit command hash: `llm-d-async:drain-limit:dynamo-batch`
- Drain-limit token state: `llm-d-async:drain-limit:dynamo-batch:state`

## Deployment boundary

The historical procedure installed the stock stack first and captured the
baseline. Then, from `examples/deployments/llm-d-batch-gateway`, it upgraded
only the Async release with the locally modified sibling chart and both values
files, in this order:

```bash
# Historical only; do not run as the current deployment recipe.
helm upgrade --install async-dispatch \
  ../../../../llm-d-async/charts/llm-d-async \
  --namespace default \
  --values llm-d-async-values.yaml \
  --values llm-d-async-planner-values.yaml \
  --rollback-on-failure --cleanup-on-fail --wait=watcher --timeout 5m
```

The validated linux/amd64 image was built from the companion llm-d Async
branch and verified with a one-shot cluster smoke Job that ran
`/llm-d-async --help`. Registry coordinates from the private POC environment
are intentionally omitted; publish the companion image to a registry reachable
by the target cluster and update the overlay before rollout.
The gate fails closed until Planner writes a valid lease, so seed or start the
controller before expecting batch work to drain.

## Effect wire contract

Planner publishes one Redis hash transaction followed by an absolute key
expiry. `max_admission_rps=0` is an explicit pause; a missing, malformed,
mismatched, or expired lease also pauses.

```text
api_version=llm-d.ai/v1alpha1
pool_id=dynamo-batch
max_admission_rps=<finite number >= 0>
valid_until_unix_ms=<absolute Unix milliseconds>
decision_id=<non-empty audit ID>
```

This controls attempts sent by llm-d Async to the Dynamo frontend. It does not
replace EDF ordering in the Batch Gateway queue, normal Dynamo routing, or the
Planner's replica decision.
