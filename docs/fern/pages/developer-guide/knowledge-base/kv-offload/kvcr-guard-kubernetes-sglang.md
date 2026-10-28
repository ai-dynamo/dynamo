---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Run KVCR Guard on Kubernetes with SGLang
subtitle: Deploy SGLang Linker-mode workers whose KVCR cache survives an engine crash
---

> [!WARNING]
> Preview: needs Dynamo [#16827](https://github.com/ai-dynamo/dynamo/pull/16827) and SGLang [#45650](https://github.com/sgl-project/sglang/pull/45650) with the KVCR changes it is stacked on ([#45646](https://github.com/sgl-project/sglang/pull/45646) to [#45649](https://github.com/sgl-project/sglang/pull/45649)), none merged, and KVCR v0.2.0.
>
> Not yet run end to end on Kubernetes. Validated: the manifest renders with `envsubst` and passes `kubectl apply --dry-run=server` against the DynamoGraphDeployment CRD; the same worker flags passed on one server in [Configure KVCR Linker Mode for SGLang](configure-kvcr-sglang-linker.md), including crash recovery. Whether `main` restarts on its own after the engine is killed is not yet verified on Kubernetes.

The [`sglang-agg-linker-guard.yaml`](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/sglang/deploy/kvcr/sglang-agg-linker-guard.yaml) example deploys a frontend and two SGLang workers in Linker mode on two nodes. Guard needs Linker mode; HiCache mode has no Guard support. Each worker Pod has two containers:

- `main` runs `dynamo.sglang` with the KVCR Linker. `--kvcr-state-agent-owner` registers the worker's cache with the [KV state agent](kvcr.md#the-kv-state-agent) under a stable ID taken from the Pod's Grove index.
- `kvcr-services` runs the Guard service, which owns the KVCR pool. On the first replica only, it also runs the KV state agent, which keeps the router sending peers to a crashed worker's Guard; one state agent serves the whole DynamoGraphDeployment.

SGLang workers on one host can share a Guard service through `guard_index`, but on Kubernetes keep one sidecar per Pod, as in this example.

When the engine in `main` crashes, `main` restarts and reattaches to its pool; while it is down, the other worker can fetch the crashed worker's blocks from its Guard. Deleting the Pod loses the cache.

This example uses Kubernetes discovery; it does not need etcd.

## Before You Begin

- Two GPU nodes, each with one free GPU and two shares of an RDMA resource (`rdma/shared_ib` by default; one for each container).
- The Dynamo operator with the Grove workload provider, and `kubectl` set to the target cluster context.
- An `hf-token-secret` in the target namespace.
- A runtime image with the SGLang Linker, Dynamo with #16827, KVCR v0.2.0, and NIXL; see [Configure KVCR Linker Mode for SGLang](configure-kvcr-sglang-linker.md).

## Deploy

1. From `examples/backends/sglang/deploy/kvcr` in the Dynamo repository, set the variables and apply the manifest. `envsubst` fills in only the listed variables; the startup scripts use other `$` variables that must stay as they are.

   ```bash
   export NAMESPACE=<namespace>
   export DYNAMO_SGLANG_IMAGE=<image>@sha256:<digest>
   export DYNAMO_VERSION=<dynamo-version>
   export DYNAMO_KVCR_COMPATIBILITY_DIGEST=qwen3-0.6b-linker-v1
   export DYNAMO_UCX_NET_DEVICES=<nic>:1
   export DYNAMO_RDMA_RESOURCE=rdma/shared_ib

   envsubst '${DYNAMO_SGLANG_IMAGE} ${DYNAMO_VERSION} ${DYNAMO_RDMA_RESOURCE} ${DYNAMO_UCX_NET_DEVICES} ${DYNAMO_KVCR_COMPATIBILITY_DIGEST}' \
     < sglang-agg-linker-guard.yaml | kubectl apply -n "$NAMESPACE" -f -
   ```

   | Variable | Value |
   |---|---|
   | `DYNAMO_VERSION` | The Dynamo version in the image, for example `1.7.0`. The operator rejects an image without a Dynamo version tag unless `runtimeVersionOverride` is set. |
   | `DYNAMO_KVCR_COMPATIBILITY_DIGEST` | Any string shared by `main` and `kvcr-services`. Change it when the model, dtype, or block layout changes. |
   | `DYNAMO_UCX_NET_DEVICES` | The example requires it. NIXL picks network devices on its own, but on nodes with several RDMA NICs, workers that see every NIC can stall at startup, so pin the NIC attached directly to the worker's GPU (marked `PIX` for that GPU in `nvidia-smi topo -m`), for example `mlx5_5:1`. On a multi-GPU node the right NIC depends on which GPU the Pod gets. |
   | `DYNAMO_RDMA_RESOURCE` | The RDMA extended resource your cluster exposes. |

2. Wait until both worker Pods show `2/2 Running`:

   ```bash
   kubectl get pods -n "$NAMESPACE" -w
   ```

3. Port-forward port 8000 of the frontend Pod and confirm a cross-worker transfer as in [Run KVCR on Kubernetes with SGLang](kvcr-kubernetes-sglang.md#check-it-works). In Linker mode the second response shows `cached_tokens` greater than 0; the worker log does not show the HiCache prefetch line.

## Verify Recovery

Pin each request to a worker as in [Check It Works](kvcr-kubernetes-sglang.md#check-it-works).

1. Send `request.json` to worker A and wait 15 seconds.

2. Find worker A's engine process and stop it:

   ```bash
   kubectl exec -n "$NAMESPACE" <worker-a-pod> -c main -- nvidia-smi --query-compute-apps=pid --format=csv,noheader
   kubectl exec -n "$NAMESPACE" <worker-a-pod> -c main -- kill -9 <pid>
   ```

   `kvcr-services` logs `guard_promoted ... recovered_blocks=<n>`.

3. Wait 75 seconds, so that the router has removed worker A, then send `request.json` to worker B. `cached_tokens` is positive: B fetched the blocks from worker A's Guard.

4. After worker A is Ready again, send `request.json` to it. `cached_tokens` is positive, served from its own pool.

## Limits

- If `kvcr-services` restarts, recovery no longer works for that Pod. Replace the whole Pod.
- Without the KV state agent, peers stop finding a crashed worker's blocks once the router removes that worker.

## Related Topics

- [Configure KVCR Linker Mode for SGLang](configure-kvcr-sglang-linker.md)
- [Run KVCR on Kubernetes with SGLang](kvcr-kubernetes-sglang.md)
- [KV Cache Runner (KVCR)](kvcr.md)
