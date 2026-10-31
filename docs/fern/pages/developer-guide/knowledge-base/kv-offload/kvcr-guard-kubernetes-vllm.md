---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Run KVCR Guard on Kubernetes with vLLM
subtitle: Deploy vLLM workers whose KVCR cache survives an engine crash
---

The [`agg-memory-service.yaml`](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/deploy/kvcr/agg-memory-service.yaml) example deploys a frontend and two vLLM workers on two nodes. Each worker Pod has two containers:

- `main` runs `dynamo.vllm`.
- `kvcr-services` runs the Guard service, which owns the KVCR pool. On the first replica only, it also runs the [KV state agent](kvcr.md#the-kv-state-agent), which keeps the router routing to a crashed worker's cache; one state agent serves the whole DynamoGraphDeployment.

When the engine in `main` crashes, `main` restarts and reattaches to its pool; while it is down, the other worker can fetch the crashed worker's blocks from its Guard. Deleting the Pod loses the cache.

This example uses Kubernetes discovery; it does not need etcd.

## Before You Begin

- Two GPU nodes, each with one free GPU and two shares of an RDMA resource (`rdma/shared_ib` by default; one for each container).
- The Dynamo operator with the Grove workload provider, and `kubectl` set to the target cluster context.
- An `hf-token-secret` in the target namespace.
- A runtime image with matching Dynamo, vLLM, KVCR v0.2.0, and NIXL builds (see [Configure KVCR for vLLM](configure-kvcr-vllm.md#before-you-begin)). If its tag is not a Dynamo release version, set `runtimeVersionOverride` on every component.

## Deploy

1. From `examples/backends/vllm/deploy/kvcr` in the Dynamo repository, set the variables and apply `agg-memory-service.yaml`. `envsubst` fills in only the listed variables; the manifest's startup scripts use other `$` variables that must stay as they are.

   ```bash
   export NAMESPACE=<namespace>
   export DYNAMO_VLLM_IMAGE=<image>@sha256:<digest>
   export DYNAMO_KVCR_COMPATIBILITY_DIGEST=qwen3-0.6b-v1
   export DYNAMO_UCX_NET_DEVICES=<hca>:1
   export DYNAMO_RDMA_RESOURCE=rdma/shared_ib

   envsubst '${DYNAMO_VLLM_IMAGE} ${DYNAMO_RDMA_RESOURCE} ${DYNAMO_UCX_NET_DEVICES} ${DYNAMO_KVCR_COMPATIBILITY_DIGEST}' \
     < agg-memory-service.yaml | kubectl apply -n "$NAMESPACE" -f -
   ```

   | Variable | Value |
   |---|---|
   | `DYNAMO_KVCR_COMPATIBILITY_DIGEST` | Any string shared by `main` and `kvcr-services`. Change it when the model, dtype, or block layout changes. |
   | `DYNAMO_UCX_NET_DEVICES` | The example requires it. NIXL picks network devices on its own, but on nodes with several RDMA NICs, workers that see every NIC can stall at startup, so pin the NIC attached directly to the worker's GPU (marked `PIX` for that GPU in `nvidia-smi topo -m`), for example `mlx5_5:1`. On a multi-GPU node the right NIC depends on which GPU the Pod gets. |
   | `DYNAMO_RDMA_RESOURCE` | The RDMA extended resource your cluster exposes. |

2. Wait until both worker Pods show `2/2 Running`:

   ```bash
   kubectl get pods -n "$NAMESPACE" -w
   ```

3. Port-forward port 8000 of the frontend Pod, then confirm a cross-worker transfer with [Verify and Monitor KVCR with vLLM](kvcr-testing-monitoring-vllm.md), using the two worker IDs. Each worker serves its metrics on port 9090 of its Pod; reach a worker's metrics with `kubectl port-forward -n "$NAMESPACE" pod/<worker-pod> 9090` and read them at `localhost:9090/metrics`.

## Verify Recovery

Pin each request to a worker as in [Verify a Transfer](kvcr-testing-monitoring-vllm.md#verify-a-transfer).

1. Send a prompt to worker A, then crash its engine:

   ```bash
   kubectl exec -n "$NAMESPACE" <worker-a-pod> -c main -- pkill -9 -f '[V]LLM::EngineCore'
   ```

2. Check that only `main` restarted:

   ```bash
   kubectl get pod -n "$NAMESPACE" <worker-a-pod> \
     -o jsonpath='{range .status.containerStatuses[*]}{.name}={.restartCount}{"\n"}{end}'
   ```

   `kvcr-services` logs `guard_promoted ... recovered_blocks=<n>`.

3. Send the same prompt to worker B. `cached_tokens` is positive, and B's `vllm:kvcr_transfer_blocks_total{operation="remote_deliver"}` increases.

4. When worker A's Pod is Ready again, send the prompt to A. `cached_tokens` is positive, served from its pool (`operation="local_deliver"`).

## Limits

- Each worker Pod has its own Guard service; several workers cannot share one Guard service yet.
- If `kvcr-services` restarts, recovery no longer works for that Pod. Replace the whole Pod.
- Each crash leaves the dead engine's CPU-offload file (`cpu_bytes_to_use`, 2 GiB in the example) in `main`'s `/dev/shm` until vLLM [#53073](https://github.com/vllm-project/vllm/pull/53073) is in your vLLM build. Size `/dev/shm` for it, or remove `/dev/shm/vllm_offload_*` before starting vLLM.

## Related Topics

- [Configure KVCR for vLLM](configure-kvcr-vllm.md)
- [Verify and Monitor KVCR with vLLM](kvcr-testing-monitoring-vllm.md)
- [Run KVCR on Kubernetes with vLLM](kvcr-kubernetes-vllm.md)
