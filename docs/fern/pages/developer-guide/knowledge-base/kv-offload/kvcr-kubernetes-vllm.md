---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Run KVCR on Kubernetes with vLLM
subtitle: Deploy vLLM workers with an in-process KVCR pool
---

The [`agg.yaml`](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/deploy/kvcr/agg.yaml) example deploys a frontend with the KV router and two vLLM workers on two nodes. Each worker runs KVCR in-process with its own KVCR pool (`secondary_g2_slots: 1024`). The cache is lost when a worker restarts; for crash recovery, see [Run KVCR Guard on Kubernetes with vLLM](kvcr-guard-kubernetes-vllm.md).

This example uses etcd discovery.

## Before You Begin

- Two GPU nodes, each with one free GPU and one share of an RDMA resource (`rdma/shared_ib` by default).
- The Dynamo operator with the Grove workload provider, and `kubectl` set to the target cluster context.
- etcd. If the operator has no etcd configured, run one in the namespace and set `ETCD_ENDPOINTS` (for example `http://etcd:2379`) in the `env` of every container in the manifest; otherwise pods log `etcd not reachable`.
- An `hf-token-secret` in the target namespace.
- A runtime image with matching Dynamo, vLLM, KVCR v0.2.0, and NIXL builds (see [Configure KVCR for vLLM](configure-kvcr-vllm.md#before-you-begin)). If its tag is not a Dynamo release version, set `runtimeVersionOverride` on every component.

## Deploy

1. From `examples/backends/vllm/deploy/kvcr` in the Dynamo repository, set the variables and apply `agg.yaml`. `envsubst` fills in only the listed variables; the manifest's startup scripts use other `$` variables that must stay as they are.

   ```bash
   export NAMESPACE=<namespace>
   export DYNAMO_VLLM_IMAGE=<image>@sha256:<digest>
   export DYNAMO_UCX_NET_DEVICES=<hca>:1
   export DYNAMO_RDMA_RESOURCE=rdma/shared_ib

   envsubst '${DYNAMO_VLLM_IMAGE} ${DYNAMO_RDMA_RESOURCE} ${DYNAMO_UCX_NET_DEVICES}' \
     < agg.yaml | kubectl apply -n "$NAMESPACE" -f -
   ```

   | Variable | Value |
   |---|---|
   | `DYNAMO_UCX_NET_DEVICES` | The example requires it. NIXL picks network devices on its own, but on nodes with several RDMA NICs, workers that see every NIC can stall at startup, so pin the NIC attached directly to the worker's GPU (marked `PIX` for that GPU in `nvidia-smi topo -m`), for example `mlx5_5:1`. On a multi-GPU node the right NIC depends on which GPU the Pod gets. |
   | `DYNAMO_RDMA_RESOURCE` | The RDMA extended resource your cluster exposes. |

2. Wait until the frontend and both worker Pods are `1/1 Running`:

   ```bash
   kubectl get pods -n "$NAMESPACE" -w
   ```

3. Port-forward port 8000 of the frontend Pod, then confirm a cross-worker transfer with [Verify and Monitor KVCR with vLLM](kvcr-testing-monitoring-vllm.md), using the two worker IDs. Each worker serves its metrics on port 9090 of its Pod; reach a worker's metrics with `kubectl port-forward -n "$NAMESPACE" pod/<worker-pod> 9090` and read them at `localhost:9090/metrics`.

   The example sets no `--block-size`, so blocks are 16 tokens and `cached_tokens` is a multiple of 16.

To change the pool size, edit `secondary_g2_slots` in the worker's `kv-transfer-config`; see [Size the Tiers](configure-kvcr-vllm.md#size-the-tiers).

## Related Topics

- [Configure KVCR for vLLM](configure-kvcr-vllm.md)
- [Run KVCR Guard on Kubernetes with vLLM](kvcr-guard-kubernetes-vllm.md)
- [Verify and Monitor KVCR with vLLM](kvcr-testing-monitoring-vllm.md)
