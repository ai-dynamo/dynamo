---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Run KVCR on Kubernetes with SGLang
subtitle: Deploy SGLang workers that share KV cache through KVCR in HiCache mode
---

> [!WARNING]
> Preview. Needs SGLang [#45647](https://github.com/sgl-project/sglang/pull/45647) (stacked on [#45646](https://github.com/sgl-project/sglang/pull/45646)) and the image recipe in KVCR [#114](https://github.com/ai-dynamo/kvcr/pull/114), both unmerged, and Dynamo with [#16117](https://github.com/ai-dynamo/dynamo/pull/16117).
>
> Run end to end on two H200 nodes on 2026-10-29: the second worker reused 2,112 of 2,161 prompt tokens from the first through KVCR. The worker's liveness probe allows 10 minutes, because SGLang builds a kernel on its first request and the default probe restarts the worker before it finishes.

The [`sglang-agg-hicache.yaml`](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/sglang/deploy/kvcr/sglang-agg-hicache.yaml) example deploys a frontend with the KV router and two SGLang workers on two nodes. Each worker runs KVCR as its HiCache storage backend with its own KVCR pool (`local_dram_bytes`, 8 GiB in the example). The cache is lost when a worker restarts; for crash recovery, see [Run KVCR Guard on Kubernetes with SGLang](kvcr-guard-kubernetes-sglang.md).

This example uses Kubernetes discovery; it does not need etcd.

## Before You Begin

- Two GPU nodes, each with one free GPU and one share of an RDMA resource (`rdma/shared_ib` by default).
- The Dynamo operator with the Grove workload provider, and `kubectl` set to the target cluster context.
- An `hf-token-secret` in the target namespace.
- A runtime image built from `Dockerfile.quick-start-sglang` in [KVCR #114](https://github.com/ai-dynamo/kvcr/pull/114); see [Configure KVCR for SGLang](configure-kvcr-sglang.md#before-you-begin).

## Deploy

1. From `examples/backends/sglang/deploy/kvcr` in the Dynamo repository, set the variables and apply the manifest. `envsubst` fills in only the listed variables; the worker command uses `$POD_IP`, which must stay as it is.

   ```bash
   export NAMESPACE=<namespace>
   export DYNAMO_SGLANG_IMAGE=<image>@sha256:<digest>
   export DYNAMO_VERSION=<dynamo-version>
   export DYNAMO_UCX_NET_DEVICES=<nic>:1
   export DYNAMO_RDMA_RESOURCE=rdma/shared_ib

   envsubst '${DYNAMO_SGLANG_IMAGE} ${DYNAMO_VERSION} ${DYNAMO_RDMA_RESOURCE} ${DYNAMO_UCX_NET_DEVICES}' \
     < sglang-agg-hicache.yaml | kubectl apply -n "$NAMESPACE" -f -
   ```

   | Variable | Value |
   |---|---|
   | `DYNAMO_VERSION` | The Dynamo version in the image, for example `1.7.0`. The operator reads it from `runtimeVersionOverride`, because the image tag is not a Dynamo version. |
   | `DYNAMO_UCX_NET_DEVICES` | The example requires it. NIXL picks network devices on its own, but on nodes with several RDMA NICs, workers that see every NIC can stall at startup, so pin the NIC attached directly to the worker's GPU (marked `PIX` for that GPU in `nvidia-smi topo -m`), for example `mlx5_5:1`. On a multi-GPU node the right NIC depends on which GPU the Pod gets. |
   | `DYNAMO_RDMA_RESOURCE` | The RDMA extended resource your cluster exposes. |

2. Wait until the frontend and both worker Pods are `1/1 Running`:

   ```bash
   kubectl get pods -n "$NAMESPACE" -w
   ```

## Check It Works

1. Port-forward port 8000 of the frontend Pod.

2. Get the two worker IDs. Send any short request a few times and read `"decode_worker_id"` in each response:

   ```bash
   curl -s localhost:8000/v1/completions -H 'content-type: application/json' \
     -d '{"model":"Qwen/Qwen3-0.6B","prompt":"hello","max_tokens":1,"nvext":{"extra_fields":["worker_id"]}}'
   ```

3. Create `request.json` with a prompt of at least 1,000 tokens that you have not sent before:

   ```json
   {"model": "Qwen/Qwen3-0.6B", "prompt": "<at least 1,000 tokens of new text>", "max_tokens": 8}
   ```

4. Send it to worker 0, wait 15 seconds, then send it to worker 1:

   ```bash
   curl -s localhost:8000/v1/completions -H 'content-type: application/json' \
     -H 'x-dynamo-worker-instance-id: <worker-0-id>' -d @request.json

   curl -s localhost:8000/v1/completions -H 'content-type: application/json' \
     -H 'x-dynamo-worker-instance-id: <worker-1-id>' -d @request.json
   ```

5. In the second response, `cached_tokens` is greater than 0 and a multiple of 64. The second worker's log shows the transfer:

   ```bash
   kubectl logs -n "$NAMESPACE" <worker-1-pod> | grep 'HiCache prefetch success'
   ```

   `loaded=<n>` with `matched=0` means all `<n>` tokens came from the other worker through KVCR.

To change the pool size, edit `local_dram_bytes` in the worker's `--hicache-storage-backend-extra-config`; see [Size the Tiers](configure-kvcr-sglang.md#size-the-tiers).

## Related Topics

- [Configure KVCR for SGLang](configure-kvcr-sglang.md)
- [Run KVCR Guard on Kubernetes with SGLang](kvcr-guard-kubernetes-sglang.md)
- [KV Cache Runner (KVCR)](kvcr.md)
