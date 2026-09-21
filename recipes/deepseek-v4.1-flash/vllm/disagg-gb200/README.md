<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek-V4.1-Flash disaggregated serving on GB200 (vLLM)

Serves `deepseek-ai/DeepSeek-V4.1-Flash` through the Dynamo KV-aware router on
12 GB200 GPUs: two TP4 prefill workers and one TP4 decode worker, with KV
handed over NIXL/UCX.

| Role | Replicas | TP | Batch budget | Long-prefill cap | Decode CUDA-graph cap |
| --- | ---: | ---: | --- | --- | --- |
| Prefill | 2 | 4 | 32,768 | 4,096 | 512 |
| Decode | 1 | 4 | default | — | 1,024 |

Both pools carry the [aggregated recipe](../agg-gb200/)'s full flag set:
MXFP4 experts, MXFP4 sparse-indexer KV, `deep_gemm_mega_moe` with expert
parallelism and EPLB, DSpark with 3 draft tokens, and a 1,024-token
prefix-cache retention interval. The speculative config is identical on both
roles so the draft state and KV layout match across the handoff.

This is the highest measured throughput per GPU of the four vLLM variants, at
roughly 30 percent above the aggregated recipe on 50 percent more hardware, and
it also wins the inter-token latency tail by a wide margin. It pays for that
with a far higher time to first token. The two are not interchangeable; pick by
the TTFT your workload accepts.

## Clique placement

A GB200 node holds 4 GPUs, so each TP4 worker is one node and KV always crosses
nodes.

> [!WARNING]
> **All pools must share one NVLink clique.** A fabric memory handle cannot be
> imported across cliques, and a split placement does **not** error: the
> deployment reports Ready and chat completions return HTTP 200 with
> `content: null`. The `ComputeDomain` in this manifest is the supported
> mechanism for that guarantee — the driver sizes it from the pods that claim
> its channel.

This recipe requires the NVIDIA DRA driver with ComputeDomain support. A
dangling `resourceClaim` leaves pods `SchedulingGated` indefinitely with no
error, so verify the driver is installed before applying.

`UCX_CUDA_IPC_ENABLE_MNNVL` is `y` here, because multi-node NVLink is a valid
transport between pools in one clique. The B200 recipe sets it to `n`. There is
deliberately no `NCCL_*` tuning: the KV transfer is NIXL over UCX, and NCCL
only forms the intra-worker TP4 group, which lives inside one node. Setting
`NCCL_MNNVL_ENABLE`, `NCCL_NVLS_ENABLE`, or `NCCL_P2P_LEVEL` broke that group
with "unhandled system error" in `ncclCommInitRank`.

## Variants

| Variant | Adds |
| --- | --- |
| `deploy-generic.yaml` | Nothing. Requests `rdma/ib` on each worker. |
| `deploy-gke-rdma.yaml` | GKE multi-network RDMA, replacing `rdma/ib`. |

GKE GB200 node pools advertise four RDMA networks rather than an `rdma/ib`
resource. The `gke-rdma` Component attaches all four, drops the base's
`rdma/ib` request, and sets `UCX_IB_GID_INDEX=3` — index 3 is the RoCEv2
convention, which applies on those networks and does not apply on InfiniBand.

> [!WARNING]
> Without an RDMA device on either path, UCX falls back to TCP silently — no
> error, and the deployment still reports Ready — and time to first token
> degrades by roughly an order of magnitude. `rdma/ib` is what the reference
> non-GKE cluster advertises; other providers differ. This is the first field
> to retarget.

## Deploy

The namespace needs the `shared-model-cache` PVC populated with the pinned
model snapshot — see [`../../model-cache/`](../../model-cache/). Twelve GB200
GPUs across three ARM64 nodes in one NVLink clique must be schedulable, with
512 GiB of host memory and RDMA devices per worker. Set `CONTEXT` and
`NAMESPACE`, then run from this directory:

```bash
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" apply -f deploy-generic.yaml
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" wait --for=condition=Ready pod \
  -l nvidia.com/dynamo-graph-deployment-name=dsv41-flash-vllm-gb200-disagg \
  --timeout=7200s
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" port-forward \
  service/dsv41-flash-vllm-gb200-disagg-frontend 8000:8000
```

> [!IMPORTANT]
> First boot takes roughly an hour per worker. The checkpoint is 510 GB over 48
> shards, and a large silent host-memory load runs with the GPUs at 0 percent.

After the workers are Ready, send a real chat completion and check the response
body is non-null — that is the check that catches a cross-clique placement,
which no readiness signal reports. Then confirm KV is moving over the expected
transport from the NIXL metrics on the worker `/metrics` endpoint, with
`nixl_num_failed_transfers` and `nixl_num_failed_notifications` both zero.

## Benchmark

See [`../../perf/README.md`](../../perf/README.md). The validated operating
point for this variant is concurrency 168.

## Edit and render

Edit `kustomize/base/deploy.yaml` or the Component under
`kustomize/components/`; the `deploy-*.yaml` files are generated. From the
repository root:

```bash
python3 scripts/kustomize-matrix.py unfold recipes/deepseek-v4.1-flash/vllm/disagg-gb200/.kustomize-matrix.yaml
python3 scripts/kustomize-matrix.py render recipes/deepseek-v4.1-flash/vllm/disagg-gb200/.kustomize-matrix.yaml
```

Either checked-in overlay can be applied directly:

```bash
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" apply -k kustomize/overlays/gke-rdma
```

To compose extra Components without checking in an overlay, use
`scripts/kustomize-matrix.py compose`.
