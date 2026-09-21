<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek-V4.1-Flash disaggregated serving on B200 (vLLM)

Serves `deepseek-ai/DeepSeek-V4.1-Flash` through the Dynamo KV-aware router on
12 B200 GPUs: two TP4 prefill workers and one TP4 decode worker, with KV
handed over NIXL/UCX.

| Role | Replicas | TP | Batch budget | Long-prefill cap | Decode CUDA-graph cap |
| --- | ---: | ---: | --- | --- | --- |
| Prefill | 2 | 4 | 32,768 | 4,096 | 512 |
| Decode | 1 | 4 | default | — | 1,024 |

Both pools carry the [aggregated recipe](../agg-b200/)'s full flag set:
MXFP4 experts, MXFP4 sparse-indexer KV, `deep_gemm_mega_moe` with expert
parallelism and EPLB, DSpark with 3 draft tokens, and a 1,024-token
prefix-cache retention interval. The speculative config is identical on both
roles so the draft state and KV layout match across the handoff.

Against the aggregated recipe this gains roughly a quarter more throughput per
GPU and a much shorter inter-token latency tail, on 50 percent more hardware,
and pays for it with a far higher time to first token. The two are not
interchangeable; pick by the TTFT your workload accepts.

## Ratio and transport

2 prefill to 1 decode is balanced for this workload, not a free parameter. Both
sides saturate together at the ship point: the prefill queue grows into TTFT,
which still holds headroom, while decode inter-token latency grows into the
per-user token rate, which is what binds. Other ratios measured worse or the
same per GPU — scaling moves the absolute number, not the number per GPU.

> [!WARNING]
> Each worker requests an `rdma/ib` device. **Without an RDMA device, UCX falls
> back to TCP on `eth0` silently** — no error, and the deployment still reports
> Ready — and time to first token degrades by roughly an order of magnitude.
> `rdma/ib` is what the reference cluster advertises; other providers expose
> RDMA differently. This is the first field to retarget.

`UCX_TLS` keeps `tcp` in the list deliberately. `cuda_ipc` and `cuda_copy` are
memory-type transports only and cannot carry NIXL's UCX active-message control
plane, so removing `tcp` breaks the handshake rather than just the data path.
`UCX_IB_GID_INDEX` is deliberately unset: index 3 is the RoCEv2 convention, and
these ports report `link_layer: InfiniBand`.

## Variants

| Variant | Adds |
| --- | --- |
| `deploy-generic.yaml` | Nothing. Portable starting point. |
| `deploy-ib-device-pin.yaml` | Pins `UCX_NET_DEVICES` to a named set of HCAs. |

The reference nodes expose twelve `mlx5` devices at three speeds plus an
Ethernet bond, and UCX selects by heuristic when `UCX_NET_DEVICES` is unset, so
it can pick a slow port. The pinned variant names only the fastest eight.
**That device list is a property of the node hardware, not of this recipe** —
read your own nodes with `ibstat` before using it, and edit the value to match.

## Deploy

The namespace needs the `shared-model-cache` PVC populated with the pinned
model snapshot — see [`../../model-cache/`](../../model-cache/). Twelve B200
GPUs in three groups of four must be schedulable, with 512 GiB of host memory
and RDMA devices per worker. Set `CONTEXT` and `NAMESPACE`, then run from this
directory:

```bash
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" apply -f deploy-generic.yaml
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" wait --for=condition=Ready pod \
  -l nvidia.com/dynamo-graph-deployment-name=dsv41-flash-vllm-b200-disagg \
  --timeout=7200s
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" port-forward \
  service/dsv41-flash-vllm-b200-disagg-frontend 8000:8000
```

> [!IMPORTANT]
> First boot takes roughly an hour per worker. The checkpoint is 510 GB over 48
> shards, and a large silent host-memory load runs with the GPUs at 0 percent.

After the workers are Ready, confirm KV is actually moving over RDMA rather
than TCP before trusting any measurement: check the NIXL transfer metrics on
the worker `/metrics` endpoint, and confirm `nixl_num_failed_transfers` and
`nixl_num_failed_notifications` are both zero.

## Benchmark

See [`../../perf/README.md`](../../perf/README.md). The validated operating
point for this variant is concurrency 160.

## Edit and render

Edit `kustomize/base/deploy.yaml` or the Component under
`kustomize/components/`; the `deploy-*.yaml` files are generated. From the
repository root:

```bash
python3 scripts/kustomize-matrix.py unfold recipes/deepseek-v4.1-flash/vllm/disagg-b200/.kustomize-matrix.yaml
python3 scripts/kustomize-matrix.py render recipes/deepseek-v4.1-flash/vllm/disagg-b200/.kustomize-matrix.yaml
```

Either checked-in overlay can be applied directly:

```bash
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" apply -k kustomize/overlays/ib-device-pin
```

To compose extra Components without checking in an overlay, use
`scripts/kustomize-matrix.py compose`.
