# Qwen3-32B 1P1D Cross-Node Disagg — Nebius InfiniBand Variant

This is the **Nebius MK8S NDR InfiniBand** member of the cross-provider benchmark family. For the family-wide topology, perf-measurement protocol, and result format, see [`../README.md`](../README.md). This README only documents the Nebius-specific overrides + measured results.

## What this variant overrides (vs. the family template)

| Override | Value |
|---|---|
| **RDMA resource** | `rdma/ib: "1"` under `resources.limits.custom`. This is the Network Operator's RDMA-shared-device-plugin slot count, **not** an HCA count. One slot grants the pod access to all 8 HCAs via `/dev/infiniband/*`. Do NOT request `rdma/ib: 8` — pod will be unschedulable. |
| **NIXL backend selection** | Defaults (UCX). No `DYN_KVBM_NIXL_BACKEND_LIBFABRIC` / `_UCX` env. |
| **Transport env** | `UCX_NET_DEVICES=mlx5_0:1..7` (pins KV transfer to the 8 NDR HCAs), `UCX_IB_GPU_DIRECT_RDMA=yes`, `NCCL_SOCKET_IFNAME=eth0`. **No `UCX_TLS`** — let UCX auto-probe. |
| **HostPath volumes** | `/dev/infiniband` (hostPath) and `/dev/shm` (tmpfs emptyDir, 64 GiB) bind-mounted into both worker pods. |



## Nebius-specific gotchas

- **`privileged: true` + `IPC_LOCK`** are required for UCX `ucp_mem_map` on VRAM. `IPC_LOCK` alone is not enough.


## Results

H200 + 8-NIC NDR IB (TP=4)

**Workload:** Mooncake conversation trace, 12,031 requests at 3.23 req/s fixed schedule, aiperf v0.6.0.

| Metric | Value |
|---|---:|
| **Trace completion** | **12,031 / 12,031 (100%)** |
| **Errors / cancellations** | 0 |
| **Aggregate NIXL KV BW** | **5.96 GB/s** |
| **Per-rank KV BW** | 1.49 GB/s |
| **Mean TTFT** | 1,510 s (queue-bound — Mooncake arrival rate too aggressive for 1P1D at TP=4 on H200) |
| **Mean ITL** | **9.14 ms** (fastest of the cross-CSP family) |
| Output throughput per user | 110 tok/s (mean) |