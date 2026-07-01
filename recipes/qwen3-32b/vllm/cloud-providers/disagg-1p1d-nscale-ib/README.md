# Qwen3-32B 1P1D Cross-Node Disagg — Nscale NDR-IB Variant (B200)

This is the **Nscale managed K8s** member of the cross-provider benchmark family. For the family-wide topology, perf-measurement protocol, and result format, see [`../README.md`](../README.md). This README only documents the Nscale-specific overrides + measured results.

## What this variant overrides

| Override | Value |
|---|---|
| **RDMA resource** | `rdma/ib: "1"` per pod (Network Operator shared-device plugin slot; grants all 8 HCAs via `/dev/infiniband/*`) |
| **NIXL backend selection** | Defaults — UCX backend (no `DYN_KVBM_NIXL_BACKEND_*` env overrides) |
| **Transport env** | `UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_10:1,mlx5_11:1` (the 8 compute-fabric NICs only). **Explicit list is load-bearing on Nscale**  The side fabric `mlx5_6..9` (MTU 512, different IB subnet) must also be excluded — mixing fabrics causes `NIXL_ERR_REMOTE_DISCONNECT` at first transfer (observed historically with `mlx5_0..7` list). **No `UCX_TLS`** — let UCX auto-probe transports (allowlists break wireup AM selection). |
| **Host volume mount** | `/dev/infiniband` (hostPath) + `/dev/shm` (emptyDir, 64Gi) |


## Nscale-specific findings

- **`rdma/ib: "1"` is a slot count, not an HCA count.** Requesting `rdma/ib: "8"` will leave pods Pending (no node will satisfy it under the shared-device plugin model). One slot grants access to all 8 NDR HCAs via `/dev/infiniband/`.
- **`privileged: true` + `IPC_LOCK`** are still required for NIXL VRAM memory registration via UCX `ucp_mem_map` on B200.
- **`UCX_NET_DEVICES` must list the compute-fabric NICs explicitly** — otherwise workers won't start.

## Results

Nscale B200 + 8-NIC NDR IB (TP=4)


**Workload:** Mooncake conversation trace, 12,031 requests at 3.23 req/s fixed schedule, aiperf v0.6.0.

| Metric | Value |
|---|---:|
| **Wall time** | 3,557 s (~59 min) |
| **Trace completion** | **12,031 / 12,031 (100%)** |
| **Errors / cancellations** | 0 |
| **NIXL agent_rx_bytes (decode, rank-0)** | 9.52 TB |
| **Aggregate NIXL KV BW** | **10.71 GB/s** (`9.52 TB × TP=4 / 3,557 s`)  |
| **Per-rank KV BW** | 2.68 GB/s |
| **Mean TTFT** | 8.35 s (mildly queue-bound) |
| **Mean ITL** | ~16 ms (steady-state) |
| Output throughput per user | ~75 tok/s |
