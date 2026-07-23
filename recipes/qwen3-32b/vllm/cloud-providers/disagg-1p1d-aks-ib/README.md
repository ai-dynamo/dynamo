# Qwen3-32B 1P1D Cross-Node Disagg — Azure AKS InfiniBand Variant

This is the **Azure AKS + InfiniBand** member of the cross-provider benchmark family. For the family-wide topology, perf-measurement protocol, and result format, see [`../README.md`](../README.md). This README documents the AKS-specific overrides.

## What this variant overrides

| Override | Value |
|---|---|
| **RDMA resource** | `rdma/shared_ib: "1"` per pod (boolean scheduling hint; the NVIDIA Network Operator's RDMA shared device plugin grants the pod access to the IB rails) |
| **NIXL backend** | _(default UCX)_ — the runtime defaults to UCX when `DYN_KVBM_NIXL_BACKEND_LIBFABRIC` is unset; no selector env required |
| **Transport env** | `UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1` (compute-fabric NICs), `UCX_IB_GPU_DIRECT_RDMA=yes`; do **not** set `UCX_TLS` |
| **Volume mounts** | `/dev/infiniband` from host (hostPath) per worker pod — the shared-device plugin grants access, but the device files still need to be in the pod's mount namespace |

## AKS-specific findings

- **`UCX_NET_DEVICES` is required.** It restricts KV transfer to the compute-fabric NICs (`mlx5_0..3`). Without it, UCX auto-probe may bind a non-compute (side-fabric) NIC — KV-transfer performance degrades sharply and the NIXL KV-bandwidth telemetry counter (`agent_rx_bytes`) reads zero. Setting the allowlist restores both. (`UCX_RNDV_THRESH` is not required once `UCX_NET_DEVICES` is set.)
- **Do not set `UCX_TLS`.** An explicit transport allowlist can omit the internal transports UCX needs for connection wireup; let UCX choose transports.
- **`rdma/shared_ib: "1"` is a boolean scheduling hint**, not a count of NICs. The shared-device plugin admits the pod and exposes the IB rails (`mlx5_0..mlx5_7`) at once; do not set this to `8`.
- **Mount `/dev/infiniband` from the host (hostPath).** The shared-device plugin grants access, but the kernel device files still need to be in the pod's mount namespace.

## Results

A100-SXM4-80GB + 8-NIC ConnectX-6 HDR IB (TP=4)

**Workload:** Mooncake conversation trace, 12,031 requests at 3.23 req/s fixed schedule, aiperf v0.6.0.

| Metric | Value |
|---|---:|
| **Trace completion** | queue-bound — the full trace does not drain in a bounded window |
| **With-output success (steady-state window)** | 100% |
| **Aggregate NIXL KV BW** | **~1.7 GB/s** |
| **Per-rank KV BW** | ~0.42 GB/s |
| **Mean TTFT** | 902s (queue-bound) |
| **Mean ITL (steady-state)** | **~22 ms** (p50 20, p99 49) |

The A100 is GPU-class limited: prefill cannot keep pace with the Mooncake arrival rate at TP=4, so TTFT runs into the minutes and the full trace does not complete in a bounded window. This is a GPU limit, not a transport one — IB/NIXL utilization stays a small fraction of fabric headroom.
