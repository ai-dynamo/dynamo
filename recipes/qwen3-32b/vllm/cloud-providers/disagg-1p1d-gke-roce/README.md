# Qwen3-32B 1P1D Cross-Node Disagg — GCP GKE A4X (GB200 NVL72, RoCEv2) Variant

This is the **GCP GKE / A4X / RoCEv2-over-ConnectX-7** member of the cross-provider benchmark family. For the family-wide topology, perf-measurement protocol, and result format, see [`../README.md`](../README.md). This README only documents the GKE-A4X-specific overrides + measured results.

## What this variant overrides (vs. the family template)

| Override | Value |
|---|---|
| **RDMA NIC attachment** | Pod-level Multus annotation `networking.gke.io/interfaces` listing `rdma-0..rdma-3` (4 NICs) |
| **NIXL backend selection** | Defaults (UCX); no `DYN_KVBM_NIXL_BACKEND_*` env |
| **Transport env** | None overridden — UCX auto-probe binds the 4 RoCE NICs (rc_mlx5 over mlx5_0..3) on A4X, so `UCX_TLS`/`UCX_NET_DEVICES` stay unset, and the image's default `LD_LIBRARY_PATH` (already carries the NIXL libs) is used as-is. |
| **HostPath mounts** | `/home/kubernetes/bin/gib` → `/usr/local/gib`, `/home/kubernetes/bin/nvidia` → `/usr/local/nvidia` (GIB NCCL plugin + GKE-injected NVIDIA driver libs) |

## GKE-A4X-specific findings

- **All-or-nothing GPU allocation.** GKE GPUDirect-RDMA requires a pod to use *all* GPUs **and** all RDMA NICs on the node — RDMA can't be shared between pods on a node. Request exactly `gpu: '4'` (a full A4X node).
- **No cross-node NVLink shortcut without a ComputeDomain.** GB200 NVL72 racks share NVLink across the rack, but cross-node GPU P2P (MNNVL / `cuda_ipc`) requires an IMEX channel provisioned by a `ComputeDomain`, which this recipe does not declare. So cross-node KV transfer stays on RoCE regardless of rack placement — no rack-level separation is needed. (On A4X's 4-GPU nodes the two TP=4 workers can't co-locate anyway, so they always land on separate nodes.)
- **`networking.gke.io/default-interface: eth0`** must be set or the pod gets a default route through one of the RDMA NICs and TCP traffic (control plane, etcd, NATS, HF download) breaks.

## Results

GKE A4X GB200 + 4-NIC ConnectX-7 RoCE

**Workload:** Mooncake conversation_trace (12,031 sessions, 3.23 req/s fixed schedule), aiperf v0.6.0.

### Headline (aggregate transport + steady-state)

| Metric | Value |
|---|---|
| **Aggregate NIXL KV BW** | **~10.72 GB/s**  |
| Benchmark wall-clock duration | 3555 sec |
| Request throughput | 3.38 req/s (kept pace with 3.23 req/s arrival, no queue buildup) |
| Successful request count | **12,031 / 12,031 (100%)** |
| mean TTFT |  3,733 ms (P50: 2,599, P90: 8,975, P99: 16,074) |
| mean ITL  | 11.38 ms (P50: 10.09, P99: 30.45) |

### Goodput

| Metric | Value |
|---|---|
| **Goodput @ TTFT<2s, ITL<25ms** | **1.42 req/s** |
| GoodRequestCount | 5,031 / 12,031 (41.8% met SLA) |

**This is the only cluster in the family with non-zero meaningful goodput on 1P1D Mooncake.**
