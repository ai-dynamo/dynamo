# Qwen3-32B 1P1D Cross-Node Disagg — GCP GKE A4X (GB200 NVL72, RoCEv2) Variant

This is the **GCP GKE / A4X / RoCEv2-over-ConnectX-7** member of the cross-provider benchmark family. For the family-wide topology, perf-measurement protocol, and result format, see [`../README.md`](../README.md). This README only documents the GKE-A4X-specific overrides + measured results.

## What this variant overrides (vs. the family template)

| Override | Value |
|---|---|
| **RDMA NIC attachment** | Pod-level Multus annotation `networking.gke.io/interfaces` listing `rdma-0..rdma-3` (4 NICs); |
| **NIXL backend selection** | Defaults (UCX); no `DYN_KVBM_NIXL_BACKEND_*` env |
| **Transport env** | `LD_LIBRARY_PATH=/opt/nvidia/nvda_nixl/lib64:/usr/local/gib/lib64:/usr/local/nvidia/lib64`. **No `UCX_TLS`** — let UCX auto-probe (allowlists break wireup AM selection).  |
| **HostPath mounts** | `/home/kubernetes/bin/gib` → `/usr/local/gib`, `/home/kubernetes/bin/nvidia` → `/usr/local/nvidia` (GIB NCCL plugin + GKE-injected NVIDIA driver libs) |

## GKE-A4X-specific findings

- **All-or-nothing GPU allocation.** Partial GPU pods cannot get RDMA NICs on GKE. Request exactly `gpu: '4'`.
- **NVL72 NVLink shortcut.** GB200 NVL72 racks expose NVLink across the entire 72-GPU rack. If both pods land in the same rack, NIXL/UCX may pick `cuda_ipc` over RoCE — measured BW will reflect NVLink (~900 GB/s peer-to-peer), not the RoCE fabric (~50 GB/s peak per pod).
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
