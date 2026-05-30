# Qwen3-32B 1P1D Cross-Node Disagg — Azure AKS InfiniBand Variant

This is the **Azure AKS + InfiniBand** member of the cross-provider benchmark family. For the family-wide topology, perf-measurement protocol, and result format, see [`../README.md`](../README.md). This README only documents the AKS-specific overrides + measured results.

## What this variant overrides

| Override | Value |
|---|---|
| **RDMA resource** | `rdma/shared_ib: "1"` per pod (boolean scheduling hint; the NVIDIA Network Operator's RDMA shared device plugin grants the pod access to all 8 IB rails) |
| **NIXL backend selection** | _(default)_ — runtime defaults to UCX when `DYN_KVBM_NIXL_BACKEND_LIBFABRIC` is unset. No `DYN_KVBM_NIXL_BACKEND_*` env required. |
| **Transport env** | `UCX_IB_GPU_DIRECT_RDMA=yes`, `NCCL_IB_HCA=mlx5_0,...,mlx5_7`, `NCCL_SOCKET_IFNAME=eth0` (control plane only). **No `UCX_TLS`** — let UCX auto-probe (an explicit allowlist breaks the wireup AM transport selection) |
| **Volume mounts** | `/dev/infiniband` from host (hostPath) and a 64 GiB tmpfs `/dev/shm` per worker pod — Azure shared-device plugin grants access but the device files still need to be in the pod namespace |


## AKS-specific findings

- **`rdma/shared_ib: "1"` is a boolean scheduling hint**, not a count of NICs. The shared-device plugin admits the pod and exposes all 8 IB rails (`mlx5_0..mlx5_7` — ConnectX-6 HDR-class on AKS-dev A100; ConnectX-7 NDR on H100/H200 ND_isr variants) at once; do not set this to `8`.
- **Do NOT set `UCX_NET_DEVICES`.** Auto-probe correctly picks the 8 compute IB NICs at both TP=4 and TP=8 on AKS-dev a100a (verified by smoke tests; see Results section's Test trail). The earlier-documented "must list all 8 rails" advice was wrong — omitting the env var is correct, and `UCX_TLS` is what would have caused a silent TCP fallback (separate issue). The earlier-documented "TP=8 requires a 4-NIC explicit list" gotcha was also wrong: that failure mode is specific to an *explicit 8-NIC list*; auto-probe stays under the DMA-BUF kernel registration threshold on its own.
- **Mount `/dev/infiniband` from the host (hostPath)** on Azure. The shared-device plugin grants access, but the kernel device files still need to be in the pod's mount namespace.

## Results 

A100-SXM4-80GB + 8-NIC ConnectX-6 HDR IB (TP=4)

**Workload:** Mooncake conversation trace, 12,031 requests at 3.23 req/s fixed schedule, aiperf v0.6.0.

| Metric | Value |
|---|---:|
| **Wall time** | 7,194 s (~120 min) |
| **Trace completion** | **7,000 / 12,031 (58%)** — back half timed out under queue pressure |
| **Errored requests** | 39 connection timeouts |
| **NIXL agent_rx_bytes (decode, rank-0)** | 3.39 TB |
| **Aggregate NIXL KV BW** | **1.88 GB/s** (`3.39 TB × TP=4 / 7,194 s`) |
| **Per-rank KV BW** | 0.47 GB/s |
| **vs prior TP=8 (3.23 GB/s)** | ratio 0.58 — close to half (rank count halved); per-rank actually rose 0.40 → 0.47 GB/s |
| Mean ITL (steady-state) | n/a (not captured in cross-CSP CSV) |
| Status | queue-bound; A100 prefill cannot keep up with Mooncake arrival rate at TP=4 |

**aiperf artifact:** `dynamo-aks-dev / jihao / perf-cache PVC / artifacts/Qwen3-32B_aks-ib_20260527-2102/`

### Cross-CSP family context

| Cluster | GPU | KV BW (TP=4) | Per-rank | Completion |
|---|---|---:|---:|---:|
| GKE A4X | GB200 | 10.71 | 2.68 | 100% |
| Nscale | B200 | 10.71 | 2.68 | 100% |
| Nebius | H200 | 5.96 | 1.49 | 100% |
| AWS p5 | H100 | 5.64 | 1.41 | 100% |
| **AKS** | **A100** | **1.88** | **0.47** | **58%** |

AKS A100-IB lands at the bottom of the family on both aggregate and completion. The gap is **GPU-generation-dominated** (A100 prefill can't keep pace), not fabric-dominated — the IB NIC utilization here is sub-1% of headroom (1.6 Tb/s available, ~1.9 GB/s used = ~1%). The TP=4 result also drops 5,031 requests because the queue overflows past the engine's per-request timeout.

### Deploy outcome (history)

Initial recipe (TP=8, cross-pool) failed three times before reaching Ready. Final working configuration required four cluster-specific fixes documented under "AKS-dev gotchas":

1. ~~Frontend `nodeAffinity` to GPU pool~~ — AKS default-pool CPU nodes (123 GiB OS disk) evict the 15 GB `vllm-runtime` image. **Superseded 2026-05-29:** replaced by a generic `ephemeral-storage: 30Gi` request on Frontend, which makes the scheduler exclude too-small-disk nodes on any cluster (CSP-generic; no `agentpool` pinning needed). Tune the value up if your cluster's GPU pool nodes have <60 GiB allocatable ephemeral-storage.
2. Worker `--model` as local PVC path with `served-model-name` — ModelExpress always fetches from HF even when cache complete, hits 429 rate limits.
3. ~~Worker `nodeAffinity` to single agentpool `a100a`~~ — claim was: `a100a` and `a100b` are on different IB partitions (pkey 0x8019 vs 0x8018); cross-pool RDMA handshake fails. **Superseded 2026-05-29:** dropped from the recipe because the cross-pool failure was never independently verified by us, and pinning to a specific pool name (`a100a`) was AKS-cluster-specific (broke CSP-generic portability). The GPU resource request alone restricts workers to GPU-capable nodes. **Per-cluster override:** if your AKS cluster has confirmed cross-pool IB-partition isolation, re-add `nodeAffinity` selecting your specific GPU pool.
4. ~~Worker `UCX_NET_DEVICES` limited to 4 NICs~~ — historically this was added at TP=8 because an *explicit* 8-NIC list hit the DMA-BUF kernel registration timeout (8 ranks × 8 NICs = 64 regs). **Superseded 2026-05-29:** UCX auto-probe stays under the threshold without an explicit list at both TP=4 and TP=8. Recipe ships without `UCX_NET_DEVICES`.
