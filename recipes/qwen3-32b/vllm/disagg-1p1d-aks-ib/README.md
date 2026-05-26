# Qwen3-32B 1P1D Cross-Node Disagg — Azure AKS InfiniBand Variant

This is the **Azure AKS + InfiniBand** member of the cross-provider benchmark family. For the family-wide topology, perf-measurement protocol, and result format, see [`../disagg-1p1d-base/README.md`](../disagg-1p1d-base/README.md). This README only documents the AKS-specific overrides + measured results.

## What this variant overrides (vs. the base template)

| Override | Value |
|---|---|
| **Container image** | `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.1` (standard image — no `-efa` suffix; UCX/IB are built in) |
| **RDMA resource** | `rdma/shared_ib: "1"` per pod (boolean scheduling hint; the NVIDIA Network Operator's RDMA shared device plugin grants the pod access to all 8 IB rails) |
| **NIXL backend selection** | _(default)_ — runtime defaults to UCX when `DYN_KVBM_NIXL_BACKEND_LIBFABRIC` is unset. No `DYN_KVBM_NIXL_BACKEND_*` env required. |
| **Transport env** | `UCX_TLS=rc,cuda_copy,cuda_ipc`, `UCX_NET_DEVICES=mlx5_ib0:1,...,mlx5_ib7:1`, `UCX_IB_GPU_DIRECT_RDMA=yes`, `NCCL_IB_HCA=mlx5_ib0,...,mlx5_ib7`, `NCCL_IB_GID_INDEX=3`, `NCCL_SOCKET_IFNAME=eth0` (control plane only) |
| **Volume mounts** | `/dev/infiniband` from host (hostPath) and a 64 GiB tmpfs `/dev/shm` per worker pod — Azure shared-device plugin grants access but the device files still need to be in the pod namespace |
| **Model PVC** | `model-cache` (per-cluster PVC name; differs from AWS `shared-model-cache`) |
| **Cluster** | `dynamo-aks-dev` (4 nodes, 8x A100-SXM4-80GB, `rdma/ib: 8` + `rdma/shared_ib: 63`); fallback `dynamo-aks-exp` (2 nodes, same shape) |
| **Dropped** | EFA-specific `FI_*` env block; EFA `LD_LIBRARY_PATH=/opt/amazon/efa/...` (not present in standard image); `DYN_KVBM_NIXL_BACKEND_LIBFABRIC/UCX` selector envs |

## Prereqs

- `dynamo-aks-dev` (or `-exp`) access via Teleport (proxy `nv-prd-dgxc.teleport.sh:443`).
- Namespace `jihao`.
- `model-cache` PVC populated with `Qwen/Qwen3-32B`. Verify with `tsh kubectl -n jihao get pvc`. If missing, create the PVC and run a model-download Job; see `recipes/qwen3-32b/model-cache/`.
- `perf-cache` PVC for benchmark artifacts (manifest below).
- `hf-token-secret` Secret with `HF_TOKEN` in the `jihao` namespace.
- Dynamo Platform (operator, etcd, NATS) running in the cluster.
- NVIDIA Network Operator installed with `rdmaSharedDevicePlugin` exposing `rdma/shared_ib` on the IB nodes (see [`/notes/csp-rdma/azure-aks-ib.md`](../../../../notes/csp-rdma/azure-aks-ib.md) §3b).

## Deploy

```bash
export NAMESPACE=jihao
tsh kube login dynamo-aks-dev    # or dynamo-aks-exp if dev short on capacity
tsh kubectl config current-context     # sanity check before applying

tsh kubectl apply -f deploy.yaml -n ${NAMESPACE}

tsh kubectl wait --for=condition=ready pod \
  -l nvidia.com/dynamo-graph-deployment-name=q32b-1p1d-aks-ib \
  -n ${NAMESPACE} --timeout=1800s
```

## Verify (run all four checks before benchmarking — see base README §"Sanity-check")

```bash
# 1. Pods Ready + on different nodes (pod anti-affinity working)
tsh kubectl -n ${NAMESPACE} get pods -l nvidia.com/dynamo-graph-deployment-name=q32b-1p1d-aks-ib -o wide

# 2. UCX backend chosen with rc (IB), not tcp fallback
tsh kubectl -n ${NAMESPACE} logs -l nvidia.com/dynamo-component-sub-type=decode --tail=2000 \
  | grep -iE 'nixl|ucx|backend|transport' | head -40
# Expect: "selected transport: rc" / "selected backend: ucx" — NOT "tcp".

# 3. All 8 IB ports Active inside the pod
PREFILL_POD=$(tsh kubectl -n ${NAMESPACE} get pods \
  -l nvidia.com/dynamo-component-sub-type=prefill -o jsonpath='{.items[0].metadata.name}')
tsh kubectl -n ${NAMESPACE} exec $PREFILL_POD -- bash -c 'ibstat | grep -c "State.*Active"'
# Expect 8

tsh kubectl -n ${NAMESPACE} exec $PREFILL_POD -- bash -c 'ucx_info -d | grep -E "Device:|Transport:" | head -40'
# Expect rc_verbs / rc_mlx5 on mlx5_ib0..mlx5_ib7

# 4. NIXL Prometheus exporter responding
tsh kubectl -n ${NAMESPACE} exec $PREFILL_POD -- curl -s localhost:19090/metrics | grep -c '^nixl_'
# Expect > 10
```

## Benchmark

```bash
# perf-cache PVC if missing:
tsh kubectl -n ${NAMESPACE} apply -f - <<'EOF'
apiVersion: v1
kind: PersistentVolumeClaim
metadata: { name: perf-cache }
spec:
  accessModes: [ReadWriteOnce]
  resources: { requests: { storage: 10Gi } }
EOF

tsh kubectl apply -f perf.yaml -n ${NAMESPACE}

tsh kubectl -n ${NAMESPACE} exec -it q32b-1p1d-aks-ib-benchmark -- tmux a -t benchmark
# Detach: Ctrl+B then D

# Collect:
tsh kubectl cp ${NAMESPACE}/q32b-1p1d-aks-ib-benchmark:/perf-cache/artifacts ./bench-aks-ib
```

## AKS-specific gotchas (from the Azure-AKS-IB playbook)

- **`rdma/shared_ib: "1"` is a boolean scheduling hint**, not a count of NICs. The shared-device plugin admits the pod and exposes all 8 ConnectX-7 rails (`mlx5_ib0..mlx5_ib7`) at once; do not set this to `8`.
- **`UCX_NET_DEVICES` must list all 8 IB rails by name** with port `:1`. Omitting it lets UCX pick `eth0` and you silently land on TCP — exactly the "silent fallback" failure mode the base README warns about.
- **Mount `/dev/infiniband` from the host (hostPath)** on Azure. The shared-device plugin grants access, but the kernel device files still need to be in the pod's mount namespace.
- **`privileged: true`** required for VRAM `ibv_reg_mr`; `IPC_LOCK` alone fails on UCX/IB too (same as EFA).
- **Do NOT set `DYN_KVBM_NIXL_BACKEND_LIBFABRIC=true`** — that forces the AWS-only libfabric backend. UCX is the IB path; defaults are correct.
- **`NCCL_SOCKET_IFNAME=eth0` is control plane only.** The actual NCCL data plane uses `NCCL_IB_HCA=mlx5_ib0..7`. Setting `NCCL_IB_HCA` is what selects IB; the socket IF is just for bootstrap.
- **`NCCL_IB_GID_INDEX=3`** on Azure ND_H100/H200/A100isr (RoCEv2-style GID). Other GID indices will fail link bring-up.
- **One Dynamo worker per node in practice.** The base template's pod anti-affinity already enforces this; the shared-device plugin technically allows multiple, but PKey limits and rail sharing make it unreliable.
- **A100 80GB at TP=8 with Qwen3-32B and max-model-len 131072 may OOM** on KV cache. If you see the engine die during KV cache profiling, reduce `--max-model-len` to `32768` (Qwen3 native context) and rerun. Document the change here.
- **NIXL `transport="rc"` metric label is the canary** — `nixl_backend_transport_active{transport="rc"}=1` means real IB; `transport="tcp"` means you're measuring the eth0 fallback (~200 Gb/s, no GPU-Direct). Throw away the run and fix.
- **Cluster context safety:** `tsh kube login dynamo-aks-dev` switches the kube context but does NOT take a `--kube-cluster` flag. Always confirm with `tsh kubectl config current-context` before destructive ops.

## Results — measured 2026-05-23, AKS-dev A100 + 4-NIC NDR IB

**Topology:** Prefill TP=8 on `vmss000003`, Decode TP=8 on `vmss000002`, both in agentpool `a100a`. UCX limited to 4 of 8 IB NICs (see "AKS-dev gotchas" below).

**Workload:** Mooncake conversation trace, 12,031 requests at 3.23 req/s fixed schedule, aiperf v0.6.0. Run took 122 min wall-clock.

| Metric | Value |
|---|---|
| **Aggregate NIXL KV BW** | **~3.23 GB/s** (2752.9 GiB rank-0 × TP=8 / 7320 s) |
| Mean ITL (steady-state) | 15.34 ms |
| P50 ITL | 14.27 ms |
| P99 ITL | 30.61 ms |
| Output throughput per user (p50) | 70.10 tok/s |
| Output throughput per user (mean) | 68.53 tok/s |
| Mean time-to-second-token | 15.05 ms |
| Successful requests | 7,021 / 12,031 (58%) |
| Errored requests | 1,779 (mostly >50K-token prompts that exceeded engine timeout) |
| Mean input length | 12,571 tokens |
| Mean output tokens | 341 |
| Mean TTFT (queue-bound) | 44 min |
| P99 TTFT (queue-bound) | 82 min |
| Goodput @ TTFT<2s, ITL<25ms | 0 req/s (queue-bound) |

**aiperf artifact:** `dynamo-aks-dev / jihao / perf-cache PVC / artifacts/Qwen3-32B_aks-ib_20260523-0555/`

### Comparison with AWS EFA baseline

| | AWS dev-02 (p5/H100/EFA) | AKS-dev (A100/4-NIC IB) |
|---|---|---|
| Aggregate NIXL KV BW | ~10.7 GB/s | ~3.23 GB/s |
| Mean ITL | 12.63 ms | 15.34 ms |
| P50 ITL | 12.43 ms | 14.27 ms |
| Per-user throughput (mean) | 82.75 tok/s | 68.53 tok/s |
| TTFT, goodput | queue-bound | queue-bound |

AKS A100-IB landed at **~30% of AWS H100-EFA** for transport BW. The gap reflects two factors:
1. Hardware: AWS p5 has 32× EFA NICs (3.2 Tb/s); AKS ND_isr A100 has 8× ConnectX-7 NDR (1.6 Tb/s theoretical, ≈ half AWS).
2. Workaround: UCX limited to 4 of 8 NICs (see gotcha below) — another ~2× factor.

If the DMA-BUF reg limit could be raised to allow all 8 NICs, expected AKS BW ≈ 6 GB/s (~55% of AWS). The ITL and per-user throughput are comparable, consistent with the BW differences only mattering at the transport layer, not the steady-state decode path.

### Deploy outcome (history)

Initial recipe (TP=8, cross-pool) failed three times before reaching Ready. Final working configuration required four cluster-specific fixes documented under "AKS-dev gotchas":

1. Frontend `nodeAffinity` to GPU pool — AKS default-pool CPU nodes (123 GiB OS disk) evict the 15 GB `vllm-runtime` image; GPU pool nodes have 1744 GiB.
2. Worker `--model` as local PVC path with `served-model-name` — ModelExpress always fetches from HF even when cache complete, hits 429 rate limits.
3. Worker `nodeAffinity` to single agentpool `a100a` — `a100a` and `a100b` are on different IB partitions (pkey 0x8019 vs 0x8018); cross-pool RDMA handshake fails.
4. Worker `UCX_NET_DEVICES` limited to 4 NICs — at TP=8 × 8 NICs = 64 simultaneous DMA-BUF registrations, kernel times out on `mlx5_7`. 4 NICs × 8 ranks = 32 regs works.
