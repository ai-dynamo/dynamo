# Qwen3-32B 1P1D Cross-Node Disagg — GCP GKE A4X (GB200 NVL72, RoCEv2) Variant

This is the **GCP GKE / A4X / RoCEv2-over-ConnectX-7** member of the cross-provider benchmark family. For the family-wide topology, perf-measurement protocol, and result format, see [`../disagg-1p1d-base/README.md`](../disagg-1p1d-base/README.md). This README only documents the GKE-A4X-specific overrides + measured results.

## What this variant overrides (vs. the base template)

| Override | Value |
|---|---|
| **Cluster** | `dynamo-gcp-dev-02` (A4X / GB200 NVL72, 4 GPUs + 4 CX-7 NICs per node) |
| **Container image** | `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.1` (vanilla; GKE GPU device plugin manages drivers) |
| **TP / GPUs per pod** | **TP=4** (not 8) — A4X nodes have 4 GPUs |
| **RDMA NIC attachment** | Pod-level Multus annotation `networking.gke.io/interfaces` listing `rdma-0..rdma-3` (4 NICs); **no `resources.limits.custom.rdma/*` key** |
| **NIXL backend selection** | Defaults (UCX); no `DYN_KVBM_NIXL_BACKEND_*` env |
| **Transport env** | `UCX_TLS=rc,cuda_copy,cuda_ipc`, `UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1`, `NCCL_CROSS_NIC=0`, `LD_LIBRARY_PATH=/usr/local/gib/lib64:/usr/local/nvidia/lib64` |
| **HostPath mounts** | `/home/kubernetes/bin/gib` → `/usr/local/gib`, `/home/kubernetes/bin/nvidia` → `/usr/local/nvidia` (GIB NCCL plugin + GKE-injected NVIDIA driver libs) |
| **Model PVC** | `model-cache` (per-cluster PVC on `dynamo-gcp-dev-02`) |

## Prereqs

- `dynamo-gcp-dev-02` access via Teleport.
- `model-cache` PVC populated with Qwen3-32B in the `jihao` namespace (create + warm if missing).
- `perf-cache` PVC for benchmark artifacts (create if missing — manifest below).
- `hf-token-secret` Secret with `HF_TOKEN` in the `jihao` namespace.
- Dynamo Platform (operator, etcd, NATS) running.
- **GIB NCCL plugin DaemonSet** installed cluster-wide (`nccl-rdma-installer-a4x.yaml` from `container-engine-accelerators/gpudirect-rdma/`). Without it the hostPaths are empty and the workers fall back to TCP.
- **Network CRDs** `rdma-0..rdma-3` (`networking.gke.io/v1` `Network` + `GKENetworkParamSet`, `deviceMode: RDMA`) provisioned by Cluster Toolkit when the cluster was built.

## Deploy

```bash
export NAMESPACE=jihao
tsh kube login dynamo-gcp-dev-02
tsh kubectl apply -f deploy.yaml -n ${NAMESPACE}

tsh kubectl wait --for=condition=ready pod \
  -l nvidia.com/dynamo-graph-deployment-name=q32b-1p1d-gke-roce \
  -n ${NAMESPACE} --timeout=1800s
```

## Verify (run all four checks before benchmarking — see base README §"Sanity-check")

```bash
# 1. Pods Ready + on different nodes (and ideally different superpods / NVL72 racks)
tsh kubectl -n ${NAMESPACE} get pods \
  -l nvidia.com/dynamo-graph-deployment-name=q32b-1p1d-gke-roce -o wide

# Optional: check NVL72 / superpod topology — if both pods land in the same
# NVL72 NVLink domain, NIXL may pick NVLink/cuda_ipc over RoCE (faster, but
# measures NVLink instead of RoCE). Inspect node labels:
tsh kubectl get nodes -L topology.kubernetes.io/zone \
  -L cloud.google.com/gke-nodepool -L cloud.google.com/gce-topology-block

# 2. UCX backend chosen (not TCP) + GIB NCCL plugin loaded
tsh kubectl -n ${NAMESPACE} logs \
  -l nvidia.com/dynamo-component-sub-type=decode --tail=3000 \
  | grep -iE 'nixl|ucx|NET/gIB|NET/Socket|backend|mlx5' | head -40
# Expect:  "NCCL INFO NET/gIB"   (not NET/Socket)
# Expect:  NIXL line referencing UCX, no LIBFABRIC

# 3. RDMA devices visible in pod (4 mlx5 NICs Active)
PREFILL_POD=$(tsh kubectl -n ${NAMESPACE} get pods \
  -l nvidia.com/dynamo-component-sub-type=prefill -o jsonpath='{.items[0].metadata.name}')
tsh kubectl -n ${NAMESPACE} exec $PREFILL_POD -- bash -c \
  'ibv_devinfo 2>&1 | grep -E "hca_id|state" | head -20'
# Expect 4 mlx5 devices, all PORT_ACTIVE.

# 4. NIXL Prometheus exporter responding
tsh kubectl -n ${NAMESPACE} exec $PREFILL_POD -- \
  curl -s localhost:19090/metrics | grep -c '^nixl_'
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

tsh kubectl -n ${NAMESPACE} exec -it q32b-1p1d-gke-roce-benchmark -- tmux a -t benchmark
# Detach: Ctrl+B then D

# Collect:
tsh kubectl cp ${NAMESPACE}/q32b-1p1d-gke-roce-benchmark:/perf-cache/artifacts ./bench-gke-roce
```

## GKE-A4X-specific gotchas

- **TP=4, not TP=8.** A4X (GB200 NVL72) exposes 4 GPUs per VM. Setting TP=8 errors at vLLM init with "not enough GPUs".
- **RDMA NICs are NOT a K8s resource on GKE.** They attach via `networking.gke.io/interfaces` Multus annotation. A `rdma/ib: 4` style key is rejected by the device manager; pods schedule but never get NICs.
- **GIB plugin DaemonSet is load-bearing.** The hostPath `/home/kubernetes/bin/gib` is populated by `nccl-rdma-installer-a4x` DaemonSet. If the DS isn't installed/Ready, `LD_LIBRARY_PATH=/usr/local/gib/lib64:...` points at empty dirs and NCCL falls back to `NET/Socket` (kernel TCP) — silent perf cliff.
- **All-or-nothing GPU allocation.** Partial GPU pods cannot get RDMA NICs on GKE. Request exactly `gpu: '4'`.
- **One worker pod per node.** RDMA NICs aren't shareable; the GKE device manager enforces this. Pod anti-affinity in the base spec already guarantees this for prefill vs. decode.
- **COS-only nodes.** Ubuntu node pools don't support the GIB DaemonSet.
- **NVL72 NVLink shortcut.** GB200 NVL72 racks expose NVLink across the entire 72-GPU rack. If both pods land in the same rack, NIXL/UCX may pick `cuda_ipc` over RoCE — measured BW will reflect NVLink (~900 GB/s peer-to-peer), not the RoCE fabric (~50 GB/s peak per pod). Cross-rack scheduling is required to measure RoCE; if not achievable, document the result as "intra-rack NVLink, not RoCE".
- **`networking.gke.io/default-interface: eth0`** must be set or the pod gets a default route through one of the RDMA NICs and TCP traffic (control plane, etcd, NATS, HF download) breaks.

## Results — measured 2026-05-26, GKE A4X GB200 + 4-NIC ConnectX-7 RoCE

**Topology:** Prefill TP=4 on `customer-gpu-o7v` pool, rack `f01beb15700434ef...`. Decode TP=4 on `customer-gpu-w0e` pool, rack `9fd9a52c852d6f77...`. **Confirmed cross-rack placement** via `cloud.google.com/gce-topology-block` topologyKey — both prefill and decode are in different NVL72 racks, so KV transfer goes over RoCE (not the intra-rack NVLink shortcut). Frontend on a `customer-cpu` node.

**Workload:** Mooncake conversation_trace (12,031 sessions, 3.23 req/s fixed schedule), aiperf v0.6.0.

### Headline (aggregate transport + steady-state)

| Metric | Value |
|---|---|
| **Aggregate NIXL KV BW** | **~10.72 GB/s** (8867.66 GiB rank-0 × TP=4 / 3555 s) |
| Benchmark wall-clock duration | 3555 sec (~59 min — *faster than AWS at 62 min*) |
| Request throughput | 3.38 req/s (kept pace with 3.23 req/s arrival, no queue buildup) |
| Successful request count | **12,031 / 12,031 (100%)** |

### TTFT (NOT queue-bound — first cluster in family to achieve this on 1P1D)

| Statistic | Value |
|---|---|
| Mean TTFT | 3,733 ms (~3.7 s) |
| P50 TTFT | 2,599 ms |
| P90 TTFT | 8,975 ms |
| P99 TTFT | 16,074 ms |
| Min TTFT | 78 ms |
| Max TTFT | 21,161 ms |

### ITL + per-user (steady-state decode)

| Statistic | Value |
|---|---|
| Mean ITL | 11.38 ms |
| P50 ITL | 10.09 ms |
| P99 ITL | 30.45 ms |
| Mean per-user throughput | 96.53 tok/s |
| P50 per-user throughput | 99.13 tok/s |
| Total output throughput (aggregate) | 1140.85 tok/s |

### Goodput

| Metric | Value |
|---|---|
| **Goodput @ TTFT<2s, ITL<25ms** | **1.42 req/s** |
| GoodRequestCount | 5,031 / 12,031 (41.8% met SLA) |

**This is the only cluster in the family with non-zero meaningful goodput on 1P1D Mooncake.** AWS H100 1P+1D had 0.02 req/s goodput (queue-bound).

aiperf artifact: `dynamo-gcp-dev-02 / jihao / perf-cache PVC / artifacts/Qwen3-32B_gke-roce_20260527-0223/`. Includes:
- `profile_export_aiperf.csv` — full per-percentile stats
- `profile_export_aiperf.json` — same in JSON
- `profile_export.jsonl` — per-request records
- `server_metrics_export.csv` — GPU telemetry

### Comparison across the four measured clusters

| | AWS H100/EFA TP=8 | AKS A100/IB TP=8 (4 NICs) | Nscale B200/TCP TP=4 | **GKE GB200/RoCE TP=4** |
|---|---|---|---|---|
| Aggregate KV BW | 10.70 GB/s | 3.23 GB/s | ~1.0 GB/s (TCP) | **10.72 GB/s** |
| Mean ITL | 12.63 ms | 15.34 ms | 8.45 ms | **11.38 ms** |
| Mean per-user throughput | 82.75 tok/s | 68.53 tok/s | 133.77 tok/s | **96.53 tok/s** |
| Mean TTFT | 108 s (queue-bound) | 44 min (queue-bound) | 53 min (queue-bound) | **3.7 s (not queue-bound)** |
| Goodput @ TTFT<2s | 0.02 req/s | 0 | 0 | **1.42 req/s** |
| Completion rate | 100% | 58% | 17% | **100%** |
| TP | 8 | 8 | 4 | 4 |

**Takeaway:** GKE GB200 with TP=4 and 4 ConnectX-7 NICs matched AWS p5/H100 TP=8 / 32-EFA on aggregate transport (10.72 vs 10.70 GB/s), beat it on TTFT/goodput (queue stayed bounded), and on per-user throughput (96 vs 83 tok/s). The combination of: (a) faster B200 compute, (b) Lustre-backed model load, (c) GB200-RoCE multi-rail NIC, (d) cross-rack placement that forced real RoCE traversal — together produced the best-rounded measurement in the family.
