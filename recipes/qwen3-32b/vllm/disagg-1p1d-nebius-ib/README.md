# Qwen3-32B 1P1D Cross-Node Disagg — Nebius InfiniBand Variant

This is the **Nebius MK8S NDR InfiniBand** member of the cross-provider benchmark family. For the family-wide topology, perf-measurement protocol, and result format, see [`../disagg-1p1d-base/README.md`](../disagg-1p1d-base/README.md). This README only documents the Nebius-specific overrides + measured results.

## What this variant overrides (vs. the base template)

| Override | Value |
|---|---|
| **Container image** | `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.1` (stock; no provider tag) |
| **RDMA resource** | `rdma/ib: "1"` under `resources.limits.custom`. This is the Network Operator's RDMA-shared-device-plugin slot count, **not** an HCA count. One slot grants the pod access to all 8 HCAs via `/dev/infiniband/*`. Do NOT request `rdma/ib: 8` — pod will be unschedulable. |
| **NIXL backend selection** | Defaults (UCX). No `DYN_KVBM_NIXL_BACKEND_LIBFABRIC` / `_UCX` env. |
| **Transport env** | `UCX_TLS=rc,cuda_copy,cuda_ipc`, `UCX_NET_DEVICES=mlx5_0:1,...,mlx5_7:1` (8 HCAs), `UCX_IB_GPU_DIRECT_RDMA=yes`, `UCX_RNDV_THRESH=8192`. Plus `NCCL_IB_HCA=mlx5_0,...,mlx5_7`, `NCCL_IB_DISABLE=0`, `NCCL_SOCKET_IFNAME=eth0`. **No `FI_*` env**, **no EFA `LD_LIBRARY_PATH`** — those are AWS-only. **No `NCCL_IB_GID_INDEX`** — that's RoCE-only; native IB uses GID 0. |
| **HostPath volumes** | `/dev/infiniband` (hostPath) and `/dev/shm` (tmpfs emptyDir, 64 GiB) bind-mounted into both worker pods. |
| **Model PVC** | `model-cache` (PVC name on `dynamo-nebius-2`; verify with `tsh kubectl -n jihao get pvc` and adjust if your cluster uses a different name). |
| **Cluster** | `dynamo-nebius-2` — 8 nodes of 8× H200 SXM (141 GB) + 8× NDR400 IB (3.2 Tb/s aggregate per node). |

H200 (141 GB HBM) has plenty of headroom for TP=8 + `--max-model-len 131072`; we keep the family-wide settings unchanged.

## Prereqs

- `dynamo-nebius-2` access via Teleport.
- Node group created with Nebius's `--gpu-cluster <id>` flag (creation-time only — IB cannot be retrofitted onto an existing node group).
- GPU Operator + Network Operator deployed by Nebius's managed-K8s stack — verify with `tsh kubectl get pods -n nvidia-network-operator -n nvidia-gpu-operator`.
- `model-cache` PVC populated with `Qwen/Qwen3-32B` (verify with `tsh kubectl -n jihao get pvc`).
- `perf-cache` PVC for benchmark artifacts (create if missing — manifest below).
- `hf-token-secret` Secret with `HF_TOKEN` in the `jihao` namespace (only needed if the model isn't fully cached).
- Dynamo Platform (operator, etcd, NATS) running.

## Deploy

```bash
export NAMESPACE=jihao
tsh kube login dynamo-nebius-2
tsh kubectl apply -f deploy.yaml -n ${NAMESPACE}

tsh kubectl wait --for=condition=ready pod \
  -l nvidia.com/dynamo-graph-deployment-name=q32b-1p1d-nebius-ib \
  -n ${NAMESPACE} --timeout=1800s
```

## Verify (run all four checks before benchmarking — see base README §"Sanity-check")

```bash
# 1. Pods Ready + on different nodes (pod anti-affinity)
tsh kubectl -n ${NAMESPACE} get pods -l nvidia.com/dynamo-graph-deployment-name=q32b-1p1d-nebius-ib -o wide

# 2. UCX backend chosen (not LIBFABRIC). Also confirm NCCL is using NET/IB (not NET/Socket).
tsh kubectl -n ${NAMESPACE} logs -l nvidia.com/dynamo-component-sub-type=decode --tail=2000 \
  | grep -iE 'nixl|ucx|backend|NET/IB|NET/Socket' | head -30

# 3. IB devices visible in pod, 8 ports Active at NDR 400 Gb/s
PREFILL_POD=$(tsh kubectl -n ${NAMESPACE} get pods \
  -l nvidia.com/dynamo-component-sub-type=prefill -o jsonpath='{.items[0].metadata.name}')
tsh kubectl -n ${NAMESPACE} exec $PREFILL_POD -- ibstat | grep -E 'CA |State|Rate'
# Expect 8 CAs, "State: Active", "Rate: 400" (4xNDR)

tsh kubectl -n ${NAMESPACE} exec $PREFILL_POD -- ibv_devinfo | grep -c '^hca_id'
# Expect 8

# 4. NIXL Prometheus exporter responding
tsh kubectl -n ${NAMESPACE} exec $PREFILL_POD -- curl -s localhost:19090/metrics | grep -c '^nixl_'
# Expect > 10
```

If any check fails, fix and redeploy — do not record a benchmark from a misconfigured pod (it'll silently fall back to TCP / UD and ruin the cross-provider comparison).

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

tsh kubectl -n ${NAMESPACE} exec -it q32b-1p1d-nebius-ib-benchmark -- tmux a -t benchmark
# Detach: Ctrl+B then D

# Collect:
tsh kubectl cp ${NAMESPACE}/q32b-1p1d-nebius-ib-benchmark:/perf-cache/artifacts ./bench-nebius-ib
```

## Nebius-specific gotchas

- **`rdma/ib: 1` is a slot count, not an HCA count.** The shared-device plugin gives the pod `/dev/infiniband/*` with all 8 HCAs visible. Requesting `rdma/ib: 8` makes the pod unschedulable.
- **`--gpu-cluster` is creation-time-only.** If the node group wasn't created with `--gpu-cluster <id>`, the nodes have no IB and the only fix is to recreate the node group. Symptom: `ibstat` inside the pod shows zero CAs or `State: Down`.
- **PKey limit:** one PKey per pod via the shared-device plugin. Not relevant for 1P1D (single PKey is fine); flag for future multi-partition recipes.
- **Do NOT install your own GPU Operator or Network Operator.** Nebius pre-installs both on `--gpu-cluster` node groups and a second copy conflicts.
- **`privileged: true` + `IPC_LOCK`** are required for UCX `ucp_mem_map` on VRAM (same as the EFA `fi_mr_reg` requirement). `IPC_LOCK` alone is not enough.
- **No `FI_*` env, no Sara's libfabric patch, no EFA `LD_LIBRARY_PATH`.** These are AWS-only.

## Results

_(filled in after benchmark runs — current state: recipe authored, deploy + benchmark BLOCKED, see below)_

| Metric | Value |
|---|---|
| Mean NIXL KV transfer BW (GB/s) | TBD |
| P50 TTFT (ms) | TBD |
| P99 TTFT (ms) | TBD |
| P50 ITL (ms) | TBD |
| Goodput (req/s) | TBD |
| aiperf artifact | TBD |

**Status:** Recipe files authored from the base template + the AKS-IB sibling pattern (which is the closest UCX/IB precedent in this repo) + the Nebius section of the cross-provider RDMA playbook. Deploy + verify + benchmark were not performed in this session because the harness blocked all `tsh kubectl` access; rerun the Deploy / Verify / Benchmark sections above in a shell with cluster access to fill in the Results table.
