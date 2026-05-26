# Qwen3-32B 1P1D Cross-Node Disagg — Nscale NDR-IB Variant (B200)

This is the **Nscale managed K8s** member of the cross-provider benchmark family. For the family-wide topology, perf-measurement protocol, and result format, see [`../disagg-1p1d-base/README.md`](../disagg-1p1d-base/README.md). This README only documents the Nscale-specific overrides + measured results.

## What this variant overrides (vs. the base template)

| Override | Value |
|---|---|
| **Container image** | `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.1` (standard image; vLLM v0.19.0 / CUDA 12.9 supports B200 SM 10.0) |
| **RDMA resource** | `rdma/ib: "1"` per pod (Network Operator shared-device plugin slot; grants all 8 HCAs via `/dev/infiniband/*`) |
| **NIXL backend selection** | Defaults — UCX backend (no `DYN_KVBM_NIXL_BACKEND_*` env overrides) |
| **Transport env** | `UCX_TLS=rc,cuda_copy,cuda_ipc`, `UCX_NET_DEVICES=mlx5_0:1,...,mlx5_7:1`, `NCCL_IB_HCA=mlx5_0,...,mlx5_7` |
| **Host volume mount** | `/dev/infiniband` (hostPath) + `/dev/shm` (emptyDir, 64Gi) |
| **Model PVC** | `model-cache` (per-cluster Nscale PVC; verify Qwen3-32B is cached or use the model-download job under `../../model-cache/`) |
| **Cluster** | `dynamo-nscale-dev-cluster` (B200 192GB HBM × 8 per node, 8× NDR IB HCAs/node) |

## Nscale cluster specifics

- **Hardware:** B200 (Blackwell, SM 10.0, 192 GB HBM per GPU). TP=8 fits Qwen3-32B at 128K context with headroom — no need to drop to TP=4.
- **RDMA resources on each node:** `rdma/ib: 8`, `rdma/ib_hdr: 4`, `rdma/shared_ib: 63`, `rdma/shared_ib_hdr: 63`.
  - We use `rdma/ib: "1"` — the standard NVIDIA Network Operator RDMA shared-device-plugin convention treats this as a "slot" (not an HCA count); one request gives the pod the full set of 8 IB devices via `/dev/infiniband/`.
  - The `rdma/ib_hdr` and `rdma/shared_ib_hdr` resources are **unconfirmed** in purpose at the time of writing — likely either a separate IB fabric (HDR vs NDR) or a higher-priority partition queue. We do not request them. If the first deploy attempt fails RDMA bring-up, revisit `ib_hdr` as the next candidate.

## Image compatibility note (B200)

The standard `vllm-runtime:1.1.1` image ships vLLM `v0.19.0` on CUDA `v12.9`. CUDA 12.9 includes PTX/SASS targets for `sm_100` (Blackwell B200) so the image should JIT-compile / load kernels on B200 without an arch-specific tag. If pods CrashLoopBackOff during engine init with errors like `no kernel image is available for execution on the device` or `CUDA error: invalid device function`, the SM 10.0 toolchain coverage is incomplete — fall back to a newer image tag (e.g. an experimental `v1.2.0-deepseek-v4-dev.*` Blackwell-targeted tag, or a nightly `*.devYYYYMMDD` wheel) and document the switch here.

## Prereqs

- `dynamo-nscale-dev-cluster` access via Teleport.
- Network Operator + GPU Operator pre-installed on the cluster (Nscale managed). Verify with:
  ```bash
  tsh kubectl get nodes -o json | jq '.items[0].status.allocatable | with_entries(select(.key|test("rdma|nvidia")))'
  ```
- `hf-token-secret` Secret with `HF_TOKEN` in namespace `jihao`.
- `model-cache` PVC populated with `Qwen/Qwen3-32B`. If not present, create + populate it using the `model-download` Job pattern under `recipes/qwen3-32b/model-cache/` adapted to the Nscale PVC name.
- `perf-cache` PVC (10 Gi RWO) for benchmark artifacts (manifest below).
- Dynamo Platform (operator, etcd, NATS) running in `jihao`.

## Deploy

```bash
export NAMESPACE=jihao
tsh kube login dynamo-nscale-dev-cluster
tsh kubectl config current-context   # sanity-check before any apply/delete

tsh kubectl apply -f deploy.yaml -n ${NAMESPACE}

tsh kubectl wait --for=condition=ready pod \
  -l nvidia.com/dynamo-graph-deployment-name=q32b-1p1d-nscale-ib \
  -n ${NAMESPACE} --timeout=1800s
```

## Verify (run all four checks before benchmarking — see base README §"Sanity-check")

```bash
# 1. Pods Ready + on different nodes
tsh kubectl -n ${NAMESPACE} get pods -l nvidia.com/dynamo-graph-deployment-name=q32b-1p1d-nscale-ib -o wide

# 2. UCX/IB backend chosen + NCCL on IB (not socket)
tsh kubectl -n ${NAMESPACE} logs -l nvidia.com/dynamo-component-sub-type=decode --tail=2000 \
  | grep -iE 'nixl|ucx|NET/IB|backend|mlx5' | head -30
# Expect: NIXL backend = UCX, NCCL NET/IB selected, mlx5_0..mlx5_7 enumerated.

# 3. IB devices visible and Active in the pod
PREFILL_POD=$(tsh kubectl -n ${NAMESPACE} get pods \
  -l nvidia.com/dynamo-component-sub-type=prefill -o jsonpath='{.items[0].metadata.name}')
tsh kubectl -n ${NAMESPACE} exec $PREFILL_POD -- ibstat | grep -E 'CA |State|Rate'
# Expect 8 CAs, all "State: Active", "Rate: 400" (NDR).

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

tsh kubectl -n ${NAMESPACE} exec -it q32b-1p1d-nscale-ib-benchmark -- tmux a -t benchmark
# Detach: Ctrl+B then D

# Collect:
tsh kubectl cp ${NAMESPACE}/q32b-1p1d-nscale-ib-benchmark:/perf-cache/artifacts ./bench-nscale-ib
```

## Nscale-specific gotchas

- **`rdma/ib: "1"` is a slot count, not an HCA count.** Requesting `rdma/ib: "8"` will leave pods Pending (no node will satisfy it under the shared-device plugin model). One slot grants access to all 8 NDR HCAs via `/dev/infiniband/`. (Same gotcha as Nebius — see playbook.)
- **Unknown resource: `rdma/ib_hdr: 4`** on each node alongside the standard `rdma/ib: 8`. Not used here; appears to be either a separate IB fabric tier (e.g. HDR vs NDR, or a high-priority queue). Don't request it without first confirming what it is — it may be reserved for a control-plane service.
- **B200 image coverage:** vLLM `v0.19.0` on `vllm-runtime:1.1.1` is CUDA 12.9 which covers SM 10.0. If init errors reference missing kernel images for the device, switch to a Blackwell-tagged image (see "Image compatibility note" above).
- **`privileged: true` + `IPC_LOCK`** are still required for NIXL VRAM memory registration via UCX `ucp_mem_map` on B200 (same as on H200/H100 NCPs).
- **No `FI_*` env, no `LD_LIBRARY_PATH=/opt/amazon/efa/lib`** — those are AWS-EFA only; would silently break UCX device selection or load unrelated libfabric on this cluster.
- **`--gpu-cluster`-style placement domain (Nebius):** Nscale's analog (rack/island grouping) is *unverified*; pod anti-affinity on `kubernetes.io/hostname` is enough for the 1P1D measurement here. If you scale to many workers, verify IB-island alignment with Nscale support.

## Status: cluster-blocked at IB SR-IOV layer (2026-05-22)

**The recipe deploys cleanly and serves locally, but cross-pod NIXL KV transfer cannot complete on this cluster.** This is a cluster-side limitation, not a recipe bug. Do not retry UCX/NIXL tuning here until the underlying cluster issue is fixed.

**Symptom:** First chat completion request returns HTTP 500. Decode worker's `nixl_connector.py` logs:

```
NIXL transfer failure: transfer_exception
remote_engine_id: <prefill engine>  remote_port: 5600
nixl_cu12._bindings.nixlRemoteDisconnectError: NIXL_ERR_REMOTE_DISCONNECT
```

Engine restarts in-place; the next request hits the same failure. `agent_rx_bytes` stays at 0.

**Root cause (from UCX trace logs on both sides):**

1. `mlx5dv_devx_general_cmd(ALLOW_OTHER_VHCA_ACCESS) failed on mlx5_0, syndrome 0x172df6: Remote I/O error` — the Nscale IB driver/SR-IOV config refuses to grant cross-VHCA memory key permissions. UCX's `rc_mlx5` transport sets up locally but cross-pod RDMA establishment fails when actually used. NCCL works on the same hardware because it uses a different RDMA path that doesn't require this permission.
2. `dlopen('libuct_cuda_gdrcopy.so.0') failed: libgdrapi.so.2: cannot open shared object file` — `vllm-runtime:1.1.1` is missing GDRCopy. Lower priority — fallback path exists if cross-VHCA were allowed.

**What didn't fix it (already tried, leave alone):**

- `UCX_IB_GPU_DIRECT_RDMA=yes`
- `kv_connector_extra_config:{"backends":["UCX"]}` in `--kv-transfer-config`
- Clean delete + reapply (no rolling-update race)
- Picking specific HCAs via `UCX_NET_DEVICES`

**What's needed to unblock:**

- Nscale ops to enable `ALLOW_OTHER_VHCA_ACCESS` on the SR-IOV / IB partition config, OR move from `rdma/ib` (shared-device-plugin) to a per-pod exclusive HCA mode if available.
- Optional: image rebuild with `libgdrapi.so.2` baked in — won't fix the VHCA issue alone, but needed alongside the cluster fix for full GPUDirect.

## Recipe state (current best known)

The following recipe-side fixes are persisted in `deploy.yaml`. They are correct for Nscale; the recipe will deploy and serve fine, just won't cross-node-transfer.

- `--gpu-memory-utilization 0.70` (not 0.90): Nscale is a multi-tenant cluster; other tenants leave ~45 GiB/GPU of ghost CUDA contexts that the K8s `nvidia.com/gpu` accounting doesn't reflect. 0.90 OOMs intermittently depending on which node decode lands on.
- `NCCL_DEBUG=INFO` (was WARN): surfaces NCCL TP-init progress if engine hangs; doesn't affect performance once running.
- `UCX_IB_GPU_DIRECT_RDMA=yes`, `UCX_LOG_LEVEL=info`, `UCX_PROTO_INFO=y`: kept in for future debugging; do no harm.
- TP=4 (not 8): historical, reflecting prior session's cluster capacity. Cluster has had unreserved 8-GPU nodes available since then — could bump to TP=8 to match AWS once the SR-IOV issue is fixed.

## Results

_(blocked — see "Status" above; will fill in once cross-VHCA access is enabled at the cluster)_

| Metric | Value |
|---|---|
| Mean NIXL KV transfer BW (GB/s) | blocked |
| P50 TTFT (ms) | blocked |
| P99 TTFT (ms) | blocked |
| P50 ITL (ms) | blocked |
| Goodput (req/s) | blocked |
| aiperf artifact | n/a |
