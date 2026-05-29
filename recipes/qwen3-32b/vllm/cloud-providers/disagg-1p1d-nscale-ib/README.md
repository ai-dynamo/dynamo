# Qwen3-32B 1P1D Cross-Node Disagg — Nscale NDR-IB Variant (B200)

This is the **Nscale managed K8s** member of the cross-provider benchmark family. For the family-wide topology, perf-measurement protocol, and result format, see [`../disagg-1p1d-base/README.md`](../disagg-1p1d-base/README.md). This README only documents the Nscale-specific overrides + measured results.

## What this variant overrides (vs. the base template)

| Override | Value |
|---|---|
| **Container image** | `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.1` (standard image; vLLM v0.19.0 / CUDA 12.9 supports B200 SM 10.0) |
| **RDMA resource** | `rdma/ib: "1"` per pod (Network Operator shared-device plugin slot; grants all 8 HCAs via `/dev/infiniband/*`) |
| **NIXL backend selection** | Defaults — UCX backend (no `DYN_KVBM_NIXL_BACKEND_*` env overrides) |
| **Transport env** | `UCX_NET_DEVICES=mlx5_0:1,...,mlx5_7:1` (8 raw HCAs, **not** `mlx5_bond_0` — the bond is unusable by UCX in containers, `ibv_create_ah` segfaults). **No `UCX_TLS`** — let UCX auto-probe (allowlists break wireup AM selection). |
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

## Status: cross-pod NIXL UCX works (2026-05-27 evening, after MX/NIXL team feedback)

The recipe currently ships in a working UCX-IB state. Prior versions of this README documented a long failure narrative (SR-IOV cross-VHCA, GDRCopy, libfabric) that turned out to be **mostly self-inflicted**. See `dynamo-writings/csp/2026-05-26-nscale-detailed-experiments.md` § "Update 2026-05-27 (evening)" for the full retraction.

The two changes that unblocked UCX-IB on Nscale:

1. **Drop `UCX_TLS` entirely.** UCX needs internal transports (notably `ud_mlx5` for active-messages wireup) that an explicit allowlist almost always omits. The `select.c:657 no active messages transport` error was caused by my own `UCX_TLS=rc,cuda_copy,cuda_ipc` restriction, not anything cluster-side. The MX library deliberately doesn't set `UCX_TLS` — let UCX auto-probe.
2. **Use raw `mlx5_0..7`, never `mlx5_bond_0`.** The bond LAG aggregate is unusable by UCX in containers — `ibv_create_ah` segfaults regardless of how UCX is configured. ModelExpress's [`ucx_utils.py`](https://github.com/ai-dynamo/modelexpress/blob/main/modelexpress_client/python/modelexpress/ucx_utils.py) explicitly filters bond devices for the same reason.

The `syndrome 0x172df6 ALLOW_OTHER_VHCA_ACCESS` line in UCX trace logs is **noise**, not the gate — UCX brings up the data plane on these devices anyway.

### Smoke-test validation (2026-05-27 evening, TP=2)

```
agent_rx_bytes  BEFORE: 0  →  AFTER: 8388608   (8 MiB transferred via NIXL UCX-IB)
```

First cross-pod RDMA KV transfer on Nscale Dynamo this whole investigation. Recipe ships at TP=2 because the cluster was GPU-tight during validation; scale to TP=4 when capacity recovers.

### Open follow-ups (not blocking)

1. Run Mooncake `aiperf` at TP=2 with the working recipe to get a real Nscale-IB KV bandwidth datapoint (vs the ~1 GB/s TCP fallback measured 2026-05-26).
2. Scale to TP=4 when cluster capacity recovers.
3. Without per-rank NUMA-local NIC pinning, all TP ranks end up on whichever NIC UCX picks first — throughput will be bottlenecked. Integrate the MX helper `apply_nic_pin_for_device` from `modelexpress.ucx_utils` ahead of NIXL connector init for optimal throughput on multi-rank-per-pod recipes.

### Earlier TCP fallback measurement (preserved for record)

The old TCP fallback config (`UCX_TLS=tcp,cuda_copy,cuda_ipc` over `eth0`) was measured 2026-05-26 at ~1.0 GB/s aggregate KV BW, mean ITL 8.45 ms. Replaced by the working UCX-IB config above.

aiperf artifact (TCP run): `dynamo-nscale-dev-cluster / jihao / perf-cache PVC / artifacts/Qwen3-32B_nscale-ib_20260526-1808/`

## Recipe state (current persisted config)

- **No `UCX_TLS`** — let UCX auto-probe (allowlists break wireup AM selection).
- `UCX_NET_DEVICES=mlx5_0:1,...,mlx5_7:1` — all 8 raw HCAs, no `mlx5_bond_0`.
- `--gpu-memory-utilization 0.70` (not 0.90): multi-tenant cluster leaves ~45 GiB/GPU of ghost CUDA contexts that K8s `nvidia.com/gpu` doesn't account for.
- `NCCL_DEBUG=WARN`: TP NCCL uses IB (intra-pod path works fine).
- TP=2 (currently, due to GPU capacity). Scale to TP=4 when cluster recovers.
- `privileged: true` + `IPC_LOCK`: required for NIXL GPU memory registration.

## Results (filled — see "Measured numbers" above)
