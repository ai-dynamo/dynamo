# Qwen3-32B 1P1D Cross-CSP Benchmark Family

A family of disaggregated prefill+decode (1P1D) recipes for **Qwen3-32B (BF16)**, one per cloud/RDMA fabric. Same workload, same topology, same NIXL KV-transfer protocol — only the provider-specific bits change. Designed so the per-CSP delta in KV bandwidth, TTFT, and goodput is directly comparable.

## What's in this folder

| Variant | Provider | Testing Hardware | Transport | 
|---|---|---|---|
| [`disagg-1p1d-aws-efa/`](disagg-1p1d-aws-efa/) | AWS EKS (p5.48xlarge) | H100 | AWS EFA + libfabric | 
| [`disagg-1p1d-gke-roce/`](disagg-1p1d-gke-roce/) | GCP GKE (A4X) | GB200 | RoCEv2 (CX-7) + UCX | 
| [`disagg-1p1d-aks-ib/`](disagg-1p1d-aks-ib/) | Azure AKS (ND H100 v5) | H100 | NDR InfiniBand + UCX | 
| [`disagg-1p1d-nebius-ib/`](disagg-1p1d-nebius-ib/) | Nebius MK8S | H200 | NDR InfiniBand + UCX | 
| [`disagg-1p1d-nscale-ib/`](disagg-1p1d-nscale-ib/) | Nscale | B200 | NDR InfiniBand + UCX | 

Each subdirectory ships:
- **`deploy.yaml`** — full `DynamoGraphDeployment` 
- **`perf.yaml`** — Mooncake-trace aiperf benchmark pod (identical workload across variants)
- **`README.md`** — provider-specific overrides + measured numbers

## Fixed topology (all variants)

| Role | Replicas | TP | Constraint |
|---|---|---|---|
| Frontend | 1 | — | any CPU node |
| VllmPrefillWorker | 1 | **4** | one GPU node |
| VllmDecodeWorker | 1 | **4** | **different** GPU node — pod anti-affinity on `kubernetes.io/hostname` |

We set pod anti-affinity is load-bearing to forces KV transfer across the provider's RDMA fabric, not the intra-node NVLink/CUDA-IPC shortcut. 

## What differs per variant

Each variant overrides **only** what its fabric requires:

1. **Container image** — `vllm-runtime:1.1.1` everywhere except AWS (`1.1.1-efa-amd64` for libfabric)
2. **K8s RDMA resource key** — `vpc.amazonaws.com/efa: N` (AWS), `rdma/ib: 1` or `rdma/shared_ib: 1` (Azure/Nebius/Nscale), Multus `networking.gke.io/interfaces` (GKE)
3. **NIXL backend** — `DYN_KVBM_NIXL_BACKEND_LIBFABRIC=true` on AWS; UCX default elsewhere
4. **Transport env** — `FI_PROVIDER=efa` + `LD_LIBRARY_PATH` (AWS); `UCX_NET_DEVICES=mlx5_*:1` (IB/RoCE)
5. **Hostpath mounts** — `/dev/infiniband` on IB variants; AWS device plugin handles it
6. **PVC name** — model cache name differs per cluster

Everything else (vLLM args, anti-affinity, `IPC_LOCK`, NIXL telemetry, Prometheus annotations, etcd/NATS discovery) is identical and copied verbatim.

## Workload

All `perf.yaml` files run the same aiperf profile against the [Mooncake conversation trace](https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/conversation_trace.jsonl): 12,031 requests, ISL avg ~12k tokens, OSL avg ~340 tokens, streaming, goodput SLA `TTFT≤2000ms ∧ ITL≤25ms`.

## Measured KV bandwidth (TP=4, Mooncake trace, formula: `rank0_bytes × TP / wall_time`)

| Variant | KV BW | Per-rank | Mean ITL | Mean TTFT | Completion | Notes |
|---|---:|---:|---:|---:|---:|---|
| GKE A4X (GB200) | **10.71 GB/s** | 2.68 GB/s | 10.28 ms | 3.85 s | 100% | RoCE on CX-7; best aggregate + TTFT |
| Nscale (B200) | **10.71 GB/s** | 2.68 GB/s | ~16 ms | 8.35 s | 100% | NDR-IB on raw HCAs (compute fabric only)|
| Nebius (H200) | 5.96 GB/s | 1.49 GB/s | **9.14 ms** | 1510 s (queue) | 100% | fastest ITL of family |
| AWS p5 (H100) | 5.64 GB/s | 1.41 GB/s | 13.62 ms | 1703 s (queue) | 100% | LIBFABRIC over EFA |
| AKS (A100) | 1.88 GB/s | 0.47 GB/s | n/a | n/a | 58% (queue) | A100/CX-6; queue-bound by GPU class |

Per-rank ordering follows GPU generation: GB200 (2.68) > H200 (1.49) > H100 (1.41) > A100 (0.47). Aggregate is per-rank × TP, so it tracks the same order except where wall-time differs across clusters. KV traffic was NOT fabric-bound on any cluster — utilization was <1% of NIC headroom; the bottleneck is GPU/HBM/PCIe.

## Cross-cutting findings (apply to every IB/RoCE variant)

These are deliberately documented once here and **not re-stated in each variant** — variants only cite this section.

1. **Do NOT set `UCX_TLS`.** UCX needs internal transports (notably `ud_mlx5` for active-messages wireup) that an explicit allowlist almost always omits, breaking transport selection. Restrict via `UCX_NET_DEVICES` only and let UCX auto-probe transports.
2. **Exclude `mlx5_bond_*` LAG devices** from `UCX_NET_DEVICES`. The bond is unusable by UCX in containers (`ibv_create_ah` segfaults). Pin to raw HCAs instead. Same rule [ModelExpress's `ucx_utils.py`](https://github.com/ai-dynamo/modelexpress/blob/main/modelexpress_client/python/modelexpress/ucx_utils.py) enforces.
3. **On clusters with multiple IB fabrics** (e.g. Nscale has compute-IB MTU 4096 + side-fabric MTU 512), the `UCX_NET_DEVICES` list must include only the **compute-fabric** NICs. Mixing fabrics causes `NIXL_ERR_REMOTE_DISCONNECT` at the first transfer.
4. **`privileged: true` + `IPC_LOCK`** are required on every variant for GPU-memory registration, not just AWS EFA.
5. **Recipe pod-anti-affinity must match operator labels** (`prefill`/`decode`) — the generic `nvidia.com/dynamo-component-type=worker` selector silently matches nothing and lets prefill+decode co-locate, invalidating the BW measurement.

## Adding a new CSP variant

1. `cp -r disagg-1p1d-aws-efa disagg-1p1d-<csp>-<fabric>` (use whichever existing variant matches your fabric class)
2. Replace the per-variant fields per the "What differs" table above
3. Update the new variant's `README.md` Overrides table and (after first deploy) the Measured Results table
4. Add the row to the variant table in this README
5. Run the variant's `perf.yaml` and confirm the workload completes 12,031/12,031 — that's the validation gate before adding to the comparison table

## Related

- Cross-CSP design rationale, debugging stories, and per-CSP gotchas: `dynamo-writings/csp/` (sibling repo)
- llm-d kustomize evaluation: `dynamo-writings/csp/2026-05-28-llm-d-kustomize-eval.md`
- Cross-framework CSP survey: `dynamo-writings/csp/2026-05-29-cross-framework-csp-survey.md`
