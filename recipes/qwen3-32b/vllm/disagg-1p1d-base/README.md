# Qwen3-32B 1P1D Cross-Node Disagg — Cross-Provider Benchmark Family

This directory is the **transport-neutral reference** for a family of recipes that measure how the underlying RDMA transport affects NIXL KV-cache transfer in a Dynamo disaggregated deployment. The recipe here is a template — **not directly deployable**. Use a sibling variant for your provider.

| Variant | Provider | Transport | Status |
|---|---|---|---|
| [`../disagg-1p1d-aws-efa/`](../disagg-1p1d-aws-efa/) | AWS EKS (p5/p5e/p5en) | AWS EFA + libfabric | ✅ reference |
| `../disagg-1p1d-aks-ib/` | Azure AKS (ND H100/H200/A100 isr) | InfiniBand NDR + UCX | (in progress) |
| `../disagg-1p1d-gke-roce/` | GCP GKE (A3 Ultra / A4 / A4X GB200) | RoCEv2 on CX-7 + UCX | (in progress) |
| `../disagg-1p1d-nebius-ib/` | Nebius MK8S (H200) | NDR IB + UCX | (in progress) |
| `../disagg-1p1d-nscale-ib/` | Nscale (B200) | NDR/XDR IB + UCX | (in progress) |

## Fixed topology (do NOT change per provider)

| Role | Replicas | TP | GPUs/pod | Constraint |
|---|---|---|---|---|
| Frontend | 1 | — | 0 | any CPU node |
| VllmPrefillWorker | **1** | 8 | 8 | one full GPU node |
| VllmDecodeWorker | **1** | 8 | 8 | **different** GPU node (pod anti-affinity, `topologyKey: kubernetes.io/hostname`) |

The pod anti-affinity is the load-bearing piece — it forces prefill and decode onto different hosts so the KV transfer between them must traverse the provider's RDMA fabric (not the NVLink/CUDA-IPC intra-node shortcut). Without it the benchmark measures intra-host bandwidth and the per-provider deltas disappear.

**TP=8 assumes 8 GPUs/node** (H100, H200, A100, B200 SXM5/SXM6). On GB200 nodes that expose 4 GPUs per VM (A4X / `BM.GPU.GB200.4`), use **TP=4** and document the change in the variant's README.

## What's the same across all variants (transport-neutral)

The deploy.yaml in this directory (`deploy.template.yaml`) captures these. Every variant copies them verbatim:

- vLLM args: model, disaggregation-mode, kv-transfer-config (NixlConnector, kv_both), TP=8, max-model-len 131072 with YaRN rope_scaling, block-size 64, gpu-memory-utilization 0.90, async-scheduling.
- Anti-affinity block.
- `privileged: true` + `IPC_LOCK` (required for VRAM memory registration on every RDMA transport, not just EFA).
- `NIXL_TELEMETRY_ENABLE=y` + Prometheus exporter on port 19090.
- Prometheus annotations on the worker pods.
- Service discovery via the Dynamo Platform's etcd + NATS (no per-provider tweaks).

## What differs per variant (transport-specific)

Each variant overrides **only** these things:

1. **Container image** — base `vllm-runtime:1.1.1` for most providers; `vllm-runtime:1.1.1-efa-amd64` for AWS (libfabric pre-installed).
2. **K8s RDMA resource key** (under `resources.limits.custom`):
   - AWS EFA: `vpc.amazonaws.com/efa: <N>` (32 on p5.48xlarge, 16 on p5en)
   - Azure AKS / Nebius / Nscale: `rdma/ib: 1` or `rdma/shared_ib: 1`
   - GKE A4X: pod-level Multus annotation `networking.gke.io/interfaces` instead of a resource
3. **NIXL backend selection env**:
   - AWS: `DYN_KVBM_NIXL_BACKEND_LIBFABRIC=true`, `DYN_KVBM_NIXL_BACKEND_UCX=false`
   - Everyone else: defaults (UCX backend)
4. **Transport provider env**:
   - AWS: `FI_PROVIDER=efa`, `FI_EFA_USE_DEVICE_RDMA=1`, `FI_EFA_ENABLE_SHM_TRANSFER=0`, `LD_LIBRARY_PATH=/opt/amazon/efa/lib:...`
   - IB / RoCE (UCX): `UCX_TLS=rc,cuda_copy,cuda_ipc`, `UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,...` (number of mlx5 ports depends on provider — 8 on AKS/Nebius/Nscale H200, 4 on GKE A4X GB200)
5. **Hostpath/volume mounts** for special device files when the provider needs them (`/dev/infiniband` on most IB providers; not needed on AWS EFA since the device plugin handles it).
6. **PVC name for the model** — each provider has its own model cache:
   - AWS dev-02: `shared-model-cache`
   - Other clusters: per-cluster PVC, named in that variant's README.

## Perf-measurement protocol (same for every variant)

Each variant ships its own `perf.yaml` with the same workload — only the FRONTEND service name + artifact directory name differ. The workload is:

- **Dataset:** Mooncake conversation trace (`https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/conversation_trace.jsonl`). 12,031 requests, avg input 12,035 tokens, avg output 343 tokens, 3.4 req/s arrival rate. Same trace used in `../disagg-kv-router/perf.yaml` so cross-recipe comparison is consistent.
- **Client:** `aiperf 0.6.0` with `--fixed-schedule` (replays at original timestamps) so throughput is identical across variants and only latency / KV BW vary.
- **Streaming:** on.
- **Goodput target:** `time_to_first_token:2000 inter_token_latency:25` (TTFT < 2s, ITL < 25ms).

### Numbers to capture for the comparison report

Every variant reports exactly these 6 numbers + 1 artifact, in the variant's README under a `## Results` section.

| # | Metric | Where it comes from |
|---|---|---|
| 1 | **Mean NIXL KV transfer BW (GB/s)** | NIXL Prometheus `nixl_bytes_transferred_count` rate over a steady-state window (skip warmup). Sum prefill+decode pods. |
| 2 | **P50 TTFT (ms)** | aiperf summary |
| 3 | **P99 TTFT (ms)** | aiperf summary |
| 4 | **P50 ITL (ms)** | aiperf summary |
| 5 | **Goodput (req/s meeting TTFT<2s, ITL<25ms)** | aiperf `--goodput` output |
| 6 | **`profile_export.json`** | `${ARTIFACT_DIR}/profile_export.json`, copied out via `kubectl cp` |

### Sanity-check before benchmarking (every variant)

Don't run the benchmark until all four of these pass — otherwise you're measuring the fallback path, not the provider's RDMA path:

1. **Both worker pods are `Ready`** and `kubectl logs` shows the model finished loading.
2. **Prefill and decode are on different nodes** (`kubectl get pods -o wide`).
3. **NIXL backend log line shows the expected backend** (LIBFABRIC for AWS; UCX for everyone else). Failure = backend env didn't apply.
4. **NIXL Prometheus endpoint responds:** `curl <pod>:19090/metrics | grep -c '^nixl_'` returns > 10.

If any of these fail, fix and redeploy — do not record a benchmark from a misconfigured pod.

## How to add a new provider variant

1. Copy `deploy.template.yaml` into a new sibling directory `disagg-1p1d-<provider>-<transport>/deploy.yaml`.
2. Fill in the five `# REPLACE` blocks (image, resource key, NIXL backend selection env, transport env, model PVC name).
3. Copy `perf.yaml` template into the same directory; change only the pod name and the `FRONTEND` env value to match the variant's DGD name.
4. Write a short variant README that links back here and documents only the **provider-specific overrides + measured Results**.
5. Run the sanity-check + benchmark; record Results.

## References

- [Dynamo disagg communication guide](../../../../docs/kubernetes/disagg-communication-guide.md) — measured AWS EFA numbers, troubleshooting tree.
- [Mooncake conversation trace](https://github.com/kvcache-ai/Mooncake) — dataset source.
- Sibling recipes: `../disagg-kv-router/`, `../agg-round-robin/` — the original Qwen3-32B disagg recipes; this one is a 1P1D simplification for transport-layer measurement.
