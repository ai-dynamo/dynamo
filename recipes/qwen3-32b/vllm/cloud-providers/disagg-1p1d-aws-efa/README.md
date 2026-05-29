# Qwen3-32B 1P1D Cross-Node Disagg — AWS EFA Variant

This is the **AWS EFA** member of the cross-provider benchmark family. For the family-wide topology, perf-measurement protocol, and result format, see [`../disagg-1p1d-base/README.md`](../disagg-1p1d-base/README.md). This README only documents the AWS-specific overrides + measured results.

## What this variant overrides (vs. the base template)

| Override | Value |
|---|---|
| **Container image** | `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.1-efa-amd64` (libfabric + EFA bits pre-installed) |
| **RDMA resource** | `vpc.amazonaws.com/efa: "32"` per pod (all 32 NICs on p5.48xlarge) |
| **NIXL backend selection** | `DYN_KVBM_NIXL_BACKEND_LIBFABRIC=true`, `DYN_KVBM_NIXL_BACKEND_UCX=false` |
| **Transport env** | `FI_PROVIDER=efa`, `FI_EFA_USE_DEVICE_RDMA=1`, `FI_EFA_ENABLE_SHM_TRANSFER=0`, `LD_LIBRARY_PATH=/opt/amazon/efa/lib:...` |
| **Model PVC** | `shared-model-cache` (already populated with Qwen3-32B on `dynamo-aws-dev-02`) |
| **Cluster** | `dynamo-aws-dev-02` (p5.48xlarge: 8× H100 80GB, 32 EFA NICs/node) |

## Prereqs

- `dynamo-aws-dev-02` access via Teleport.
- `shared-model-cache` PVC populated with Qwen3-32B (exists on dev-02).
- `perf-cache` PVC for benchmark artifacts (create if missing — manifest below).
- `hf-token-secret` Secret with `HF_TOKEN` (only needed if model isn't fully cached).
- Dynamo Platform (operator, etcd, NATS) running.

## Deploy

```bash
export NAMESPACE=jihao
tsh kube login dynamo-aws-dev-02
tsh kubectl apply -f deploy.yaml -n ${NAMESPACE}

tsh kubectl wait --for=condition=ready pod \
  -l nvidia.com/dynamo-graph-deployment-name=q32b-1p1d-efa \
  -n ${NAMESPACE} --timeout=1800s
```

## Verify (run all four checks before benchmarking — see base README §"Sanity-check")

```bash
# 1. Pods Ready + on different nodes
tsh kubectl -n ${NAMESPACE} get pods -l nvidia.com/dynamo-graph-deployment-name=q32b-1p1d-efa -o wide

# 2. LIBFABRIC backend chosen (not UCX)
tsh kubectl -n ${NAMESPACE} logs -l nvidia.com/dynamo-component-sub-type=decode --tail=2000 \
  | grep -iE 'nixl|libfabric|backend|EFA' | head -30

# 3. EFA devices visible in pod
PREFILL_POD=$(tsh kubectl -n ${NAMESPACE} get pods \
  -l nvidia.com/dynamo-component-sub-type=prefill -o jsonpath='{.items[0].metadata.name}')
tsh kubectl -n ${NAMESPACE} exec $PREFILL_POD -- bash -c 'fi_info -p efa 2>&1 | grep -c fi_addr_efa'
# Expect 32

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

tsh kubectl -n ${NAMESPACE} exec -it q32b-1p1d-efa-benchmark -- tmux a -t benchmark
# Detach: Ctrl+B then D

# Collect:
tsh kubectl cp ${NAMESPACE}/q32b-1p1d-efa-benchmark:/perf-cache/artifacts ./bench-aws-efa
```

## AWS-specific gotchas (discovered during bringup 2026-05-21)

Recording the issues that blocked the very first deploy and how the recipe was fixed.

- **`LD_LIBRARY_PATH` must prepend `/opt/nvidia/nvda_nixl/lib64`** before the EFA lib paths — otherwise `NIXL_TELEMETRY_ENABLE=y` silently fails to load `libtelemetry_exporter_prometheus.so` (missing `libcore.so.1.3`) and **the vLLM engine cores crash on init** with `RuntimeError: Engine core initialization failed. Failed core proc(s): {}` plus `resource_tracker: 8 leaked shared_memory objects`. The crash happens AFTER weight loading completes, ~10 min into deploy.
- **NIXL backend selection requires `--kv-transfer-config` to carry `kv_connector_extra_config:{"backends":["LIBFABRIC"]}`.** The env vars `DYN_KVBM_NIXL_BACKEND_LIBFABRIC=true` / `DYN_KVBM_NIXL_BACKEND_UCX=false` are **not sufficient** — without the kv-transfer-config flag, NIXL silently picks UCX and you measure UCX-over-EFA (~1 GB/s) instead of libfabric-over-EFA (~9.6 GB/s).
- **`Frontend` must NOT set `command: [python, -m, dynamo.frontend]`** on the `-efa-amd64` image. That image has a built-in ENTRYPOINT (Rust binary `dynamo.frontend`) and the explicit command makes it parse `-m dynamo.frontend` as CLI args, crashing with `Unknown arguments specified`.
- **`Frontend` MUST mount the model cache PVC.** Without it the Dynamo runtime fails to register the model card with `Failed to create cache directory "/shared-models/hub": Permission denied`. Symptom: workers register fine, but `/v1/models` returns `{"data":[]}` and chat completions 404.
- **`FI_EFA_ENABLE_SHM_TRANSFER` must be `0`.** SHM-on causes silent GPU memory corruption under load.
- **`NIXL_TELEMETRY_ENABLE=y`**, not `=1`. Only `=y` starts the Prometheus exporter.
- **`vpc.amazonaws.com/efa`** lives under `resources.limits.custom`, not at the top of `limits`. CRD schema rejects the top-level form.
- **`privileged: true`** is required for VRAM `fi_mr_reg`. `IPC_LOCK` alone fails registration.
- **`gdrcopy_dl_hmem_init failed!`** warning at startup is non-fatal — libfabric falls back to dmabuf path for GPU memory, which still gives full RDMA performance on kernel ≥ 5.12.
- **p5en uses EFAv3, 16 NICs/node** (not 32). If targeting p5en, set `vpc.amazonaws.com/efa: "16"`.
- **Bake Sara's `efa_mr_is_cuda` patch** ([ofiwg/libfabric#12019](https://github.com/ofiwg/libfabric/pull/12019)) on arm64+64K-page kernels (GB200). The `1.1.1-efa-amd64` image is amd64-only.

## Historical gotchas (prior EFA work, may not all apply to this variant)

- **`FI_EFA_ENABLE_SHM_TRANSFER` must be `0`.** SHM-on causes silent GPU memory corruption under load.
- **`NIXL_TELEMETRY_ENABLE=y`**, not `=1`. Only `=y` starts the Prometheus exporter.
- **`vpc.amazonaws.com/efa`** lives under `resources.limits.custom`, not at the top of `limits`. CRD schema rejects the top-level form.
- **`privileged: true`** is required for VRAM `fi_mr_reg`. `IPC_LOCK` alone fails registration.
- **p5en uses EFAv3, 16 NICs/node** (not 32). If targeting p5en, set `vpc.amazonaws.com/efa: "16"`.
- **Bake Sara's `efa_mr_is_cuda` patch** ([ofiwg/libfabric#12019](https://github.com/ofiwg/libfabric/pull/12019)) on arm64+64K-page kernels (GB200). The `1.1.1-efa-amd64` image is amd64-only; if you switch to arm64 (dev-01 GB200) you need a different image tag and the patch.

## Results

### Bringup verification (2026-05-21, dynamo-aws-dev-02)

After applying the gotcha fixes above, all four sanity checks passed:

| Check | Result |
|---|---|
| Both worker pods Ready | ✅ (1/1 on different p5.48xlarge nodes) |
| Anti-affinity (cross-node) | ✅ prefill on ip-100-66-190-59, decode on ip-100-66-163-68 |
| LIBFABRIC backend selected | ✅ `libfabric:…:efa:` log lines on both workers, `fi_info -p efa` lists 32 EFA devices with `protocol: FI_PROTO_EFA`, `fabric: efa-direct` |
| NIXL Prometheus exporter on :19090 | ✅ `agent_*` metrics responding |
| Model registered at frontend | ✅ `/v1/models` returns `Qwen/Qwen3-32B` with `context_window: 131072` |
| End-to-end inference works | ✅ test prompt (21 in / 40 out) returned in 3.2 s |
| KV transfer over EFA | ✅ after the test prompt, decode reports `agent_rx_bytes=2097152` (2 MiB), `agent_xfer_time=12.5 ms`, `agent_memory_registered=58 GiB` (VRAM KV cache) |

### Mooncake-trace benchmark (2026-05-21, 12,031 requests over 62 min)

| Metric | avg | P50 | P99 |
|---|---|---|---|
| **TTFT (ms)** | 107,983 | 115,185 | 185,203 |
| **Time-to-Second-Token (ms)** | 16.66 | 9.42 | 123.00 |
| **Request latency (ms)** | 112,163 | 119,342 | 189,093 |
| **ITL (ms)** | **12.63** | 12.43 | 21.85 |
| **Per-user output throughput (tok/s)** | 82.75 | 80.46 | 124.32 |
| **Total output throughput (tok/s)** | **1,096** | — | — |
| **Request throughput (req/s)** | 3.23 | — | — |
| **Goodput @ TTFT<2s, ITL<25ms** | 0.02 | — | — |

**NIXL transport-layer measurement (the actual point of this benchmark):**

| Metric | Value | Source |
|---|---|---|
| **NIXL agent_rx_bytes (decode, rank-0)** | 4.76 TB | `nixl_bytes_transferred_count` after 12,031 requests |
| **Approx aggregate KV transfer BW** | **~10.7 GB/s** | `4.76 TB × TP=8 / 3,726 s benchmark duration` |
| **Reference baseline (from `disagg-communication-guide.md`)** | ~9.6 GB/s | Llama-3.1-8B / EFA / p5 |
| **NIXL backend confirmed** | LIBFABRIC | `libfabric:…:efa:` log lines + `fi_info -p efa` shows 32 `FI_PROTO_EFA` devices |

aiperf artifact: `/perf-cache/artifacts/Qwen3-32B_aws-efa_20260521-2147/`

### Interpreting the results

**The high TTFT and low goodput are NOT EFA-related — they're 1P1D + Mooncake-trace queueing.**

Mooncake's fixed-schedule replays 12,031 requests at the original arrival rate of 3.23 req/s. A single prefill replica at TP=8 processing 12K-token-avg prompts cannot keep up; the queue builds, TTFT inflates. That's working as designed for a 1P1D *transport-layer* benchmark — the goal here is to measure the NIXL/EFA path under sustained KV-transfer load, not end-to-end serving SLOs.

**The metric that matters for the cross-provider comparison is "Approx aggregate KV transfer BW" (~10.7 GB/s).** It validates that the LIBFABRIC backend is engaged and saturating roughly the documented baseline. ITL (12.6 ms once decode starts) and per-user output throughput (~83 tok/s) are clean since they aren't queue-bound.
