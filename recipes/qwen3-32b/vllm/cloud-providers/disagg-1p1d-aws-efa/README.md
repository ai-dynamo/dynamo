# Qwen3-32B 1P1D Cross-Node Disagg — AWS EFA Variant

This is the **AWS EFA** member of the cross-provider benchmark family. For the family-wide topology, perf-measurement protocol, and result format, see [`../README.md`](../README.md). This README only documents the AWS-specific overrides + measured results.

## What this variant changes

| Override | Value |
|---|---|
| **Container image** | `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.1-efa-amd64` (libfabric + EFA bits pre-installed) |
| **RDMA resource** | `vpc.amazonaws.com/efa: "32"` per pod (all 32 NICs on p5.48xlarge) |
| **NIXL backend selection** | `DYN_KVBM_NIXL_BACKEND_LIBFABRIC=true`, `DYN_KVBM_NIXL_BACKEND_UCX=false` |
| **Transport env** | `FI_PROVIDER=efa`, `FI_EFA_USE_DEVICE_RDMA=1`, `FI_EFA_ENABLE_SHM_TRANSFER=0`, `LD_LIBRARY_PATH=/opt/amazon/efa/lib:...` |


## AWS-specific findings

Recording the issues that blocked the very first deploy and how the recipe was fixed.

- **NIXL backend selection requires `--kv-transfer-config` to carry `kv_connector_extra_config:{"backends":["LIBFABRIC"]}`.** The env vars `DYN_KVBM_NIXL_BACKEND_LIBFABRIC=true` / `DYN_KVBM_NIXL_BACKEND_UCX=false` are **not sufficient** — without the kv-transfer-config flag, NIXL silently picks UCX and you measure UCX-over-EFA (~ 1 GB/s) instead of libfabric-over-EFA (~ 9.6 GB/s).
- **`gdrcopy_dl_hmem_init failed!`** warning at startup is non-fatal — libfabric falls back to dmabuf path for GPU memory, which still gives full RDMA performance on kernel ≥ 5.12.
- For the image used in the recipe: **`LD_LIBRARY_PATH` must prepend `/opt/nvidia/nvda_nixl/lib64`** before the EFA lib paths — otherwise `NIXL_TELEMETRY_ENABLE=y` silently fails to load `libtelemetry_exporter_prometheus.so` (missing `libcore.so.1.3`) and **the vLLM engine cores crash on init** with `RuntimeError: Engine core initialization failed. Failed core proc(s): {}` plus `resource_tracker: 8 leaked shared_memory objects`. The crash happens AFTER weight loading completes.


## Results

### Bringup verification

After applying the gotcha fixes above, all four sanity checks passed:

| Check | Result |
|---|---|
| LIBFABRIC backend selected | ✅ `libfabric:…:efa:` log lines on both workers, `fi_info -p efa` lists 32 EFA devices with `protocol: FI_PROTO_EFA`, `fabric: efa-direct` |
| NIXL Prometheus exporter on :19090 | ✅ `agent_*` metrics responding |
| KV transfer over EFA | ✅ after the test prompt, decode reports `agent_rx_bytes=2097152` (2 MiB), `agent_xfer_time=12.5 ms`, `agent_memory_registered=58 GiB` (VRAM KV cache) |

### Mooncake-trace benchmark

H100 (P5.48xlarge), TP=4

| Metric | Value |
|---|---:|
| **Wall time** | 6,753 s (~113 min) |
| **Trace completion** | 12,031 / 12,031 (100%) |
| **Errors / cancellations** | 0 |
| **Mean TTFT** | **1,703 s** (~28 min — queue-bound) |
| **Mean ITL** | **13.62 ms** |
| **Per-user output throughput** | ~75 tok/s |

**NIXL transport-layer measurement (the actual point of this benchmark):**

| Metric | Value | Source |
|---|---|---|
| **NIXL agent_rx_bytes (decode, rank-0)** | 9.52 TB | `agent_rx_bytes` after 12,031 requests |
| **Aggregate KV transfer BW** | **5.64 GB/s** | `9.52 TB × TP=4 / 6,753 s benchmark duration` |
| **Per-rank KV BW** | 1.41 GB/s | aggregate / TP |


### Interpreting the results

**The high TTFT is NOT EFA-related — it's 1P1D + Mooncake-trace queueing on H100.**

Mooncake's fixed-schedule replays 12,031 requests at the original arrival rate of 3.23 req/s. A single prefill replica at TP=4 processing 12k-token-avg prompts on H100 cannot keep up; the queue accumulates monotonically. Cluster-wise, only GB200 keeps TTFT bounded at this load; H100/H200/A100 are all queue-bound.

**The metric that matters for the cross-provider comparison is "Aggregate KV transfer BW" (5.64 GB/s).** EFA NIC headroom is 3.2 Tb/s aggregate per node — utilization here is ~0.18%; the fabric is nowhere near saturated. Per-rank KV BW (1.41 GB/s) is the right number to compare across CSPs since it factors out TP differences.

ITL (13.62 ms steady-state) and per-user output throughput (~75 tok/s) are clean since they aren't queue-bound.
