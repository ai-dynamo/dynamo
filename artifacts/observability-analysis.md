# Dynamo Observability: Status, Assessment & Strategic Roadmap

**Date:** 2026-04-07
**Scope:** Full code-level audit of metrics, tracing, and logging across the Dynamo codebase. All technical claims are based on code inspection. GitHub issues are referenced for context but their open/closed status may change; do not rely on issue state as evidence for code behavior.

---

## Executive Summary

Dynamo's observability stack is **unevenly mature**. The **metrics layer covers the happy-path request lifecycle well** — 133+ Prometheus metrics with proper hierarchy, auto-labeling, and Grafana dashboards. **Distributed tracing propagates context via TCP** for the P/D (prefill/decode) path across all three backends; the E/P/D (encoder/prefill/decode) path has partial coverage (SGLang propagates, vLLM encode worker does not forward context). Remaining tracing gaps include NATS egress, sampling, and Python-side span creation.

However, **GPU-level visibility relies entirely on external DCGM**, **failure/saturation-path metrics are absent** ([#7826](https://github.com/ai-dynamo/dynamo/issues/7826)), and **six major infrastructure subsystems have little or no observability** (NATS, etcd, LoRA/model lifecycle, multimodal, audit, connection pools). Additionally, 2 defined metrics are never populated (dead metrics), and 6 of 21 metric categories are completely undocumented. The system is approximately **55% complete** for production-grade observability.


| Pillar               | Maturity                                                                                                 | Grade  |
| -------------------- | -------------------------------------------------------------------------------------------------------- | ------ |
| Metrics (Prometheus) | Strong happy-path, critical failure-path gaps ([#7826](https://github.com/ai-dynamo/dynamo/issues/7826)) | **B+** |
| Distributed Tracing  | P/D propagation works; E/P/D partial (vLLM gap); no sampling/NATS/gRPC                                  | **B-** |
| Logging              | Functional, well-structured                                                                              | **B+** |
| Dashboards (Grafana) | Good coverage, gaps remain                                                                               | **B**  |
| GPU Observability    | External dependency only                                                                                 | **C-** |


---

## 1. Metrics: Strong Foundation, Failure-Path Gaps

### Strengths

- **Architecture:** `MetricsHierarchy` trait provides a clean `DistributedRuntime → Namespace → Component → Endpoint` hierarchy with automatic label injection (`dynamo_namespace`, `dynamo_component`, `dynamo_endpoint`, `model`, `model_name`)
- **Callback-based exposition:** Backends (vLLM, SGLang, TRT-LLM) plug in without duplicating infrastructure
- **Auto-generated Python constants:** `prometheus_names.rs` → `prometheus_names.py` prevents naming drift
- **133+ metric definitions** across 19 namespaces covering request lifecycle, LLM golden signals (TTFT, ITL, ISL, OSL), per-worker load, KVBM transfers, Tokio runtime internals, routing overhead, task tracker lifecycle
- **Model load time** is tracked for all backends (`vllm/main.py:584`, `trtllm/engine.py:192`, `sglang/init_llm.py:91`)
- **Event loop canary** measures delay and stall count in `tokio_perf.rs:267-269`

### Critical Gap: 250K Post-Mortem ([#7826](https://github.com/ai-dynamo/dynamo/issues/7826))

A 300K-request benchmark at 100K concurrency exposed that **failure/saturation paths are unmonitored**:


| Gap                        | Severity | What's Missing                                                       | Location                      |
| -------------------------- | -------- | -------------------------------------------------------------------- | ----------------------------- |
| Worker pool queue depth    | CRITICAL | No gauge on bounded mpsc channel (default 6000)                      | `shared_tcp_endpoint.rs:107`  |
| `rx_subjects` HashMap size | CRITICAL | No gauge on pending oneshot entries; explicit TODO comments unfilled | `tcp/server.rs:122-124`       |
| Response-plane wait time   | CRITICAL | Frontend awaits `oneshot::Receiver` with NO TIMEOUT and NO HISTOGRAM | `addressed_router.rs:264-267` |
| Semaphore permit wait time | HIGH     | Worker pool semaphore (1500 permits) has no wait-time histogram      | `shared_tcp_endpoint.rs:156`  |
| Instance down events       | HIGH     | `report_instance_down()` logged at DEBUG only; no counter or gauge   | `client.rs:204-217`           |
| Migration outcomes         | MEDIUM   | Tracks migration starts but not success/failure/exhaustion           | `migration.rs:130-178`        |


### Other Gaps

- **No pre-computed throughput metric** — `tokens_per_second` gauge does not exist; must be derived from counters
- **Forward pass metrics only exist for vLLM** — `InstrumentedScheduler` provides per-iteration batch/queue metrics via ZMQ; SGLang and TRT-LLM lack equivalents
- **No request-level memory estimation** or batch efficiency metrics
- **`tcp_pool_active` and `tcp_pool_idle` are dead metrics** — defined in `prometheus_names.rs` but never incremented
- **Migration metrics are not exported in gRPC and Dynamic engine modes** — `Metrics::new()` is called off-registry in `grpc.rs:121` and `common.rs:102`, so migration counters (`migration_ongoing_request`, `migration_new_request`) are tracked in memory but not served via `/metrics`. Only the HTTP frontend mode exports them.

**Key files:** `lib/runtime/src/metrics/prometheus_names.rs`, `lib/llm/src/http/service/metrics.rs`, `lib/kv-router/src/indexer/metrics.rs`, `lib/llm/src/block_manager/metrics_kvbm.rs`, `components/src/dynamo/common/utils/prometheus.py`

---

## 2. Distributed Tracing: Functional, with Remaining Gaps

### What Works

**Rust-side (TCP transport):** `inject_trace_headers_into_map()` is called in `addressed_router.rs:243` before every TCP request. Ingress extraction works on HTTP (Axum TraceLayer), NATS, and TCP. W3C Trace Context compliant.

**Python-side P/D path (all backends):** `build_trace_headers(context)` builds W3C traceparent from the Rust `Context` object:

- **vLLM:** 3 call sites in `handlers.py` (generate, prefill, decode)
- **SGLang:** 8+ call sites across prefill, decode, embedding, diffusion, multimodal, video, image handlers
- **TRT-LLM:** Called in `handler_base.py:801`, passed to `generate_async()`

**Python-side E/P/D path (partial):**

- **SGLang multimodal:** Encode worker (`encode_worker_handler.py:413`) passes `context=context` to the PD worker client; the PD worker then calls `build_trace_headers(context)` — full E/P/D propagation.
- **vLLM multimodal:** Encode worker (`encode_worker_handler.py`) does NOT forward trace context — it yields encoded embeddings back to the Rust frontend without propagating the trace to downstream workers. The Rust frontend re-injects context on the next TCP hop, but there is a gap in the encode worker span.

**Infrastructure:** OpenTelemetry SDK (`tracing` + `tracing-opentelemetry` + `opentelemetry-otlp`), OTLP export (`OTEL_EXPORT_ENABLED=true`), Tempo + OTel Collector + Loki in deploy stack, JSONL logging with trace_id/span_id/parent_id.

### Remaining Gaps

| Gap | Impact |
|-----|--------|
| **vLLM E/P/D encode worker does not propagate trace** | `encode_worker_handler.py` yields embeddings without passing trace context; span gap in encode step |
| **NATS egress does not inject trace context** | `inject_current_trace_into_nats_headers()` exists but is never called in `push_router.rs` or `nats_client.rs` |
| **Limited `#[instrument]` usage** | Only the task tracker uses it (~3 call sites). Hot-path functions lack function-level spans. |
| **No gRPC tracing** | Tonic/kserve endpoints have no trace middleware |
| **No sampling configuration** | All-or-nothing; no head/tail sampling. Uncontrolled volume has bottlenecked pods at ~1600 rps in practice. |
| **Python components don't create child spans** | They propagate context but don't initialize OTEL SDK or create spans for Python-side processing |
| **NIXL transfers invisible in traces** | No spans on KV cache transfers (see Section 7) |


---

## 3. Logging: Functional but Under-leveraged

- Structured logging via `tracing` crate, JSONL format with full trace context
- OTLP log export via `opentelemetry-appender-tracing`
- Loki integration in deploy stack

**Gaps:** No structured logging convention in Python; no runtime log-level management; no sensitive data scrubbing — the audit sink worker path (`lib/llm/src/audit/sink.rs:94,98`) forwards raw `AuditRecord` values directly to sinks with no redaction transform ([#3447](https://github.com/ai-dynamo/dynamo/issues/3447)).

---

## 4. Dashboards: Good Start, Incomplete Coverage

Six Grafana dashboards ship with the project:


| Dashboard               | Coverage                                                                                      |
| ----------------------- | --------------------------------------------------------------------------------------------- |
| `dynamo.json`           | Frontend golden signals (TTFT, ITL, request rate, throughput)                                 |
| `disagg-dashboard.json` | Disaggregated deployment — 21 panels across 7 rows, well-documented in `DASHBOARD_METRICS.md` |
| `kvbm.json`             | KV block manager transfer metrics                                                             |
| `dcgm-metrics.json`     | GPU hardware metrics                                                                          |
| `dynamo-operator.json`  | Kubernetes operator reconciliation                                                            |
| `sglang.json`           | SGLang backend metrics                                                                        |


**Missing:** No vLLM-specific dashboard, no TRT-LLM dashboard (despite custom `trtllm_kv_transfer_`* metrics), no planner dashboard (16 gauges with zero visualization), no error analysis dashboard, no capacity planning view.

---

## 5. GPU Observability: External Dependency

GPU visibility comes entirely from external DCGM exporter in Kubernetes (`DCGM_FI_DEV_GPU_UTIL`, `DCGM_FI_DEV_MEM_COPY_UTIL`, `DCGM_FI_DEV_FB_USED`, NVLink TX/RX). In-process, only `dynamo_kvstats_gpu_cache_usage_percent` exists. [#6241](https://github.com/ai-dynamo/dynamo/issues/6241) tracks deeper DCGM/k8s-device-plugin integration.

**Missing:** No local dev GPU monitoring, no OOM prediction, no GPU health events, no per-request GPU utilization.

---

## 6. Backend Parity Analysis


| Capability                                 | vLLM                      | SGLang             | TRT-LLM                                    |
| ------------------------------------------ | ------------------------- | ------------------ | ------------------------------------------ |
| Basic Prometheus metrics                   | ✅                         | ✅                  | ✅                                          |
| Forward pass instrumentation               | ✅ (InstrumentedScheduler) | ❌                  | ❌                                          |
| KV transfer metrics                        | Via KVBM                  | Via KVBM           | ✅ (custom histograms)                      |
| Custom error metrics                       | Via engine                | Via engine         | ✅ (aborted, multimodal, structured output) |
| Dedicated Grafana dashboard                | ❌                         | ✅                  | ❌                                          |
| Trace propagation to engine                | ✅                         | ✅                  | ✅                                          |
| Worker load metrics (active_decode_blocks) | ✅                         | ✅ (via KV metrics) | ✅                                          |
| Model load time tracking                   | ✅                         | ✅                  | ✅                                          |


SGLang is the most under-instrumented backend: [#5796](https://github.com/ai-dynamo/dynamo/issues/5796) reports that SGLang worker load metrics are always zero — the `active_prefill_tokens` and `active_decode_blocks` gauges are registered in the frontend (`lib/llm/src/kv_router/metrics.rs:124`, wired in `service_v2.rs:469`, updated via `worker_monitor.rs:194`), but the issue is that SGLang backends may not be publishing the underlying `ActiveLoad` values correctly. Forward pass metrics are also absent for SGLang.

---

## 7. NIXL Transfer Observability: The Disaggregation Blind Spot

NIXL moves KV cache blocks between prefill/decode workers. Transfer latency directly contributes to TTFT.

### What Exists

- **NIXL's own Prometheus exporter** on separate port (default 19090), disabled by default (`NIXL_TELEMETRY_ENABLE=y`)
- **TRT-LLM KV transfer metrics** (Python): `trtllm_kv_transfer_success_total`, `trtllm_kv_transfer_latency_seconds`, `trtllm_kv_transfer_bytes`, `trtllm_kv_transfer_speed_gb_s`
- **Grafana dashboard** Row 6 uses DCGM proxy metrics (GPU mem bandwidth, NVLink) as indirect indicators

### What's Missing


| Gap                                             | Impact                                                                                            |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| Zero instrumentation in Rust NIXL transfer code | `lib/memory/src/nixl.rs`, `lib/kvbm-physical/src/transfer/executor/nixl.rs` — no metrics or spans |
| No vLLM or SGLang NIXL transfer metrics         | Only TRT-LLM has application-level KV transfer metrics                                            |
| NIXL telemetry isolated from Dynamo metrics     | Different port, registry, scrape target; can't correlate with request traces                      |
| No frontend-level transfer latency              | [#6181](https://github.com/ai-dynamo/dynamo/issues/6181) (tagged `Dynamo 1.0.0`)            |
| **Transport selection is opaque to Dynamo**      | NIXL's API (`XferRequest`, `XferStatus`) returns only success/in-progress — no information about which transport (RDMA, TCP, CUDA IPC, GDS) was chosen. `agent.backends()` lists available backends but not which one was selected for a specific transfer. If NIXL silently falls back to TCP when RDMA is misconfigured, Dynamo has no way to detect it. UCX is adding transport/device table introspection (branch `print-transports-devices`), but this info does not yet flow through NIXL's API to Dynamo. |
| **No per-transfer transport annotation**         | NIXL can select different transports for different block ranges within the same logical transfer (e.g., CUDA IPC for same-node blocks, RDMA for cross-node). Even if transport metadata is eventually exposed, per-descriptor transport breakdown would require additional API surface. |
| **Dynamo does not surface NIXL/UCX transport info via structured logging** | UCX is adding transport/device tables at context and endpoint creation. Even once available, Dynamo has no mechanism to capture this info from NIXL and emit it through Dynamo's structured logging system. Dynamo should expose transport selection as structured log events so operators can see which transports are active without enabling UCX debug logs. |


---

## 8. Infrastructure & Subsystem Observability Gaps

### 8.1 NATS Messaging (~5% observable)

Single error counter (`dynamo_transport_nats_errors_total`). Missing: message rates, connection state, reconnection tracking, queue depth, publish/subscribe latency, JetStream metrics.

### 8.2 etcd Service Discovery (0% metricsized)

Comprehensive DEBUG logging but zero Prometheus metrics. Missing: registration latency, watch event throughput, lease health, lock contention, active registrations gauge.

### 8.3 LoRA / Model Lifecycle (~15% observable)

`LoadEstimator` tracks per-adapter counts in memory but never exports to Prometheus. Downloads have retry logic but no latency/cache/bandwidth metrics. Runtime config watch has no change event counters. GPU Memory Service has no allocation metrics.

### 8.4 Multimodal Pipeline (~10% observable)

NVTX profiling markers exist (`mm:img:`*, `mm:nixl:*`), and embedding cache tracks stats in memory, but neither is exported to Prometheus. Rust media preprocessor has explicit TODO for observability. No HTTP download, image/video decode, or encoder throughput metrics.

### 8.5 Audit System (~15% observable)

Broadcast bus with stderr/NATS sinks works, but zero self-observability: no event counter, failure counter, latency histogram, bus depth gauge, or dropped record counter.

### 8.6 Preprocessing & Postprocessing (~40% observable)

Tokenization and template rendering are measured. Detokenization is cumulative. Request error types ARE classified — `requests_total` is labeled with `error_type` (`validation`, `not_found`, `overload`, `cancelled`, `internal`) and populated via `mark_error()` in request paths. **Missing:** `stage_duration_seconds` "postprocess" label never recorded; tool call parsing has zero metrics across 7+ parser types.

### 8.7 TCP/HTTP Connection Pools (~20% observable)

Byte counters and error counter exist. `tcp_pool_active` and `tcp_pool_idle` are **dead metrics** (defined, never incremented). No connection establishment latency, reuse rate, pool exhaustion, or health gauge.

### 8.8 Request Cancellation (~50% observable)

`model_cancellation_total` counter with labels exists, plus disconnect detection via oneshot channels. Missing: cancellation propagation latency, phase distinction (pre-routing vs. in-flight), cancellation success rate.

### Dead Metrics Summary


| Metric            | Defined In            | Status            |
| ----------------- | --------------------- | ----------------- |
| `tcp_pool_active` | `prometheus_names.rs` | Never incremented |
| `tcp_pool_idle`   | `prometheus_names.rs` | Never incremented |


---

## 9. Metrics Documentation Audit

**10 of 21 metric categories are fully documented. 5 are partial. 6 are completely undocumented.**


| #   | Metric Category                                | Documented | Enable      | View | Gap               |
| --- | ---------------------------------------------- | ---------- | ----------- | ---- | ----------------- |
| 1   | Frontend (`dynamo_frontend_`*)                 | ✅          | ✅           | ✅    | None              |
| 2   | Component (`dynamo_component_*`)               | ✅          | ✅           | ✅    | None              |
| 3   | Router (`dynamo_router_*`)                     | ✅          | ✅           | ✅    | None              |
| 4   | Routing Overhead (`overhead_*_ms`)             | ✅          | ✅           | ✅    | None              |
| 5   | vLLM native (`vllm:*`)                         | ✅          | ✅           | ✅    | None              |
| 6   | SGLang native (`sglang:*`)                     | ✅          | ✅           | ✅    | None              |
| 7   | LMCache (`lmcache:*`)                          | ✅          | ✅           | ✅    | None              |
| 8   | TRT-LLM additional (`trtllm_*`)                | ✅          | ✅           | ✅    | None              |
| 9   | NIXL telemetry                                 | ✅          | ✅           | ✅    | None              |
| 10  | DCGM GPU metrics                               | ✅          | ✅           | ✅    | None              |
| 11  | Planner (`planner:*`)                          | ⚠️         | ⚠️ K8s only | ⚠️   | No local dev docs |
| 12  | KV Indexer (`dynamo_kvindexer_*`)              | ⚠️         | ❌           | ❌    | **High**          |
| 13  | Frontend Perf (`stage_duration`, `event_loop`) | ⚠️         | ❌           | ❌    | **High**          |
| 14  | Tokio runtime (`dynamo_tokio_`*)               | ⚠️         | ❌           | ❌    | **High**          |
| 15  | Transport (`dynamo_transport_`*)               | ⚠️         | ❌           | ❌    | **High**          |
| 16  | KVBM (`dynamo_kvbm_`*)                         | ❌          | ❌           | ❌    | **Critical**      |
| 17  | KV Stats (`dynamo_kvstats_`*)                  | ❌          | ❌           | ❌    | **Critical**      |
| 18  | Request Plane (`dynamo_request_plane_`*)       | ❌          | ❌           | ❌    | **Critical**      |
| 19  | Task Tracker (`tasks_*_total`)                 | ❌          | ❌           | ❌    | **Critical**      |
| 20  | KV Publisher (`kv_publisher_`*)                | ❌          | ❌           | ❌    | **Critical**      |
| 21  | Model Info (`model_load_time_seconds`)         | ❌          | ❌           | ❌    | **Critical**      |


The local observability stack (`deploy/docker-observability.yml`) provides Prometheus, Grafana, Tempo, Loki, OTEL Collector with scrape targets for frontend, backend, NIXL, KVBM, NATS, etcd, DCGM. Setup is two docker-compose commands. But **users can't use what they can't discover** — 6 metric categories are completely undocumented and 5 more are only partially documented.

---

## 10. Related GitHub Issues

The following issues provide context for the gaps identified above. Check each issue on GitHub for current status — issue states change frequently and this document does not track them.

| # | Issue | Title | Priority | Category |
|---|-------|-------|----------|----------|
| 1 | [#7826](https://github.com/ai-dynamo/dynamo/issues/7826) | Observability Gaps: 250K Concurrency Post-Mortem | **P0** | Metrics (failure path) |
| 2 | [#5796](https://github.com/ai-dynamo/dynamo/issues/5796) | SGLang worker load metrics always 0 | **P0** | Backend parity |
| 3 | [#7030](https://github.com/ai-dynamo/dynamo/issues/7030) | OTEL trace context not propagated in E/P/D flow | **P1** | Tracing |
| 4 | [#3131](https://github.com/ai-dynamo/dynamo/issues/3131) | Planner skips adjustments due to NaN metrics | **P1** | Planner |
| 5 | [#3112](https://github.com/ai-dynamo/dynamo/issues/3112) | Planner division by zero in SGLang disagg | **P1** | Planner |
| 6 | [#6985](https://github.com/ai-dynamo/dynamo/issues/6985) | Throughput scaling can't scale-down at zero traffic | **P1** | Planner |
| 7 | [#6181](https://github.com/ai-dynamo/dynamo/issues/6181) | Add frontend metrics for KV cache transfer latency | **P1** | Metrics (Dynamo 1.0.0) |
| 8 | [#7753](https://github.com/ai-dynamo/dynamo/issues/7753) | Request rejection broken — ActiveLoad publisher race | **P1** | Load shedding |
| 9 | [#7787](https://github.com/ai-dynamo/dynamo/issues/7787) | Disaggregated Topology Readiness | **P2** | Readiness |
| 10 | [#7762](https://github.com/ai-dynamo/dynamo/issues/7762) | Frontend Ready before discovery completes | **P2** | Readiness |
| 11 | [#7798](https://github.com/ai-dynamo/dynamo/issues/7798) | KV indexer lacks /ready and /status endpoints | **P2** | Readiness |
| 12 | [#7645](https://github.com/ai-dynamo/dynamo/issues/7645) | Media handler doesn't call mark_error() | **P2** | Metrics (accuracy) |
| 13 | [#7622](https://github.com/ai-dynamo/dynamo/issues/7622) | Planner Advisory Mode | **P2** | Planner |
| 14 | [#4132](https://github.com/ai-dynamo/dynamo/issues/4132) | Observability metrics for media decoding / HTTP downloads | **P2** | Metrics (multimodal) |
| 15 | [#3447](https://github.com/ai-dynamo/dynamo/issues/3447) | Support redaction for sensitive info | **P2** | Logging / Compliance |
| 16 | [#6241](https://github.com/ai-dynamo/dynamo/issues/6241) | Integrate advanced GPU monitoring (DCGM/k8s-device-plugin) | **P2** | GPU observability |
| 17 | [#5782](https://github.com/ai-dynamo/dynamo/issues/5782) | Fine-grained error classification for request metric | **P2** | Metrics |
| 18 | [#1456](https://github.com/ai-dynamo/dynamo/issues/1456) | TUI Metrics (TTFT, TPOT, throughput in terminal) | **P3** | Developer experience |
| 19 | [#6383](https://github.com/ai-dynamo/dynamo/issues/6383) | Dynamo Mocker Enhancements (telemetry simulation) | **P2** | Testing |
| 20 | [#6770](https://github.com/ai-dynamo/dynamo/issues/6770) | Too many spans logged, overflowing router | **P2** | Tracing volume |

### Key Patterns (from code inspection, not issue states)

1. **SGLang is the most under-instrumented backend** — worker load metrics report zero values, forward pass metrics absent, planner integration has edge cases
2. **Planner depends on metrics correctness** — NaN handling, division by zero, and zero-traffic transitions cause wrong scaling decisions
3. **Failure-path metrics are the critical gap** — the 250K post-mortem ([#7826](https://github.com/ai-dynamo/dynamo/issues/7826)) documented 6 specific metric gaps that prevented incident detection
4. **Audit sink has no redaction** — `lib/llm/src/audit/sink.rs` forwards raw `AuditRecord` to sinks with no PII scrubbing

---

## 11. Strategic Recommendations

### Phase 1: Fix the Critical (Weeks 1–3)

**P0 — Implement 250K post-mortem metrics** ← [#7826](https://github.com/ai-dynamo/dynamo/issues/7826)

- Worker pool queue depth gauge + queue-full counter
- `rx_subjects`/`tx_subjects` size gauges + TCP connection counters (implements existing TODOs)
- Instance down counter + available/inhibited instance gauges
- Response-plane wait histogram
- Semaphore permit wait histogram + pool active tasks gauge
- Migration outcome counters
- ~400 lines of Rust across 7 files

**P0 — Fix SGLang worker load metrics** ← [#5796](https://github.com/ai-dynamo/dynamo/issues/5796)

- The frontend gauges (`active_prefill_tokens`, `active_decode_blocks`) are registered and wired, but SGLang backends report zero values, making request rejection non-functional

### Phase 2: Close Parity Gaps (Weeks 4–8)

**P1 — Harden planner metric consumption** ← [#3131](https://github.com/ai-dynamo/dynamo/issues/3131), [#3112](https://github.com/ai-dynamo/dynamo/issues/3112), [#6985](https://github.com/ai-dynamo/dynamo/issues/6985)

- NaN → 0.0 for idle metrics; division-by-zero guards; zero-traffic scale-down

**P1 — KV cache transfer latency metrics** ← [#6181](https://github.com/ai-dynamo/dynamo/issues/6181)

- Instrument Rust NIXL transfer executor for uniform backend coverage
- Tagged `Dynamo 1.0.0`

**P1 — Forward pass metrics for SGLang and TRT-LLM**

- Port `InstrumentedScheduler` pattern

**P1 — Complete trace coverage**

- Wire NATS egress injection; add sampling controls; fix vLLM E/P/D encode worker gap

**P1 — Planner dashboard + advisory mode** ← [#7622](https://github.com/ai-dynamo/dynamo/issues/7622)

**P2 — Documentation for undocumented metric categories** (6 of 21)

**P2 — Multimodal observability** ← [#4132](https://github.com/ai-dynamo/dynamo/issues/4132), [#7645](https://github.com/ai-dynamo/dynamo/issues/7645)

**P2 — Readiness and topology** ← [#7787](https://github.com/ai-dynamo/dynamo/issues/7787), [#7762](https://github.com/ai-dynamo/dynamo/issues/7762), [#7798](https://github.com/ai-dynamo/dynamo/issues/7798)

### Phase 3: Production Hardening (Weeks 9–16)

- Python component tracing (OTEL SDK init, child spans) + gRPC trace middleware
- In-process GPU metrics for local dev (pynvml)
- Remove dead metrics (`tcp_pool_active`, `tcp_pool_idle`) or wire them up

### Phase 4: Differentiation (Ongoing)

- Request-level cost attribution
- Mocker telemetry enrichment ← [#6383](https://github.com/ai-dynamo/dynamo/issues/6383)
- Adaptive sampling
- NATS/etcd infrastructure metrics
- Anomaly detection on key metrics

---

## Architecture Diagram: Current vs. Target State

### Current State

```
Frontend ──metrics+traces──► Prometheus ──► Grafana
    │          │                  ▲
    │ traceparent (TCP)           │
    ▼          ▼                  │
Router ──metrics+traces──────────┘
    │          │                  ▲
    │ traceparent (TCP)           │
    ▼          ▼                  │
Worker ──metrics─────────────────┘
    │
    ├── vLLM: forward pass metrics (ZMQ) ✅, P/D traces ✅, E/P/D traces ⚠️ (encode gap)
    ├── SGLang: basic metrics ⚠️ (load metrics zero), P/D+E/P/D traces ✅
    └── TRT-LLM: custom KV metrics ✅, P/D traces ✅, no forward pass ⚠️

Logging: Structured (JSONL) → Loki ✅
Tracing: TCP propagation ✅, NATS ✗, sampling ✗
Failure-path metrics: None ✗
```

### Target State

```
Frontend ──metrics+traces──► Prometheus ──► Grafana (dashboards)
    │          │                  ▲
    │ traceparent (all)           │
    ▼          ▼                  │
Router ──metrics+traces──────────┘
    │          │
    ▼          ▼
Worker ──metrics+traces──► OTel Collector ──► Tempo (traces) + Loki (logs)
    │
    ├── All backends: forward pass metrics ✅
    ├── NIXL: instrumented transfers with spans ✅
    ├── GPU: pynvml (local) + DCGM (K8s) ✅
    └── Failure path: queue depth, rx_subjects, permit wait ✅
```

---

## Conclusion

Dynamo's observability has strong foundations — the metrics hierarchy, auto-labeling, and trace propagation across the P/D path represent real engineering investment. The biggest remaining gaps are:

**Priority order:**

1. **250K post-mortem metrics** — failure/saturation path is unmonitored ([#7826](https://github.com/ai-dynamo/dynamo/issues/7826))
2. **SGLang load metrics** — overload protection non-functional ([#5796](https://github.com/ai-dynamo/dynamo/issues/5796))
3. **Planner metric hardening** — NaN/div-by-zero breaks autoscaling ([#3131](https://github.com/ai-dynamo/dynamo/issues/3131)/[#3112](https://github.com/ai-dynamo/dynamo/issues/3112)/[#6985](https://github.com/ai-dynamo/dynamo/issues/6985))
4. **KV transfer latency** — Dynamo 1.0.0 requirement ([#6181](https://github.com/ai-dynamo/dynamo/issues/6181))
5. **Metrics documentation** — 6 of 21 categories undocumented
6. **Trace completion** — vLLM E/P/D encode gap, NATS egress, sampling, Python spans, gRPC
7. **Backend parity** — forward pass metrics for SGLang/TRT-LLM
8. **Infrastructure subsystem metrics** — NATS, etcd, LoRA, multimodal, audit

The project is ~55% of the way to production-grade observability. With focused effort on Phase 1 (post-mortem metrics, SGLang fix), it could reach 80% within 6 weeks.