# TTFT breakdown — p50 / p95 / p99

Benchmark: `openai/gpt-oss-120b` · disagg (1 ctx + 1 gen) · KV router · 48 users · 1800 s
Source logs: `frontend.log`, `ctx_worker.log`, `gen_worker.log`
Total requests parsed: 10,913
Aggregate p50 / p90 / p95 / p99 / max TTFT (ms): 1128 / 2958 / 3513 / 4069 / 4740

**Inherent costs** (any inference server / disagg setup pays these):
- Tokenize
- ctx `engine_ms` (prefill forward pass)
- gen `engine_ms` (bundles NIXL KV transfer wait + first decode forward — not separable without TRT-LLM internal hooks)

**Pure Dynamo overhead** = TTFT − (tokenize + ctx prefill engine + gen engine).
This bucket contains: HTTP entry, preprocessor handoffs, KV router, RPC plane, Python handler dispatch, return path, decode dispatch, detokenize, SSE emit.

\* Rows marked with `*` are cross-process — cross-host clock skew may shift a few ms in either direction.

---

## p50 — `8c8cc7a7-ed1d-4667-89b5-8600c23b3cf8` · 22,495 tokens · TTFT 1190 ms

| # | Phase | Duration | % TTFT | Kind |
|---|---|---|---|---|
| 1 | HTTP entry → preprocessor | 0 ms | 0.0% | Dynamo |
| 2 | Tokenize | 60 ms | 5.0% | Inherent |
| 3 | `tokenize_done` → `prefill_router_entry` | 1 ms | 0.1% | Dynamo |
| 4 | Prefill router (resolve + select + dispatch + rpc_send) | <1 ms | 0.0% | Dynamo |
| 5 | RPC out (frontend → ctx)\* | 3 ms | 0.3% | Dynamo |
| 6 | ctx Python handler dispatch | 1 ms | 0.1% | Dynamo |
| 7 | ctx prep + engine submit | <1 ms | 0.0% | Dynamo |
| 8 | ctx prefill engine | 1057 ms | 88.8% | Inherent |
| 9 | Return path (ctx → frontend)\* | 3 ms | 0.3% | Dynamo |
| 10 | Prefill → decode handoff | 2 ms | 0.2% | Dynamo |
| 11 | RPC out (frontend → gen)\* | 4 ms | 0.3% | Dynamo |
| 12 | gen Python handler + prep | 2 ms | 0.2% | Dynamo |
| 13 | gen engine submit | <1 ms | 0.0% | Dynamo |
| 14 | gen engine (KV xfer + 1st decode forward) | 53 ms | 4.5% | Inherent |
| 15 | Detokenize + SSE emit | 0 ms | 0.0% | Dynamo |
| | **→ Pure Dynamo overhead (sum of Dynamo rows)** | **20 ms** | **1.7%** | |


---

## p95 — `fbb79613-1e1a-4997-92b6-a708cf5d3e47` · 39,609 tokens · TTFT 3586 ms

| # | Phase | Duration | % TTFT | Kind |
|---|---|---|---|---|
| 1 | HTTP entry → preprocessor | 1 ms | 0.0% | Dynamo |
| 2 | Tokenize | 161 ms | 4.5% | Inherent |
| 3 | `tokenize_done` → `prefill_router_entry` | 12 ms | 0.3% | Dynamo |
| 4 | Prefill router (resolve + select + dispatch + rpc_send) | <1 ms | 0.0% | Dynamo |
| 5 | RPC out (frontend → ctx)\* | 7 ms | 0.2% | Dynamo |
| 6 | ctx Python handler dispatch | 2 ms | 0.1% | Dynamo |
| 7 | ctx prep + engine submit | <1 ms | 0.0% | Dynamo |
| 8 | ctx prefill engine | 3322 ms | 92.6% | Inherent |
| 9 | Return path (ctx → frontend)\* | 4 ms | 0.1% | Dynamo |
| 10 | Prefill → decode handoff | 2 ms | 0.1% | Dynamo |
| 11 | RPC out (frontend → gen)\* | 6 ms | 0.2% | Dynamo |
| 12 | gen Python handler + prep | 4 ms | 0.1% | Dynamo |
| 13 | gen engine submit | <1 ms | 0.0% | Dynamo |
| 14 | gen engine (KV xfer + 1st decode forward) | 59 ms | 1.6% | Inherent |
| 15 | Detokenize + SSE emit | 1 ms | 0.0% | Dynamo |
| | **→ Pure Dynamo overhead (sum of Dynamo rows)** | **44 ms** | **1.2%** | |


---

## p99 — `750624ff-2f75-469b-bd55-76cc9c804ed9` · 34,214 tokens · TTFT 4129 ms

| # | Phase | Duration | % TTFT | Kind |
|---|---|---|---|---|
| 1 | HTTP entry → preprocessor | 0 ms | 0.0% | Dynamo |
| 2 | Tokenize | 84 ms | 2.0% | Inherent |
| 3 | `tokenize_done` → `prefill_router_entry` | 1 ms | 0.0% | Dynamo |
| 4 | Prefill router (resolve + select + dispatch + rpc_send) | <1 ms | 0.0% | Dynamo |
| 5 | RPC out (frontend → ctx)\* | 4 ms | 0.1% | Dynamo |
| 6 | ctx Python handler dispatch | 2 ms | 0.1% | Dynamo |
| 7 | ctx prep + engine submit | <1 ms | 0.0% | Dynamo |
| 8 | ctx prefill engine | 3969 ms | 96.1% | Inherent |
| 9 | Return path (ctx → frontend)\* | 5 ms | 0.1% | Dynamo |
| 10 | Prefill → decode handoff | 2 ms | 0.1% | Dynamo |
| 11 | RPC out (frontend → gen)\* | 6 ms | 0.1% | Dynamo |
| 12 | gen Python handler + prep | 4 ms | 0.1% | Dynamo |
| 13 | gen engine submit | <1 ms | 0.0% | Dynamo |
| 14 | gen engine (KV xfer + 1st decode forward) | 48 ms | 1.2% | Inherent |
| 15 | Detokenize + SSE emit | 1 ms | 0.0% | Dynamo |
| | **→ Pure Dynamo overhead (sum of Dynamo rows)** | **28 ms** | **0.7%** | |


---

## Cross-cut

| | p50 | p95 | p99 |
|---|---|---|---|
| Input tokens | 22,495 | 39,609 | 34,214 |
| TTFT | 1190 ms | 3586 ms | 4129 ms |
| ctx prefill engine | 1057 ms (88.8%) | 3322 ms (92.6%) | 3969 ms (96.1%) |
| Tokenize + gen engine | 113 ms (9.5%) | 220 ms (6.1%) | 132 ms (3.2%) |
| **Pure Dynamo overhead** | **20 ms (1.7%)** | **44 ms (1.2%)** | **28 ms (0.7%)** |

## Conclusions

- TTFT is **engine-bound at every percentile**. ctx prefill engine alone is 89% at p50 and 92–96% at p95/p99; engine compute time grows with input size and queueing/batching at tail load.
- **Pure Dynamo overhead is small and roughly fixed**: 20 ms at p50, 44 ms at p95, 28 ms at p99 — all under 2% of TTFT. The largest line items are the three RPC hops (~10–15 ms total) and Python handler dispatch on the workers (~3–6 ms total). These costs don't scale with input size, so their *relative* share shrinks as TTFT grows.
- The one **anomaly worth a look** is the `tokenize_done → prefill_router_entry` gap. Distribution across all 10,913 requests (mean 2.44 ms, max 45 ms):

  | metric | gap |
  |---|---|
  | p50 | 2 ms |
  | p75 | 3 ms |
  | p90 | 4 ms |
  | p95 | 6 ms |
  | p99 | 19 ms |
  | p99.9 | 34 ms |
  | max | 45 ms |

  Bucketed: 92.8% of requests sit at <5 ms, but **3.3% spill above 10 ms** and a few percent (1.0%) reach 20–45 ms. The 12 ms observed in the picked p95-TTFT request lies in that elevated tail. Whatever runs between those events in [`lib/llm/src/preprocessor.rs`](../lib/llm/src/preprocessor.rs) — likely chat-template post-processing or the KV router handoff — occasionally stalls. Mean impact on TTFT is tiny (2.4 ms), but the bimodal tail (sub-5 ms vs 20–45 ms) suggests a contention or blocking issue worth poking at.
- Tail TTFT optimization for this workload should focus on the engine path (prefill batching, scheduling, KV cache hit rate), not the Dynamo plane.

## Reproduce

```bash
# Per-request waterfall (ASCII):
python3 analyze_traces.py --request-id 8c8cc7a7-ed1d-4667-89b5-8600c23b3cf8 --no-aggregate
python3 analyze_traces.py --request-id fbb79613-1e1a-4997-92b6-a708cf5d3e47 --no-aggregate
python3 analyze_traces.py --request-id 750624ff-2f75-469b-bd55-76cc9c804ed9 --no-aggregate

# Interactive Gantt (HTML):
python3 plot_trace.py --request-id 8c8cc7a7-ed1d-4667-89b5-8600c23b3cf8
python3 plot_trace.py --request-id fbb79613-1e1a-4997-92b6-a708cf5d3e47
python3 plot_trace.py --request-id 750624ff-2f75-469b-bd55-76cc9c804ed9

# Aggregate stage percentiles across all requests:
python3 analyze_traces.py --requests 0
```
