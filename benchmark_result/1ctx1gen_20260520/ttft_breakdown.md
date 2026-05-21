# TTFT breakdown — p50 / p95 / p99

Benchmark: `openai/gpt-oss-120b` · disagg (1 ctx + 1 gen) · KV router · 48 users · Phase 8 (measurement window)
Source logs: `1ctx1gen_20260520/frontend.log`, `ctx_worker.log`, `gen_worker.log`
Phase 8 window: 18:24:36 → 18:41:53 PDT (measurement only)
Total requests parsed (full TTFT span, in window): 8,599
Aggregate p50 / p90 / p95 / p99 / max TTFT (ms): 3922 / 6803 / 7278 / 7862 / 8864

**Inherent costs** (any inference server / disagg setup pays these):
- Tokenize
- ctx `engine_ms` (prefill forward pass)
- gen `engine_ms` (bundles NIXL KV transfer wait + first decode forward — not separable without TRT-LLM internal hooks)

**Pure Dynamo overhead** = TTFT − (tokenize + ctx prefill engine + gen engine).
This bucket contains: HTTP entry, preprocessor handoffs, KV router, RPC plane, Python handler dispatch, return path, decode dispatch, detokenize, SSE emit.

\* Rows marked with `*` are cross-process — cross-host clock skew may shift a few ms in either direction.

---

## p50 — `3a512b7b-2039-4a2b-b5b8-b946a1a42197` · 42,846 tokens · TTFT 3922 ms

| # | Phase | Duration | % TTFT | Kind |
|---|---|---|---|---|
| 1 | HTTP entry → preprocessor | 1 ms | 0.0% | Dynamo |
| 2 | Tokenize | 92 ms | 2.4% | Inherent |
| 3 | `tokenize_done` → `prefill_router_entry` | 1 ms | 0.0% | Dynamo |
| 4 | Prefill router (resolve + select + dispatch + rpc_send) | <1 ms | 0.0% | Dynamo |
| 5 | RPC out (frontend → ctx)* | 7 ms | 0.2% | Dynamo |
| 6 | ctx Python handler dispatch | 2 ms | 0.1% | Dynamo |
| 7 | ctx prep + engine submit | <1 ms | 0.0% | Dynamo |
| 8 | ctx prefill engine | 3696 ms | 94.3% | Inherent |
| 9 | Return path (ctx → frontend)* | 6 ms | 0.2% | Dynamo |
| 10 | Prefill → decode handoff | <1 ms | 0.0% | Dynamo |
| 11 | RPC out (frontend → gen)* | 9 ms | 0.2% | Dynamo |
| 12 | gen Python handler + prep | 5 ms | 0.1% | Dynamo |
| 13 | gen engine submit | <1 ms | 0.0% | Dynamo |
| 14 | gen engine (KV xfer + 1st decode forward) | 99 ms | 2.5% | Inherent |
| 15 | Detokenize + SSE emit | <1 ms | 0.0% | Dynamo |
| | **→ Pure Dynamo overhead (sum of Dynamo rows)** | **31 ms** | **0.8%** | |

---

## p95 — `96831fe4-b2b2-40cb-891a-db063e56fb6f` · 28,356 tokens · TTFT 7278 ms

| # | Phase | Duration | % TTFT | Kind |
|---|---|---|---|---|
| 1 | HTTP entry → preprocessor | <1 ms | 0.0% | Dynamo |
| 2 | Tokenize | 68 ms | 0.9% | Inherent |
| 3 | `tokenize_done` → `prefill_router_entry` | 1 ms | 0.0% | Dynamo |
| 4 | Prefill router (resolve + select + dispatch + rpc_send) | <1 ms | 0.0% | Dynamo |
| 5 | RPC out (frontend → ctx)* | 5 ms | 0.1% | Dynamo |
| 6 | ctx Python handler dispatch | 1 ms | 0.0% | Dynamo |
| 7 | ctx prep + engine submit | <1 ms | 0.0% | Dynamo |
| 8 | ctx prefill engine | 7132 ms | 98.0% | Inherent |
| 9 | Return path (ctx → frontend)* | 5 ms | 0.1% | Dynamo |
| 10 | Prefill → decode handoff | <1 ms | 0.0% | Dynamo |
| 11 | RPC out (frontend → gen)* | 7 ms | 0.1% | Dynamo |
| 12 | gen Python handler + prep | 4 ms | 0.1% | Dynamo |
| 13 | gen engine submit | <1 ms | 0.0% | Dynamo |
| 14 | gen engine (KV xfer + 1st decode forward) | 52 ms | 0.7% | Inherent |
| 15 | Detokenize + SSE emit | <1 ms | 0.0% | Dynamo |
| | **→ Pure Dynamo overhead (sum of Dynamo rows)** | **23 ms** | **0.3%** | |

---

## p99 — `f3004830-2d82-4db2-beec-958a2c12b9e9` · 45,686 tokens · TTFT 7862 ms

| # | Phase | Duration | % TTFT | Kind |
|---|---|---|---|---|
| 1 | HTTP entry → preprocessor | 1 ms | 0.0% | Dynamo |
| 2 | Tokenize | 98 ms | 1.2% | Inherent |
| 3 | `tokenize_done` → `prefill_router_entry` | 1 ms | 0.0% | Dynamo |
| 4 | Prefill router (resolve + select + dispatch + rpc_send) | 1 ms | 0.0% | Dynamo |
| 5 | RPC out (frontend → ctx)* | 16 ms | 0.2% | Dynamo |
| 6 | ctx Python handler dispatch | 2 ms | 0.0% | Dynamo |
| 7 | ctx prep + engine submit | <1 ms | 0.0% | Dynamo |
| 8 | ctx prefill engine | 7200 ms | 91.6% | Inherent |
| 9 | Return path (ctx → frontend)* | 6 ms | 0.1% | Dynamo |
| 10 | Prefill → decode handoff | <1 ms | 0.0% | Dynamo |
| 11 | RPC out (frontend → gen)* | 8 ms | 0.1% | Dynamo |
| 12 | gen Python handler + prep | 6 ms | 0.1% | Dynamo |
| 13 | gen engine submit | <1 ms | 0.0% | Dynamo |
| 14 | gen engine (KV xfer + 1st decode forward) | 517 ms | 6.6% | Inherent |
| 15 | Detokenize + SSE emit | <1 ms | 0.0% | Dynamo |
| | **→ Pure Dynamo overhead (sum of Dynamo rows)** | **41 ms** | **0.5%** | |

---

## Cross-cut

| | p50 | p95 | p99 |
|---|---|---|---|
| Input tokens | 42,846 | 28,356 | 45,686 |
| TTFT | 3922 ms | 7278 ms | 7862 ms |
| ctx prefill engine | 3696 ms (94.3%) | 7132 ms (98.0%) | 7200 ms (91.6%) |
| Tokenize + gen engine | 191 ms (4.9%) | 120 ms (1.7%) | 615 ms (7.8%) |
| **Pure Dynamo overhead** | **31 ms (0.8%)** | **23 ms (0.3%)** | **41 ms (0.5%)** |

## Tokenize → prefill_router gap distribution

| metric | gap |
|---|---|
| p50 | 1 ms |
| p75 | 1 ms |
| p90 | 2 ms |
| p95 | 2 ms |
| p99 | 3 ms |
| p99.9 | 3 ms |
| max | 5 ms |
| mean | 1.00 ms |
| n | 8,599 |

## Aggregate engine_ms across all phase-8 requests

| stage | p50 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|
| ctx engine | 3681 ms | 6501 ms | 6985 ms | 7593 ms | 8197 ms |
| gen engine | 101 ms | 272 ms | 368 ms | 662 ms | 1299 ms |
| tokenize | 78 ms | 125 ms | 144 ms | 209 ms | 301 ms |

## Conclusions

- TTFT is **engine-bound at every percentile**, even more so than the earlier sweep run. ctx prefill engine alone is 94% at p50 and 92–98% at p95/p99. With 48 concurrent users sharing one ctx worker, prefill engine time roughly **triples** vs the earlier run (p50 3681 ms vs 1057 ms, p99 7593 ms vs 3969 ms) — queueing/batching inside the ctx engine is the dominant cost.
- **Pure Dynamo overhead stays small and roughly flat**: 31 ms at p50, 23 ms at p95, 41 ms at p99 — all under 1% of TTFT. Same shape as the earlier run (≤50 ms). The three RPC hops and Python handler dispatch remain the largest line items; none of them scale with concurrency.
- **gen `engine_ms` is the only inherent component that visibly degrades at tail**: p99 jumps to 517 ms in the picked request (and 662 ms across all phase-8 requests) vs ~50 ms in the earlier run. Since `gen engine_ms` bundles **NIXL KV transfer wait + first decode forward**, this points to KV-transfer queueing at the decode worker under load, not a Dynamo control-plane regression.
- The `tokenize_done → prefill_router_entry` anomaly from the earlier analysis has **disappeared**: p50=1 ms, p99=3 ms, max=5 ms across 8,599 requests (vs p99=19 ms, max=45 ms before). Whatever was occasionally stalling in the preprocessor is not firing in this run.
- Tail TTFT optimization at concurrency=48 should focus on:
  1. ctx prefill scheduling / batching (94–98% of TTFT)
  2. KV transfer queueing into the gen worker (drives the gen-engine p99 spike)
  - The Dynamo plane itself contributes <1% of TTFT and is not worth tuning further at this concurrency.

## Reproduce

```bash
# In this directory:
python3 analyze_phase8.py > ttft_breakdown.md

# Per-request waterfalls (uses the parent analyze_traces.py):
python3 ../analyze_traces.py --dir . --request-id 3a512b7b-2039-4a2b-b5b8-b946a1a42197 --no-aggregate
python3 ../analyze_traces.py --dir . --request-id 96831fe4-b2b2-40cb-891a-db063e56fb6f --no-aggregate
python3 ../analyze_traces.py --dir . --request-id f3004830-2d82-4db2-beec-958a2c12b9e9 --no-aggregate
```
