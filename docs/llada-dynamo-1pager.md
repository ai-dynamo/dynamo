# LLaDA 2.0 on Dynamo — 1-Page Summary

**Question**: Does Dynamo deliver real serving wins for the LLaDA 2.0 diffusion LLM?
**Answer**: **Yes — ~8× over native SGLang on the same 2 GPUs**, from multi-worker scaling + FP8.
KV-aware routing modes were tested empirically and **did NOT deliver value** at this fleet size — explained below.

## Throughput / latency — headline numbers

| Configuration | Concurrency | req/s | tok/s | TTFT avg |
|---|---:|---:|---:|---:|
| Native SGLang, 1 GPU | 8 | 0.98 | 62.9 | 7.3 s |
| Native SGLang, TP=2 (both GPUs) | 8 | 0.82 | 52.1 | 8.8 s |
| Dynamo 2-worker, round-robin | 8 | 4.05 | 259.1 | 0.86 s |
| **Dynamo 2-worker, round-robin** | **16** | **7.74** | **495.1** | **0.78 s** |
| **Dynamo 2-FP8-worker, round-robin** | **16** | **8.15** | ~520 | ~0.78 s |

**Best ratio: 8.3× throughput, 9.4× TTFT** vs native single-GPU.

## Why it's so much better than the naive 2×

The Dynamo worker code is the *same SGLang engine* as native. The only difference: Dynamo's frontend splits the concurrency across two workers, so each worker sees a smaller batch.

| Setup | Worker engine | Offered conc | Batch each worker actually processes |
|---|---|---:|---:|
| Native single-GPU | sglang.Engine | 8 | **8** |
| Dynamo 2-worker | sglang.Engine × 2 | 8 | **4** per worker |

A pure-linear cost model (`step_time ∝ batch × seqlen`) would predict only **2× total throughput** (smaller per-worker step × 2 parallel workers). We measure **4× at conc=8 and 8× at conc=16**. The extra factor comes from things the linear model misses:

1. **Native at conc=8 over-saturates `chunked_prefill_size=8192`**: 8 × 2065 = 16,520 prefill tokens forces 2 chunked iterations on native, but 4 × 2065 = 8,260 fits in (or barely overflows) one chunk on each Dynamo worker. The prefill-chunking penalty hits native disproportionately.
2. **Past SM saturation, per-step cost is super-linear in batch size** — kernels degrade when the batch grows beyond peak-efficiency size. Native at conc=8 is on the bad side of that curve; Dynamo at conc=4 per worker isn't.
3. **At conc=16 the gap widens to 8×** because native is fully GPU-saturated and additional requests just queue (TTFT goes 7.3 s → 16.2 s while throughput stays flat at ~0.9 req/s). Dynamo's two workers absorb the extra load instead of queueing.

**Honest framing**: the 4-8× win is real and reproducible, but it isn't "2× from parallel workers × 2× from smaller batch" as if those are independent multipliers. It's "2× from parallel workers plus a regime change where native saturates and Dynamo doesn't." For diffusion LLMs specifically, the per-step batch sensitivity is what makes data-parallel scaling pay off way more than the obvious 2×.

## KV routing experiments — what we tried, why none helped

Tested all 3 router modes at warm-trio conc=8 and at conc=16 with mixed-OSL workloads, plus the new `--router-osl-load-weight` and `--dllm-load-multiplier` knobs. **Round-robin won or tied every comparison.**

| Mode | conc | tok/s | req/s | Δ vs RR |
|---|---:|---:|---:|---:|
| round-robin | 8 | 294.1 | 4.60 | baseline |
| kv-approx | 8 | 298.1 | 4.66 | +1.3% (noise) |
| kv-events | 8 | 297.2 | 4.65 | +1.1% (noise) |
| round-robin | 16 | 495.1 | 7.74 | baseline |
| kv-approx | 16 | 462.9 | 7.23 | **−6.6% (regression)** |

**Corrected mechanism** — *the radix cache IS active*, but at 2-worker fleet RR already saturates its benefit:

| Workload | radix-off (ChunkCache forced) | radix-on (default, RadixCache) | Δ |
|---|---|---|---|
| Chat (P=2000, OSL=64) | 4.0 req/s | 4.9 req/s | **+24% throughput** |
| RAG (P=6000, OSL=32) | 2.3 req/s | 5.7 req/s | **+149% throughput** |

The radix cache is ON by default in SGLang 0.5.11 for DLLM. Our prior "ChunkCache structural blocker" claim was based on outdated source — the disable was removed upstream. Cross-request prefix reuse is real and big.

So why don't kv-approx / kv-events beat RR? Because at 2 workers × 4 prefixes under RR:
- Each (prefix, worker) pair gets ~25 same-prefix hits in 200 reqs → only 8 cold starts out of 200 → **96% cache hit rate already with RR**.
- "Perfect" pin-by-prefix routing reaches ~98% hit rate — a 2% theoretical improvement.
- And the pin produces a 107/93 load imbalance, which costs more than the 2% gain.

That's the −6.6% kv-approx regression: small cache-hit improvement minus larger load imbalance. **Round-robin is empirically optimal at this fleet size *because* the cache is so effective that random distribution already captures most of the value.**

Additional knobs tested:
- `--dllm-load-multiplier 8.0` (Tier-1 LLaDA load multiplier) — same 107/93 routing decisions as multiplier=1.0; the overlap-score term dwarfs the load term regardless.
- `--router-osl-load-weight ∈ {0.5, 1, 2, 5}` — every value regressed vs RR by 1.7-15% on p95.

**Verdict**: KV routing intelligence is empirically inert at our 2-worker fleet *because the cache is so effective that there's nothing left to optimize*. The unlock for routing intelligence is **more workers** (4+ with more prefix variety, so RR's random distribution starts missing cache hits). Not engine changes — those are already done by upstream.

## Production recommendation

```bash
# Workers: one per GPU, FP8
QUANTIZATION=fp8 bash examples/backends/sglang/launch/diffusion_llada_multi.sh --gpu-id 0 &
QUANTIZATION=fp8 bash examples/backends/sglang/launch/diffusion_llada_multi.sh --gpu-id 1 &
# Frontend: stay with round-robin
bash examples/backends/sglang/launch/frontend_router.sh --mode round-robin
```

## What we shipped (Dynamo-side, this hardware)

- **FP8 quantization on Blackwell** — 60.6→44.45 GB weights, +5-22% per-worker throughput, quality intact.
- **`DllmRegressionModel`** (`components/src/dynamo/planner/core/perf_model/dllm.py`) — deterministic latency predictor `α·num_blocks + β·per_step_tokens + γ`. **Holdout validation on 6 fresh workloads (not in training set): 30.8% MAPE vs AR baseline 46.9%** — 1.5× better than AR, best on OSL-dominated workloads (9.1% vs 63.1%) but **fails the ≤15% production-grade target** outside the training-distribution box. Use with 15-20% SLA headroom.
- **`sla_sizing.py` deployment-sizing CLI** — uses the fitted cost model to recommend worker count for `(ISL, OSL, RPS, SLA)`. Validated: prediction 2429ms p50, measured 2685ms (+10.5%). Directionally correct.
- **OSL-aware router knob** (`--router-osl-load-weight`) — plumbed Rust→Python→CLI. **Live benchmark at fleet=2: RR wins every weight value by 1.7-15% on p95.** Knob is correct but inert at this fleet size; waits for fleet ≥ 4 to demonstrate value.
- **2-worker FP8 launch scripts** + parametric `aiperf` driver + custom OSL loadgen (`osl_loadgen.py`) — reproducible benchmark harness.

## What needs more hardware

| To prove | Need |
|---|---|
| KV routing actually delivering value | 4+ GPUs (so RR random distribution starts missing cache hits) |
| 4-worker FP8 stability | SGLang DLLM scheduler fixes (separate workstream) |
| ~~Engine-side prefix caching~~ | ~~SGLang radix cache for DLLM~~ — **already enabled by default in 0.5.11**, contributes +149% throughput on RAG workloads. No change needed. |

## Bottom line

**Ship Dynamo + FP8 + 1 worker/GPU + round-robin** for any LLaDA deployment with concurrency ≥ 4. The 8× win is real, reproducible, and comes from data-parallel scaling — exactly what Dynamo is for. KV-aware routing's value for LLaDA is a future-fleet question, not a today question.

---
*Full report: `docs/llada-dynamo-vs-native-sglang.md` · Routing structural analysis: `docs/llada-dynamo-routing-analysis.md` · Raw artifacts: `bench/llada-approx-kv/results/`*
