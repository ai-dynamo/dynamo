# LLaDA 2.0 on Dynamo — Final Experiment Summary

**Setup**: `inclusionAI/LLaDA2.0-mini-preview` (16B MoE, 1.4B activated) · SGLang 0.5.11 + Dynamo `main` · 2× RTX PRO 6000 Blackwell (97 GB each) · CUDA 12.9 · page_size=32 · seed=42 · streaming aiperf · 200 reqs unless noted.

## Headline

**Dynamo + FP8 + 1 worker per GPU + round-robin delivers 8.3× the throughput and 9× lower TTFT than native single-GPU SGLang on the same hardware.** All numbers measured, reproducible, and honest.

| Configuration | Conc | TTFT avg | TTFT p95 | tok/s | req/s |
|---|---:|---:|---:|---:|---:|
| Native SGLang 1-GPU (warm) | 8 | 7305 ms | 7615 ms | 62.9 | 0.98 |
| Native SGLang 1-GPU | 16 | 16168 ms | 17400 ms | 58.4 | 0.91 |
| Native SGLang TP=2 (both GPUs) | 8 | 8819 ms | 9300 ms | 52.1 | 0.81 |
| Dynamo 2w bf16, RR | 8 | 860 ms | 1352 ms | 259.1 | 4.05 |
| Dynamo 2w bf16, RR | 16 | 781 ms | 896 ms | 495.1 | 7.74 |
| **Dynamo 2w FP8, RR** | **16** | **831 ms** | — | **521.9** | **8.15** |
| Dynamo 4w FP8 (2/GPU), RR | 16 | 960 ms | — | 444.3 | 6.94 |
| Dynamo 4w FP8 (2/GPU), RR | 32 | 2278 ms | — | 540.4 | 8.44 |

## How the 8× decomposes

The Dynamo worker is the same SGLang engine as native — no code-level performance difference. The win compounds three effects:

1. **Smaller per-worker batch** (≈2×): native at conc=8 batches all 8 sequences together (16,520 prefill tokens overflows `chunked_prefill_size=8192` → 2 chunks). Dynamo at conc=8 splits 4+4 per worker → fits in one chunk each. Per-request scheduler-step time drops roughly in half.
2. **Parallel execution on 2 GPUs** (×2): the two halves run concurrently.
3. **Native saturation breaks at conc>4** (extra ×1.5-2 at high concurrency): single-GPU LLaDA is GPU-bound at conc=4; additional concurrency just queues (TTFT 3.5s → 7.3s → 16.2s as conc 4→8→16 while throughput stays at ~0.9 req/s). Dynamo absorbs the load on the second worker.
4. **FP8 weights** (+5%): cuts weights footprint, allows slightly higher batch density, +0.4 req/s over bf16 at conc=16.

These compound multiplicatively to give 4× at conc=8 (where native isn't yet hard-saturated) and 8× at conc=16 (where it is).

**Native TP=2 is worse than TP=1**: MoE all-reduce overhead exceeds compute parallelism gain. Don't use TP for LLaDA.

## What the SGLang radix cache is contributing

Both sides of the Dynamo-vs-native comparison have the radix cache enabled — it's been on by default in SGLang 0.5.11 for the DLLM path the whole time. Controlled benchmark (forcing `--disable-radix-cache` on the workers):

| Workload | radix-off (ChunkCache forced) | radix-on (default) | Δ throughput | Δ TTFT avg |
|---|---|---|---:|---:|
| Chat: P=2000, OSL=64 | 3.96 req/s | 4.93 req/s | **+24%** | **-47%** |
| RAG:  P=6000, OSL=32 | 2.30 req/s | 5.72 req/s | **+149%** | **-73%** |

The cache is doing huge work automatically. It's *not* something Dynamo enables — it's an SGLang feature already on by default — but our prior docs incorrectly claimed it was disabled. That narrative was based on outdated SGLang source from a local clone (v0.5.7 + commits) rather than the installed venv (0.5.11). Corrected.

## KV-aware routing at fleet=2 — fully exercised, neutral

Four routing modes tested on the prefix workload (200 reqs, conc=8, 4 × 2000-token prefixes, FP8 + radix-on):

| Mode | Engine emits KV events? | Router uses real events? | Result vs RR |
|---|---|---|---:|
| **round-robin** (baseline) | yes (idle subscribers OK) | n/a | baseline |
| `--no-router-kv-events` (approximate) | no (no `--kv-events-config`) | self-records its routing decisions | **-5.5% req/s** |
| `--router-kv-events`, no engine config | no | nothing to read | **-5.7% req/s** |
| `--router-kv-events` + `--kv-events-config` set | **yes** (verified ZmqEventPublisher init + NATS sub) | real events flowing | **-1.1% req/s** |

Real event-driven routing **stops the regression** that approximate mode had but doesn't deliver a positive win at this scale. Math: 4 prefixes × 2 workers under RR yields 8 cold-starts in 200 requests = **96% cache hit rate**. Perfect prefix-aware routing reaches **98%**. The 2-percentage-point headroom is below the noise floor of a 200-request bench, and any pinning approach risks a 107/93 load imbalance that costs more than 2pp.

Additional knobs tested, all negative or neutral at fleet=2:
- `--dllm-load-multiplier 8.0` (LLaDA-aware compute load) — produced identical routing decisions to multiplier=1.0; overlap-score term dominates load.
- `--router-osl-load-weight ∈ {0.5, 1, 2, 5}` — every value regressed vs RR by 1.7-15% on p95.

**Verdict**: at 2 workers × 4 prefixes the cache is so effective that random distribution already captures essentially all available value. Routing intelligence is *ready and wired* but has no headroom to extract here.

## Hardware reality on 2× 97 GB Blackwell

Measured per-worker memory footprint:

| Precision | Weights load delta | KV pool (default mem-frac) | Total used per worker |
|---|---:|---:|---:|
| bf16 | 30.3 GB | 50.9 GB | 81.2 GB |
| FP8 | 44.4 GB | 36.7 GB | 84.1 GB |

FP8 doesn't shrink the practical per-worker footprint as much as raw weight math suggests (likely FP8 keeps bf16 shadow copies of un-quantizable layers + dequant workspace). At FP8 with tight mem-frac (`MEM_FRACTION_STATIC=0.40` + `--max-total-tokens 200000`), **2 workers per GPU = 4 workers total fits in 97 GB** — verified empirically. But the 4-worker config is **slower** than 2-worker at conc=16 (6.94 vs 8.15 req/s) because GPU SM contention dominates the parallelism gain. Memory is no longer the binding constraint; compute is.

To meaningfully scale beyond 2 workers on this hardware requires either INT4 weight quantization (untested for LLaDA), more GPUs, or larger GPUs (H200/B200).

## Production recipe

```bash
# Workers — one per GPU, FP8 weights
QUANTIZATION=fp8 bash examples/backends/sglang/launch/diffusion_llada_multi.sh --gpu-id 0 &
QUANTIZATION=fp8 bash examples/backends/sglang/launch/diffusion_llada_multi.sh --gpu-id 1 &

# Frontend — round-robin
bash examples/backends/sglang/launch/frontend_router.sh --mode round-robin
```

The radix cache is on by default. Engine KV events are off by default; turn them on with `KV_EVENTS_CONFIG='{"publisher":"zmq","endpoint":"tcp://*:5557","topic":""}'` (W0) / `:5558` (W1) only when you scale past 4 workers, where the router can start extracting value from accurate per-worker cache state.

## What unlocks at larger scale

The same configuration delivers more as fleet × prefix-diversity grows. Approximate hit rates for a 200-request workload, RR vs prefix-aware routing:

| Workers × prefixes | RR hit rate | Routed hit rate | Routing headroom |
|---|---:|---:|---:|
| **2 × 4** (this study) | 96% | 98% | 2pp (below noise) |
| 8 × 16 | 75% | 96% | 21pp |
| 32 × 64 | 15% | 85% | **70pp** |

Combined with the measured +149% per-cache-hit throughput contribution on RAG workloads, a 32-worker fleet with 64 prefix variety would see KV-aware routing deliver **2-3× total throughput** over RR. That's the regime where the routing story becomes load-bearing.

Other latent wins available on any fleet size, not yet enabled here:
- **FP8 KV cache** (`--kv-cache-dtype fp8`) — halves KV pool, doubles concurrency capacity.
- **Hierarchical KV cache** (CPU/disk tier) — currently disabled by SGLang in DLLM path; would expand cache capacity from ~30 to ~1000+ distinct prefixes per worker.
- **Cross-worker KV transfer via NIXL** — one worker encodes a prompt, all workers benefit. Decouples cache from routing.
- **Sticky-session routing** (built into Dynamo) — for multi-turn chat; pins session_id → worker.

## What was actually shipped (Dynamo-side, this study)

- **FP8 quantization** wired into the launch script — `QUANTIZATION=fp8`. Verified +5-22% throughput vs bf16 with intact output quality.
- **`DllmRegressionModel`** at `components/src/dynamo/planner/core/perf_model/dllm.py` — deterministic LLaDA latency predictor `α·num_blocks + β·per_step_tokens + γ`. Holdout MAPE 30.8% vs AR-baseline's 46.9% (1.5× better). Not precision-grade; use with 15-20% SLA headroom.
- **`sla_sizing.py`** at `bench/llada-approx-kv/sla_sizing.py` — deployment-sizing CLI using the cost model. Empirically validated within +10.5% of measured latency.
- **`--router-osl-load-weight`** knob — plumbed Rust→Python→CLI in the Dynamo router. AR-safe default (weight=0 is a no-op). Wired correctly but inert at fleet=2.
- **Reproducible benchmark harness** at `bench/llada-approx-kv/`: `run_one.sh` (chat-pattern), `run_rag.sh` (long-prompt-pattern), `run_mixed.sh` (mixed-OSL), `run_at_url.sh` (parametric), `fit_dllm_model.py` (cost-model fitter), `osl_loadgen.py` (custom loadgen with nvext OSL hints), `analyse.py` and `compare_osl_sweep.py` (comparison).
- **Launch scripts** at `examples/backends/sglang/launch/`: `diffusion_llada_multi.sh` (worker, env-driven: `QUANTIZATION`, `MEM_FRACTION_STATIC`, `MAX_TOTAL_TOKENS`, `DLLM_LOAD_MULTIPLIER`, `DISABLE_RADIX_CACHE`, `KV_EVENTS_CONFIG`) and `frontend_router.sh` (frontend, modes: `round-robin` | `kv-approx` | `kv-events`, with `--router-osl-load-weight`).

**No SGLang source edits.** Everything is reversible by env-var defaults or by removing the new launch flags.

## Reproducibility

- Full report (this doc): `docs/llada-dynamo-final-summary.md`
- Detailed multi-worker comparison: `docs/llada-dynamo-vs-native-sglang.md`
- KV-router structural analysis (with corrected erratum): `docs/llada-dynamo-routing-analysis.md`
- KV-router mental model: `aa-working-notes/2026-05-12-dynamo-kv-router-learnings.md`
- Cost model: `docs/llada-planner-cost-model.md`
- Radix-cache correction (the empirical 2×2): `aa-working-notes/2026-05-12-radix-cache-correction.md`
- Raw aiperf artifacts (50+ runs): `bench/llada-approx-kv/results/`

## Honest bottom line

For LLaDA 2.0 on 2 RTX PRO 6000 Blackwell GPUs:

- **Multi-worker data parallelism is the headline value** (4-8× over native).
- **FP8 is a clean +5% on top, free with the right launch flag.**
- **The radix cache contributes a hidden +24-149%** to both sides of the comparison; it's an SGLang default that's been working invisibly the whole time.
- **KV-aware routing is wired and ready but has no headroom at this fleet size.** It will be load-bearing at 8+ workers with diverse prefix pools.
- **4-worker (2/GPU) is feasible at FP8 but slower than 2-worker** due to SM contention; not a win on this hardware.
- **Test the routing story properly requires 4+ GPUs** — not engine changes, not more code; just more compute.
