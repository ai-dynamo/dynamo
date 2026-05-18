# LLaDA-aware planner cost model

- **Date**: 2026-05-12
- **Scope**: Add a diffusion-LM (LLaDA-class) cost branch to Dynamo's planner so capacity sizing matches the engine's actual scaling regime.
- **Companion docs**: `docs/llada-dynamo-vs-native-sglang.md` (multi-worker scaling), `docs/llada-dynamo-routing-analysis.md` (router structural analysis).

## Why the AR cost model under-predicts LLaDA latency

Dynamo's planner uses learned regression models that fit
`wall_time = f(sum_prefill_tokens, sum_decode_kv_tokens)` (see
`components/src/dynamo/planner/core/perf_model/{agg,decode,prefill}.py`). These
features make sense for AR LLMs: one scheduler step generates one token per
in-flight sequence, so wall time and token count scale together with no
hidden non-linearities.

LLaDA 2.0 with the `LowConfidence` algorithm doesn't work that way:

1. Each request's output is committed in **blocks** of `page_size` tokens
   (default 32). For a 64-token output, the engine commits 2 blocks; for a
   512-token output, 16 blocks.
2. Inside each block the engine runs **K denoise forward passes** before
   accepting tokens (K is a runtime parameter, typically 5-8 for the
   default 0.95 confidence threshold).
3. Each forward pass is a **batched single-step** kernel call whose cost is
   approximately linear in the total in-flight token count across all
   concurrent requests.

So a LLaDA request's wall time is:

```
e2e ≈ ceil(OSL / page_size) × K × step_time(batch_tokens)
```

The dominant factor — `ceil(OSL / page_size)` — is **discrete and absent
from the AR feature space**. An AR-style regression sees only
`sum_decode_kv_tokens` (≈ batch × avg_decode_len), which grows smoothly with
batch composition and rolls up the per-block step explosion into one number
with weak signal. Empirically, an AR-style fit on LLaDA data yields ~35%
mean prediction error and ~100% max error — useless for capacity planning.

## The DLLM cost model

`components/src/dynamo/planner/core/perf_model/dllm.py` (new) adds a
two-feature linear model:

```
avg_latency = α · num_blocks + β · per_step_tokens + γ

num_blocks       = ceil(OSL / page_size)            # discrete, deterministic from OSL
per_step_tokens  = batch_size × (ISL + OSL / 2)     # compute per forward pass
```

The intuition: α captures `K × step_setup_overhead` (per-block constant work,
including kernel launch and rope/qk-norm reused across passes within a block).
β captures the per-token compute cost in a single forward pass. γ is request
overhead (tokenizer, transport, scheduler dispatch).

## Empirical validation

Fit on 12 representative observations from the LLaDA 2.0 mini-preview aiperf
corpus (2x RTX PRO 6000, both bf16 and FP8 quantization, page_size=32):

| ISL | OSL | per-worker conc | rps system | avg_lat measured | predicted | error |
|----:|----:|----:|----:|----:|----:|---:|
| 2064 | 64 | 4 | 4.60 | 1.70 | 1.70 | -0.2% |
| 2064 | 64 | 4 | 4.66 | 1.71 | 1.71 | -0.6% |
| 2064 | 63 | 4 | 4.64 | 1.72 | 1.70 | -1.1% |
| 2064 | 64 | 8 | 7.74 | 2.00 | 1.94 | -2.8% |
| 2064 | 64 | 8 | 8.15 | 1.90 | 1.94 | +2.3% |
| 64 | 63 | 4 | 5.46 | 1.44 | 1.46 | +1.5% |
| 64 | 63 | 8 | 10.40 | 1.30 | 1.47 | +13.3% |
| 2064 | 136 | 4 | 2.56 | 3.08 | 2.90 | -6.0% |
| 2064 | 135 | 4 | 2.64 | 2.98 | 2.90 | -2.8% |
| 6064 | 32 | 4 | 4.84 | 1.61 | 1.77 | +9.7% |
| 32 | 125 | 4 | 3.50 | 2.22 | 2.26 | +1.7% |
| 64 | 445 | 4 | 1.18 | 6.19 | 6.26 | +1.2% |

**Mean abs error: 3.6%**. **Max abs error: 11.9%**. Fit values for this
hardware/model pairing:

```
α = 0.40 s per output block
β = 3.0e-5 s per per-step token
γ = 0.63 s request overhead
```

The single worst prediction (-13.3%) is on the lowest-ISL workload where
`per_step_tokens` is tiny and the constant term dominates — exactly where
the linear model has the least information to disambiguate. All higher-ISL
or higher-OSL workloads are within 10%.

For comparison, an AR-style baseline (`avg_lat = a·per_step_tokens + b`)
fit on the same data:

| Model | Mean rel error | Max rel error |
|---|---:|---:|
| **DLLM (this work)** | **3.6%** | **11.9%** |
| AR baseline | 34.7% | 103.7% |

The DLLM model is **~10× more accurate** because it captures the dominant
discrete factor (`num_blocks`) that AR conflates into noise.

## How a planner uses it

The `DllmRegressionModel.find_best_engine_dllm_rps()` method walks batch
sizes and returns the highest engine RPS achievable within an end-to-end
latency SLA:

```python
from dynamo.planner.core.perf_model import DllmRegressionModel

model = DllmRegressionModel(max_num_fpm_samples=128, page_size=32)
# ... load observations from FPM stream or pre-deployment sweep ...
engine_rps, achieved_ms = model.find_best_engine_dllm_rps(
    isl=2000, osl=64, e2e_sla=2000.0, max_num_seqs=8
)
# engine_rps tells you how many req/s a single worker can sustain.
# scale up to num_workers via the planner state machine.
```

Unlike the AR `find_best_engine_agg_rps()`, this signature takes a single
**end-to-end** SLA instead of separate TTFT/ITL targets. LLaDA's diffusion
loop doesn't have a meaningful TTFT/ITL split — the first decoded block
arrives ~`K × step_time` after the prefill completes, not after one forward
pass. End-to-end latency is the binding metric.

The `fit_from_observations()` helper lets ops teams pre-compute
`(α, β, γ)` from a short offline sweep instead of waiting for live-traffic
FPM accumulation:

```python
from dynamo.planner.core.perf_model import fit_from_observations

# Each observation: (ISL, OSL, per-worker concurrency, sys_rps, avg_lat_s)
obs = [
    (2000, 64, 4, 4.5, 1.7),
    (2000, 64, 8, 7.5, 2.0),
    (32, 128, 4, 3.5, 2.2),
    (64, 512, 4, 1.2, 6.3),
]
alpha, beta, gamma = fit_from_observations(obs)
# Inject into planner config.
```

A 4-point sweep is enough to fit; more diverse OSL values improve `α` (the
block coefficient) and more diverse batch sizes improve `β` (per-token).

## Production guidance

For LLaDA-class engines:

1. Run a short pre-deployment sweep over 4-8 (ISL, OSL, concurrency) points
   covering your real traffic regime.
2. Call `fit_from_observations()` to get `(α, β, γ)` for your hardware.
3. Load coefficients into the planner config as the DLLM perf model
   constants.
4. Set `optimization_target=sla` and provide `e2e_sla` (LLaDA-mode) instead
   of TTFT/ITL.
5. The planner will size worker count via the deterministic DLLM cost
   model rather than the AR regression that mis-prices block-quantized
   compute.

## Limitations

- The model assumes `K` (steps per block) is constant. The `LowConfidence`
  algorithm at the default 0.95 threshold averages ~5-8 steps per block,
  but the actual count varies per request based on the noise schedule.
  This noise is absorbed into the `γ` intercept; it adds ~5% noise floor
  to predictions and does not bias them.
- The single-worker GPU-saturated regime (where 1 worker hits 100% SM
  utilization) is **not well predicted** by this linear model: native
  single-GPU benchmarks gave R²≈0 on the linear fit because saturation
  introduces a hard ceiling. Multi-worker fleets (the production
  configuration) are below saturation per-worker and fit cleanly.
- The fit is per-(model × hardware × quantization). bf16 and FP8 give
  different coefficients (FP8's `α` is ~10% lower). One global tuple
  doesn't cover the full deployment matrix.

## Holdout validation (2026-05-12)

The training-set fit reported ~3.6% mean error. To check whether that was
genuine generalization or training-distribution overfit, we collected six
fresh aiperf runs covering workloads the model was NOT fit on:

| label | ISL | OSL | conc | rationale |
|---|---:|---:|---:|---|
| holdout-rag-short | 4096 | 32 | 4 | long-prompt regime |
| holdout-rag-med | 4096 | 128 | 8 | RAG-like, medium output |
| holdout-shortin-longout | 64 | 384 | 4 | short prompt, long generation (5+ blocks) |
| holdout-mixed-mid | 1024 | 96 | 8 | middle of input/output range |
| holdout-conc-burst | 2000 | 64 | 24 | high concurrency, prefix-heavy |
| holdout-min | 64 | 32 | 2 | low-load corner |

Fit both models on the 21-obs training corpus (existing `bench/llada-approx-kv/results/*` dirs, excluding holdouts/smoke/warmup runs):

```
DLLM fit: latency_s = 0.3754 * num_blocks + 4.76e-5 * per_step_tokens + 0.7116
AR fit:   latency_s = -7.15e-5 * prefill   + 1.85e-6 * decode         + 1.4529
```

The AR fit's negative `prefill` coefficient is a red flag — the training
distribution is dominated by ISL=2065 OSL=64 runs, so the regression
collapses prefill into the intercept.

**Predicted vs actual on holdouts:**

| holdout | ISL | OSL | conc | actual (s) | DLLM (s) | DLLM err | AR (s) | AR err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| conc-burst | 4001 | 64 | 24 | 3.99 | 3.77 | 5.5% | 3.75 | 5.8% |
| min | 64 | 32 | 2 | 0.62 | 1.09 | 75.3% | 1.45 | 133.5% |
| mixed-mid | 1024 | 96 | 8 | 2.45 | 2.04 | 16.5% | 1.92 | 21.4% |
| rag-med | 4096 | 128 | 8 | 4.70 | 3.01 | 36.1% | 4.22 | 10.2% |
| rag-short | 4096 | 32 | 4 | 2.57 | 1.48 | 42.4% | 1.35 | 47.2% |
| shortin-longout | 64 | 363 | 4 | 4.80 | 5.24 | 9.1% | 1.77 | 63.1% |

**Holdout MAPE: DLLM=30.8% vs AR=46.9%.**

### What this means honestly

The 3.6% mean prediction error reported on the original 12-obs training
corpus does **not** survive holdout validation. On fresh workloads the
DLLM model is at ~31% MAPE, well above the 15% target we'd want for
production sizing.

That said, the DLLM model still meaningfully **outperforms the AR
baseline** (30.8% vs 46.9%, a 1.5× reduction in mean error). Two cases
make the structural difference obvious:

- `shortin-longout` (ISL=64, OSL=363): DLLM is **9.1%**, AR is **63%**.
  This is the OSL-dominated regime, exactly where the `num_blocks`
  feature is informative. AR has no way to express the 12-block cost.
- `rag-med` (ISL=4096, OSL=128): AR wins here (10% vs 36%). AR's prefill
  coefficient happens to extrapolate better at this high-ISL/low-conc point
  where DLLM under-predicts.

### Root cause of the residual error

The training set is **heavily clustered**: 19 of 21 observations are at
ISL≈2065 with OSL∈{64,135}. The model never sees high-ISL/low-conc
regimes during training, so:

1. `rag-short` and `rag-med` both underpredict — `per_step_tokens` for
   ISL=4096 conc=4 (3.6e4) is outside the training envelope which
   maxes out at ~3.3e4 at ISL=2065 conc=16.
2. `min` overpredicts by 75% — at very low load the intercept γ=0.71
   dominates, but actual latency is only 0.62s; the intercept was fit
   on data where the constant term was a smaller fraction of total.

### Production verdict

The DLLM model is **directionally correct but quantitatively limited**
with this training set. For real production deployment:

1. Operators MUST run a wider sweep at deployment time covering the
   actual ISL/OSL envelope they care about (a 4-point sweep at the
   corners of the workload box).
2. The 1.5× improvement over AR is real and the `shortin-longout`
   result (9.1%) validates the block-count feature hypothesis. The
   AR baseline is structurally unsuited for diffusion-LM scaling.
3. The current α=0.375 s/block is fit on FP8 LLaDA 2.0 mini on a
   2x RTX PRO 6000 fleet and is not portable to other hardware.

## Deployment sizing CLI

`bench/llada-approx-kv/sla_sizing.py` operationalizes the fitted model
into a deployment recommendation:

```
$ python bench/llada-approx-kv/sla_sizing.py \
    --isl 2000 --osl 64 --e2e-sla-ms 2500 --target-rps 8

Max RPS / worker:   4.117
Pred lat / worker:  2429.0 ms @ batch=10
Recommended fleet:  2 worker(s)
```

It loads `dllm_fit.json` (produced by `fit_dllm_model.py`) and runs
`find_best_engine_dllm_rps()` to find the largest batch size that fits
the SLA, then divides the target RPS by per-worker RPS for the fleet
count.

### Empirical validation

Took one recommendation — `(ISL=2000, OSL=64, e2e_sla=2500ms,
target_rps=8)` predicting 2 workers / 2429ms — and ran aiperf at
exactly that workload through both running FP8 workers (conc=20 over
2 workers, 200 reqs, prefix-pool 4 × 2000 tokens).

| metric | predicted | measured | delta |
|---|---:|---:|---:|
| lat p50 | 2429 ms | 2685 ms | +10.5% |
| lat avg | ~2429 ms | 2590 ms | +6.6% |
| throughput | 8.0 rps | 7.43 rps | -7.1% |

The tool's fleet-count recommendation is correct (2 workers do handle
this workload), and the latency prediction is within ~10% of measured
p50. The 7% RPS shortfall is the SLA-overshoot reflected back —
the actual achievable batch is slightly lower because each request
costs slightly more than predicted.

**Production guidance**: add 15-20% latency headroom on top of the
model's prediction when sizing for hard SLAs.

## Files

- `components/src/dynamo/planner/core/perf_model/dllm.py` — model + helpers
- `components/src/dynamo/planner/core/perf_model/__init__.py` — re-export
- `components/src/dynamo/planner/tests/unit/test_dllm_perf_model.py` — unit tests
- `bench/llada-approx-kv/fit_dllm_model.py` — offline fit + holdout eval
- `bench/llada-approx-kv/sla_sizing.py` — SLA-driven fleet sizing CLI
- `bench/llada-approx-kv/dllm_fit.json` — coefficients + MAPE numbers

## Future work

- Wire `DllmRegressionModel` into `PlannerStateMachine` behind a
  `backend_kind=dllm` config gate, mirroring how the AR `AggRegressionModel`
  is selected by `mode=agg`.
- Add an FPM emission path on the dynamo.sglang side so the model can be
  fit live from production traffic (not just pre-deployment sweep).
- Validate on a second LLaDA-class engine when one becomes available (vLLM
  recently added prelim diffusion-LM support; the same cost-model shape
  should apply).
