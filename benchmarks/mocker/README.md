# AIC vs Mocker Pareto Comparison

Run the same parallelism × batch-size sweep through two perf predictors —
AIC's analytical model and dynamo's mocker simulator — and plot their
Pareto fronts on a single figure.

## Scripts

| File | Purpose |
|---|---|
| `lib/bindings/python/src/dynamo/replay/pareto.py` | Mocker-only sweep. Importable as a library; runnable as `python -m dynamo.replay.pareto`. Produces `raw_mocker.csv` + `pareto_mocker.csv`. |
| `benchmarks/mocker/pareto_comparison.py` | Driver that calls mocker pareto + AIC's `InferenceSession` Python API, then plots both. Produces additional `raw_aic.csv` + `pareto_aic.csv` + `pareto_plot.png` + `sweep_meta.json`. |

## Usage

```bash
python benchmarks/mocker/pareto_comparison.py \
    --model moonshotai/Kimi-K2.5 \
    --system b200_sxm \
    --backend vllm \
    --backend-version 0.19.0 \
    --total-gpus 8 \
    --isl 8192 --osl 1024 \
    --ttft 1000 --tpot 50 \
    --save-dir results/kimi_b200_compare
```

The CLI mirrors `aiconfigurator cli default` so users moving between AIC and
this tool don't have to relearn anything.

## Outputs (in `--save-dir`)

```
raw_mocker.csv             every mocker-evaluated (parallelism, bs) point
pareto_mocker.csv          mocker's Pareto-front subset (under TTFT/TPOT SLA)
best_topn_mocker.csv       top-N STRICT-SLA configs by tok/s/gpu (mocker)
raw_aic.csv                every AIC-evaluated point
pareto_aic.csv             AIC's Pareto-front subset
best_topn_aic.csv          top-N STRICT-SLA configs by tok/s/gpu (AIC)
pareto_plot.png            dual-curve plot (AIC blue / mocker orange)
sweep_meta.json            invocation args + AIC + mocker version pins
```

When `--mode {disagg, both}`, the same set is also written with a
`_disagg` suffix (`raw_mocker_disagg.csv`, etc.).

Axes are `tokens/s/user` (X) vs `tokens/s/gpu` (Y), matching AIC's own
`pareto_frontier.png` convention. SLA filtering: TTFT is always enforced
for the Pareto; pass `--strict-sla` to also enforce TPOT on the Pareto.
The **top-N CSVs always enforce both TTFT and TPOT strictly** — the top-N is
the practitioner's "which configs would I actually deploy?" answer.

## Top-N recommendations

By default both scripts also emit a top-`N` list of the SLA-compliant
configurations ranked by `tokens/s/gpu`, matching the semantics of
`aiconfigurator cli default --top-n`. Configure with `--top-n N` (default
`5`). Echoed to stdout as e.g.

```
=== Top 5 recommendations (mocker agg) ===
  #1: tp=8 dp=1 moe_tp=8 moe_ep=1 bs=32 gpus=8 | TTFT=739ms TPOT=45.6ms | tok/s/user=22.62 tok/s/gpu=86.4
  ...
```

## Current limitations (v1)

These are intentional trade-offs for the first iteration. They affect how
faithfully the two paretos can be compared:

### Sweep coverage

- **Batch size grid is powers of 2 [1, 2, 4, …, 256].**
  AIC's `find_best_agg_result_under_constraints` internally sweeps a much
  denser list (~45 values from `b_list_default` in each backend file). We
  use the coarser power-of-2 grid for speed. We stop at bs=256 because
  mocker's per-config wall time scales with `bs × num_requests` (where
  `num_requests = max(50, 5*bs)`), and bs=512 alone takes many minutes per
  config. With `--workers > 1` (default = half of `cpu_count`) the high-bs
  tier parallelizes naturally. Both AIC and mocker see this same coarse
  grid, so the comparison is well-defined on these points, but the
  resulting Pareto fronts will look coarser than AIC's own
  `aiconfigurator cli default` output, and the highest-bs regime (bs=512+)
  is not represented in v1.

- **`ctx_tokens` is held fixed per config.** AIC's sweep is 2D: for each
  bs it also sweeps `ctx_tokens` (vllm: `max_num_batched_tokens`) with
  step 512 up to `max(8192, 4*isl)`. We do not sweep this dimension at
  all — we set `max_num_batched_tokens = max(bs*2, 8192)` per point. This
  means deployments whose optimal operating point lies at unusual ctx_tokens
  values won't be discovered here.

- **`max_batch_size` capped at 512.** AIC notes in source comments that
  perf-DB coverage degrades past 1024, so it also drops 1024 in practice
  when `max_batch_size=512` (the default). We mirror this cap.

### Deployment-mode coverage

- **`--mode {agg, disagg, both}`** — agg is the default. Disagg sweeps the
  cross-product of prefill / decode parallelism × prefill_bs × decode_bs ×
  decode_workers, filtered by total_gpus and the invariant
  `decode.num_gpus × decode_workers ≥ prefill.num_gpus × prefill_workers`
  (decode is the throughput bottleneck; configs where decode has fewer GPUs
  than prefill are rarely interesting). `prefill_workers` is held at 1 in
  v1 to keep the search space tractable; bump
  `DISAGG_PREFILL_WORKERS` in `pareto.py` for asymmetric replica counts.

  Disagg constants (v1):
    - `DISAGG_PREFILL_BS = (1, 2, 4, 8)`
    - `DISAGG_DECODE_BS  = (32, 64, 128)`
    - `DISAGG_PREFILL_WORKERS = (1,)`
    - `DISAGG_DECODE_WORKERS  = (1, 2, 4)`

  Empirically on Kimi-K2.5 / b200 / 8 GPUs these produce ~540 viable points
  and finish in ~10-15 min with `--workers 6`.

- **Backend-aware parallelism grid is a subset of AIC's.** Our
  `_derive_parallel_grid` covers dense vs MoE, GB200/GB300 widening, and
  the wideep code paths for trtllm/sglang. It does **not** replicate AIC's
  full `TaskConfigFactory._agg_defaults_layer` logic (e.g., GQA+MoE vs
  MLA+MoE heuristics, system-version-specific quirks). For mainline models
  on supported systems the two grids agree; for edge cases (notably
  DeepSeek MLA on non-default configs) AIC may explore configs we skip.

### Engine-args fidelity (mocker side)

- **`dp_size` forced to 1 in MockEngineArgs.** Mocker's concurrency-replay
  validator rejects any other value (`validate.rs:55-60`). The deployment's
  actual DP world size is carried via `--num-workers N` (replay-level pool)
  and `aic_attention_dp_size: N` inside engine-args (AIC perf-model side).
  This is correct but worth knowing if you read the emitted JSON.

- **KV cache dtype is not modeled by mocker's MockEngineArgs.** AIC's perf
  model accounts for KV dtype via the `aic_backend` hookup, so the timing
  predictions still reflect KV quant correctly, but mocker's scheduling /
  KV-cache-capacity calculations use vllm/sglang defaults. If your
  comparison hinges on KV-dtype sensitivity, AIC's analytical side is the
  more trustworthy half.

- **`enable_prefix_caching` is forced false.** AIC's `--prefix` flag is
  not plumbed through. Workloads with significant prefix reuse aren't
  modeled accurately on either side.

### Workload coverage

- **No quantization sweep.** Both AIC and mocker use whatever quant the
  model's `config.json` declares. To compare quant variants, run the sweep
  multiple times.

- **No speculative decoding / MTP.** AIC supports `--nextn` (draft tokens);
  mocker does not model speculative decoding. Sweeps assume `nextn=0`.

- **Fixed ISL/OSL per run.** AIC's `cli default` supports varying request
  patterns via experiment YAML; we take a single `--isl --osl` pair.

### Numerical / reproducibility

- **Single mocker run per (parallelism, bs).** No averaging across multiple
  random seeds. Mocker's sim is deterministic per seed but the default
  could mask noise if it's introduced later.

- **AIC's pareto on our grid ≠ AIC's pareto from `cli default`.** The
  driver computes AIC's pareto from the per-point `run_agg` calls on our
  coarse grid. AIC's own `aiconfigurator cli default` uses the denser bs
  grid + ctx_tokens sweep, so its `pareto_frontier.png` may show a denser /
  shifted front than `pareto_aic.csv` here. Compare against AIC's own
  output if you want the analytical-best AIC pareto.

## Roadmap

In rough priority order, these are the natural next iterations:

1. Add disagg coverage (`--mode disagg`), wiring `--prefill-engine-args` +
   `--decode-engine-args` per config.
2. Add a `--bs-mode {pow2, aic}` flag toggling between the coarse v1 grid
   and AIC's verbatim `b_list_default`.
3. Sweep `ctx_tokens` as an inner loop, matching AIC's 2D grid.
4. Plumb `--prefix`, `--nextn`, `--enable-chunked-prefill` so they round-trip
   through both evaluators.
5. Optional parallel execution (`multiprocessing.Pool`) to cut wall time
   for the denser grids.
