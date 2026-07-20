---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Spica Search Space
subtitle: Searchable deployment, engine, router, Planner, and parallelism settings
---

> [!WARNING]
> **Experimental.** Spica is intended for evaluation and feedback, not production capacity
> planning. Its API, configuration schema, search results, and deployment output may change
> without a standard deprecation period. Spica provides no SLA, accuracy, or configuration
> optimality guarantees.

A `SearchSpace` (the `search_space:` block of a `SmartSearchConfig` YAML) is the input
to one smart-sweep run: the knobs to **explore**, plus the **pinned** context (model,
hardware, GPU budget, and the engine/router/kv-manager/planner scalars). The optimizer
runs one Vizier study per `deployment_mode` branch and ranks candidates across branches.

This document is the reference for **every knob**: its type, default, whether it is
searched or pinned, and its allowed choices. Source of truth:
`components/src/dynamo/profiler/spica/config.py`
(`SearchSpace`, `SEARCH_CHOICES`, `COMPOSITE_DICT_KEYS`, `COMPOSITE_REQUIRED_KEYS`).

## Kinds of knob

There are four kinds of knob:

| Kind | How to pin | How to search |
|---|---|---|
| **Atomic list knob** | a single-element list, `[x]` | a multi-element list, `[a, b, c]` |
| **Pinned scalar** | just set the value (always one value) | — (never searched) |
| **Composite knob** | one entry: a preset id **or** a dict | multiple entries (presets and/or dicts) |
| **`parallel_configs`** | one `dict` entry (+ single mode) | multiple `dict` entries (a custom menu) |

- **Atomic list knob** — a list whose entries must be a non-empty subset of that knob's
  `SEARCH_CHOICES`. One element pins; many search.
- **Pinned scalar** — a plain value; never enters the sweep.
- **Composite knob** — bundles several coupled *unrolled* planner/predictor fields behind a
  named preset; each list entry is a preset id **or** a self-contained dict.
- **`parallel_configs`** — the parallel shape + replica count. Left **empty** it is *derived*
  by KV-feasible enumeration (see [Parallel configs](#parallel-configs-derived)); provide a
  list of dicts to pin one or search a custom menu.

`_validate_search_choices` enforces list knobs are non-empty and every entry is legal.

## Deployment

| knob | type | default | searched / pinned | allowed choices |
|---|---|---|---|---|
| `deployment_mode` | list[str] | `["disagg", "agg"]` | searched (branch) | `disagg`, `agg` |
| `backend` | list[str] | `["vllm"]` | searched | `vllm`, `sglang`, `trtllm` |
| `model_name` | str | — (required) | pinned | HF id or private model name |
| `hardware_sku` | str | — (required) | pinned | any SKU with a system YAML (e.g. `h200_sxm`) |
| `gpu_budget` | int | `32` | pinned | max GPUs per candidate |
| `min_gpu_budget` | int? | `None` | pinned | `0 < min_gpu_budget <= gpu_budget` |
| `min_endpoint` | int? | `None` | pinned | declared-but-unused (kept as-is) |
| `context_length` | int? | `None` | pinned | `max_seq_len` for KV feasibility; `None` → model max |
| `startup_time` | float? | `None` | pinned | — |
| `aic_nextn` | int? | `None` | pinned | speculative-decode (MTP) depth (the `1..5` in the source is a comment hint; **not** validated) |

`deployment_mode` is the only knob that branches the sweep (agg / disagg have structurally
different parallel configs); `backend` is a *searched knob within each branch*, not a
separate study. `min_gpu_budget` is validated against `gpu_budget` by `_validate_gpu_budget`.

## Prefill engine (disagg branch)

| knob | type | default | searched / pinned | allowed choices |
|---|---|---|---|---|
| `prefill_max_num_batched_tokens` | list[int] | `[8192, 16384, 32768]` | searched | `8192`, `16384`, `32768` |
| `prefill_max_num_seqs` | list[int] | `[1, 2, 4, 8, 16, 32, 64, 128, 256]` | searched | `1`, `2`, `4`, `8`, `16`, `32`, `64`, `128`, `256` |
| `prefill_block_size` | int | `64` | pinned | — |
| `prefill_gpu_memory_utilization` | float | `0.9` | pinned | — |
| `prefill_enable_prefix_caching` | bool | `True` | pinned | — |

## Decode engine (disagg branch)

| knob | type | default | searched / pinned | allowed choices |
|---|---|---|---|---|
| `decode_max_num_batched_tokens` | list[int] | `[8192]` | searched | `8192` |
| `decode_max_num_seqs` | list[int] | `[256, 512, 1024]` | searched | `256`, `512`, `1024` |
| `decode_block_size` | int | `64` | pinned | — |
| `decode_gpu_memory_utilization` | float | `0.9` | pinned | — |
| `decode_enable_prefix_caching` | bool | `False` | pinned | forced off for decode workers |

## Agg engine (agg branch)

| knob | type | default | searched / pinned | allowed choices |
|---|---|---|---|---|
| `agg_max_num_batched_tokens` | list[int] | `[8192, 16384, 32768]` | searched | `8192`, `16384`, `32768` |
| `agg_max_num_seqs` | list[int] | `[256, 512, 1024]` | searched | `256`, `512`, `1024` |
| `agg_block_size` | int | `64` | pinned | — |
| `agg_gpu_memory_utilization` | float | `0.9` | pinned | — |
| `agg_enable_prefix_caching` | bool | `True` | pinned | — |

## KV manager (multi-tier offload)

All pinned. G3/G4 extend G2; offload is **off by default** (`num_g2_blocks = 0`).
When G2 is enabled, `kv_bytes_per_token` is required so replay cannot silently
disable offload if model metadata is private or unavailable.

| knob | type | default | searched / pinned | allowed choices |
|---|---|---|---|---|
| `num_g2_blocks` | int | `0` | pinned | `0` disables host offload |
| `kv_bytes_per_token` | int? | `None` | pinned | required when `num_g2_blocks > 0` |
| `bandwidth_g1_to_g2_gbps` | float? | `None` | pinned | — |
| `bandwidth_g2_to_g1_gbps` | float? | `None` | pinned | — |
| `offload_batch_size` | int? | `None` | pinned | — |

## Router

KV-router weights are ignored under `round_robin`. When `router_mode` is pinned to
`["round_robin"]`, the search-space builder removes all dependent router knobs from the
Vizier study. A mixed `["kv_router", "round_robin"]` study retains them because they are
active for its KV-router trials.

| knob | type | default | searched / pinned | allowed choices |
|---|---|---|---|---|
| `router_mode` | list[str] | `["kv_router", "round_robin"]` | searched | `kv_router`, `round_robin` |
| `overlap_score_credit` | list[float] | `[0.0, 0.5, 1.0]` | searched | `0.0`, `0.5`, `1.0` |
| `prefill_load_scale` | list[float] | `[0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]` | searched | `0.0`, `0.25`, `0.5`, `1.0`, `2.0`, `4.0`, `8.0`, `16.0`, `32.0` |
| `router_temperature` | list[float] | `[0.0, 0.2, 0.5, 1.0]` | searched | `0.0`, `0.2`, `0.5`, `1.0` |
| `host_cache_hit_weight` | list[float] | `[0.5, 0.75, 1.0]` | searched **iff** `num_g2_blocks > 0` | `0.5`, `0.75`, `1.0` |
| `disk_cache_hit_weight` | list[float] | `[0.0, 0.25, 0.5]` | searched **iff** `num_g2_blocks > 0` | `0.0`, `0.25`, `0.5` |

`host_cache_hit_weight` / `disk_cache_hit_weight` weight the **host/disk extension blocks**
in the router's scoring. With offload off (the default `num_g2_blocks = 0`) those blocks are
0, so the weights can't affect a replay and are dropped from the search to avoid dead
dimensions. They are only swept when multi-tier KV offload is enabled (`num_g2_blocks > 0`).

## Admission control (pinned)

| knob | type | default | searched / pinned | allowed choices |
|---|---|---|---|---|
| `active_decode_blocks_threshold` | int? | `None` | pinned | — |
| `active_prefill_tokens_threshold` | int? | `None` | pinned | — |
| `active_prefill_tokens_threshold_frac` | float? | `None` | pinned | — |
| `no_admission_control` | bool | `False` | pinned | — |

For a search that includes `kv_router`, these fields are reserved but currently rejected
when set: Dynamo's replay API does not accept admission-control configuration, so scoring
them would diverge from the generated frontend. Leave them at their defaults until replay
support is available. They remain inert under a round-robin-only search.

## Planner composites

Composite knobs. Each list entry is a preset id (string) **or** a dict; one entry pins,
several search; presets and dicts can be mixed. Choices below are the preset ids; the
preset → field expansions are in [Composite presets](#composite-presets).

| knob | type | default | searched / pinned | allowed preset ids |
|---|---|---|---|---|
| `planner_scaling_policy` | list[str\|dict] | `["disabled", "throughput_180_5", "throughput_600_5", "load_180_5", "load_180_10", "hybrid_180_5", "hybrid_600_5"]` | searched | `disabled`, `throughput_180_5`, `throughput_600_5`, `load_180_5`, `load_180_10`, `hybrid_180_5`, `hybrid_600_5` |
| `planner_fpm_sampling` | list[str\|dict] | `["small", "default", "large", "fine"]` | searched | `small`, `default`, `large`, `fine` |
| `planner_load_sensitivity` | list[str\|dict] | `["aggressive", "default", "conservative"]` | searched | `aggressive`, `default`, `conservative` |

When every configured `planner_scaling_policy` disables both throughput and load scaling
(for example `["disabled"]`), the search-space builder removes `planner_fpm_sampling` and
`planner_load_sensitivity` from the Vizier study. Mixed studies retain them for policies that
enable the planner.

## Load predictor

| knob | type | default | searched / pinned | allowed preset ids |
|---|---|---|---|---|
| `load_predictor_candidates` | list[str\|dict] | `["constant_last", "arima_raw", "arima_log1p", "prophet_w20_raw", "prophet_w20_log1p", "prophet_w50_raw", "prophet_w50_log1p", "kalman_default_raw", "kalman_default_log1p", "kalman_reactive_raw", "kalman_reactive_log1p"]` | swept **separately** | the 11 ids in the default |

`load_predictor_candidates` is the forecaster for predictive throughput scaling. It is
**not** part of the main Vizier study: a separate forecast-loss sub-sweep
(`sweep_load_predictor`) scores every entry against the trace by one-step-ahead loss and
pins the per-interval winner into the main sweep. Triggered only when some
`planner_scaling_policy` candidate enables throughput scaling; for a static (non-trace)
workload it short-circuits to `constant_last` for every interval.

## How the sweep uses these (branches & backend)

The sweep runs **one Vizier study per `deployment_mode`**. **`backend` is a searched knob,
not a branch**: listing multiple backends searches them *together* within each mode's study,
and the cross-branch merge picks the global best — `rank()` for a single-objective goal, or
`pareto_front()` (the non-dominated set) under a `pareto` goal. The parallel-config pool is the **union** of every
backend's KV-feasible configs. Backend is selected first and the latent shape request is
projected only onto configs that backend supports. A backend with no
perf DB / no viable config for a mode is dropped; a **mode** for which no backend is viable
is **skipped with a warning** (a viable mode still runs); only if *no* mode is viable does
the run error. A *pinned* config legal for no backend is a hard error (fail fast).

Replay capability is part of viability: TRT-LLM is excluded from disaggregated branches
because Dynamo replay currently rejects TRT-LLM disaggregation. It remains searchable in
aggregate branches.

The planner's `optimization_target` is derived from the sweep **goal**, not from
`planner_scaling_policy` (see `OptimizationTarget.planner_optimization_target`):
`throughput`/`throughput_per_gpu`/`throughput_per_user` → `"throughput"`, `e2e_latency` →
`"latency"`, `goodput`/`goodput_per_gpu` → `"sla"`, `pareto` → `"throughput"`. The policy
only decides *which* scaling loops run and their intervals.

**Predictive throughput scaling needs an SLA**, so it only works under a goodput sweep
(`optimization_target="sla"`). For a `throughput`/`e2e_latency`/`pareto` sweep,
`filter_scaling_policies(allow_throughput=False)` **drops** every throughput-scaling entry
(`throughput_*`, `hybrid_*`, or any dict with `enable_throughput_scaling: true`); only
`disabled` / `load_*` survive. Listing *only* throughput-scaling policies for a non-SLA
sweep errors (nothing left to search). Both scaling flags false ⇒ planner off (static
replica count, same as `disabled`).

An e2e-only goodput SLA can be replay-scored, but it cannot seed the planner's ttft/itl SLA
target. In that case the main sweep drops every planner-scaling entry (throughput, load, and
hybrid) and keeps only static policies such as `disabled`; if nothing static remains, it raises a
clear config error.

## Composite presets

A composite dict entry **replaces** the preset expansion (no partial / merge). Required keys
are enforced at load by `_validate_search_choices` against `COMPOSITE_REQUIRED_KEYS`: the
three **planner** composites must provide *all* their unrolled fields; a
`load_predictor_candidates` dict needs at least `load_predictor` (the family), with family
params defaulting per family. Unknown keys are rejected. Value *legality* (perfect-square fpm
bucket, positive intervals, `load_scaling_down_sensitivity` in 0–100, …) is validated
downstream by Dynamo's `PlannerConfig`.

### `planner_scaling_policy`

Decoded by `SCALING_POLICIES` in `components/src/dynamo/profiler/spica/planner.py`.

| preset | `enable_throughput_scaling` | `enable_load_scaling` | `throughput_adjustment_interval_seconds` | `load_adjustment_interval_seconds` |
|---|---|---|---|---|
| `disabled` | False | False | — | — |
| `throughput_180_5` | True | False | 180 | 5 |
| `throughput_600_5` | True | False | 600 | 5 |
| `load_180_5` | False | True | 180 | 5 |
| `load_180_10` | False | True | 180 | 10 |
| `hybrid_180_5` | True | True | 180 | 5 |
| `hybrid_600_5` | True | True | 600 | 5 |

Dict keys (all required): `enable_throughput_scaling`, `enable_load_scaling`,
`throughput_adjustment_interval_seconds`, `load_adjustment_interval_seconds`.

### `planner_fpm_sampling`

Decoded by `FPM_SAMPLING`.

| preset | `max_num_fpm_samples` | `fpm_sample_bucket_size` |
|---|---|---|
| `small` | 32 | 4 |
| `default` | 64 | 16 |
| `large` | 128 | 16 |
| `fine` | 128 | 64 |

Dict keys (all required): `max_num_fpm_samples`, `fpm_sample_bucket_size` (a perfect square,
checked downstream).

### `planner_load_sensitivity`

Decoded by `LOAD_SENSITIVITY`.

| preset | `load_scaling_down_sensitivity` | `load_min_observations` |
|---|---|---|
| `aggressive` | 70 | 3 |
| `default` | 80 | 5 |
| `conservative` | 90 | 8 |

Dict keys (all required): `load_scaling_down_sensitivity` (0–100), `load_min_observations`.

### `load_predictor_candidates`

Decoded by `LOAD_PREDICTOR_PRESETS` in
`components/src/dynamo/profiler/spica/load_predictor_sweep.py`.

| preset | `load_predictor` | `load_predictor_log1p` | extra family fields |
|---|---|---|---|
| `constant_last` | constant | False | — |
| `arima_raw` / `arima_log1p` | arima | False / True | — |
| `prophet_w20_raw` / `prophet_w20_log1p` | prophet | False / True | `prophet_window_size`=20 |
| `prophet_w50_raw` / `prophet_w50_log1p` | prophet | False / True | `prophet_window_size`=50 |
| `kalman_default_raw` / `kalman_default_log1p` | kalman | False / True | `kalman_q_level`=1.0, `kalman_q_trend`=0.1, `kalman_r`=10.0, `kalman_min_points`=5 |
| `kalman_reactive_raw` / `kalman_reactive_log1p` | kalman | False / True | `kalman_q_level`=10.0, `kalman_q_trend`=1.0, `kalman_r`=5.0, `kalman_min_points`=3 |

Dict keys: `load_predictor` (required; one of `constant` / `arima` / `prophet` / `kalman`),
`load_predictor_log1p`, `prophet_window_size` (prophet), and
`kalman_q_level` / `kalman_q_trend` / `kalman_r` / `kalman_min_points` (kalman). Omitted
family fields take the planner defaults (`prophet_window_size`=50, `kalman_q_level`=1.0,
`kalman_q_trend`=0.1, `kalman_r`=10.0, `kalman_min_points`=5).

## Parallel configs (derived)

Left empty, `parallel_configs` is **enumerated** by `parallel_configs_for`
(`components/src/dynamo/profiler/spica/model_hw.py`) on top of `enumerate_parallel_configs` /
`enumerate_disagg_configs` (`components/src/dynamo/profiler/spica/parallel_enum.py`). Per `(model, hardware,
gpu_budget, backend)`:

1. **GPUs per worker** is drawn from the ladder `{1, 2, 4, 8, 16}`
   (`_DEFAULT_GPUS_PER_WORKER`), capped at `gpu_budget`. A worker spans `tp * pp * dp` GPUs
   with `pp` pinned to 1, so `gpus_per_worker = tp * dp`.
2. **Dense models** use plain TP only: `{tp: g, dp: 1, moe_tp: 1, moe_ep: 1}`.
3. **MoE models** scan `tp / dp / moe_tp / moe_ep` over `{1, 2, 4, 8, …}` subject to the MoE
   width constraint `tp * attention_dp == moe_tp * moe_ep` (here `dp` = `attention_dp`), and
   keep only these four pure patterns:
   - **TEP** — attention-TP + expert-EP: `tp > 1`, `dp == 1`, `moe_tp == 1`, `moe_ep > 1`.
   - **DEP** — attention-DP + expert-EP: `tp == 1`, `dp > 1`, `moe_tp == 1`, `moe_ep > 1`.
   - **TP** — `tp > 1`, `dp == 1`, `moe_tp > 1`, `moe_ep == 1`.
   - **DTP** — `tp == 1`, `dp > 1`, `moe_tp > 1`, `moe_ep == 1`.
     MoE tensor parallelism is scanned for every MoE model; backend filters below can still
     remove unsupported combinations.
4. **Backend filters** (mirroring AI Configurator's `enumerate_parallel_config`):
   - `trtllm` forbids `tp > 1 & attention_dp > 1`.
   - `sglang` EP-only MoE backends (wideep / `deepep_moe` / `megamoe`) force `moe_tp == 1`.
   - `vllm` forbids `moe_tp > 1 & moe_ep > 1`.
   Large MoE that a node can't hold (`gpus_per_node * vram_per_gpu < 2 * weight_bytes`)
   auto-enables multi-node wideEP.
5. **KV feasibility** (`dynamo.profiler.spica.kv_estimate.feasible_shape_tokens`, via AI
   Configurator's `estimate_kv_cache`): when the compatible memory estimator is installed, a shape
   is kept iff its total KV capacity in tokens (after quantized weights + activation reservations)
   holds at least one longest sequence —
   `total_kv_size_tokens > max_seq_len`, where `max_seq_len` = `context_length` if set, else
   the model's max context. This is per-shape (TEP / DEP / TP differ at the same GPU count)
   and uses the real (often FP8) weights. SKUs with no perf DB raise `NoPerfDatabase` (no
   naive fallback). If no shape is feasible within budget, `NoViableParallelConfig` is raised
   for that branch. The default Planner/Profiler image retains AI Configurator 0.9, which does not
   provide `aiconfigurator.sdk.memory`; Spica warns and skips this pre-search filter in that image.
   Trace and fixed-concurrency workloads still reach Replay, while `kv_load_ratio` fails closed
   because candidate-relative concurrency cannot be derived without the estimator.
6. **Replicas** fill the budget: for each kept worker shape, replica counts
   `r ∈ 1..(gpu_budget // g)` such that `g * r` lies in
   `[min_gpu_budget, gpu_budget]`.
7. **disagg** enumerates prefill × decode **independently** (each role its own shape +
   replica count) and pairs them so `prefill.total_gpus + decode.total_gpus` fits the budget;
   prefill/decode throughput rate-matching is applied downstream at replay.

### Parallel search projection

Vizier searches model-visible latent dimensions and projects every suggestion onto the valid
parallel-config pool. There is no opaque config-index path. These dimensions are **internal
Vizier parameters**, not fields accepted under `search_space:`. Users constrain them through
the model, hardware, backend, GPU budget, context length, and optional `parallel_configs` pool.

#### Internal Vizier parameters

The projector defines up to three aggregate features, plus `agg_ffn_mode` when the valid pool
contains MoE TP or EP:

| parameter | Vizier type | actual value encoded from a valid config | range / feasible values | default |
|---|---|---|---|---|
| `used_gpu_ratio` | continuous float | `(gpus_per_worker * replicas) / gpu_budget` | minimum to maximum ratio present in the branch pool | `1.0`, clipped to the range |
| `agg_num_gpus_per_engine_target` | numeric discrete, log scale | `tp * pp * attention_dp` | sorted distinct worker sizes present in the pool; a subset of `{1, 2, 4, 8, 16}` | feasible value nearest the geometric center |
| `agg_attention_mode` | categorical | `dp` when `attention_dp > 1`, otherwise `tp` | modes present in the pool, ordered `tp`, `dp` | `tp` if present, otherwise `dp` |
| `agg_ffn_mode` | categorical; MoE only | `ep` when `moe_ep > 1`, otherwise `tp` | modes present in the pool, ordered `ep`, `tp` | `ep` if present, otherwise `tp` |

It defines up to six disaggregated features, plus one FFN-mode feature per role when the valid
pool contains MoE TP or EP:

| parameter | Vizier type | actual value encoded from a valid config | range / feasible values | default |
|---|---|---|---|---|
| `used_gpu_ratio` | continuous float | `(prefill_total_gpus + decode_total_gpus) / gpu_budget` | minimum to maximum ratio present in the branch pool | `1.0`, clipped to the range |
| `prefill_gpu_share` | continuous float | `prefill_total_gpus / (prefill_total_gpus + decode_total_gpus)` | minimum to maximum share present in the branch pool | `0.5`, clipped to the range |
| `prefill_num_gpus_per_engine_target` | numeric discrete, log scale | prefill `tp * pp * attention_dp` | sorted distinct prefill worker sizes in the pool | feasible value nearest the geometric center |
| `decode_num_gpus_per_engine_target` | numeric discrete, log scale | decode `tp * pp * attention_dp` | sorted distinct decode worker sizes in the pool | feasible value nearest the geometric center |
| `prefill_attention_mode` | categorical | prefill `tp` or `dp` | modes present in the prefill pool, ordered `tp`, `dp` | first present mode |
| `decode_attention_mode` | categorical | decode `tp` or `dp` | modes present in the decode pool, ordered `tp`, `dp` | first present mode |
| `prefill_ffn_mode` | categorical; MoE only | prefill `ep` or `tp` | modes present in the prefill pool, ordered `ep`, `tp` | first present mode |
| `decode_ffn_mode` | categorical; MoE only | decode `ep` or `tp` | modes present in the decode pool, ordered `ep`, `tp` | first present mode |

Here `prefill_total_gpus = prefill_gpus_per_engine * prefill_replicas`, with the analogous
definition for decode. A one-GPU shape is canonicalized as attention `tp` and FFN `tp`.
`*_ffn_mode` is omitted when the pool has no MoE shape. Any single-valued feature is also
omitted from Vizier and injected as a constant. A one-config pool bypasses projection entirely.

The geometric-center default for an engine-size dimension is the feasible value closest in
log space to `sqrt(min_size * max_size)`; a tie picks the smaller value. For example,
`{1, 2, 4, 8, 16}` defaults to `4`. The ranges and defaults are rebuilt for every deployment
mode from that branch's union of valid configs across its viable backends. With multiple
backends, a value may therefore exist in the study-wide range but not for the backend selected
by a particular trial. Projection handles that case instead of marking the trial infeasible.

#### Projection onto a valid config

For latent request `z` and its selected backend, `ParallelConfigProjector.project` applies the
following deterministic steps:

1. **Backend filter.** Remove every config that is not legal and KV-feasible for the selected
   backend. GPU-budget and shape validity were already enforced when this pool was built.
2. **Mode match.** For every remaining config, count mismatches across all requested
   attention and FFN modes. Keep only configs with the minimum mismatch count. Consequently,
   an exact mode combination always wins over a numerically closer config with a different
   mode. If no config realizes all requested modes, the least-mismatched mode combination is
   used and trial metadata records `mode_projected: true`.
3. **Numeric distance.** Among the mode survivors, minimize the sum of squared normalized
   deltas over `used_gpu_ratio`, `prefill_gpu_share` when applicable, and every per-engine GPU
   target. Ratios and shares use their linear value; engine sizes use `log2(size)`. Each delta
   is divided by that feature's min-to-max span **within the selected backend's pool**:

   ```text
   R_j = max_c T_j(actual(c)) - min_c T_j(actual(c))
   distance(c, z) = sum_{j: R_j > 0} ((T_j(actual(c)) - T_j(z)) / R_j)^2

   T_j(x) = log2(x)  for *_num_gpus_per_engine_target
          = x        for used_gpu_ratio and prefill_gpu_share
   ```

   A feature with zero span for that backend contributes zero. All non-constant numeric
   features otherwise have equal weight after normalization.
4. **Stable tie-break.** Equal distances use a deterministic lexicographic shape-and-replica
   key, so projection never depends on evaluation history.

Replica count, concrete `tp` / `attention_dp` / `moe_tp` / `moe_ep`, and total GPU usage are
not independent Vizier knobs. They come from the selected valid config. This preserves useful
numeric and categorical structure for Vizier while guaranteeing replay never receives an
illegal parallel combination.

For example, consider a 32-GPU disagg branch containing these two valid configs for one
backend:

| config | prefill | decode | used ratio | prefill share | decode FFN mode |
|---|---|---|---:|---:|---|
| A | 2 replicas x 4 GPUs | 1 replica x 8 GPUs | `16/32 = 0.5` | `8/16 = 0.5` | `ep` |
| B | 1 replica x 4 GPUs | 3 replicas x 8 GPUs | `28/32 = 0.875` | `4/28 = 0.143` | `tp` |

A request for A's numeric values (`used_gpu_ratio=0.5`, `prefill_gpu_share=0.5`, engine sizes
`4 + 8`), the modes shared by both configs, but decode FFN mode `tp` projects to B. Mode
matching removes A before numeric distance is considered, so B's actual ratios become `0.875`
and `0.143`; its concrete replica counts `1 + 3` are carried into replay.

Requested features, actual features, numeric distance, whether a mode fallback occurred, and
the full selected parallel config are recorded in Vizier trial metadata under
`spica_projection`.

Different latent requests may project to the same full sample. Spica reuses the successful
replay result and makes bounded replacement asks; only unique successful samples count toward
`candidates_per_round`. After `11 * candidates_per_round` attempts, the branch may stop early.
Projection never changes to the nearest *untested* config, which would make it
history-dependent.

**Derived, not settable:** `strategy` (`tp` / `tep` / `dep` / `dtp`, computed by
`ParallelShape.strategy`; it also has a `mixed` fallback that enumeration never emits) and
`used_gpus` (`gpus_per_worker × replicas`, summed across roles for disagg).

### Pinning `parallel_configs`

Provide a list of dicts to **pin** (one entry) or search a **custom menu** (several entries).
A pinned config is kept for whichever backends it is legal + feasible on (errors if none).
`_validate_parallel_configs` requires `deployment_mode` to list **exactly one mode** when
`parallel_configs` is non-empty.

With projection, one entry bypasses all parallel Vizier dimensions and projection;
several entries form the complete projection pool, so the sampler can never leave the user's
menu. The YAML schema is unchanged.

- **agg** entry — a flat shape dict: `tp` (required), `attention_dp`, `moe_tp`, `moe_ep`,
  `pp`, `replicas`. Omitted dims default to `1`; `replicas` defaults to `1`. Dense models can
  write just `{tp: N}`.
- **disagg** entry — nests two shape dicts: `{prefill: <shape>, decode: <shape>}`, each with
  its own `replicas`.

Each pinned shape is validated against the same rules the enumerator applies — MoE width
(`tp × attention_dp == moe_tp × moe_ep`), `gpus_per_worker ∈ {1, 2, 4, 8, 16}`, backend
filters, KV feasibility for the model's max sequence, `used_gpus ≤ gpu_budget`. Structural +
single-mode checks run at config load; full legality / KV-feasibility when branches are
enumerated. An illegal pin is rejected before replay.

## Examples

Pin a single deployment (one candidate — handy for a targeted re-evaluation):

```yaml
search_space:
  model_name: deepseek-ai/DeepSeek-V3
  hardware_sku: gb200
  deployment_mode: [agg]            # single mode (required to pin parallel_configs)
  backend: [trtllm]
  gpu_budget: 64
  parallel_configs:
    - {tp: 4, moe_ep: 4, replicas: 2}     # TEP, 8 GPUs
  agg_max_num_batched_tokens: [16384]
  agg_max_num_seqs: [512]
  router_mode: [round_robin]
  planner_scaling_policy: ["load_180_5"]
  planner_fpm_sampling: ["default"]
  planner_load_sensitivity: ["default"]
```

Search presets but pin one composite field with a dict (custom 240 s interval):

```yaml
  planner_scaling_policy:
    - "throughput_180_5"
    - {enable_throughput_scaling: true, enable_load_scaling: false,
       throughput_adjustment_interval_seconds: 240, load_adjustment_interval_seconds: 5}
```

Pin a disagg parallel config (prefill TEP, decode DEP):

```yaml
  deployment_mode: [disagg]
  parallel_configs:
    - prefill: {tp: 8, attention_dp: 1, moe_tp: 1, moe_ep: 8, replicas: 1}   # 8 GPUs
      decode:  {tp: 1, attention_dp: 8, moe_tp: 1, moe_ep: 8, replicas: 2}   # 16 GPUs
```

## Validation summary

- **List knob** — non-empty; string entries must be a listed `SEARCH_CHOICES` value
  (`_validate_search_choices`).
- **Composite dict** — no unknown keys, and the required keys are present (all unrolled
  fields for the planner composites; at least `load_predictor` for
  `load_predictor_candidates`). Value legality is checked downstream by Dynamo's
  `PlannerConfig`.
- **`gpu_budget` / `min_gpu_budget`** — `0 < min_gpu_budget <= gpu_budget`
  (`_validate_gpu_budget`).
- **`parallel_configs`** — structural + single-mode at config load
  (`_validate_parallel_configs`); full legality / KV-feasibility when branches are enumerated.
