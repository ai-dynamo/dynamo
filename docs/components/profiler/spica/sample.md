---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Spica Unrolled Samples
subtitle: How Spica turns a search trial into a concrete deployment configuration
---

> [!WARNING]
> **Experimental.** Spica is intended for evaluation and feedback, not production capacity
> planning. Its API, configuration schema, search results, and deployment output may change
> without a standard deprecation period. Spica provides no SLA, accuracy, or configuration
> optimality guarantees.

The [Spica search space](search-space.md) defines what gets explored. A single *trial* picks
one concrete point in it: a **selection** (one value per searched knob, chosen by the
sampler), the one **`parallel_config`** the sampler selected or projected from the branch's valid pool, and
the **load-predictor winner** from the independent forecast sub-sweep.
`unroll_sample` (`components/src/dynamo/profiler/spica/sample.py`) expands those into a **flat,
self-contained deployment-config dict** — the `config` of a `Candidate`
(`components/src/dynamo/profiler/spica/config.py`). It
unrolls every composite knob (parallel shape, planner presets, load predictor) and folds
in the pinned scalars, so downstream DGD generation never re-decodes a preset string or a
structured config object.

The flat dict carries **only the knobs relevant to the chosen
`deployment_mode` / `router_mode` / planner state** — irrelevant dimensions are simply
absent (not set to a sentinel). What follows is what a sample contains and the three ways a
user pins/overrides what gets emitted.

## What the flat sample contains

`unroll_sample(search_space, selection, parallel_config, load_predictor)` builds the dict in
this order.

### Mode + backend

| key | source |
|---|---|
| `deployment_mode` | `selection["deployment_mode"]` (`agg` / `disagg`) |
| `backend` | `selection["backend"]` (`vllm` / `sglang` / `trtllm` enum values) |

### Pinned scalars (always present, never searched)

Folded straight off the `SearchSpace` (`_DEPLOYMENT_PINNED` + `_KV_MANAGER`):

- deployment/runtime context: `model_name`, `hardware_sku`, `context_length`,
  `startup_time`, `aic_nextn`
- planner limits: `gpu_budget`, `min_gpu_budget`, `min_endpoint`
- kv-manager offload: `num_g2_blocks`, `kv_bytes_per_token`, `bandwidth_g1_to_g2_gbps`,
  `bandwidth_g2_to_g1_gbps`, `offload_batch_size`

The GPU bounds and endpoint floor constrain candidate enumeration, and they also become
live runtime policy (`max_gpu_budget`, `min_gpu_budget`, `min_endpoint`) when the selected
candidate enables the planner. Keeping them in the flat sample makes replay and generated
deployment artifacts reproduce the search contract.

### Parallel fields (`_unroll_parallel`)

Unrolled from the chosen `parallel_config` via `_shape_fields`. The keys differ by mode:

- **agg** — flat: `tp`, `pp`, `attention_dp`, `moe_tp`, `moe_ep`, `strategy`, `replicas`,
  `used_gpus`.
- **disagg** — the same per-shape keys prefixed per role plus a per-role `replicas`:
  `prefill_tp` … `prefill_strategy`, `prefill_replicas`, `decode_tp` … `decode_strategy`,
  `decode_replicas`, and a single top-level `used_gpus`.

`strategy` (`tp` / `tep` / `dep` / `dtp`, with a `mixed` fallback the enumerator never emits) and
`used_gpus` (`gpus_per_worker × replicas`, summed across roles for disagg) are **derived
from the shape**, not settable. See [Search Space](search-space.md).

### Engine knobs — the active branch only

The active mode's batching knobs come from the `selection` (`_*_SEARCHED`); the per-role
scalars are folded off the `SearchSpace` (`_*_PINNED`).

| branch | searched (from selection) | pinned (from search_space) |
|---|---|---|
| agg | `agg_max_num_batched_tokens`, `agg_max_num_seqs` | `agg_block_size`, `agg_gpu_memory_utilization`, `agg_enable_prefix_caching` |
| disagg | `prefill_max_num_batched_tokens`, `prefill_max_num_seqs`, `decode_max_num_batched_tokens`, `decode_max_num_seqs` | `prefill_block_size`, `prefill_gpu_memory_utilization`, `prefill_enable_prefix_caching`, `decode_block_size`, `decode_gpu_memory_utilization`, `decode_enable_prefix_caching` |

The inactive branch's engine keys are not emitted.

### Router knobs

`router_mode` is always present. The rest apply **only under `kv_router`**:

- `_KV_ROUTER_KNOBS` (from `selection`): `overlap_score_credit`, `prefill_load_scale`,
  `router_temperature`, and — only when present in the selection —
  `host_cache_hit_weight`, `disk_cache_hit_weight`. The two cache-hit weights are *gated
  out of the search* unless multi-tier offload is on (`num_g2_blocks > 0`; see
  `branch_knob_choices`), so the `if key in selection` guard simply skips them when absent.
- `_ROUTER_ADMISSION` (pinned, from `search_space`): `active_decode_blocks_threshold`,
  `active_prefill_tokens_threshold`, `active_prefill_tokens_threshold_frac`,
  `no_admission_control`. For searches including `kv_router`, these fields are currently
  reserved and rejected when set because Dynamo replay cannot model them yet.

Under `round_robin` none of these are emitted.

### Planner fields (`_unroll_planner`)

`planner_scaling_policy` decodes via `scaling_fields`
(`components/src/dynamo/profiler/spica/planner.py`). (For a
non-goodput goal, `filter_scaling_policies` has already dropped any throughput-scaling
policy candidates before the sampler runs, so a selection can only carry a policy the goal
actually supports.) Always emitted:

- `planner_scaling_policy` — the **raw selection entry** (preset id *or* dict), kept for
  traceability
- `enable_throughput_scaling`, `enable_load_scaling`

**If both flags are false (`disabled`), unrolling stops here** — no intervals, fpm,
sensitivity, or predictor fields. Otherwise it also emits:

- `throughput_adjustment_interval_seconds`, `load_adjustment_interval_seconds`
- fpm fields from `planner_fpm_sampling` (`fpm_fields`): `max_num_fpm_samples`,
  `fpm_sample_bucket_size`
- sensitivity fields from `planner_load_sensitivity` (`load_sensitivity_fields`):
  `load_scaling_down_sensitivity`, `load_min_observations`

**Load-predictor fields are emitted only under throughput scaling** (`enable_throughput`
and a non-`None` `load_predictor`). The winner for *this policy's*
`throughput_adjustment_interval_seconds` is looked up in
`load_predictor.best_by_interval` and expanded by `predictor_fields`
(`components/src/dynamo/profiler/spica/load_predictor_sweep.py`): always `load_predictor` (family) +
`load_predictor_log1p`; plus **only the family's own knobs** — `prophet_window_size` for
prophet, `kalman_q_level` / `kalman_q_trend` / `kalman_r` / `kalman_min_points` for kalman,
nothing extra for constant/arima. If the interval has no winner, no predictor fields are
added.

### Candidate-relative KV load

When `workload.kv_load_ratio` is set, a scalar ratio is pinned or a Pareto `[min, max]`
range becomes a continuous per-trial selection. After `unroll_sample`, the search evaluates
the selected backend, shape, and batching with AI Configurator and appends these derived fields:

- `kv_load_ratio` — the requested normalized load;
- `concurrency` — the concrete closed-loop in-flight cap;
- `kv_load_concurrency_capacity` — estimated concurrency at ratio `1`;
- `kv_load_capacity_tokens` — aggregate decode (disagg) or agg KV tokens;
- `prefill_kv_capacity_tokens` + `decode_kv_capacity_tokens` for disagg, or
  `agg_kv_capacity_tokens` for agg.

These fields are added by the search orchestration rather than `unroll_sample`, because
capacity depends on the fully selected candidate's backend, batching, shape, and replicas.
A fixed `workload.concurrency` is also copied into each candidate as `concurrency`.

## Overriding

A user controls what gets unrolled through **three mechanisms**, all in the
`search_space:` YAML. The first two are detailed in [Search Space](search-space.md);
this section is about their effect on the *unrolled* sample.

### 1. Pin an atomic knob — single-element list

Any searched list knob with one entry pins it: that one value lands in every sample. E.g.
`router_mode: [round_robin]` forces `router_mode: round_robin` (and drops all kv-router /
admission keys); `agg_max_num_seqs: [512]` pins `agg_max_num_seqs: 512`. Multiple entries
leave it for the sampler.

### 2. Pin a composite's unrolled fields — a dict entry (escape hatch)

The four composite knobs (`planner_scaling_policy`, `planner_fpm_sampling`,
`planner_load_sensitivity`, `load_predictor_candidates`) accept a **dict** in place of a
preset id. The dict **pins the unrolled fields directly**, *replacing* the preset
expansion — there is no merge. `scaling_fields` / `fpm_fields` / `load_sensitivity_fields`
/ `predictor_fields` accept a dict identically to a preset, so the unrolled sample looks
the same as if a preset had produced it.

The dict's keys are drawn from that composite's allowed field names (`COMPOSITE_DICT_KEYS`;
unknown keys are rejected), and it must be **self-contained**: the three planner composites
require *all* their fields (allowed == required), while `load_predictor_candidates` needs at
least `load_predictor` and allows a wider set (`COMPOSITE_REQUIRED_KEYS` ⊊ `COMPOSITE_DICT_KEYS`)
— remaining family params default per family. Use this to pin a value no preset offers
(e.g. a 240 s throughput interval). Value legality is validated downstream by Dynamo's
`PlannerConfig`.

### 3. Pin the parallel shape — a `parallel_configs` dict entry

`parallel_configs` (empty by default — the shape is then enumerated by KV-feasibility)
takes a list of dicts. One entry **pins** the parallel shape that unrolls into the
`tp`/`pp`/`attention_dp`/`moe_tp`/`moe_ep`/`strategy`/`replicas`/`used_gpus` fields (or the
`prefill_*`/`decode_*` pair for disagg). **Pinning `parallel_configs` requires
`deployment_mode` to list exactly one mode.** Omitted shape dims default to `1`; `strategy`
and `used_gpus` remain derived. See [Search Space](search-space.md) for the full shape
rules and legality checks.

## Never searched

These reach the sample only as **pinned scalars** (always one value, mechanism #1 does not
apply): `model_name`, `hardware_sku`, `aic_nextn`, the kv-manager offload fields, the
per-role `*_block_size` / `*_gpu_memory_utilization` / `*_enable_prefix_caching`, and the
admission-control fields. The parallel shape + replica count are *derived* (enumerated)
unless pinned via mechanism #3. See [Kinds of Knob](search-space.md#kinds-of-knob).
