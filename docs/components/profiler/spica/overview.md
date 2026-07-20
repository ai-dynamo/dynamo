---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Spica Overview
subtitle: How Spica turns a deployment-tuning problem into a replay-backed search
---

> [!WARNING]
> **Experimental.** Spica is intended for evaluation and feedback, not production capacity
> planning. Its API, configuration schema, search results, and deployment output may change
> without a standard deprecation period. Spica provides no SLA, accuracy, or configuration
> optimality guarantees.

Spica turns a deployment-tuning question into a search. You give it four things in one
YAML (`SmartSearchConfig`):

| Block | Model | What it is |
|---|---|---|
| `search_space:` | `SearchSpace` | knobs to **explore** + pinned context (model, hardware, GPU budget) |
| `workload:` | `Workload` | the **traffic** every candidate is replayed against (mostly pinned; Pareto may search KV load) |
| `goal:` | `OptimizationGoal` | what **"better"** means (the target metric) + the SLA constraint |
| `sweep:` | `SweepConfig` | run-control (`max_rounds`, `candidates_per_round`, `parallel_evals`, `random_seed`) |

Spica returns the **best deployment config(s)** — a parallel shape + replica count +
backend + engine/router/planner knobs — each scored by a **Dynamo Replay** (the
mocker bridge), not an analytical estimate. `run_smart_search`
(`components/src/dynamo/profiler/spica/search.py`)
returns a `list[Candidate]`: best-first for a scalar goal, or the non-dominated set for a
`pareto` goal.

## End-to-end flow

The steps below follow `run_smart_search`
(`components/src/dynamo/profiler/spica/search.py`). Steps **3** (load-predictor
sub-sweep) and **4** (branch enumeration) are independent pre-loop stages — the code happens to
enumerate branches first, but neither depends on the other; we describe the load-predictor sweep
first to keep the planner-scaling story together.

### 1. Parse & validate the config

`SmartSearchConfig.from_yaml` (`components/src/dynamo/profiler/spica/config.py`) loads the YAML and runs every
pydantic validator: the per-knob choice/dict-key checks (`SearchSpace._validate_search_choices`),
GPU-budget bounds, single-mode-when-pinning-`parallel_configs`, the goal's SLA requirement
(`goodput`/`goodput_per_gpu` need a `ttft_ms`+`itl_ms` or `e2e_ms` SLA), and the rule that
**`workload.concurrency` is always scalar while a ranged `workload.kv_load_ratio` is Pareto-only**.
An invalid config never reaches the search.

### 2. Filter throughput-scaling policies (`filter_scaling_policies`)

Predictive **throughput scaling** needs an SLA, so it only works when the goal maps to the
planner's `"sla"` target. `goal.target.planner_optimization_target`
(`OptimizationTarget.planner_optimization_target`) maps `goodput`/`goodput_per_gpu` → `"sla"`;
everything else (`throughput`*, `e2e_latency`, `pareto`) → `"throughput"`/`"latency"`.

`filter_scaling_policies` (`components/src/dynamo/profiler/spica/planner.py`) is called with
`allow_throughput=(planner_optimization_target == "sla")`. When `False`, every
`planner_scaling_policy` entry whose `enable_throughput_scaling` is true (the `throughput_*` /
`hybrid_*` presets, or any dict that sets it) is **dropped up front** — before either the
sampler or the load-predictor sub-sweep sees it. `disabled` / `load_*` survive. The dropped set
is logged. **Error-if-nothing-left:** if dropping leaves `kept` empty (every policy enabled
throughput scaling under a non-`sla` goal), the run raises `ValueError`. The kept list is
written back via `config.model_copy`.

For goodput with an **e2e-only SLA**, Spica applies a second filter: planner scaling is dropped
entirely and only static policies such as `disabled` survive. Replay can score e2e goodput, but
the planner's SLA scaling target needs `ttft_ms` + `itl_ms`; filtering here prevents Vizier from
sampling candidates that would fail during deployment-plan construction.

### 3. Load-predictor sub-sweep (`sweep_load_predictor`) — separate from Vizier

`sweep_load_predictor` (`components/src/dynamo/profiler/spica/load_predictor_sweep.py`) picks the forecaster for
predictive throughput scaling. This is a **standalone brute-force grid**, not part of the
main Vizier loop, and its single winner is injected into every unrolled sample.

- For each **distinct** `throughput_adjustment_interval_seconds` among the (kept) scaling
  policies (`throughput_intervals`), it aggregates the trace into per-interval windows
  (`build_windows`, via the planner's own mooncake tool) and scores **every**
  `load_predictor_candidate` by **mean one-step-ahead forecast loss** (`evaluate_preset` /
  `window_loss` — a weighted log-scale error over num_req·isl, num_req·osl, isl, osl). The
  lowest-loss entry is pinned per interval into `LoadPredictorResult.best_by_interval`. It
  reuses the Dynamo predictor classes, so the chosen preset is what the Planner will run.
- **Shortcuts:** no throughput-enabled policy → no intervals → skip entirely
  (`reason="no_throughput_scaling_candidate"`). Static (non-trace) workload → `constant_last`
  for every interval (`reason="static_workload_constant"`) — there is no series to learn. A
  short/empty trace where every preset ties at `inf` loss falls back to `constant_last` for
  that interval.

### 4. Enumerate branches (`enumerate_branches`)

`enumerate_branches` (`components/src/dynamo/profiler/spica/search_space.py`) builds **one `BranchSpace` per
`deployment_mode`** (agg / disagg) — one Vizier study each, because agg and disagg have
structurally different parallel configs. **`backend` is a searched knob, not a branch**: for
each mode the valid projection pool is the **union** of every configured backend's
KV-feasible per-worker shapes × replica counts (`parallel_configs_for`), tagged with which
backends support each. `context_length` is threaded into KV feasibility.

- A backend with no perf DB / no viable config for a mode is dropped from that mode's backend
  knob. A **mode** for which *no* backend is viable is **skipped with a warning** (a viable
  mode still runs); only if *no* mode is viable does it raise `NoViableParallelConfig`.
- A *pinned* `parallel_configs` that is legal for no backend is a **hard error** (fail fast).
- A ranged `workload.kv_load_ratio` (Pareto) becomes a continuous per-trial dimension on the
  branch. A scalar ratio is injected as a constant.

The default Planner/Profiler image currently uses AI Configurator 0.9, which does not expose
`aiconfigurator.sdk.memory`. Spica warns and skips the pre-search KV-capacity shape filter in that
environment. Trace and fixed-concurrency workloads can continue through Replay, but
`kv_load_ratio` fails closed because it cannot derive candidate-specific concurrency without the
memory estimator.

### 5. Per-branch Vizier study loop

For each branch, a `BranchSampler` (`make_branch_sampler`, study id
`spica_{mode}_{run_nonce}`) runs `sweep.max_rounds` rounds. Each round is a **barrier**:

1. **ask + project** — `sampler.suggest(per_round)` returns suggestions on the main process.
   Vizier sees resource ratios, per-engine GPU targets, and attention/FFN modes; each request
   is deterministically projected onto the nearest backend-compatible config in the valid
   pool. A single user-supplied `parallel_configs` entry remains a strict pin and bypasses
   these dimensions.
2. **deduplicate** — exact duplicate full samples reuse their cached measurement and are
   immediately told back to Vizier. They do not run replay or consume the unique replay
   budget; replacement suggestions are requested up to the per-round 11x safety cap.
3. **evaluate** — the rest fan out across worker processes: a single **spawned**
   `ProcessPoolExecutor` created once for the whole run (amortizing the per-worker Dynamo
   import) with `min(parallel_evals, per_round)` workers. The pool is used only when **both**
   `parallel_evals > 1` and `per_round > 1`; otherwise evaluation runs sequentially in-process.
   Each worker runs the pure pipeline `_evaluate_one`: `unroll_sample` → resolve candidate KV
   capacity/derived concurrency (KV-load mode) → `build_deployment` →
   `ReplayEvaluator.evaluate` (**real replay**) → score (`make_candidate`). Workers never touch
   the Vizier study; a dead pool re-raises a friendly error pointing at the `if __name__ ==
   "__main__":` guard that spawned workers require.
4. **tell** — back on the main process: a **feasible** trial is `observe`'d with its metrics;
   replay/build failures are `observe_infeasible`'d. Backend and GPU-budget gates remain as
   defensive checks, but structured projection should make them unreachable.

`is_feasible` gates on `used_gpus <= gpu_budget` only; SLA is **not** re-gated here (goodput
targets already bake the SLA into the metric). Each trial's outcome is tallied as one of
`feasible` / `infeasible` / `failed` / `unsupported`; cache hits are tallied separately.

### 6. Merge by the goal

After all branches finish, candidates from every branch are merged
(`components/src/dynamo/profiler/spica/score.py`):

- **single-objective** → `rank()` returns them best-first by signed score, ties broken toward
  fewer GPUs (across branches — backends and modes compete on one list).
- **pareto** → `pareto_front()` returns the non-dominated set over the resolved objectives
  (default `throughput_per_gpu` × `throughput_per_user`), sorted along the last objective (the
  x-axis) so the list traces the frontier left-to-right. (It only considers candidates that
  carry an `objectives` vector; each candidate's `score` still holds the *first* objective's
  value as a headline number, but that is not used for the pareto merge.)

## Flow diagram

Steps 3 (load-predictor sub-sweep) and 4 (branch enumeration) both branch off step 2 — they
are independent and both feed the per-branch loop; qualifiers (error-if-nothing-left,
skip / `constant_last` shortcuts) are described in the prose above.

```mermaid
flowchart TD
  Y(["SmartSearchConfig (YAML)"]):::io --> V["1 · parse + validate"]
  V --> F["2 · filter scaling policies<br/>drop throughput-scaling unless goal → sla"]
  F --> P["3 · load-predictor sub-sweep<br/>per throughput interval · forecast-loss winner"]
  F --> E["4 · enumerate branches<br/>one per deployment_mode · backend = knob"]
  P --> A
  E --> A
  subgraph L["5 · per branch — Vizier study (× max_rounds)"]
    direction TB
    A["ask + project · suggest(per_round)<br/>structured features → valid config"] --> D["deduplicate<br/>reuse cached measurement"]
    D --> EV["eval ⇉ parallel workers<br/>unroll → build_deployment → replay → score"]
    EV --> T["tell · feasible: observe / gated: observe_infeasible"]
  end
  T --> C(["candidates from all branches"]):::io
  C --> M{"merge by goal"}
  M -->|single-objective| R["rank()<br/>best-first, ties → fewer GPUs"]
  M -->|pareto| PF["pareto_front()<br/>non-dominated set"]
  R --> OUT(["list[Candidate]"]):::io
  PF --> OUT
  classDef io fill:#dbeafe,stroke:#60a5fa,color:#1e3a8a;
  classDef decision fill:#fef3c7,stroke:#fbbf24,color:#78350f;
  class M decision;
```

## See also

- [Optimization Goals](optimization-goal.md) describes the `OptimizationGoal` targets, SLA, scoring,
  and how the goal maps to the planner's `optimization_target`.
- [Traffic](traffic.md) describes the `Workload` load shapes (trace, request rate, fixed
  concurrency / KV load) and candidate-relative Pareto load sweep.
- [Search Space](search-space.md) lists every pinnable or searchable knob, the composite presets,
  and `parallel_configs`.
- [Unrolled Samples](sample.md) explains how a Vizier suggestion becomes a concrete deployment.
