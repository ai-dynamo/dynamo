# Profiler V2: Smart Sweeping Tool for Dynamo Deployments

## Motivation

Dynamo deployments are hard to configure. A good setup depends on multiple interacting components, including engines, routers, planners, KV policy, replica counts, and parallelization choices. The right choice can also change under dynamic traffic, where static rules of thumb are often not enough.

Replay gives us a fast way to try candidate configurations before deployment, but today the process still needs a human expert to decide what to try next. Profiler V2 should automate that loop: generate legal candidates, evaluate them with replay, and use Vizier to guide the next sweep samples.

This sweeper is not an engine/kernel autotuner. Its focus is high-level Dynamo configuration: scheduling, routing, autoscaling, deployment shape, replica counts, and runtime capacity knobs that affect serving behavior.

## Scope

This plan defines the DGDR profiler V2 smart-search surface. Profiler V2 uses Vizier as the smart sweeping optimizer. This document inventories deployment, engine, KV manager, router, and planner knobs, then classifies each knob by optimizer treatment. It does not finalize the DGDR API.

## V2 Refactor Direction

V2 should land as an opt-in implementation under `components/src/dynamo/profiler/v2/`. It should own search-space construction, candidate decoding, replay evaluation, ranking, and report generation for opt-in runs.

V2 should not change the existing V1 profiler defaults, DGDR CR behavior, or k8s controller/operator path. In the opt-in path, V2 replaces the current AIC sweeper/picker role, while AIC remains a lower-level provider for backend support checks, legal parallelism hints, perf model data, and memory/capacity estimates.

## Overview

Profiler V2 uses two sweep paths. The main Vizier sweep is manually branched by deployment mode, so `agg` and `disagg` each expose a flat search space. Vizier samples are guided by replay's virtual-clock simulation, and replay evaluations can run in parallel on CPU. Planner load prediction can be tuned separately with an independent simple grid sweep when throughput planner is enabled.

```mermaid
flowchart TD
    input["DGDR input + replay traces"] --> profiler["Profiler V2"]
    profiler --> throughputPlanner{"Throughput planner enabled?"}
    throughputPlanner -->|yes| loadPredictor["Independent planner load predictor sweep"]
    loadPredictor --> predictorPreset["Selected predictor preset"]
    predictorPreset --> branch["Manual branch on deployment_mode"]
    throughputPlanner -->|no| branch
    branch --> agg["agg branch"]
    branch --> disagg["disagg branch"]
    agg --> aggVizier["Vizier main sweep: flat agg space"]
    disagg --> disaggVizier["Vizier main sweep: flat disagg space"]
    aggVizier --> rank["Replay evaluation + direct agg/disagg comparison"]
    disaggVizier --> rank
```

| Branch | Engine | Router | Planner |
| --- | --- | --- | --- |
| `agg` | aggregated workers: `tp/pp/dp/moe_tp/moe_ep`, `workers`, agg engine args | agg-compatible router modes and queue policy | `mode=agg`, agg engine GPU shape |
| `disagg` | prefill/decode workers: separate prefill shape, decode shape, prefill worker count, decode worker count | disagg router mode, prefill/decode scoring, admission | `mode=disagg`, prefill/decode engine GPU shape |

Treatment labels:

- `Branch`: split outside a single Vizier flat study.
- `Search`: Vizier candidate dimension. All `Search` knobs can also be pinned by user override.
- `Composite Search`: one high-level candidate dimension decoded into multiple runtime fields.
- `Derived`: generated from branch, backend, shape, AIC output, or candidate generation.
- `Pinned`: fixed by user input, environment, model, or backend policy.

## Optimization Goal

Profiler V2 does not return a single tuned configuration. Its goal is to produce a ranked set of viable deployment candidates for the requested `model_name`, `hardware_sku`, and workload, so the user can compare the top recommendation against alternatives, binding constraints, and skipped-candidate reasons.

The optimization goal is user-owned and always `Pinned`; it is never a search dimension:

- `optimization_target` defines what "better" means: `throughput`, `e2e_latency`, `goodput`, or `goodput_per_gpu`.
- SLA targets (`ttft_ms`+`itl_ms`, or `e2e_ms`) are user constraints. `goodput` and `goodput_per_gpu` require an SLA target; `throughput` and `e2e_latency` may run without one.
- The GPU budget (`gpu_budget`) is the deployment `budget` knob, not part of this structure; it is applied as a feasibility constraint in the gate below.

The goal is captured by a single structure, extracted from the DGDR spec (objective + SLA) and never mutated by the search. Field names follow the DGDR-aligned lowerCamelCase convention used by `replay_optimize.specs`:

```python
class OptimizationTarget(str, Enum):
    THROUGHPUT = "throughput"            # maximize replay throughput
    E2E_LATENCY = "e2e_latency"          # minimize mean end-to-end latency
    GOODPUT = "goodput"                  # maximize SLA-satisfying throughput
    GOODPUT_PER_GPU = "goodput_per_gpu"  # maximize goodput / used_gpus


class SLATarget(BaseModel):
    """Per-request latency bounds in ms. Set ttftMs+itlMs, or e2eMs."""
    model_config = ConfigDict(extra="forbid")
    ttftMs: float | None = None
    itlMs: float | None = None
    e2eMs: float | None = None


class OptimizationGoal(BaseModel):
    """User-owned objective and SLA. Pinned; never searched."""
    model_config = ConfigDict(extra="forbid")
    target: OptimizationTarget = OptimizationTarget.THROUGHPUT
    sla: SLATarget | None = None  # required for goodput / goodput_per_gpu

    @model_validator(mode="after")
    def _require_sla_for_goodput(self) -> "OptimizationGoal":
        needs_sla = self.target in (
            OptimizationTarget.GOODPUT,
            OptimizationTarget.GOODPUT_PER_GPU,
        )
        has_sla = self.sla is not None and (
            self.sla.e2eMs is not None
            or (self.sla.ttftMs is not None and self.sla.itlMs is not None)
        )
        if needs_sla and not has_sla:
            raise ValueError(
                f"{self.target.value} requires an SLA target (ttftMs+itlMs or e2eMs)"
            )
        return self
```

Candidates are ranked in two stages:

1. Feasibility gate — a candidate must pass candidate generation and preflight (legal parallel shape, memory fit, backend/SKU support, `gpu_budget`, and any user-pinned knobs) and, when an SLA is set, satisfy it under replay. Infeasible candidates are dropped with a structured reason instead of being scored.
2. Objective ranking — feasible candidates are ordered by the `optimization_target` score computed from their replay report.

If no candidate clears the feasibility gate, V2 reports `NoViableCandidate` rather than emitting an unproven configuration. The concrete per-candidate score that the Vizier main sweep optimizes for each objective is defined in [Main Sweep Optimization Goal](#main-sweep-optimization-goal).

## Deployment Knobs

Deployment owns model identity, hardware target, topology, backend type, and budget envelope: model, GPU SKU, branch, replica counts, GPU budget, and per-engine GPU accounting. Engine and KV manager consume these derived values.

| Category | Knob | Applies | Proposed treatment | Notes |
| --- | --- | --- | --- | --- |
| model | `model_name` | all | `Pinned` | Model identity from the DGDR `model` field (HF id or private model name). Selects the AIC perf-model table and memory-fit estimates; never a search dimension. One model is shared across all components in a candidate. Maps to `EngineSpec.model`. |
| hardware | `hardware_sku` | all | `Pinned` | GPU SKU from the DGDR `hardware.gpuSku` field (e.g. `h200_sxm`, `h100_sxm`). Selects the AIC hardware system and bounds legal parallelism and `gpu_budget`; never a search dimension. Maps to `HardwareSpec.gpuSku`. |
| deployment mode | `deployment_mode`: `agg`, `disagg` | all | `Pinned`+`Branch` | Pin when user specifies one mode; otherwise split studies outside Vizier and rank branches globally. |
| backend type | `backend` + `engine_type`: `vllm`, `sglang`, `trtllm` | all | `Search` | Search only across candidate backends with comparable replay+perf model support; can be pinned by user override. Disagg first pass uses one shared backend type for prefill+decode unless mixed-backend deployment is explicitly enabled. |
| objective | `optimization_target`: `throughput`, `e2e_latency`, `goodput`, `goodput_per_gpu`; optional SLA targets: `ttft_ms`+`itl_ms` or `e2e_ms` | profiler+planner | `Pinned` | User-selected replay objective. `goodput` and `goodput_per_gpu` require an SLA target. |
| component enablement | `enable_kvrouter` | deployment | `Pinned` | Controls whether V2 emits/evaluates KV-router config; not a search knob. |
| component enablement | `enable_planner` | deployment | `Pinned` | Controls whether V2 emits planner config; not a search knob. |
| replica+parallel config | `workers`, `prefill_workers`, `decode_workers`, parallel strategy: `tp`, `tep`, `dep` (MLA) | replay+profiler+AIC | `Composite Search` | Search legal per-component parallel configs using the current profiler's TP+TEP+DEP (MLA) sweep, then compute replica counts from `gpu_budget` and combine them into one categorical candidate. PP is pinned outside this search. |
| budget | `gpu_budget`, `max_gpu_budget`, `min_gpu_budget`, `min_endpoint` | hardware+planner | `Pinned` | `gpu_budget` is the single max-GPU budget concept and maps to planner `max_gpu_budget`; min constraints are pinned deployment inputs used by candidate generation. |

## Engine Knobs

### Full Engine Inventory

| Category | Knob | Applies | Proposed treatment | Notes |
| --- | --- | --- | --- | --- |
| worker role | `worker_type`: `aggregated`, `prefill`, `decode` | replay engine | `Derived` | Derived from branch and component. |
| parallel shape | `tp`, `pp`, `dp`, `moe_tp`, `moe_ep` | planner+AIC | `Composite Search` | Search together with deployment `replica+parallel config` as legal deployment-shape composites capped by `gpu_budget`; first pass only emits TP+TEP+DEP (MLA) candidates, with PP pinned. |
| KV capacity | `num_gpu_blocks`, `total_kv_blocks`, `max_kv_tokens` | replay+runtime | `Derived` | Derived from AIC+memory-fit and block layout; not a search knob in the first design pass. |
| block layout | `block_size`, `kv_cache_block_size` | replay+runtime | `Pinned` | Backend+runtime policy. vLLM default differs from SGLang page size. |
| context | `context_length` | runtime metadata | `Pinned` | Model+runtime constraint, not a search knob. |
| memory budget policy | `gpu_memory_utilization`, `mem_fraction_static`, `free_gpu_memory_fraction` | replay+AIC+backend | `Pinned` | Different backend/config-surface names for the same KV memory budget fraction policy; pin one normalized value. |
| batching | `max_num_seqs` | replay+runtime | `Search` | Important scheduler capacity knob; controls concurrent sequence admission. |
| batching | `max_num_batched_tokens` | replay+runtime | `Search` | Important scheduler capacity knob; controls token batching capacity. |
| cache | `enable_prefix_caching` | replay+runtime | `Pinned` | Pin from backend+runtime policy; decoder may still force worker-role-specific values such as false for decode workers. |
| startup | `startup_time` | replay+planner | `Pinned` | Deployment+environment input. |
| speculative decode | `aic_nextn`, `aic_nextn_accept_rates` | `origin/main` replay+AIC | `Pinned` | `aic_nextn` validates to 1..5 when set. Replay MTP token progression support should be clarified before searching this. |

### Engine Search Knobs By Component

Parallelization mapping is searched as a composite together with deployment `replica+parallel config`; backend type is also owned by deployment. The component-level engine search below only covers runtime batching knobs.

| Component | Search knob | Treatment | Notes |
| --- | --- | --- | --- |
| prefill | `max_num_batched_tokens` | `Search` | Primary prefill batching capacity knob. |
| prefill | `max_num_seqs` | `Search` | Optional prefill admission and concurrency knob when the backend uses it. |
| decode | `max_num_seqs` | `Search` | Primary decode concurrency knob. |
| decode | `max_num_batched_tokens` | `Search` | Optional decode token batching cap when the backend exposes it. |
| agg | `max_num_seqs` | `Search` | Aggregated concurrency and admission knob. |
| agg | `max_num_batched_tokens` | `Search` | Aggregated token batching capacity knob. |

## KV Manager Knobs

KV manager owns multi-tier KV storage/offload policy. Engine search should only consume the derived capacity and transfer assumptions it needs for replay; G2/G3/G4 should not be modeled as engine knobs.

| Category | Knob | Applies | Proposed treatment | Notes |
| --- | --- | --- | --- | --- |
| KV accounting | `kv_bytes_per_token` | replay/KV manager | `Derived` | Derived from model/cache dtype and block layout; required for capacity and transfer estimates. |
| KV transfer | `kv_transfer_bandwidth` | replay/mocker | `Derived` | PD disagg KV transfer bandwidth for prefill-to-decode KV handoff latency; separate from KVBM G1/G2/G3/G4 offload tier bandwidths. |
| G2 capacity | `num_g2_blocks` | replay/offload | `Pinned` | Entry point for offload policy. G3/G4 depend on G2. |
| G2 transfer | `bandwidth_g1_to_g2_gbps`, `bandwidth_g2_to_g1_gbps` | replay/offload | `Pinned` | Tier-specific transfer model. |
| G2 batching | `offload_batch_size` | replay/offload | `Pinned` | KV policy input, not an optimizer dimension in the first design pass. |
| G3 capacity | `num_g3_blocks` | `origin/main` replay/offload | `Pinned` | Requires `num_g2_blocks`. |
| G3 transfer | `bandwidth_g2_to_g3_gbps`, `bandwidth_g3_to_g2_gbps` | `origin/main` replay/offload | `Pinned` | Applies only when G3 policy is enabled. |
| G4 enablement | `enable_g4_storage` | `origin/main` replay/offload | `Pinned` | Requires `num_g2_blocks`. |
| G4 transfer | `bandwidth_g2_to_g4_gbps`, `bandwidth_g4_to_g2_gbps` | `origin/main` replay/offload | `Pinned` | Applies only when G4 policy is enabled. |

## Router Knobs

### Full Router Inventory

| Category | Knob | Proposed treatment | Notes |
| --- | --- | --- | --- |
| router mode | `router_mode`: `round-robin`, `random`, `power-of-two`, `kv`, `direct`, `least-loaded`, `device-aware-weighted` | `Search` | Replay optimizer currently uses `kv_router` and `round_robin` names; decoder should normalize to runtime names. |
| branch strictness | `enforce_disagg` | `Derived` | Should follow disagg branch unless user wants fallback behavior. |
| admission | `active_decode_blocks_threshold`, `active_prefill_tokens_threshold`, `active_prefill_tokens_threshold_frac`, `no_admission_control` | `Pinned` | Admission policy input, not an optimizer dimension in the first design pass. |
| KV score | `overlap_score_credit` | `Search` | Current replay optimizer already searches it. |
| KV score | `prefill_load_scale` | `Search` | Current replay optimizer already searches it. |
| KV score | `host_cache_hit_weight`, `disk_cache_hit_weight` | `Search` | Rust binding exposes these; Python CLI group does not currently list them. |
| stochasticity | `router_temperature` | `Search` | Only relevant for KV scoring. |

## Planner Knobs

### Full Planner Inventory

| Category | Knob | Proposed treatment | Notes |
| --- | --- | --- | --- |
| mode/env | `mode`: `agg`, `disagg` | `Derived` | Comes from branch. |
| mode/env | `environment`, `namespace` | `Pinned` | Deployment context. Backend type is owned by deployment search. |
| per-engine GPU count | `decode_engine_num_gpu`, `prefill_engine_num_gpu` | `Derived` | Derived directly from the chosen parallel shape. |
| scaling policy | `enable_throughput_scaling`, `enable_load_scaling`, `throughput_adjustment_interval_seconds`, `load_adjustment_interval_seconds` | `Composite Search` | Search as one legal planner scaling policy tuple. If both scaling modes are enabled, load cadence must be shorter than throughput cadence. |
| FPM sampling | `max_num_fpm_samples`, `fpm_sample_bucket_size` | `Composite Search` | Search as paired presets. Bucket size must be a perfect square. |
| load scaling sensitivity | `load_scaling_down_sensitivity`, `load_min_observations` | `Search` | Applies to load scaling policy. |

## Main Sweep Optimization Goal

The main sweep objective is user configurable. Every Vizier sample is decoded into a concrete candidate and evaluated by replay virtual-clock simulation; the replay report provides the metric used as the candidate score.

| Objective | Direction | Requires SLA | Score definition |
| --- | --- | --- | --- |
| `throughput` | maximize | no | Replay throughput for the configured workload. |
| `e2e_latency` | minimize | no | Replay end-to-end request latency for the configured workload. |
| `goodput` | maximize | yes | Throughput from requests that satisfy the configured SLA. SLA can be `ttft_ms`+`itl_ms` or `e2e_ms`. |
| `goodput_per_gpu` | maximize | yes | `goodput / used_gpus`, where `used_gpus` is derived from `replica_parallel_config`. |

## Main Sweep Search Space

Engine batching values in this table are per attention-DP rank. Candidate decoding should derive global component capacity from `attention_dp_size` in the selected `replica_parallel_config`.

| Group | Search dimension | Treatment | Knobs controlled | Candidate values | Notes |
| --- | --- | --- | --- | --- | --- |
| Deployment | `backend_type` | `Search` | `backend` + `engine_type` | `vllm`, `sglang`, `trtllm` | All branches; can be pinned by user override. |
| Deployment | `replica_parallel_config` | `Composite Search` | branch-specific workers + parallel configs | Generated legal TP+TEP+DEP (MLA) configs; replica counts computed from `gpu_budget` | Branch-aware categorical candidate. Agg decodes to `workers` + agg parallel config; disagg decodes to `prefill_workers`, `decode_workers` + prefill+decode parallel configs. |
| Prefill Engine | `prefill_max_num_batched_tokens` | `Search` | `max_num_batched_tokens` | `8k`, `16k`, `32k` | Disagg branch only. |
| Prefill Engine | `prefill_max_num_seqs` | `Search` | `max_num_seqs` | `1`, `2`, `4`, `8` | Disagg branch only. |
| Decode Engine | `decode_max_num_batched_tokens` | `Search` | `max_num_batched_tokens` | `8k` | Disagg branch only; fixed search dimension for explicit config emission. |
| Decode Engine | `decode_max_num_seqs` | `Search` | `max_num_seqs` | `256`, `512`, `1024` | Disagg branch only. |
| Agg Engine | `agg_max_num_batched_tokens` | `Search` | `max_num_batched_tokens` | `8k`, `16k`, `32k` | Agg branch only. |
| Agg Engine | `agg_max_num_seqs` | `Search` | `max_num_seqs` | `256`, `512`, `1024` | Agg branch only. |
| Router | `router_mode` | `Search` | `router_mode` | `round-robin`, `kv` | Decoder normalizes runtime names to replay names such as `round_robin` and `kv_router`. |
| Router | `router_overlap_score_credit` | `Search` | `overlap_score_credit` | `0.0`, `0.5`, `1.0` | Active for KV router mode; ignored by round-robin. |
| Router | `router_prefill_load_scale` | `Search` | `prefill_load_scale` | `0.0`, `0.25`, `0.5`, `1.0`, `2.0`, `4.0` | Active for KV router mode; ignored by round-robin. |
| Router | `router_host_cache_hit_weight` | `Search` | `host_cache_hit_weight` | `0.5`, `0.75`, `1.0` | Active for KV router mode; default is `0.75`. |
| Router | `router_disk_cache_hit_weight` | `Search` | `disk_cache_hit_weight` | `0.0`, `0.25`, `0.5` | Active for KV router mode; default is `0.25`. |
| Router | `router_temperature` | `Search` | `router_temperature` | `0.0`, `0.2`, `0.5`, `1.0` | `0.0` keeps deterministic selection; higher values increase sampling randomness. |
| Planner | `planner_scaling_policy` | `Composite Search` | `enable_throughput_scaling`, `enable_load_scaling`, `throughput_adjustment_interval_seconds`, `load_adjustment_interval_seconds` | `throughput_180_5`: `{true, false, 180, 5}`; `throughput_600_5`: `{true, false, 600, 5}`; `load_180_5`: `{false, true, 180, 5}`; `load_180_10`: `{false, true, 180, 10}`; `hybrid_180_5`: `{true, true, 180, 5}`; `hybrid_600_5`: `{true, true, 600, 5}` | Composite categorical dimension. Candidate generator may filter policies by `optimization_target`. |
| Planner | `planner_fpm_sampling` | `Composite Search` | `max_num_fpm_samples`, `fpm_sample_bucket_size` | `small`: `{32, 4}`; `default`: `{64, 16}`; `large`: `{128, 16}`; `fine`: `{128, 64}` | Paired presets keep bucket size compatible with sample count. `fpm_sample_bucket_size` must be a perfect square. |
| Planner | `planner_load_sensitivity` | `Search` | `load_scaling_down_sensitivity`, `load_min_observations` | `aggressive`: `{70, 3}`; `default`: `{80, 5}`; `conservative`: `{90, 8}` | Controls scale-down conservativeness and regression cold-start threshold. |

## Planner Load Predictor Independent Grid Sweep

Planner load predictor tuning should be a separate deterministic grid search, not part of the main Vizier candidate space. It is easy to validate directly against replay/planner traces, so V2 should run a small predefined set of predictor configs and pick by forecast loss.

Scope: load predictors only matter when the planner uses predictive throughput scaling. The current planner feeds predictors from traffic windows and predicts the next window's `num_req`, `isl`, `osl`, and optionally `kv_hit_rate`. Load-only/easy scaling paths do not need this sweep.

| Category | Knob | Proposed treatment | Notes |
| --- | --- | --- | --- |
| predictor family | `load_predictor`: `constant`, `arima`, `kalman`, `prophet` | Separate Grid Search | Predefine a small config set per family. |
| predictor input transform | `load_predictor_log1p` | Separate Grid Search | Search as part of predictor presets, not independently. |
| predictor warmup | `load_predictor_warmup_trace` | Pinned+Derived | If warmup data is available, apply it to all predictor candidates equally; do not multiply the preset space by warmup path. |
| prophet preset | `prophet_window_size` | Separate Grid Search | Predictor-specific preset field. |
| kalman preset | `kalman_q_level`, `kalman_q_trend`, `kalman_r`, `kalman_min_points` | Separate Grid Search | Predictor-specific preset fields. |

Load Predictor Search Space:

| ID | `load_predictor` | Config | Notes |
| --- | --- | --- | --- |
| `constant_last` | `constant` | `load_predictor_log1p=false` | Baseline: predict last observed value. |
| `arima_raw` | `arima` | `load_predictor_log1p=false` | Auto-ARIMA on raw series. |
| `arima_log1p` | `arima` | `load_predictor_log1p=true` | Auto-ARIMA on log-scaled series. |
| `prophet_w20_raw` | `prophet` | `load_predictor_log1p=false`, `prophet_window_size=20` | Short window, raw series. |
| `prophet_w20_log1p` | `prophet` | `load_predictor_log1p=true`, `prophet_window_size=20` | Short window, log-scaled series. |
| `prophet_w50_raw` | `prophet` | `load_predictor_log1p=false`, `prophet_window_size=50` | Default-ish window, raw series. |
| `prophet_w50_log1p` | `prophet` | `load_predictor_log1p=true`, `prophet_window_size=50` | Default-ish window, log-scaled series. |
| `kalman_default_raw` | `kalman` | `load_predictor_log1p=false`, `kalman_q_level=1.0`, `kalman_q_trend=0.1`, `kalman_r=10.0`, `kalman_min_points=5` | Planner default Kalman shape. |
| `kalman_default_log1p` | `kalman` | `load_predictor_log1p=true`, `kalman_q_level=1.0`, `kalman_q_trend=0.1`, `kalman_r=10.0`, `kalman_min_points=5` | Planner default Kalman shape on log scale. |
| `kalman_reactive_raw` | `kalman` | `load_predictor_log1p=false`, `kalman_q_level=10.0`, `kalman_q_trend=1.0`, `kalman_r=5.0`, `kalman_min_points=3` | Faster response to bursts. |
| `kalman_reactive_log1p` | `kalman` | `load_predictor_log1p=true`, `kalman_q_level=10.0`, `kalman_q_trend=1.0`, `kalman_r=5.0`, `kalman_min_points=3` | Faster response to bursts on log scale. |

Grid search evaluation:

1. Build the same traffic windows used by planner throughput scaling, using `throughput_adjustment_interval_seconds`.
2. Run each predefined predictor preset in rolling one-step-ahead mode. Warm with the configured warmup prefix/trace when present, then score predictions against the next observed window.

Optimization goal:

For each evaluated non-empty window `t`, let observed traffic be `N_t`, `I_t`, `O_t` for `num_req`, `isl`, `osl`, and prediction be `N_hat_t`, `I_hat_t`, `O_hat_t`. Minimize weighted log-scale one-step-ahead error:

`loss = 0.4 * err(N_hat_t * I_hat_t, N_t * I_t) + 0.4 * err(N_hat_t * O_hat_t, N_t * O_t) + 0.1 * err(I_hat_t, I_t) + 0.1 * err(O_hat_t, O_t)`

where `err(pred, actual) = abs(log1p(max(pred, 0)) - log1p(max(actual, 0)))`. This keeps `num_req*isl`, `num_req*osl`, `isl`, and `osl` comparable despite different raw scales.

The selected predictor preset is emitted as pinned planner config for the main V2 candidate evaluation.

## Internal Entrypoint

Profiler V2 is reached through the existing profiler entrypoint, not a separate V2 CLI or a parallel sweep-spec envelope. The user still submits a `DynamoGraphDeploymentRequestSpec` to `python -m dynamo.profiler --config <json-or-yaml>`; V2 is an internal, opt-in code path inside `run_profile`, so the DGDR contract stays the single source of truth and the operator wiring is unchanged.

```bash
# Opt-in flag routes run_profile() into the V2 smart-search path.
python -m dynamo.profiler --config dgdr.yaml --smart-search
```

Dispatch:

1. `run_profile(dgdr, ops)` parses and validates the same `DynamoGraphDeploymentRequestSpec` as V1.
2. When the V2 opt-in is set (a `ProfilerOperationalConfig` flag exposed as `--smart-search`, off by default), `run_profile` delegates to the V2 module instead of the V1 AIC sweeper/picker, via `run_smart_search` (full API surface below).
3. V2 reads `model_name`, `backend`, `hardware_sku`, workload, SLA, objective, and component-enablement directly off the DGDR spec — the same fields V1 already consumes. It does **not** introduce a parallel top-level YAML envelope (`load` / `objective` / `sweep` / `searchSpaceConfigs` / `searchSpaceOverrides`); when dynamic-workload inputs and pin/override controls are finalized they extend the DGDR spec contract itself, not a V2-only schema (out of scope for this plan).
4. Sweep execution budget (max rounds, concurrent replay evaluations, candidates per round, random seed) is V2 run-control carried on `ProfilerOperationalConfig` / CLI flags, mirroring how V1 already exposes interpolation granularity and deployment timeouts. It is operational config, not part of the deployment request.
5. For each decoded Vizier sample, V2 constructs a per-candidate `replay_optimize.ReplayOptimizeSpec` (`EngineSpec` model/backend/engine args, `HardwareSpec`, `WorkloadSpec`, `SLASpec`, `RouterSpec`) and evaluates it through `optimize_dense_agg_with_replay` / `optimize_dense_disagg_with_replay`. V2 owns search-space construction, candidate decoding, ranking, and report generation around those calls; AIC and replay stay lower-level providers.
6. V2 writes its ranked-candidate report and `profiler_status.yaml` into `ops.output_dir`, exactly as V1 reports results, so the surrounding sidecar/controller flow needs no change.

The full internal API surface: the input is the existing DGDR spec, and the output is the ranked-candidate contract — which also carries the generated DGD, router, and planner config that the DEP output contract requires. `OptimizationGoal` is defined in [Optimization Goal](#optimization-goal):

```python
# components/src/dynamo/profiler/v2/__init__.py

class SweepConfig(BaseModel):
    """V2 run-control. Operational, not part of the deployment request;
    carried on ProfilerOperationalConfig and exposed as CLI flags."""
    model_config = ConfigDict(extra="forbid")
    maxRounds: int = 20                    # total Vizier suggestion/evaluation rounds
    parallelEvals: int = 16                # concurrent CPU replay evaluations
    candidatesPerRound: int | None = None  # defaults to parallelEvals
    randomSeed: int = 1


class CandidateStatus(str, Enum):
    VIABLE = "viable"    # passed the feasibility gate, scored
    SKIPPED = "skipped"  # removed by candidate generation / preflight
    FAILED = "failed"    # replay or evaluation error


class Candidate(BaseModel):
    """One evaluated deployment candidate and its generated artifacts."""
    model_config = ConfigDict(extra="forbid")
    status: CandidateStatus
    score: float | None = None             # objective score; None unless viable
    deploymentMode: str                    # "agg" | "disagg"
    backend: str                           # "vllm" | "sglang" | "trtllm"
    replicaParallelConfig: dict[str, Any]  # decoded parallel shape + replica counts
    usedGpus: int
    generatedDgd: dict[str, Any]           # DynamoGraphDeployment manifest
    routerConfig: dict[str, Any] | None = None
    plannerConfig: dict[str, Any] | None = None
    replayReportRef: str | None = None     # path/URI to the replay report
    slaMetrics: dict[str, float] = Field(default_factory=dict)  # ttft/itl/e2e/goodput
    reason: str | None = None              # set when status is skipped / failed


class RecommendationMode(str, Enum):
    ADVISORY = "advisory"
    AUTO_APPLIED = "auto_applied"


class RankedCandidates(BaseModel):
    """V2 output contract. Selected candidate first; alternatives ranked."""
    model_config = ConfigDict(extra="forbid")
    goal: OptimizationGoal
    candidates: list[Candidate]                             # viable, best-first
    skipped: list[Candidate] = Field(default_factory=list)  # skipped + failed, with reasons
    bindingConstraints: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    recommendationMode: RecommendationMode = RecommendationMode.ADVISORY


async def run_smart_search(
    dgdr: DynamoGraphDeploymentRequestSpec,  # existing V1 input, unchanged
    ops: ProfilerOperationalConfig,          # carries SweepConfig + output_dir
) -> RankedCandidates:
    """Extract the OptimizationGoal from the DGDR spec, build the branch-aware
    search space, run the Vizier + replay sweep, rank candidates, persist the
    report into ops.output_dir, and return them. Raises NoViableCandidate when
    no candidate clears the feasibility gate."""
    ...
```

This keeps V2 local/offline-first and strictly additive: it does not change V1 defaults, DGDR CR behavior, or the k8s controller/operator path, and it reuses the profiler's existing input contract instead of defining a second one. A user-facing sweep-control surface can be layered on later once the internal path is proven.
