# DGDR Profiler Smart Search Knob Plan

## Scope

这个 plan 先做 knob inventory，不直接决定最终 DGDR API。目标是把 engine、router、planner 里可能影响 replay/profiler search 的配置列出来，再按优化器视角分类。

主要来源：

- `components/src/dynamo/profiler/utils/replay_optimize/*`
- `components/src/dynamo/planner/config/*`
- `components/src/dynamo/planner/core/*`
- `components/src/dynamo/common/configuration/groups/*`
- `lib/bindings/python/rust/llm/replay.rs`
- `lib/mocker/src/common/protocols.rs`
- `lib/kv-router/src/scheduling/config.rs`
- `origin/main` 上 replay 已经 merge 但本地分支还没有的字段

分类：

- `Branch`: 不放进单个 Vizier flat study，由 DGDR profiler 在外层分支。
- `Search`: 可以成为 Vizier candidate dimension。All `Search` knobs can also be pinned by user override.
- `Composite`: 搜一个高层 shape，再 decode 成多个底层字段。
- `Derived`: 从 branch、backend、shape、AIC 或 candidate generation 结果生成。
- `Pinned`: 用户、环境、模型或 backend 固定。

## V2 Refactor Direction

大方向是 refactor DGDR profiler，但第一阶段不替换现有 V1 行为。V2 先作为 profiler 里的独立实现落地，目标是把 replay-based search、candidate decoding、ranked output 做完整，同时避免影响当前 DGDR/k8s feature。

### Guardrails

- 不改现有 DGDR V1 profiler 的默认入口和行为。
- 不接 k8s controller/operator 路径，不从 DGDR CR 自动触发 V2。
- 不改现有 `v1beta1` schema 的语义，除非只是读取/转换输入。
- 不做 deployment apply，只生成 candidate DGD/router/planner config 和 replay report。
- V2 failure 不应影响 V1 profiler、rapid/thorough profiling、planner sweep 现有流程。

### Proposed V2 Layout

V2 代码建议独立放在 profiler 下，而不是混进 `profile_sla.py` 或现有 `utils/replay_optimize` 的启发式搜索里：

```text
components/src/dynamo/profiler/v2/
  __init__.py
  spec.py              # V2 internal spec and validation
  search_space.py      # branch-specific legal latent spaces
  candidates.py        # candidate model and ranking metadata
  decoder.py           # latent candidate -> replay args, router config, planner config, DGD
  evaluator.py         # replay execution and score extraction
  optimizer.py         # Vizier adapter plus fallback enumerator for tests
  report.py            # ranked output, skipped/failed reasons, refs
```

Current `components/src/dynamo/profiler/profile_sla.py` remains V1. A local/offline V2 entrypoint can be added later, for example `profile_sla_v2.py` or a hidden CLI command, but it should not be wired into DGDR k8s until the V2 result contract is stable.

### Phase Boundary

| Phase | Scope | Non-goals |
| --- | --- | --- |
| V2 alpha | Local/offline profiler path, replay-based evaluation, candidate ranking, generated configs/reports | k8s integration, CRD behavior change, replacing rapid/thorough |
| V2 opt-in | Explicit CLI/config opt-in, compatibility adapter from existing DGDR inputs, replacement path for the current AIC sweeper/picker | automatic controller integration, changing V1 defaults |
| V2 integration | DGDR API/controller wiring after output contract is reviewed | silent fallback to V1 or hidden behavior changes |

### AIC Sweeper Replacement Boundary

In the opt-in phase, V2 should replace the current AIC-driven sweep orchestration for that explicit path:

- V2 owns search-space construction, branch splitting, candidate generation, replay evaluation, scoring, ranking, and report generation.
- V2 replaces AIC's current rapid-mode sweep/pick role for opt-in runs.
- V2 may still call AIC as a lower-level provider for model/backend support checks, legal parallelism hints, perf model data, and memory/capacity estimates.
- AIC output should be treated as input evidence for the V2 candidate decoder/evaluator, not as the final search result.
- Existing V1 rapid/thorough paths keep their current AIC behavior until V2 integration is explicitly wired.

## Branch Model

顶层先按 deployment mode 分 branch，而不是把 `agg/disagg` 放进一个 Vizier conditional space。

| Branch | Engine | Router | Planner |
| --- | --- | --- | --- |
| `agg` | aggregated workers: `tp/pp/dp/moe_tp/moe_ep`, `workers`, agg engine args | agg-compatible router modes and queue policy | `mode=agg`, agg engine GPU shape |
| `disagg` | prefill/decode workers: separate prefill shape, decode shape, prefill worker count, decode worker count | disagg router mode, prefill/decode scoring, admission | `mode=disagg`, prefill/decode engine GPU shape |

每个 branch 里给 Vizier 一个 flat search space。DGDR profiler 负责：

- 生成 branch-specific legal candidates。
- 把 composite knob decode 成 DGD、router config、planner config、replay `MockEngineArgs`。
- 对少量 runtime 失败 candidate 标记 infeasible。
- 全局合并 agg/disagg 的 ranked result。

## Deployment Knobs

Deployment owns topology and budget envelope: branch, replica counts, GPU budget, and per-engine GPU accounting. Engine and KV manager consume these derived values.

| Category | Knob | Applies | Proposed treatment | Notes |
| --- | --- | --- | --- | --- |
| deployment mode | `deployment_mode`: `agg`, `disagg` | all | `Pinned`+`Branch` | Pin when user specifies one mode; otherwise split studies outside Vizier and rank branches globally. |
| objective | `optimization_target`: `sla`, `throughput`, `latency`; SLA targets: `ttft_ms`+`itl_ms` or `e2e_ms` | profiler/planner | `Pinned` | User-selected evaluation objective; `sla` uses explicit latency constraints, while `throughput` and `latency` use replay metrics as the objective. |
| component enablement | `enable_kvrouter` | deployment | `Pinned` | Controls whether V2 emits/evaluates KV-router config; not a search knob. |
| component enablement | `enable_planner` | deployment | `Pinned` | Controls whether V2 emits planner config; not a search knob. |
| replicas | `workers`, `prefill_workers`, `decode_workers` | replay/profiler | `Composite Search` | Search together with the engine `parallel shape` as legal deployment-shape composites capped by `gpu_budget`. |
| budget | `gpu_budget`, `max_gpu_budget`, `min_gpu_budget`, `min_endpoint` | hardware/planner | `Pinned` | `gpu_budget` is the single max-GPU budget concept and maps to planner `max_gpu_budget`; min constraints are pinned deployment inputs used by candidate generation. |

## Engine Knobs

### Full Engine Inventory

| Category | Knob | Applies | Proposed treatment | Notes |
| --- | --- | --- | --- | --- |
| backend | `backend` / `engine_type`: `vllm`, `sglang`, `trtllm` | all | `Search` | Search only across candidate backends with comparable replay/perf model support. |
| worker role | `worker_type`: `aggregated`, `prefill`, `decode` | replay engine | `Derived` | Derived from branch and component. |
| parallel shape | `tp`, `pp`, `dp`, `moe_tp`, `moe_ep` | planner/AIC | `Composite Search` | Search with deployment replicas as one legal deployment-shape composite, not raw independent ints. |
| KV capacity | `num_gpu_blocks`, `total_kv_blocks`, `max_kv_tokens` | replay/runtime | `Derived` | Derived from AIC/memory-fit and block layout; not a search knob in the first design pass. |
| block layout | `block_size`, `kv_cache_block_size` | replay/runtime | `Pinned` | Backend/runtime policy. vLLM default differs from SGLang page size. |
| context | `context_length` | runtime metadata | `Pinned` | Model/runtime constraint, not a search knob. |
| memory budget policy | `gpu_memory_utilization`, `mem_fraction_static`, `free_gpu_memory_fraction` | replay/AIC/backend | `Pinned` | Different backend/config-surface names for the same KV memory budget fraction policy; pin one normalized value. |
| batching | `max_num_seqs` | replay/runtime | `Search` | Important scheduler/capacity knob; controls concurrent sequence admission. |
| batching | `max_num_batched_tokens` | replay/runtime | `Search` | Important scheduler/capacity knob; controls token batching capacity. |
| cache | `enable_prefix_caching` | replay/runtime | `Pinned` | Pin from backend/runtime policy; decoder may still force worker-role-specific values such as false for decode workers. |
| startup | `startup_time` | replay/planner | `Pinned` | Deployment/environment input. |
| speculative decode | `aic_nextn`, `aic_nextn_accept_rates` | `origin/main` replay/AIC | `Pinned` | `aic_nextn` validates to 1..5 when set. Replay MTP token progression support should be clarified before searching this. |

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
| mode/env | `environment`, `namespace`, `backend` | `Pinned` | Deployment context. |
| per-engine GPU count | `decode_engine_num_gpu`, `prefill_engine_num_gpu` | `Derived` | Derived directly from the chosen parallel shape. |
| scaling policy | `enable_throughput_scaling`, `enable_load_scaling`, `throughput_adjustment_interval_seconds`, `load_adjustment_interval_seconds` | `Composite Search` | Search as one legal planner scaling policy tuple. If both scaling modes are enabled, load cadence must be shorter than throughput cadence. |
| FPM sampling | `max_num_fpm_samples`, `fpm_sample_bucket_size` | `Search` | Bucket size must be a perfect square. |
| load scaling sensitivity | `load_scaling_down_sensitivity`, `load_min_observations` | `Search` | Applies to load scaling policy. |

## Planner Load Predictor Grid Sweep

Planner load predictor tuning should be a separate deterministic grid search, not part of the main Vizier candidate space. It is easy to validate directly against replay/planner traces, so V2 should run a small predefined set of predictor configs and pick by forecast loss.

Scope: load predictors only matter when the planner uses predictive throughput scaling. The current planner feeds predictors from traffic windows and predicts the next window's `num_req`, `isl`, `osl`, and optionally `kv_hit_rate`. Load-only/easy scaling paths do not need this sweep.

| Category | Knob | Proposed treatment | Notes |
| --- | --- | --- | --- |
| predictor family | `load_predictor`: `constant`, `arima`, `kalman`, `prophet` | Separate Grid Search | Predefine a small config set per family. |
| predictor input transform | `load_predictor_log1p` | Separate Grid Search | Search as part of predictor presets, not independently. |
| predictor warmup | `load_predictor_warmup_trace` | Pinned+Derived | If warmup data is available, apply it to all predictor candidates equally; do not multiply the preset space by warmup path. |
| prophet preset | `prophet_window_size` | Separate Grid Search | Predictor-specific preset field. |
| kalman preset | `kalman_q_level`, `kalman_q_trend`, `kalman_r`, `kalman_min_points` | Separate Grid Search | Predictor-specific preset fields. |

Predefined predictor presets:

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

## Dependency Rules To Encode

Use candidate generation and validation for these, not post-hoc Vizier learning:

1. `deployment_mode=agg` and `deployment_mode=disagg` are separate studies.
2. Worker role is derived from branch: agg uses `aggregated`; disagg uses `prefill` and `decode`.
3. Parallel shape fields are not independent. Search legal tuples from AIC/model/backend constraints.
4. Worker counts and TP/PP/DP/MoE shape must fit GPU budget.
5. Capacity fields should be derived from AIC/memory-fit unless we intentionally search memory policy.
6. Generic `enable_chunked_prefill` is always on in V2 and is not a search knob.
7. KV manager `num_g3_blocks` requires `num_g2_blocks`.
8. KV manager `enable_g4_storage` requires `num_g2_blocks`.
9. `aic_nextn`, when set, must be in 1..5.
10. Planner `optimization_target=sla` requires either `ttft_ms`+`itl_ms` or `e2e_ms`.
11. Planner `fpm_sample_bucket_size` must be a perfect square.
12. If both planner load and throughput scaling are enabled, load adjustment interval must be shorter than throughput adjustment interval.

## Proposed First Design Pass

### Latent Search Space

Keep the first Vizier space compact:

| Branch | Candidate dimensions |
| --- | --- |
| `agg` | `agg_parallel_shape`, `agg_workers`, `max_num_batched_tokens`, `max_num_seqs`, `router_mode`, `overlap_score_credit`, `prefill_load_scale` |
| `disagg` | `prefill_parallel_shape`, `decode_parallel_shape`, `prefill_workers`, `decode_workers`, `prefill_max_num_batched_tokens`, `decode_max_num_seqs`, `router_mode`, `overlap_score_credit`, `prefill_load_scale` |

### Candidate Decoder

The decoder should emit:

- Replay engine args for every component.
- KV manager policy and replay offload args when enabled.
- Router config.
- Planner config.
- DGD candidate.
- Validation metadata: branch, skipped reasons, derived capacity, GPU budget accounting.

### Later Search Extensions

Add only after replay results are stable:

- Shared cache routing policy.

## Open Questions

1. First backend scope: vLLM only, SGLang only, TRT-LLM only, or backend branch comparison?
2. Should planner be in the evaluation loop now, or should profiler only generate planner config from final ranked candidates?
3. Do we want router queue policy in the first design pass, or pin `fcfs` until queueing replay is validated for DGDR workloads?
4. Which AIC outputs should V2 consume as provider data, and which capacity/memory-fit knobs should V2 own as search dimensions?
5. Should V2 accept pinned KV manager G2/G3/G4/offload config in the first target, or keep it disabled by default?
6. Should DGDR output include all derived configs or only refs to generated files?
