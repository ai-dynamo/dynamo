# Thompson Router Parameter Reference

Complete catalog of every tunable parameter in the KvThompsonRouter.
Each entry includes: purpose, default, range, interactions, and recommended
search bounds for hyperparameter optimization.

---

## Physics Tower Weights

These control the relative importance of each physics signal. All inputs are
hardware-agnostic ratios in [0,1]. Weights should sum to ~1.0.

### physics_cache_weight
- **Purpose:** Weight on KV cache hit fraction (ρ). Higher = favor workers with cached prefixes.
- **Default:** 0.35
- **Range:** [0.0, 1.0]
- **Search bounds:** [0.1, 0.6]
- **Interactions:** Competes with compute/queue/memory weights. If too high relative to compute_weight, the router may send to a worker with a cache hit even if that worker is overloaded.
- **Intuition:** Most important for agentic workloads with high prefix reuse. Less important for one-shot traffic where cache hits are rare.

### physics_compute_weight
- **Purpose:** Weight on compute availability (1 - kv_util). Higher = avoid workers with high active decode load.
- **Default:** 0.30
- **Range:** [0.0, 1.0]
- **Search bounds:** [0.1, 0.5]
- **Interactions:** Competes with cache_weight. For latency-sensitive workloads, compute is king — an overloaded worker with a cache hit may still be slower.
- **Intuition:** Critical for tail latency. Increase for throughput-sensitive deployments.

### physics_queue_weight
- **Purpose:** Weight on queue availability (1 - prefill_util). Higher = avoid workers with prefill backlogs.
- **Default:** 0.20
- **Range:** [0.0, 1.0]
- **Search bounds:** [0.05, 0.4]
- **Interactions:** Primarily affects TTFT (time to first token). If prefill queue is full, new requests wait.
- **Intuition:** Increase if TTFT is the primary metric. Less important if throughput is the bottleneck.

### physics_memory_weight
- **Purpose:** Weight on memory availability (1 - tree_size/total_blocks). Higher = avoid workers near cache eviction.
- **Default:** 0.15
- **Range:** [0.0, 1.0]
- **Search bounds:** [0.0, 0.3]
- **Interactions:** This is a deferred cost — doesn't affect *this* request's latency but protects future cache hits. Interacts with cache_weight: high memory_weight protects the value of high cache_weight.
- **Intuition:** Important for sustained agentic workloads. Can be low for bursty one-shot traffic.

---

## Learned Tower (LinTS)

### lints_weight
- **Purpose:** Scale coefficient on the LinTS residual correction. Controls how much the learned model can override the physics tower. Always tanh-bounded: `abs(lints_weight) * tanh(sample)`.
- **Default:** 1.0
- **Range:** [0.0, 2.0]
- **Search bounds:** [0.1, 1.5]
- **Interactions:** If too large, the learner dominates physics. If too small, the learner can't correct physics errors. Should be calibrated relative to physics score range (~0-1).
- **Intuition:** Start at 1.0. After observing residual variance via /metrics, adjust: low residual variance → reduce, high → increase.
- **Note:** Requires `enable_lints=True`. When disabled, LinTS contributes zero to the score (useful for ablation).

### lints_v
- **Purpose:** Noise scale for LinTS posterior sampling. Controls exploration magnitude.
- **Default:** 0.25
- **Range:** [0.01, 2.0]
- **Search bounds:** [0.05, 0.75]
- **Interactions:** Higher v = more exploration = more variance in worker selection = slower convergence but better long-term learning. Lower v = exploitative, fast convergence, risk of getting stuck.
- **Intuition:** If /metrics shows residual variance >> v², you're under-exploring (increase v). If residual variance << v², you're over-exploring (decrease v). The sweet spot is v ≈ sqrt(residual_variance).

### lints_lambda (lints_lambda)
- **Purpose:** Ridge regularization strength for LinTS precision matrix.
- **Default:** 1.0
- **Range:** [0.1, 10.0]
- **Search bounds:** [0.5, 3.0]
- **Interactions:** Higher = more regularization = slower learning but more stable. Lower = faster adaptation but risk of ill-conditioning.
- **Intuition:** Rarely needs tuning. Increase if you see numerical warnings from Cholesky decomposition.

### lints_forget_rate
- **Purpose:** Exponential forgetting factor γ for LinTS posterior. Controls how quickly old observations are discounted.
- **Default:** 0.995
- **Range:** [0.95, 0.9999]
- **Search bounds:** [0.98, 0.999]
- **Interactions:** Effective window ≈ 1/(1-γ). At 0.995, window ≈ 200 observations. Lower = faster adaptation to changing conditions, higher = more stable posterior.
- **Intuition:** If workload patterns change frequently (different models, traffic patterns), decrease toward 0.98. If stable, keep near 0.999.

---

## Global Exploration (Beta TS)

### ts_weight
- **Purpose:** Scale coefficient on the Beta TS global exploration term. Controls how much context-free worker quality sampling affects the score.
- **Default:** 0.05
- **Range:** [0.0, 1.0]
- **Search bounds:** [0.01, 0.5]
- **Interactions:** Competes with lints_weight. If ts_weight >> lints_weight, the router mostly explores randomly rather than using contextual features. Should be smaller than lints_weight in steady state.
- **Intuition:** Higher early (more exploration), lower as system stabilizes. For production, 0.05-0.2 is typical.

### beta_decay
- **Purpose:** Exponential decay factor γ_β for Beta posterior. Prevents the Beta from concentrating and killing exploration.
- **Default:** 0.995
- **Range:** [0.95, 0.9999]
- **Search bounds:** [0.98, 0.999]
- **Interactions:** Same semantics as lints_forget_rate. Window ≈ 1/(1-γ_β).
- **Intuition:** Keep in sync with lints_forget_rate. If one forgets fast and the other doesn't, the learners have inconsistent views of history.

---

## Heuristic Adjustments

### affinity_base
- **Purpose:** Base stickiness bonus when a worker is the prefix owner and reuse > 0. Inside tanh, so the raw value is the unsaturated contribution.
- **Default:** 0.5
- **Range:** [0.0, 2.0]
- **Search bounds:** [0.2, 1.0]
- **Interactions:** Combined with affinity_reuse_weight and overlap to form the tanh argument. Saturates at ~15 reuse budget with default settings.
- **Intuition:** Higher = stickier for even low-reuse prefixes. Lower = only high-reuse prefixes get meaningful stickiness.

### affinity_reuse_weight
- **Purpose:** Per-reuse-count increment to the affinity bonus. Controls how much additional reuse budget increases stickiness.
- **Default:** 0.13
- **Range:** [0.0, 0.5]
- **Search bounds:** [0.05, 0.25]
- **Interactions:** With affinity_base, determines where the tanh saturates. At default (0.5 + 0.13*budget), saturates near budget=15.
- **Intuition:** Increase if you want saturation at fewer reuses. Decrease if you want more gradual stickiness ramp.

### switch_base
- **Purpose:** Base switching penalty when routing away from the prefix owner.
- **Default:** 0.2
- **Range:** [0.0, 2.0]
- **Search bounds:** [0.1, 0.5]
- **Interactions:** Combined with switch_reuse inside tanh. Saturates at ~15 reuse budget.
- **Intuition:** Higher = stronger penalty for any switch even with low reuse.

### switch_reuse
- **Purpose:** Per-reuse-count increment to switching penalty.
- **Default:** 0.12
- **Range:** [0.0, 0.5]
- **Search bounds:** [0.05, 0.25]
- **Interactions:** With switch_base, determines saturation point. At default (0.2 + 0.12*budget), saturates near budget=15.
- **Intuition:** Should be roughly matched to affinity_reuse_weight so the bonus and penalty scale together.

### switch_cost_weight
- **Purpose:** Global multiplier on the entire switching penalty (after tanh). Controls the penalty's strength relative to the utility score.
- **Default:** 1.0
- **Range:** [0.0, 3.0]
- **Search bounds:** [0.3, 2.0]
- **Interactions:** Since the tanh-saturated penalty is in [0,1), this coefficient directly controls the max penalty. At 1.0, the max penalty equals the max bonus. Increase to strongly discourage switching, decrease to allow more flexibility.
- **Intuition:** The most important single knob for balancing stickiness vs load balancing. Start at 1.0 and tune based on prefix scatter rates from /decisions/summary.

---

## Load Modulation

### queue_penalty_weight
- **Purpose:** Controls the steepness of the exponential load modulator: `load_mod = exp(-qpw * kv_util²)` where `kv_util ∈ [0,1]` is the hardware-agnostic KV utilization ratio from WorkerLoadMonitor.
- **Default:** 2.5
- **Range:** [0.0, 20.0]
- **Search bounds:** [0.5, 10.0]
- **Interactions:** Higher = sharper load penalty = stronger avoidance of loaded workers. At qpw=2.5 and kv_util=0.5, load_mod ≈ 0.54. At kv_util=0.8, load_mod ≈ 0.20. Interacts with physics_compute_weight (both penalize load, but through different mechanisms). Note: when kv_util is 0.0 (no load data), load_mod = 1.0 — no penalty applied.
- **Intuition:** This is a hard constraint on top of the physics tower's soft preference. Increase if workers are getting overloaded; decrease if load is well-balanced and you want more cache-locality.

### load_mod_floor
- **Purpose:** Minimum load modulator value. Prevents the score from being driven to zero for loaded workers.
- **Default:** 0.3
- **Range:** [0.0, 1.0]
- **Search bounds:** [0.1, 0.5]
- **Interactions:** Only active when enable_load_mod_floor=True. Ensures even heavily loaded workers retain 30% of their utility score.
- **Intuition:** Lower = more aggressive load avoidance. Higher = more tolerance for loaded workers (useful when cache hits on loaded workers are very valuable).

### sticky_load_floor
- **Purpose:** Minimum load modulator for the sticky (affinity) worker. Prevents load fluctuations from breaking stickiness for high-reuse prefixes.
- **Default:** 0.01
- **Range:** [0.0, 1.0]
- **Search bounds:** [0.01, 0.3]
- **Interactions:** Only active when enable_sticky_floor=True and the worker is the prefix owner with reuse > 0. Overrides load_mod for sticky workers.
- **Intuition:** Higher = stickier even under load. Very high values (>0.5) can cause load imbalance. Keep low unless prefix reuse is extremely valuable.

---

## Action Selection

### temperature
- **Purpose:** Boltzmann softmax temperature for action selection. Higher = more random, lower = more greedy.
- **Default:** 1.70
- **Range:** [0.1, 5.0]
- **Search bounds:** [0.3, 3.0]
- **Interactions:** Only active when enable_softmax=True. Interacts with adaptive_temp: when enable_adaptive_temp=True, this value is overridden by `adaptive_temp_base / (1 + reuse * iat_factor)`.
- **Intuition:** High temperature during exploration phase, low for exploitation. For production, 0.5-2.0 is typical.

### adaptive_temp_base
- **Purpose:** Base temperature for adaptive temperature scaling. Temperature = base / (1 + reuse * iat_factor).
- **Default:** 1.0
- **Range:** [0.1, 3.0]
- **Search bounds:** [0.3, 2.0]
- **Interactions:** Only active when enable_adaptive_temp=True. For high-reuse + low-IAT (rapid reuse), temperature → base / (1 + large_number) → near-greedy.
- **Intuition:** Controls the "ceiling" temperature for one-shot requests (reuse=0 → temp=base).

### temp_min / temp_max
- **Purpose:** Hard clamps on temperature.
- **Defaults:** temp_min=0.15, temp_max=2.0
- **Search bounds:** temp_min [0.05, 0.5], temp_max [1.0, 5.0]

---

## Feature Toggles

These are boolean flags (True/False). For optimization, test combinations:

| Flag | Purpose | Recommended for optimization |
|------|---------|------------------------------|
| enable_softmax | Probabilistic vs argmax selection | True (enables temperature tuning) |
| enable_beta_ts | Beta TS global exploration | True (context-free worker quality) |
| enable_lints | LinTS contextual bandit | True (core learner) |
| enable_affinity | Prefix stickiness bonus | True (for agentic workloads) |
| enable_switching_cost | Penalty for prefix migration | True (paired with affinity) |
| enable_adaptive_temp | Temperature decays with reuse | True (for mixed workloads) |
| enable_adaptive_explore | Beta TS weight decays with reuse | True (requires enable_beta_ts) |
| enable_cold_start | Round-robin when no cache hits | False (physics tower handles cold start) |
| enable_idle_boost | Floor on overlap for idle workers | False (physics tower handles this) |
| enable_load_mod_floor | Floor on load modulator | True (prevents zero scores) |
| enable_sticky_floor | Sticky worker load floor | True (prevents thrashing) |

---

## Recommended Search Strategy

### Phase 1: Physics tower weights (fast, no learner involvement)
Optimize physics_cache_weight, physics_compute_weight, physics_queue_weight,
physics_memory_weight with lints disabled. This finds the best "no learning"
baseline. Use hot-reload (no restart needed).

### Phase 2: Learner scale (enable lints, tune contribution)
Enable lints. Sweep lints_weight and lints_v. Compare against Phase 1 baseline.
Hot-reload works here too.

### Phase 3: Heuristic calibration (affinity + switching)
Enable affinity + switching_cost. Sweep switch_cost_weight, affinity_base,
affinity_reuse_weight. Measure prefix stickiness rate and latency jointly.

### Phase 4: Full sweep (all features enabled)
Joint optimization of top 5-8 most impactful params from Phases 1-3.
Use nat optimize for Pareto frontier search.

### Key params for initial sweep (ordered by expected impact):
1. physics_cache_weight + physics_compute_weight (ratio matters most)
2. switch_cost_weight (stickiness vs load balance)
3. lints_weight + lints_v (learner contribution)
4. queue_penalty_weight (load avoidance sharpness)
5. temperature / adaptive_temp_base (exploration)
