# Thompson Router: Two-Tower Scoring Architecture

## Final Score Formula

```
score(w) = utility(w) × load_mod(w) − switching_penalty(w)
```

Where `utility = physics + learned + heuristics`, `load_mod` multiplicatively
penalizes loaded workers, and the switching penalty is subtracted outside to
avoid sign inversion.

---

## Tower 1: Physics (deterministic, no learning)

Computes a weighted sum of 4 hardware-agnostic signals, all in [0, 1]. This is
the stable anchor — correct from the first request.

```
physics(w) = cache_w × cache_hit  +  compute_w × compute_avail
           + queue_w × queue_avail  +  memory_w × memory_avail
```

| Signal | Formula | Source | What it captures |
|--------|---------|--------|-----------------|
| `cache_hit` | overlap ∈ [0,1] | KvIndexer RadixTree | Fraction of request tokens already cached on worker `w` |
| `compute_avail` | 1 − kv_util ∈ [0,1] | WorkerLoadMonitor (NATS) | Free KV capacity — low when worker has many active decode blocks |
| `prefill_avail` | 1 − prefill_util ∈ [0,1] | WorkerLoadMonitor (NATS) | Free prefill capacity — low when worker has a prefill backlog |
| `memory_avail` | 1 − tree_size/total_blocks ∈ [0,1] | KvIndexer + WorkerLoadMonitor | Eviction headroom — low when worker's cache is full (future hits at risk) |

**Tunable weights** (sum to ~0.85 in current config):

| Weight | Default | Phase 1 Best | Role |
|--------|---------|-------------|------|
| `physics_cache_weight` | 0.35 | 0.55 | Dominant — cache hit is most impactful for agentic workloads |
| `physics_compute_weight` | 0.30 | 0.10 | Penalize active decode load |
| `physics_queue_weight` | 0.20 | 0.15 | Penalize prefill backlogs (primarily affects TTFT) |
| `physics_memory_weight` | 0.15 | 0.05 | Protect against eviction (deferred cost) |

---

## Tower 2: Learned Residual (optional, currently disabled)

Two independent learners, both bounded via `tanh` to prevent dominating physics.

### Beta TS (`enable_beta_ts`)

Context-free per-worker quality:

```
utility += ts_weight × tanh(Beta(α_w, β_w).sample())
```

- No features — just "is this worker generally good?"
- `beta_decay` controls memory window (~1/(1−decay) observations)
- Bounded contribution: ±ts_weight (typically 0.05–0.2)

### LinTS (`enable_lints`)

Contextual bandit with 9-dim feature vector:

```
utility += |lints_weight| × tanh(θ_wᵀ x + noise)
```

- Per-worker linear model: θ_w = A_w⁻¹ b_w
- Trains on calibrated residual: `reward − calibrated_physics + 0.5`
- `lints_v` controls exploration noise (adaptive option: v = √(residual_ema_var))
- `lints_forget_rate` controls memory window

**Feature vector** (9 dims, all [0,1]):

| # | Feature | Formula | What it captures |
|---|---------|---------|-----------------|
| 0 | bias | 1.0 | Per-worker intercept |
| 1 | overlap × idle | overlap × (1 − kv_util) | Cache value discounted by load (interaction) |
| 2 | load rank | rank(kv_util) / (n−1) | Relative load position among workers |
| 3 | overlap rank | rank(overlap) / (n−1) | Relative cache position among workers |
| 4 | selection pressure | EMA of pick frequency | Herding detection — "am I overusing this worker?" |
| 5 | prefill fraction | prefill_tokens / tokens_in | Actual uncached token fraction |
| 6 | inflight share | inflight[w] / total_inflight | Direct queue depth from router's own tracking |
| 7 | osl × load | (osl/1024) × kv_util | Long output + loaded worker = bad |
| 8 | iat × reuse | iat_norm × tanh(reuse/4) | Rapid reuse + high budget = sticky |

### Phase 1 Ablation Result

LinTS hurts performance (TTFT +35ms, TPS −2.0 on 20% dataset; worse on full
dataset). The learner correctly identifies congestion on popular workers but its
correction undermines cache locality. **Disabled in the winning config.**

See `WORKLOAD_AWARE_ROUTING.md` for proposed deterministic alternatives.

---

## Heuristic Layer (deterministic, workload-aware)

### Affinity Bonus (`enable_affinity`)

Rewards sticking to the prefix's previous worker:

```
if last_worker == w and reuse_budget > 0:
    utility += tanh((affinity_base + affinity_reuse_weight × reuse) × (0.5 + 0.5 × overlap))
```

- Saturates via `tanh` at ~15 reuse budget
- Scaled by overlap: stickiness is more valuable when the worker actually has cache
- `reuse_budget = total_requests − 1` from prediction trie

### Switching Penalty (`enable_switching_cost`)

Penalizes migrating away from the prefix owner:

```
if last_worker ≠ w and reuse_budget > 0:
    score −= switch_cost_weight × tanh(switch_base + switch_reuse × reuse)
```

- Applied OUTSIDE load_mod to avoid sign inversion (historical bug fix)
- Only activates when switching to a *different* worker for a reusable prefix

---

## Load Modulator (multiplicative gate)

Exponential penalty applied to the positive utility term:

```
load_mod = exp(−qpw × kv_util²)
```

| kv_util | qpw=2.5 | Effect |
|---------|---------|--------|
| 0.0 (idle) | 1.00 | No penalty |
| 0.3 | 0.80 | Mild |
| 0.5 | 0.54 | Moderate |
| 0.8 | 0.20 | Heavy |
| 1.0 (full) | 0.08 | Near-zero |

`kv_util` is the hardware-agnostic ratio `active_decode_blocks / total_kv_blocks`
from WorkerLoadMonitor, delivered via NATS ActiveLoad events.

**Optional floors:**

| Flag | What it does |
|------|-------------|
| `enable_load_mod_floor` | Clamp to `load_mod_floor` (prevents zero scores) |
| `enable_sticky_floor` | Clamp to `sticky_load_floor` for the prefix owner (prevents load fluctuations from breaking stickiness) |

---

## Action Selection

| Mode | Behavior | Phase 1 Result |
|------|----------|---------------|
| `enable_softmax=false` (default) | **Argmax** — pick the highest-scoring worker deterministically | **Winner** |
| `enable_softmax=true` | Boltzmann sampling: `π(w) ∝ exp(score(w) / τ)` | +161ms TTFT, −9.7 TPS |

When softmax is enabled, temperature `τ` can optionally adapt via
`enable_adaptive_temp`: `τ = τ_base / (1 + reuse × iat_factor)`, making
high-reuse rapid-fire sessions near-greedy.

---

## Workload Hints (from prediction trie)

The NAT prediction trie provides per-call predictions injected via
`nvext.agent_hints` and `nvext.annotations`:

| Hint | Trie Range | Used in routing | Used in feedback |
|------|-----------|----------------|-----------------|
| `reuse_budget` | 4–10 | Scales affinity bonus + switching penalty | — |
| `osl` | 78–282 tokens | LinTS feature only (disabled) | Reward baseline bucketing |
| `iat` | 90–247ms | LinTS feature only (disabled) | — |
| `latency_sensitivity` | 1–5 | Not yet used in routing | Engine scheduling priority |
| `tokens_in` | varies | prefill_tokens computation | Reward baseline bucketing |
| `prefix_id` | unique per conversation | Worker affinity tracking | — |

**Note:** `iat` was incorrectly derived from `latency_sensitivity` until the
`hints.py` fix (Issue #18). Now correctly reads from annotations.

See `WORKLOAD_AWARE_ROUTING.md` for proposals to use osl/iat/latency_sensitivity
directly in the physics tower and heuristic adjustments.

---

## Data Flow

```
Request arrives with token_ids + nvext hints (osl, iat, prefix_id, reuse_budget)
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  Per-worker signals (parallel for all workers):         │
│    KvIndexer.find_matches()  →  overlap, tree_sizes     │
│    KvRouter.get_potential_loads()  →  decode_blocks      │
│    WorkerLoadMonitor.get_all()  →  kv_util, prefill_util│
│    _prefix_workers[prefix_id]  →  last_worker           │
└─────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  Scoring (per worker):                                  │
│    Physics Tower:  Σ(weight × signal) → physics_score   │
│    + Affinity bonus (if sticky worker, reuse > 0)       │
│    × Load modulator: exp(−qpw × kv_util²)              │
│    − Switching penalty (if different worker, reuse > 0) │
│    = final_score                                        │
└─────────────────────────────────────────────────────────┘
     │
     ▼
  argmax(final_scores) → chosen worker
     │
     ▼
  Forward request → observe latency → update learners + baselines
```

---

## Phase 1 Ablation Summary (32 trials, grid search)

| Feature | TTFT Impact | TPS Impact | Status |
|---------|------------|-----------|--------|
| **affinity** | −20ms | +1.3 | **Enabled** |
| **switching_cost** | −14ms | +0.4 | **Enabled** |
| softmax | +161ms | −9.7 | Disabled (harmful) |
| lints | +35ms | −2.0 | Disabled (harmful) |
| load_mod_floor | −9ms | +0.2 | Optional |
| beta_ts | −8ms | +0.3 | Optional |

**Pareto winners:**
- Best TTFT: `beta_ts + load_mod_floor` → 0.19s / 54.61 TPS
- Best TPS: `affinity + switching_cost` → 0.20s / 57.78 TPS
