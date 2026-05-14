# Workload-Aware Routing Without LinTS

## Status: Proposed Next Step

Based on Phase 1 ablation (32 trials) and LinTS convergence analysis (6807 observations),
the winning router config is `affinity + switching_cost` with LinTS disabled. However,
this leaves three trie-predicted workload signals unused in routing decisions.

## The Gap

The prediction trie provides rich per-call hints that the router currently ignores:

| Signal | Trie Range | Used in routing? | Used elsewhere? |
|--------|-----------|-----------------|-----------------|
| `reuse_budget` | 4-10 | **Yes** — scales affinity + switching_cost | — |
| `osl` | 78-282 tokens | **No** | Reward baseline bucketing only |
| `iat` | 90-247ms | **No** | Nowhere (was broken until IAT fix) |
| `latency_sensitivity` | 1-5 | **No** | Engine scheduling priority only |

## Why LinTS Failed

LinTS correctly learned that high-overlap workers have worse residual rewards
(because they're overloaded from being selected too often). Its correction
— avoid high-overlap workers — is locally optimal for the residual but globally
destructive: it undermines cache locality, the router's most valuable signal.

This is a fundamental tension: the learner optimizes per-decision reward, but
routing quality depends on the *population* of decisions (cache locality is a
collective property, not a per-request one).

## Proposed: Deterministic Workload Adjustments

Instead of learning, use the trie signals as **deterministic modifiers** on
existing physics tower and heuristic parameters. These are interpretable,
don't require convergence, and can't develop adversarial dynamics.

### 1. IAT-Aware Affinity Scaling

**Idea:** Rapid-fire requests (low IAT) benefit more from cache stickiness
because the cache is still warm. Infrequent requests (high IAT) should be
less sticky because the cache may have been evicted.

**Implementation:** Scale the affinity bonus by `iat_factor`:
```python
# In _score_worker, modify the affinity block:
if self.enable_affinity and last_worker == wid and reuse_budget > 0:
    raw_affinity = (
        self.affinity_base + self.affinity_reuse_weight * float(reuse_budget)
    ) * (0.5 + 0.5 * overlap)
    utility += iat_factor * math.tanh(raw_affinity)  # iat_factor scales stickiness
```

`iat_factor` ranges from 1.5 (rapid, IAT=50ms) to 0.6 (infrequent, IAT=1000ms).
This makes affinity 2.5x stronger for rapid-fire calls than infrequent ones.

**Expected impact:** Better cache utilization for multi-turn rapid conversations,
more load-balancing flexibility for slow/one-shot requests.

### 2. OSL-Aware Load Penalty

**Idea:** A long-output request (high OSL) will occupy a worker for many seconds.
Sending it to an already-loaded worker compounds the congestion. Short requests
are less risky to stack.

**Implementation:** Scale `queue_penalty_weight` by osl_norm:
```python
# In _score_worker, modify the load_mod block:
osl_norm = min(osl, 1024) / 1024.0
effective_qpw = qpw * (1.0 + osl_norm)  # long OSL → stronger load avoidance
load_mod = math.exp(-effective_qpw * u * u)
```

At OSL=128 (short): effective_qpw = qpw * 1.125 (minimal change).
At OSL=512 (medium): effective_qpw = qpw * 1.5 (50% stronger penalty).
At OSL=1024 (long): effective_qpw = qpw * 2.0 (double the penalty).

**Expected impact:** Long-running requests spread more evenly across workers,
reducing tail latency. Short requests still follow cache locality.

### 3. Latency-Sensitivity-Aware Compute Weight

**Idea:** High latency_sensitivity requests need fast TTFT. For these, compute
availability (idle worker) matters more than cache hit. Low-sensitivity requests
can tolerate queueing to benefit from cache hits.

**Implementation:** Temporarily boost physics_compute_weight for sensitive calls:
```python
# In _physics_score, before computing the weighted sum:
sensitivity_boost = latency_sensitivity / max_sensitivity  # [0, 1]
effective_compute_weight = self.physics_compute_weight + sensitivity_boost * 0.2
# Re-normalize to maintain weight sum
```

**Expected impact:** Latency-critical early-conversation calls get routed to
idle workers; later calls (lower sensitivity) favor cache hits.

### 4. IAT-Aware Switching Cost

**Idea:** The switching penalty should be stronger when the next request arrives
soon (low IAT → cache still warm if we stay) and weaker when the gap is long
(high IAT → cache may be cold anyway, switching is less costly).

**Implementation:** Scale switch_cost_weight by iat_factor:
```python
# In _score_worker, modify the switching penalty block:
if self.enable_switching_cost and last_worker is not None and wid != last_worker and reuse_budget > 0:
    raw_penalty = self.switch_base + self.switch_reuse * float(reuse_budget)
    score -= (self.switch_cost_weight * iat_factor) * math.tanh(raw_penalty)
```

**Expected impact:** Pairs naturally with IAT-aware affinity. Rapid sessions
get both stronger stickiness bonus AND stronger switching penalty.

## Implementation Priority

1. **IAT-aware affinity** (highest impact, simplest change, one line)
2. **IAT-aware switching cost** (pairs with #1, one line)
3. **OSL-aware load penalty** (medium impact, helps tail latency)
4. **Latency-sensitivity compute boost** (requires passing hint through to physics tower)

## Validation Plan

- Run Phase 1 ablation baseline (`affinity + switching_cost`, no workload mods) on full 500-item dataset
- Enable workload mods one at a time, compare TTFT/TPS
- If positive, add to Phase 2 optimization as tunable coefficients

## Related Issues

- IAT was broken until the `hints.py` fix (priority_jump conflation) — Issue #18 in tracker
- Trie-predicted values confirmed reaching the router: osl=134, iat=5ms, reuse=4
- `latency_sensitivity` now available as a separate field in hints dict
