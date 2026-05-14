# Learner Analysis: Beta and LinTS in the Two-Term Router

## Status: Reference Document (April 2026)

Analysis of how the Beta and LinTS learners interact with the two-term scoring
model, and whether they can correctly learn residual loss from the physics tower.

## Current Architecture

```
score(w) = [ λ₁ × R(w) + λ₁ × ε × tanh(B(w)) + λ₂ × S(w) ] × D(w)
```

- `R(w)`: physics ranking (cache overlap, queue availability, sensitivity boost)
- `B(w)`: Beta(α_w, β_w) sample (context-free, per-worker)
- `S(w)`: stickiness (future cache value, IAT urgency, sticky bonus)
- `D(w)`: load discount exp(-w_osl_load × kv_util × (1 + osl_norm))
- `ε = 0.05`: beta contribution scale (primarily for cold-start tiebreaking)

The **only active learner** is the Beta bandit. LinTS was removed.

## Why the Beta Learner Is Not a True Residual Learner

### Problem 1: Training signal is raw reward, not residual

The beta learner trains on:
```
reward = 1 / (1 + metric / baseline)
```

This is the marginal expected reward E[reward | worker = w], averaging over all
request types, overlap levels, and load conditions that worker w happened to
receive. It confounds:
- Worker intrinsic quality (what we want)
- Request assignment bias (high-overlap workers look "better")
- Load effects (underloaded workers are faster)

A correct residual would be `reward - f(physics_prediction)`, capturing only
what the physics model missed.

### Problem 2: No counterfactual reasoning

Classical Thompson Sampling assumes rewards are independent of the selection
policy. In our setting:
- Reward depends on load (pulling arm w too often makes it slow)
- Reward depends on cache state (affected by prior routing decisions)
- Reward depends on other workers' states (relative, not absolute)

The beta learner treats reward as an intrinsic property of the worker, but it's
actually a property of the (worker, routing_policy, workload) tuple.

### Problem 3: Redundancy with physics

The beta posterior converges to E[reward | worker = w], which is highly
correlated with the physics ranking R(w). Both capture "workers with cached
requests are fast." The beta term adds a noisy duplicate.

### Why ε=0.05 makes this acceptable

At ε=0.05 with load_discount applied:
- Cold start: max contribution = 0.05 × tanh(1) × 1.0 = 0.038
- Converged: max contribution = 0.05 × tanh(0.55) × 1.0 = 0.025
- Under load (kv_util=0.5): max contribution ≈ 0.004

The beta term cannot materially influence routing decisions once physics signals
differentiate workers. It functions as **decaying structured noise** for initial
scatter, not as a learner.

## LinTS: Why It Was Removed and Potential Roles

### What happened (empirical)

LinTS was tested extensively with two different feature sets (7-dim and 9-dim),
physics calibration, and adaptive v. In all configurations:
- TTFT worsened by 35ms+ and TPS dropped by 2-20
- LinTS learned to anti-cache: high overlap → negative residual → penalize cached workers
- The anti-caching is structurally correct (congested workers DO have worse
  residuals) but globally destructive (undermines cache locality)

### Potential roles for LinTS in the current architecture

**Role 1: Residual learner on ranking term**
Train on `reward - R(w)`. Failed before — routing-induced congestion dominates
the residual. The load discount in the new architecture reduces this but doesn't
eliminate it. Same structural risk of anti-caching.

**Role 2: Contextual extension of Beta**
Replace the context-free Beta with a contextual LinTS that conditions on request
features. Problem: worker-differentiating features (overlap, load) are already in
physics; non-differentiating features (osl, iat) are the same for all workers.
Nothing new to learn.

**Role 3: Interaction learner**
Learn cross-term interactions (overlap × load). Already captured by the
multiplicative load discount. When tried with interaction features, LinTS still
converged to anti-caching.

**Role 4: Reward normalization (most promising)**
Use LinTS NOT in the scoring path but to improve the reward signal:
- LinTS learns `E[reward | request_features]` (expected reward given request type)
- Beta trains on `reward - LinTS_prediction` (request-normalized residual)
- Separates "was this a hard request?" from "was this worker good?"

This doesn't have the anti-caching problem because LinTS isn't in the scoring
path. The LatencyTracker's bucket baselines already do a crude version of this.
LinTS could do it more precisely with continuous features.

**Role 5: Strict-guardrails residual**
Bring back residual learning but hard-clamp contribution to ±0.02 and only
activate after 500+ observations per worker. Limits damage while allowing
gradual learning. Essentially a bigger Beta tiebreaker with context.

## Fundamental Barrier

The core issue is separating routing-induced effects from worker-intrinsic
quality in the reward signal. A worker that receives many high-overlap requests
(because the router prefers it) will have better rewards — not because it's
intrinsically better, but because the router gave it easy work.

This is a **confounded observational study** problem. Solutions require either:
- Counterfactual estimation (inverse propensity weighting, doubly robust methods)
- Randomized exploration (costly in production)
- Session-level rewards (slow convergence, ~10-100x more data needed)
- Instrumental variables (no obvious instrument in this setting)

## Recommendation

Keep ε=0.05 with raw reward for cold-start tiebreaking. Don't treat the Beta
learner as a principled value estimator — it's structured noise that decays.
The physics tower + stickiness + load discount handles routing correctly without
learning. If residual learning is revisited, start with Role 4 (reward
normalization) as it has the lowest risk of adversarial dynamics.
