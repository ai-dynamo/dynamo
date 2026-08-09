// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Built-in routing policies.
//!
//! These are native implementations behind [`RoutingPolicy`]. Switchyard slots
//! in as a third implementation of the same trait rather than replacing it,
//! which is what keeps a pre-alpha dependency from becoming load-bearing.

use async_trait::async_trait;

use crate::routing::{Candidate, Decision, RoutingContext, RoutingError, RoutingPolicy};

/// Normalize a set of values to 0.0..=1.0 by min-max.
///
/// Prefill tokens, milliseconds, and dollars cannot be summed directly, so each
/// axis is normalized across the candidate set before weighting. A degenerate
/// set (all equal) contributes nothing rather than dividing by zero.
fn normalize(values: &[f64]) -> Vec<f64> {
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let span = max - min;
    if !span.is_finite() || span <= f64::EPSILON {
        return vec![0.0; values.len()];
    }
    values.iter().map(|v| (v - min) / span).collect()
}

/// Relative importance of each axis. All weights are non-negative; larger means
/// the axis matters more.
#[derive(Debug, Clone, Copy)]
pub struct Weights {
    pub prefill: f64,
    pub cost: f64,
    pub ttft: f64,
}

impl Default for Weights {
    fn default() -> Self {
        // Prefill-dominant by default: it is both the cheapest signal to get
        // right and the one the rest of the system is built to exploit.
        Self {
            prefill: 1.0,
            cost: 0.5,
            ttft: 0.25,
        }
    }
}

/// Prefer the target with the best expected cache position, subject to load and
/// quality constraints.
pub struct AffinityPolicy {
    weights: Weights,
    /// Skip local candidates whose load exceeds this. Frontier candidates
    /// report no load and are never excluded by it.
    max_load: Option<f64>,
    min_quality: f64,
}

impl Default for AffinityPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl AffinityPolicy {
    pub fn new() -> Self {
        Self {
            weights: Weights::default(),
            max_load: None,
            min_quality: 0.0,
        }
    }

    pub fn with_weights(mut self, weights: Weights) -> Self {
        self.weights = weights;
        self
    }

    pub fn with_max_load(mut self, max_load: f64) -> Self {
        self.max_load = Some(max_load);
        self
    }

    pub fn with_min_quality(mut self, min_quality: f64) -> Self {
        self.min_quality = min_quality;
        self
    }

    fn admissible<'a>(&self, candidates: &'a [Candidate]) -> Vec<&'a Candidate> {
        candidates
            .iter()
            .filter(|c| c.quality_prior >= self.min_quality)
            .filter(|c| match (self.max_load, c.load) {
                (Some(ceiling), Some(load)) => load <= ceiling,
                _ => true,
            })
            .collect()
    }
}

#[async_trait]
impl RoutingPolicy for AffinityPolicy {
    fn name(&self) -> &str {
        "affinity"
    }

    async fn choose(&self, ctx: &RoutingContext<'_>) -> Result<Decision, RoutingError> {
        if ctx.candidates.is_empty() {
            return Err(RoutingError::NoCandidates);
        }
        let pool = self.admissible(ctx.candidates);
        if pool.is_empty() {
            return Err(RoutingError::NoViableCandidate);
        }

        let prefill = normalize(
            &pool
                .iter()
                .map(|c| c.expected_prefill_tokens)
                .collect::<Vec<_>>(),
        );
        let cost = normalize(&pool.iter().map(|c| c.expected_cost_usd).collect::<Vec<_>>());
        let ttft = normalize(&pool.iter().map(|c| c.expected_ttft_ms).collect::<Vec<_>>());

        let mut best_index = 0;
        let mut best_score = f64::INFINITY;
        for index in 0..pool.len() {
            let score = self.weights.prefill * prefill[index]
                + self.weights.cost * cost[index]
                + self.weights.ttft * ttft[index];
            if score < best_score {
                best_score = score;
                best_index = index;
            }
        }

        let winner = pool[best_index];
        let hit_ratio = winner.cache_hit_ratio(ctx.isl_tokens);
        Ok(Decision {
            target: winner.target.clone(),
            rationale: format!(
                "score {:.4} over {} candidate(s); expected prefill {:.0} of {} tokens ({:.0}% cached), ${:.5}",
                best_score,
                pool.len(),
                winner.expected_prefill_tokens,
                ctx.isl_tokens,
                hit_ratio * 100.0,
                winner.expected_cost_usd,
            ),
        })
    }
}

/// Serve most turns from the cheaper pool, but hand every `audit_every`-th turn
/// to the highest-quality target available.
///
/// This is the shape Switchyard's escalation router implements, reproduced
/// natively so the behavior is available without the dependency. The periodic
/// audit is the cheap approximation; the richer version latches to the strong
/// target on a confirmed streak of trouble, which needs a quality signal we do
/// not yet collect.
pub struct EscalationPolicy {
    inner: AffinityPolicy,
    audit_every: u64,
}

impl EscalationPolicy {
    pub fn new(inner: AffinityPolicy, audit_every: u64) -> Self {
        Self {
            inner,
            audit_every: audit_every.max(1),
        }
    }

    fn is_audit_turn(&self, turn_index: u64) -> bool {
        // Never audit the opening turn: there is no history to check yet, and
        // paying frontier prices for turn zero of every session is exactly the
        // cost this design exists to avoid.
        turn_index > 0 && turn_index.is_multiple_of(self.audit_every)
    }
}

#[async_trait]
impl RoutingPolicy for EscalationPolicy {
    fn name(&self) -> &str {
        "escalation"
    }

    async fn choose(&self, ctx: &RoutingContext<'_>) -> Result<Decision, RoutingError> {
        if ctx.candidates.is_empty() {
            return Err(RoutingError::NoCandidates);
        }
        if !self.is_audit_turn(ctx.turn_index) {
            return self.inner.choose(ctx).await;
        }

        let best = ctx
            .candidates
            .iter()
            .max_by(|a, b| {
                a.quality_prior
                    .partial_cmp(&b.quality_prior)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or(RoutingError::NoCandidates)?;

        Ok(Decision {
            target: best.target.clone(),
            rationale: format!(
                "audit turn (every {}); escalated to highest quality prior {:.2}",
                self.audit_every, best.quality_prior
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::SessionId;
    use crate::routing::{CacheLedger, Target};

    fn local(worker_id: u64, prefill: f64, load: f64) -> Candidate {
        Candidate {
            target: Target::Local {
                worker_id,
                dp_rank: 0,
                model: "llama".into(),
            },
            expected_prefill_tokens: prefill,
            matched_prefix_tokens: 0,
            expected_ttft_ms: 100.0,
            expected_cost_usd: 0.0,
            quality_prior: 0.6,
            load: Some(load),
        }
    }

    fn frontier(prefill: f64, cost: f64) -> Candidate {
        Candidate {
            target: Target::Frontier {
                provider: "anthropic".into(),
                model: "claude".into(),
            },
            expected_prefill_tokens: prefill,
            matched_prefix_tokens: 0,
            expected_ttft_ms: 500.0,
            expected_cost_usd: cost,
            quality_prior: 0.95,
            load: None,
        }
    }

    async fn choose(policy: &dyn RoutingPolicy, candidates: &[Candidate], turn: u64) -> Decision {
        let session_id = SessionId::new("s");
        let ledger = CacheLedger::new();
        let ctx = RoutingContext {
            session_id: &session_id,
            turn_index: turn,
            isl_tokens: 10_000,
            candidates,
            ledger: &ledger,
        };
        policy.choose(&ctx).await.unwrap()
    }

    #[tokio::test]
    async fn the_warmest_worker_wins_all_else_equal() {
        let candidates = vec![local(1, 9_000.0, 0.1), local(2, 500.0, 0.1)];
        let decision = choose(&AffinityPolicy::new(), &candidates, 1).await;
        assert_eq!(decision.target, candidates[1].target);
    }

    #[tokio::test]
    async fn an_overloaded_worker_is_excluded_even_when_warmest() {
        let candidates = vec![local(1, 500.0, 0.99), local(2, 9_000.0, 0.05)];
        let policy = AffinityPolicy::new().with_max_load(0.8);
        let decision = choose(&policy, &candidates, 1).await;
        assert_eq!(decision.target, candidates[1].target);
    }

    #[tokio::test]
    async fn a_free_local_worker_beats_a_paid_frontier_at_equal_prefill() {
        let candidates = vec![local(1, 1_000.0, 0.1), frontier(1_000.0, 0.30)];
        let decision = choose(&AffinityPolicy::new(), &candidates, 1).await;
        assert!(decision.target.is_local());
    }

    #[tokio::test]
    async fn a_warm_frontier_beats_a_cold_local_worker() {
        // The whole point of one comparable axis: a large enough cache
        // advantage should pull the turn to the frontier despite it costing
        // real money and having worse TTFT.
        let candidates = vec![local(1, 100_000.0, 0.5), frontier(200.0, 0.02)];
        let decision = choose(&AffinityPolicy::new(), &candidates, 1).await;
        assert!(!decision.target.is_local(), "{}", decision.rationale);
    }

    #[tokio::test]
    async fn no_admissible_candidate_is_an_error_not_a_bad_choice() {
        let candidates = vec![local(1, 500.0, 0.99)];
        let policy = AffinityPolicy::new().with_max_load(0.5);
        let session_id = SessionId::new("s");
        let ledger = CacheLedger::new();
        let ctx = RoutingContext {
            session_id: &session_id,
            turn_index: 1,
            isl_tokens: 1_000,
            candidates: &candidates,
            ledger: &ledger,
        };
        assert!(matches!(
            policy.choose(&ctx).await,
            Err(RoutingError::NoViableCandidate)
        ));
    }

    #[tokio::test]
    async fn escalation_audits_periodically_but_never_on_the_first_turn() {
        let candidates = vec![local(1, 100.0, 0.1), frontier(50_000.0, 5.0)];
        let policy = EscalationPolicy::new(AffinityPolicy::new(), 4);

        assert!(choose(&policy, &candidates, 0).await.target.is_local());
        assert!(choose(&policy, &candidates, 1).await.target.is_local());
        assert!(choose(&policy, &candidates, 3).await.target.is_local());
        // Turn 4 is an audit: the expensive, high-quality target wins despite
        // being far worse on every cost axis.
        assert!(!choose(&policy, &candidates, 4).await.target.is_local());
        assert!(choose(&policy, &candidates, 5).await.target.is_local());
        assert!(!choose(&policy, &candidates, 8).await.target.is_local());
    }

    #[test]
    fn normalizing_a_degenerate_set_does_not_divide_by_zero() {
        assert_eq!(normalize(&[5.0, 5.0, 5.0]), vec![0.0, 0.0, 0.0]);
        assert_eq!(normalize(&[]), Vec::<f64>::new());
        assert_eq!(normalize(&[0.0, 10.0]), vec![0.0, 1.0]);
    }
}
