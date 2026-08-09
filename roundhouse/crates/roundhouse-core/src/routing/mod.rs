// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Routing: putting local workers and frontier models on one comparable axis.
//!
//! The problem this layer solves is that "serve this turn from our own Llama on
//! worker 7" and "send it to Anthropic" are not obviously comparable. They
//! become comparable once both are expressed as *cache-adjusted expected
//! prefill*, plus latency, price, and a quality prior.
//!
//! The two sides get that number very differently:
//!
//! - **Local** — Dynamo's selection service answers directly. `POST /select` is
//!   query-only and returns `effective_prefill_tokens`, the scheduler's own
//!   cache-credit-weighted prefill cost, without booking anything. That is
//!   already the currency we want.
//! - **Frontier** — no provider exposes its cache, so we *model* it from the
//!   routing ledger: what we last sent to that target, when, and under which
//!   provider TTL. See [`ledger`].
//!
//! Costing happens in `roundhouse-fleet`, which owns the transports. This
//! module owns the vocabulary and the choice.
//!
//! [`RoutingPolicy`] is the seam that keeps Switchyard optional. Its
//! `Algorithm` trait is a good fit — the algorithm emits `Step::CallLlm` with a
//! semantic target and the *host* executes it — but `libsy::State` is in-memory
//! with no pluggable persistence, which collides with surviving process death.
//! Keeping libsy behind this trait means our session core never depends on it.

pub mod ledger;
pub mod policy;

pub use ledger::{CacheLedger, CacheModel, LedgerEntry, ProviderPricing};
pub use policy::{AffinityPolicy, EscalationPolicy};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::ids::SessionId;

/// Where a turn can be sent.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Target {
    /// A worker in the local Dynamo fleet.
    ///
    /// `worker_id` and `dp_rank` mirror Dynamo's `WorkerId` (`u64`) and
    /// `DpRank` (`u32`) exactly, so no conversion is needed at the fleet
    /// boundary.
    Local {
        worker_id: u64,
        dp_rank: u32,
        model: String,
    },
    /// A hosted model behind a provider API.
    Frontier { provider: String, model: String },
}

impl Target {
    pub fn model(&self) -> &str {
        match self {
            Target::Local { model, .. } | Target::Frontier { model, .. } => model,
        }
    }

    pub fn is_local(&self) -> bool {
        matches!(self, Target::Local { .. })
    }

    /// Stable key for ledger lookups.
    ///
    /// Deliberately excludes `dp_rank`: cache residency is a property of the
    /// worker, and we want a turn that previously landed on a given worker to
    /// be recognized as warm regardless of rank.
    pub fn ledger_key(&self) -> String {
        match self {
            Target::Local { worker_id, model, .. } => format!("local:{model}:{worker_id}"),
            Target::Frontier { provider, model } => format!("frontier:{provider}:{model}"),
        }
    }
}

/// One option, priced.
///
/// Every field is an *expectation* — the point is that local and frontier
/// candidates are filled in by completely different machinery yet end up
/// directly comparable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Candidate {
    pub target: Target,

    /// Cache-adjusted prefill tokens. Locally this is the scheduler's
    /// `effective_prefill_tokens`; for frontier targets it is
    /// `isl - p_hit * cached_prefix_tokens`.
    pub expected_prefill_tokens: f64,

    /// Raw matched prefix tokens, before cache weighting. Observability only —
    /// decisions use `expected_prefill_tokens`.
    pub matched_prefix_tokens: u64,

    pub expected_ttft_ms: f64,
    pub expected_cost_usd: f64,

    /// Relative capability, 0.0..=1.0. Supplied by configuration, not measured.
    pub quality_prior: f64,

    /// Queue depth or equivalent pressure signal. `None` for frontier targets,
    /// whose load is not observable to us.
    pub load: Option<f64>,
}

impl Candidate {
    /// Fraction of the prompt expected to be served from cache.
    pub fn cache_hit_ratio(&self, isl_tokens: usize) -> f64 {
        if isl_tokens == 0 {
            return 0.0;
        }
        let saved = (isl_tokens as f64 - self.expected_prefill_tokens).max(0.0);
        (saved / isl_tokens as f64).clamp(0.0, 1.0)
    }
}

/// Inputs available to a policy for one turn.
pub struct RoutingContext<'a> {
    pub session_id: &'a SessionId,
    /// 0 for the first turn of a session.
    pub turn_index: u64,
    pub isl_tokens: usize,
    pub candidates: &'a [Candidate],
    pub ledger: &'a CacheLedger,
}

impl RoutingContext<'_> {
    pub fn candidate_for(&self, target: &Target) -> Option<&Candidate> {
        self.candidates.iter().find(|c| &c.target == target)
    }
}

/// A policy's choice, with the reasoning preserved for the audit trail.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Decision {
    pub target: Target,
    pub rationale: String,
}

/// The persisted form of a decision, written into the session event log.
///
/// Carries the losing options as well as the winner: without them the ledger
/// cannot answer "was that the right call?" after the fact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionRecord {
    pub chosen: Target,
    pub rationale: String,
    pub policy: String,
    pub isl_tokens: u64,
    pub expected_prefill_tokens: f64,
    pub expected_cost_usd: f64,
    pub considered: Vec<Candidate>,
}

#[derive(Debug, thiserror::Error)]
pub enum RoutingError {
    #[error("no candidates available for routing")]
    NoCandidates,
    #[error("no candidate satisfied the policy constraints")]
    NoViableCandidate,
    #[error("policy failure: {0}")]
    Policy(#[from] anyhow::Error),
}

/// Chooses a target for one turn.
///
/// Implementations must be pure with respect to the context: the session layer
/// records the returned [`Decision`] before any execution begins, so a policy
/// that mutated shared state here would produce an audit trail that disagrees
/// with what actually happened.
#[async_trait]
pub trait RoutingPolicy: Send + Sync {
    /// Stable name, recorded in [`DecisionRecord::policy`].
    fn name(&self) -> &str;

    async fn choose(&self, ctx: &RoutingContext<'_>) -> Result<Decision, RoutingError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_hit_ratio_reflects_prefill_savings() {
        let candidate = Candidate {
            target: Target::Frontier {
                provider: "anthropic".into(),
                model: "claude".into(),
            },
            expected_prefill_tokens: 250.0,
            matched_prefix_tokens: 750,
            expected_ttft_ms: 400.0,
            expected_cost_usd: 0.01,
            quality_prior: 0.95,
            load: None,
        };
        assert!((candidate.cache_hit_ratio(1000) - 0.75).abs() < 1e-9);
        assert_eq!(candidate.cache_hit_ratio(0), 0.0);
    }

    #[test]
    fn ledger_key_ignores_dp_rank_but_separates_workers() {
        let a = Target::Local { worker_id: 1, dp_rank: 0, model: "llama".into() };
        let b = Target::Local { worker_id: 1, dp_rank: 3, model: "llama".into() };
        let c = Target::Local { worker_id: 2, dp_rank: 0, model: "llama".into() };
        assert_eq!(a.ledger_key(), b.ledger_key());
        assert_ne!(a.ledger_key(), c.ledger_key());
    }
}
