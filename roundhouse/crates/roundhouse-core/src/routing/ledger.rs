// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The routing ledger: modelling a cache we cannot query.
//!
//! For local workers the selection service reports prefix overlap directly. For
//! a frontier provider there is no such query, so the ledger reconstructs the
//! answer from what we know: what we last sent to that target, how long ago,
//! and how that provider's cache expires.
//!
//! The append-only property of a session makes this tractable. Within one
//! session we send the whole conversation every turn, so whatever we sent to a
//! target last time is a *prefix* of what we are about to send. The expected
//! cached portion is therefore `p_hit(elapsed) * last_prefix_tokens`, and the
//! only hard part is `p_hit`.
//!
//! That property breaks if the context is compacted or truncated — dropping
//! early turns makes the old prompt no longer a prefix of the new one, and the
//! model would overestimate. [`CacheLedger::invalidate`] exists for that case
//! and must be called whenever the assembler rewrites history rather than
//! appending to it.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::routing::Target;

/// How a target's prefix cache expires.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CacheModel {
    /// Explicit breakpoints with a deterministic lifetime, refreshed on each
    /// hit. Anthropic's `cache_control` behaves this way, which makes it the
    /// easiest provider to route against: within the window a hit is a
    /// near-certainty rather than a hope.
    Deterministic { ttl_ms: u64 },

    /// Automatic caching above a minimum prefix length, evicted after a period
    /// of inactivity that is not contractually fixed. OpenAI behaves this way;
    /// a stable `prompt_cache_key` improves the odds by steering requests to
    /// the same cache node, which is why the executor always sends one.
    InactivityDecay {
        half_life_ms: u64,
        max_ttl_ms: u64,
        min_prefix_tokens: u64,
    },

    /// The router reports overlap directly, so no model is needed. Present so
    /// local targets can share the ledger's bookkeeping.
    Observed,
}

impl CacheModel {
    /// Probability that a prefix of `prefix_tokens` is still cached after
    /// `elapsed_ms`.
    pub fn hit_probability(&self, elapsed_ms: u64, prefix_tokens: u64) -> f64 {
        if prefix_tokens == 0 {
            return 0.0;
        }
        match *self {
            CacheModel::Deterministic { ttl_ms } => {
                if elapsed_ms < ttl_ms {
                    1.0
                } else {
                    0.0
                }
            }
            CacheModel::InactivityDecay {
                half_life_ms,
                max_ttl_ms,
                min_prefix_tokens,
            } => {
                if prefix_tokens < min_prefix_tokens || elapsed_ms >= max_ttl_ms {
                    return 0.0;
                }
                if half_life_ms == 0 {
                    return 0.0;
                }
                0.5f64.powf(elapsed_ms as f64 / half_life_ms as f64)
            }
            // Never guessed at; callers use the router's reported overlap.
            CacheModel::Observed => 0.0,
        }
    }
}

/// Per-million-token prices.
///
/// These are configuration, not constants: provider prices change, and baking
/// them into code guarantees they go stale. [`ProviderPricing::free`] is the
/// right default for local targets, whose marginal cost we account for in
/// prefill tokens rather than dollars.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ProviderPricing {
    pub input_per_mtok_usd: f64,
    /// Reads against a warm prefix. The gap between this and
    /// `input_per_mtok_usd` is the entire economic lever this design pulls.
    pub cached_input_per_mtok_usd: f64,
    /// Writing a prefix into the cache, which some providers price at a
    /// premium over ordinary input.
    pub cache_write_per_mtok_usd: f64,
    pub output_per_mtok_usd: f64,
}

impl ProviderPricing {
    pub fn free() -> Self {
        Self {
            input_per_mtok_usd: 0.0,
            cached_input_per_mtok_usd: 0.0,
            cache_write_per_mtok_usd: 0.0,
            output_per_mtok_usd: 0.0,
        }
    }
}

/// What we last sent to a target.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TargetState {
    pub last_call_at_ms: u64,
    /// Prompt length of that call, and therefore the longest prefix that could
    /// still be warm.
    pub last_prefix_tokens: u64,
}

/// One recorded dispatch, projected from the session event log.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LedgerEntry {
    pub target_key: String,
    pub at_ms: u64,
    pub isl_tokens: u64,
    pub output_tokens: u64,
    pub cost_usd: f64,
}

/// Per-session record of what went where, and the cache models to reason with.
#[derive(Debug, Clone, Default)]
pub struct CacheLedger {
    state: HashMap<String, TargetState>,
    models: HashMap<String, (CacheModel, ProviderPricing)>,
}

impl CacheLedger {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register how a target's cache behaves and what it costs.
    ///
    /// Unregistered targets are treated as [`CacheModel::Observed`] and free,
    /// which is the correct fallback for local workers.
    pub fn register(&mut self, target: &Target, model: CacheModel, pricing: ProviderPricing) {
        self.models.insert(target.ledger_key(), (model, pricing));
    }

    pub fn model_for(&self, target: &Target) -> (CacheModel, ProviderPricing) {
        self.models
            .get(&target.ledger_key())
            .copied()
            .unwrap_or((CacheModel::Observed, ProviderPricing::free()))
    }

    /// Record a dispatch. Called as the session projects its event log.
    pub fn record(&mut self, target: &Target, at_ms: u64, isl_tokens: u64) {
        self.state.insert(
            target.ledger_key(),
            TargetState {
                last_call_at_ms: at_ms,
                last_prefix_tokens: isl_tokens,
            },
        );
    }

    /// Drop cached-prefix assumptions for every target.
    ///
    /// Required whenever the conversation stops being append-only — a
    /// compaction, a summarization, an edited history. Without this the model
    /// keeps claiming a warm prefix that no longer exists.
    pub fn invalidate(&mut self) {
        self.state.clear();
    }

    pub fn state_for(&self, target: &Target) -> Option<TargetState> {
        self.state.get(&target.ledger_key()).copied()
    }

    /// Expected number of prompt tokens served from cache.
    pub fn expected_cached_tokens(&self, target: &Target, now_ms: u64, isl_tokens: u64) -> f64 {
        let Some(state) = self.state_for(target) else {
            return 0.0;
        };
        // The warm prefix cannot exceed what we are about to send.
        let prefix = state.last_prefix_tokens.min(isl_tokens);
        let elapsed = now_ms.saturating_sub(state.last_call_at_ms);
        let (model, _) = self.model_for(target);
        model.hit_probability(elapsed, prefix) * prefix as f64
    }

    /// Expected dollar cost of one call to `target`.
    pub fn estimate_cost_usd(
        &self,
        target: &Target,
        now_ms: u64,
        isl_tokens: u64,
        expected_output_tokens: u64,
    ) -> f64 {
        let (_, pricing) = self.model_for(target);
        let cached = self.expected_cached_tokens(target, now_ms, isl_tokens);
        let uncached = (isl_tokens as f64 - cached).max(0.0);

        // Uncached prompt tokens are also written into the cache, so they are
        // priced at the write rate when the provider charges one.
        let write_rate = if pricing.cache_write_per_mtok_usd > 0.0 {
            pricing.cache_write_per_mtok_usd
        } else {
            pricing.input_per_mtok_usd
        };

        let per_mtok = 1e-6;
        uncached * write_rate * per_mtok
            + cached * pricing.cached_input_per_mtok_usd * per_mtok
            + expected_output_tokens as f64 * pricing.output_per_mtok_usd * per_mtok
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINUTE: u64 = 60_000;

    fn frontier(provider: &str) -> Target {
        Target::Frontier {
            provider: provider.into(),
            model: "m".into(),
        }
    }

    #[test]
    fn deterministic_cache_is_a_cliff_at_the_ttl() {
        let model = CacheModel::Deterministic { ttl_ms: 5 * MINUTE };
        assert_eq!(model.hit_probability(0, 1000), 1.0);
        assert_eq!(model.hit_probability(5 * MINUTE - 1, 1000), 1.0);
        assert_eq!(model.hit_probability(5 * MINUTE, 1000), 0.0);
    }

    #[test]
    fn inactivity_decay_falls_off_and_respects_the_minimum_prefix() {
        let model = CacheModel::InactivityDecay {
            half_life_ms: 5 * MINUTE,
            max_ttl_ms: 10 * MINUTE,
            min_prefix_tokens: 1024,
        };
        assert!((model.hit_probability(0, 2048) - 1.0).abs() < 1e-9);
        assert!((model.hit_probability(5 * MINUTE, 2048) - 0.5).abs() < 1e-9);
        assert_eq!(model.hit_probability(10 * MINUTE, 2048), 0.0);
        // Below the provider's minimum cacheable prefix nothing is cached.
        assert_eq!(model.hit_probability(0, 512), 0.0);
    }

    #[test]
    fn an_unseen_target_has_no_warm_prefix() {
        let ledger = CacheLedger::new();
        assert_eq!(
            ledger.expected_cached_tokens(&frontier("anthropic"), 0, 5_000),
            0.0
        );
    }

    #[test]
    fn a_warm_prefix_is_capped_by_the_current_prompt_length() {
        let mut ledger = CacheLedger::new();
        let target = frontier("anthropic");
        ledger.register(
            &target,
            CacheModel::Deterministic { ttl_ms: 5 * MINUTE },
            ProviderPricing::free(),
        );
        ledger.record(&target, 0, 4_000);

        // Prompt shrank below what we last sent; only the overlap can be warm.
        assert_eq!(
            ledger.expected_cached_tokens(&target, MINUTE, 1_000),
            1_000.0
        );
        // Prompt grew; the warm part is still the earlier prefix.
        assert_eq!(
            ledger.expected_cached_tokens(&target, MINUTE, 9_000),
            4_000.0
        );
    }

    #[test]
    fn a_cold_prefix_costs_full_price_and_a_warm_one_costs_the_cached_rate() {
        let mut ledger = CacheLedger::new();
        let target = frontier("anthropic");
        let pricing = ProviderPricing {
            input_per_mtok_usd: 3.0,
            cached_input_per_mtok_usd: 0.3,
            cache_write_per_mtok_usd: 3.75,
            output_per_mtok_usd: 15.0,
        };
        ledger.register(
            &target,
            CacheModel::Deterministic { ttl_ms: 5 * MINUTE },
            pricing,
        );

        // Never seen: the whole prompt is a cache write.
        let cold = ledger.estimate_cost_usd(&target, 0, 100_000, 500);
        assert!((cold - (100_000.0 * 3.75e-6 + 500.0 * 15e-6)).abs() < 1e-9);

        // Seen a minute ago with a 100k prefix: reads at the cached rate.
        ledger.record(&target, 0, 100_000);
        let warm = ledger.estimate_cost_usd(&target, MINUTE, 100_000, 500);
        assert!((warm - (100_000.0 * 0.3e-6 + 500.0 * 15e-6)).abs() < 1e-9);
        assert!(warm < cold, "a warm prefix must be cheaper than a cold one");
    }

    #[test]
    fn invalidation_clears_warm_prefixes_after_a_compaction() {
        let mut ledger = CacheLedger::new();
        let target = frontier("anthropic");
        ledger.register(
            &target,
            CacheModel::Deterministic { ttl_ms: 5 * MINUTE },
            ProviderPricing::free(),
        );
        ledger.record(&target, 0, 50_000);
        assert!(ledger.expected_cached_tokens(&target, MINUTE, 50_000) > 0.0);

        ledger.invalidate();
        assert_eq!(ledger.expected_cached_tokens(&target, MINUTE, 50_000), 0.0);
    }

    #[test]
    fn expiry_returns_the_target_to_cold_pricing() {
        let mut ledger = CacheLedger::new();
        let target = frontier("anthropic");
        ledger.register(
            &target,
            CacheModel::Deterministic { ttl_ms: 5 * MINUTE },
            ProviderPricing::free(),
        );
        ledger.record(&target, 0, 10_000);

        assert_eq!(
            ledger.expected_cached_tokens(&target, 4 * MINUTE, 10_000),
            10_000.0
        );
        assert_eq!(
            ledger.expected_cached_tokens(&target, 6 * MINUTE, 10_000),
            0.0
        );
    }
}
