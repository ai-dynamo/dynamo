// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frontier providers.
//!
//! A frontier model cannot be asked what it has cached, so its candidate is
//! priced entirely from the routing ledger: what we last sent it, when, and how
//! that provider's cache expires. The executor's job is then to make the model
//! self-fulfilling — sending a stable `prompt_cache_key` and, where the
//! provider supports explicit breakpoints, `cache_control` markers at the same
//! prefix boundary each turn. Routing on a predicted cache hit and then
//! prompting in a way that defeats it is the obvious failure mode.

use async_trait::async_trait;
use roundhouse_core::routing::{CacheLedger, CacheModel, Candidate, ProviderPricing, Target};

/// A frontier model we may route to.
#[derive(Debug, Clone)]
pub struct FrontierModelSpec {
    pub provider: String,
    pub model: String,
    pub cache_model: CacheModel,
    pub pricing: ProviderPricing,
    /// Relative capability, 0.0..=1.0. Configuration, not measurement.
    pub quality_prior: f64,
    /// Latency floor before any prefill, i.e. network plus queueing.
    pub base_ttft_ms: f64,
    /// Additional TTFT per uncached prompt token.
    pub ttft_ms_per_uncached_token: f64,
}

impl FrontierModelSpec {
    pub fn target(&self) -> Target {
        Target::Frontier {
            provider: self.provider.clone(),
            model: self.model.clone(),
        }
    }
}

/// The set of frontier models available to a session.
///
/// Static because provider capability and pricing are deployment
/// configuration. Prices are supplied by the caller rather than hardcoded here:
/// baking a rate card into source guarantees it goes stale.
#[derive(Debug, Clone, Default)]
pub struct StaticFrontierCatalog {
    models: Vec<FrontierModelSpec>,
}

impl StaticFrontierCatalog {
    pub fn new(models: Vec<FrontierModelSpec>) -> Self {
        Self { models }
    }

    pub fn models(&self) -> &[FrontierModelSpec] {
        &self.models
    }

    /// Seed a ledger with this catalog's cache models and pricing.
    ///
    /// Must run before the session replays its log, so replayed dispatches are
    /// interpreted under the right TTL.
    pub fn apply_to_ledger(&self, ledger: &mut CacheLedger) {
        for spec in &self.models {
            ledger.register(&spec.target(), spec.cache_model, spec.pricing);
        }
    }

    /// Price every frontier model against the current prompt.
    pub fn quote(
        &self,
        ledger: &CacheLedger,
        now_ms: u64,
        isl_tokens: u64,
        expected_output_tokens: u64,
    ) -> Vec<Candidate> {
        self.models
            .iter()
            .map(|spec| {
                let target = spec.target();
                let cached = ledger.expected_cached_tokens(&target, now_ms, isl_tokens);
                let uncached = (isl_tokens as f64 - cached).max(0.0);
                Candidate {
                    target: target.clone(),
                    // The same axis the router reports for local workers:
                    // prompt tokens that actually have to be processed.
                    expected_prefill_tokens: uncached,
                    matched_prefix_tokens: cached as u64,
                    expected_ttft_ms: spec.base_ttft_ms
                        + uncached * spec.ttft_ms_per_uncached_token,
                    expected_cost_usd: ledger.estimate_cost_usd(
                        &target,
                        now_ms,
                        isl_tokens,
                        expected_output_tokens,
                    ),
                    quality_prior: spec.quality_prior,
                    // Provider-side load is not observable to us. Reporting a
                    // guess here would let a fabricated number gate routing.
                    load: None,
                }
            })
            .collect()
    }
}

/// One streamed chunk from a provider.
#[derive(Debug, Clone, PartialEq)]
pub enum FrontierChunk {
    OutputText(String),
    Done {
        input_tokens: u64,
        cached_input_tokens: u64,
        output_tokens: u64,
    },
}

/// What a provider was asked to do.
#[derive(Debug, Clone)]
pub struct FrontierQuote {
    pub target: Target,
    pub prompt: String,
    /// Stable per-session key. Providers use it to steer requests to the same
    /// cache node, so it must not vary turn to turn.
    pub prompt_cache_key: String,
    pub expected_output_tokens: Option<u32>,
}

#[derive(Debug, thiserror::Error)]
pub enum FrontierError {
    #[error("provider `{0}` is not configured")]
    UnknownProvider(String),
    #[error("provider call failed: {0}")]
    Upstream(String),
}

/// Executes a turn against a hosted provider.
///
/// Streaming is modelled as a whole-response `Vec` for the walking skeleton;
/// the session layer already appends deltas incrementally, so swapping this for
/// a real byte stream does not change the state machine.
#[async_trait]
pub trait FrontierClient: Send + Sync + 'static {
    async fn execute(&self, quote: &FrontierQuote) -> Result<Vec<FrontierChunk>, FrontierError>;
}

/// Deterministic [`FrontierClient`] for tests and offline runs.
pub struct EchoFrontierClient {
    reply: String,
}

impl EchoFrontierClient {
    pub fn new(reply: impl Into<String>) -> Self {
        Self {
            reply: reply.into(),
        }
    }
}

#[async_trait]
impl FrontierClient for EchoFrontierClient {
    async fn execute(&self, quote: &FrontierQuote) -> Result<Vec<FrontierChunk>, FrontierError> {
        let input_tokens = quote.prompt.len() as u64;
        Ok(vec![
            FrontierChunk::OutputText(self.reply.clone()),
            FrontierChunk::Done {
                input_tokens,
                cached_input_tokens: 0,
                output_tokens: self.reply.len() as u64,
            },
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINUTE: u64 = 60_000;

    fn catalog() -> StaticFrontierCatalog {
        StaticFrontierCatalog::new(vec![FrontierModelSpec {
            provider: "anthropic".into(),
            model: "claude".into(),
            cache_model: CacheModel::Deterministic { ttl_ms: 5 * MINUTE },
            pricing: ProviderPricing {
                input_per_mtok_usd: 3.0,
                cached_input_per_mtok_usd: 0.3,
                cache_write_per_mtok_usd: 3.75,
                output_per_mtok_usd: 15.0,
            },
            quality_prior: 0.95,
            base_ttft_ms: 350.0,
            ttft_ms_per_uncached_token: 0.002,
        }])
    }

    #[test]
    fn a_cold_frontier_prices_the_whole_prompt_as_prefill() {
        let catalog = catalog();
        let mut ledger = CacheLedger::new();
        catalog.apply_to_ledger(&mut ledger);

        let quotes = catalog.quote(&ledger, 0, 50_000, 500);
        assert_eq!(quotes.len(), 1);
        assert_eq!(quotes[0].expected_prefill_tokens, 50_000.0);
        assert_eq!(quotes[0].matched_prefix_tokens, 0);
        assert_eq!(quotes[0].load, None, "provider load is not observable");
    }

    #[test]
    fn a_warm_frontier_prices_far_less_prefill_and_far_less_money() {
        let catalog = catalog();
        let mut ledger = CacheLedger::new();
        catalog.apply_to_ledger(&mut ledger);

        let cold = catalog.quote(&ledger, 0, 50_000, 500).remove(0);

        ledger.record(&catalog.models()[0].target(), 0, 50_000);
        let warm = catalog.quote(&ledger, MINUTE, 50_000, 500).remove(0);

        assert_eq!(warm.expected_prefill_tokens, 0.0);
        assert_eq!(warm.matched_prefix_tokens, 50_000);
        assert!(warm.expected_cost_usd < cold.expected_cost_usd);
        assert!(warm.expected_ttft_ms < cold.expected_ttft_ms);
    }

    #[test]
    fn cache_expiry_returns_the_frontier_to_cold_pricing() {
        let catalog = catalog();
        let mut ledger = CacheLedger::new();
        catalog.apply_to_ledger(&mut ledger);
        ledger.record(&catalog.models()[0].target(), 0, 50_000);

        let inside = catalog.quote(&ledger, 4 * MINUTE, 50_000, 500).remove(0);
        let outside = catalog.quote(&ledger, 6 * MINUTE, 50_000, 500).remove(0);

        assert_eq!(inside.expected_prefill_tokens, 0.0);
        assert_eq!(outside.expected_prefill_tokens, 50_000.0);
    }

    #[tokio::test]
    async fn the_echo_client_reports_usage() {
        let client = EchoFrontierClient::new("hello");
        let chunks = client
            .execute(&FrontierQuote {
                target: Target::Frontier {
                    provider: "anthropic".into(),
                    model: "claude".into(),
                },
                prompt: "some prompt".into(),
                prompt_cache_key: "sess_x".into(),
                expected_output_tokens: None,
            })
            .await
            .unwrap();

        assert_eq!(chunks[0], FrontierChunk::OutputText("hello".into()));
        assert!(matches!(
            chunks[1],
            FrontierChunk::Done {
                output_tokens: 5,
                ..
            }
        ));
    }
}
