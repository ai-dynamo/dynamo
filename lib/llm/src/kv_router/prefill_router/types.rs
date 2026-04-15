// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::config::RouterConfigOverride;

use crate::protocols::common::preprocessor::{BootstrapInfo, PrefillResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ConditionalPrefillDecisionInput {
    pub prompt_tokens: usize,
    pub overlap_tokens: usize,
}

impl ConditionalPrefillDecisionInput {
    pub fn net_new_tokens(self) -> usize {
        self.prompt_tokens.saturating_sub(self.overlap_tokens)
    }
}

pub(super) trait ConditionalPrefillPolicy {
    fn is_enabled(&self) -> bool;

    fn should_bypass_remote_prefill(&self, input: ConditionalPrefillDecisionInput) -> bool;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct TokenCapConditionalPrefillPolicy {
    enabled: bool,
    max_new_tokens: usize,
}

impl TokenCapConditionalPrefillPolicy {
    pub fn from_config(config: Option<&KvRouterConfig>) -> Self {
        let Some(config) = config else {
            return Self::default();
        };

        Self {
            enabled: config.conditional_prefill_enabled,
            max_new_tokens: config.conditional_prefill_max_new_tokens,
        }
    }
}

impl Default for TokenCapConditionalPrefillPolicy {
    fn default() -> Self {
        Self {
            enabled: false,
            max_new_tokens: dynamo_kv_router::config::DEFAULT_CONDITIONAL_PREFILL_MAX_NEW_TOKENS,
        }
    }
}

impl ConditionalPrefillPolicy for TokenCapConditionalPrefillPolicy {
    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn should_bypass_remote_prefill(&self, input: ConditionalPrefillDecisionInput) -> bool {
        self.enabled && input.net_new_tokens() <= self.max_new_tokens
    }
}

/// Errors that can occur during prefill routing
#[derive(Debug, thiserror::Error)]
pub enum PrefillError {
    /// Prefill router has not been activated yet
    #[error("Prefill router not yet activated")]
    NotActivated,

    /// TODO: Separate prefill worker error from prefill router error
    /// Error during prefill execution
    #[error("Prefill execution failed: {0}")]
    PrefillError(
        String,
        #[source] Option<Box<dyn std::error::Error + Send + Sync + 'static>>,
    ),

    /// Disaggregated params not found in prefill response
    #[error("No disaggregated params in prefill response: {0}")]
    NoDisaggregatedParams(String),
}

/// Result of the prefill phase in `generate()`.
pub(super) enum PrefillOutcome {
    /// Bootstrap optimization: prefill spawned in background, bootstrap info ready
    Bootstrap(BootstrapInfo),
    /// Synchronous prefill completed with result
    Completed(PrefillResult),
}

pub(super) enum PrefillResolveDecision {
    Resolved {
        worker_id: u64,
        dp_rank: Option<u32>,
        bootstrap_info: BootstrapInfo,
    },
    Unavailable,
    NotActivated,
    NoBootstrapEndpoint,
}

pub(super) fn build_decode_router_override(
    existing_override: Option<RouterConfigOverride>,
) -> RouterConfigOverride {
    RouterConfigOverride {
        overlap_score_weight: Some(0.0),
        assume_kv_reuse: Some(false),
        track_prefill_tokens: Some(false),
        ..existing_override.unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_kv_router::config::DEFAULT_CONDITIONAL_PREFILL_MAX_NEW_TOKENS;

    #[test]
    fn token_cap_policy_is_disabled_by_default() {
        let policy = TokenCapConditionalPrefillPolicy::default();

        assert!(!policy.is_enabled());
        assert_eq!(
            policy.max_new_tokens,
            DEFAULT_CONDITIONAL_PREFILL_MAX_NEW_TOKENS
        );
        assert!(
            !policy.should_bypass_remote_prefill(ConditionalPrefillDecisionInput {
                prompt_tokens: 1,
                overlap_tokens: 0,
            })
        );
    }

    #[test]
    fn token_cap_policy_bypasses_at_or_below_cap() {
        let policy = TokenCapConditionalPrefillPolicy {
            enabled: true,
            max_new_tokens: 10,
        };

        assert!(
            policy.should_bypass_remote_prefill(ConditionalPrefillDecisionInput {
                prompt_tokens: 15,
                overlap_tokens: 5,
            })
        );
        assert!(
            !policy.should_bypass_remote_prefill(ConditionalPrefillDecisionInput {
                prompt_tokens: 16,
                overlap_tokens: 5,
            })
        );
    }

    #[test]
    fn token_cap_policy_allows_no_overlap() {
        let policy = TokenCapConditionalPrefillPolicy {
            enabled: true,
            max_new_tokens: 10,
        };

        assert!(
            policy.should_bypass_remote_prefill(ConditionalPrefillDecisionInput {
                prompt_tokens: 10,
                overlap_tokens: 0,
            })
        );
    }
}
